"""
model_runner.py
---------------
Prefill and decode runners for the ToyLLM inference engine.

Process/thread topology
-----------------------
In a disaggregated deployment, prefill and decode live on separate nodes.
This module exposes two independent runner classes — ``PrefillRunner`` and
``DecodeRunner`` — each of which is started in its own thread by
``launch.py``.

                  ┌──────────────────────────────────┐
                  │          Prefill Node             │
                  │                                   │
   prefill_queue ─►  PrefillChunkerThread             │
                  │        │ chunk_prefill_queue       │
                  │        ▼                           │
                  │  PrefillForwardThread  ────────────┼──► decode_schedule_queue
                  └──────────────────────────────────┘
                                                             │
                  ┌──────────────────────────────────┐      ▼
                  │          Decode Node              │  Scheduler
                  │                                   │      │
  decode_worker_queue ◄──────────────────────────────────────┘
                  │        │
                  │        ▼
                  │  DecodeWorkerThread  ─────────────┼──► decode_schedule_queue
                  └──────────────────────────────────┘

NOTE: In production the three cross-node queues (prefill_queue,
decode_schedule_queue, decode_worker_queue) would be replaced by network
transports (gRPC, ZMQ, NCCL for KV tensors, etc.).  For this toy the same
``queue.Queue`` objects are shared across threads in one process.
"""

from __future__ import annotations

import queue
import threading
from queue import Queue

from toy_llm_model import ToyLLMModel

from block_manager import BlockManager
from config import ToyLLMConfig
from sequence import Sequence, SequenceStatus


# ---------------------------------------------------------------------------
# Sentinel types
# ---------------------------------------------------------------------------

class _Sentinel:
    """
    Unique stop-signal object placed on a queue to trigger graceful shutdown.

    Using a dedicated class (rather than ``None`` or a bare ``object()``)
    makes ``isinstance`` checks unambiguous even when queues are typed and
    might legitimately carry ``None`` values.
    """

    def __repr__(self) -> str:
        return "<SENTINEL>"


class _PrefillDone:
    """
    Signals that all query-chunk forward passes for a sequence have finished.

    Placed on ``_chunk_prefill_queue`` by the chunker thread after it has
    enqueued every ``(seq, chunk_idx)`` pair for a sequence.  The forward
    thread handles this by transitioning the sequence to ``DECODE_PENDING``
    and forwarding it to ``decode_schedule_queue``.
    """
    __slots__ = ("seq",)

    def __init__(self, seq: Sequence):
        self.seq = seq

    def __repr__(self) -> str:
        return f"<PrefillDone seq_id={self.seq.seq_id}>"


# Module-level singleton used as a stop signal on all queues.
STOP = _Sentinel()


# ---------------------------------------------------------------------------
# Prefill runner
# ---------------------------------------------------------------------------

class PrefillRunner:
    """
    Owns the prefill-side model weights and KV-cache block manager.

    Internally runs two threads:

    * **Chunker thread** – reads :class:`~sequence.Sequence` objects from
      ``prefill_queue``, allocates KV-cache blocks, and fans out
      ``(seq, chunk_idx)`` work items onto the internal
      ``_chunk_prefill_queue``.  After all chunks for a sequence have been
      enqueued it appends a :class:`_PrefillDone` marker.

    * **Forward thread** – reads work items from ``_chunk_prefill_queue``,
      runs the model forward pass for each chunk, and — upon receiving a
      :class:`_PrefillDone` marker — transitions the sequence to
      ``DECODE_PENDING`` and sends it to ``decode_schedule_queue`` so the
      scheduler can dispatch it for decoding.

    Parameters
    ----------
    llm_config : ToyLLMConfig
        Unified model and inference configuration.
    prefill_queue : Queue
        Inbound queue.  Carries :class:`~sequence.Sequence` objects tokenised
        by the scheduler, plus a :data:`STOP` sentinel to trigger shutdown.
    decode_schedule_queue : Queue
        Outbound queue shared with the scheduler.  Completed sequences are
        placed here after all prompt chunks have been processed.
    """

    def __init__(
            self,
            llm_config: ToyLLMConfig,
            prefill_queue: Queue,
            decode_schedule_queue: Queue,
    ):
        self.llm_config = llm_config
        self.prefill_queue = prefill_queue
        self.decode_schedule_queue = decode_schedule_queue

        # Internal pipeline queue between the two prefill threads.
        # Carries (Sequence, int) chunk work items, _PrefillDone markers,
        # and the _Sentinel to shut down the forward thread.
        self._chunk_prefill_queue: Queue = Queue()

        # Both prefill threads share the same block pool and model weights,
        # exactly as they would on a single prefill node in production.
        self.block_manager = BlockManager(
            llm_config.max_block_size,
            llm_config.max_token_size_per_kv_cache_block,
            llm_config.num_transformer_layers,
        )
        self.model = ToyLLMModel(llm_config, self.block_manager)

        self._chunker_thread: threading.Thread | None = None
        self._forward_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn and start both prefill worker threads."""
        self._chunker_thread = threading.Thread(
            target=self._run_chunker,
            name="PrefillChunker",
            daemon=True,
        )
        self._forward_thread = threading.Thread(
            target=self._run_forward,
            name="PrefillForward",
            daemon=True,
        )
        self._chunker_thread.start()
        self._forward_thread.start()

    def join(self) -> None:
        """Block until both prefill threads have exited."""
        if self._chunker_thread:
            self._chunker_thread.join()
        if self._forward_thread:
            self._forward_thread.join()

    # ------------------------------------------------------------------
    # Chunker thread
    # ------------------------------------------------------------------

    def _run_chunker(self) -> None:
        """
        Chunker thread main loop.

        Reads sequences from ``prefill_queue`` and fans each one out into
        ``_chunk_prefill_queue`` as ``(seq, chunk_idx)`` work items, followed
        by a :class:`_PrefillDone` marker.

        On :data:`STOP`, propagates a stop signal to the forward thread before
        exiting.
        """
        while True:
            item = self.prefill_queue.get()
            if isinstance(item, _Sentinel):
                # Forward thread must also be told to stop.
                self._chunk_prefill_queue.put(STOP)
                break

            self._chunk_sequence(item)

    def _chunk_sequence(self, seq: Sequence) -> None:
        """
        Allocate KV-cache blocks for *seq* and enqueue its query chunks.

        If block allocation fails (pool exhausted), the sequence is marked
        ``FAILED``, its reference counts are released, and it is forwarded
        directly to ``decode_schedule_queue`` so the scheduler observes the
        failure rather than waiting indefinitely.

        Parameters
        ----------
        seq : Sequence
            The sequence to prepare for chunked prefill.
        """
        if not self.block_manager.allocate_blocks(seq):
            seq.status = SequenceStatus.FAILED
            seq.release()
            # Notify the scheduler so it does not wait for this sequence.
            self.decode_schedule_queue.put(seq)
            return

        token_ids_in_chunks = seq.token_ids_in_chunks()
        for chunk_idx in range(len(token_ids_in_chunks)):
            self._chunk_prefill_queue.put((seq, chunk_idx))

        # Mark the end of this sequence's chunk stream.
        self._chunk_prefill_queue.put(_PrefillDone(seq))

    # ------------------------------------------------------------------
    # Forward thread
    # ------------------------------------------------------------------

    def _run_forward(self) -> None:
        """
        Prefill forward thread main loop.

        Processes three kinds of items from ``_chunk_prefill_queue``:

        * ``(Sequence, int)`` – run one model forward pass for the chunk.
        * :class:`_PrefillDone` – all chunks done; transition to
          ``DECODE_PENDING`` and send to ``decode_schedule_queue``.
        * :data:`STOP` – exit the loop.
        """
        while True:
            item = self._chunk_prefill_queue.get()

            if isinstance(item, _Sentinel):
                break

            if isinstance(item, _PrefillDone):
                seq = item.seq
                seq.status = SequenceStatus.DECODE_PENDING
                self.decode_schedule_queue.put(seq)
                continue

            seq, chunk_idx = item
            self.model([seq], [chunk_idx], is_prefill=True)


# ---------------------------------------------------------------------------
# Decode runner
# ---------------------------------------------------------------------------

class DecodeRunner:
    """
    Owns the decode-side model weights and KV-cache block manager.

    Runs a single worker thread that drains up to ``decode_max_batch_size``
    sequences from ``decode_worker_queue``, executes one batched decode step,
    seals any newly filled KV-cache blocks, and returns each sequence to
    ``decode_schedule_queue`` for the scheduler to decide the next action.

    NOTE on disaggregation: in production the decode node receives only
    sequence metadata and the pre-computed KV tensors from the prefill node
    via RDMA / NCCL.  For this toy both runners share the same Python objects,
    so no explicit KV-tensor transfer is implemented.

    Parameters
    ----------
    llm_config : ToyLLMConfig
        Unified model and inference configuration.
    decode_worker_queue : Queue
        Inbound queue.  Carries :class:`~sequence.Sequence` objects ready for
        their next decode step, plus a :data:`STOP` sentinel to trigger
        shutdown.
    decode_schedule_queue : Queue
        Outbound queue shared with the scheduler.  Sequences are returned here
        after each decode step.
    """

    def __init__(
            self,
            llm_config: ToyLLMConfig,
            decode_worker_queue: Queue,
            decode_schedule_queue: Queue,
    ):
        self.llm_config = llm_config
        self.decode_worker_queue = decode_worker_queue
        self.decode_schedule_queue = decode_schedule_queue

        # In a real disaggregated setup this would be a separate block pool
        # on the decode node, populated by KV-transfer from the prefill node.
        self.block_manager = BlockManager(
            llm_config.max_block_size,
            llm_config.max_token_size_per_kv_cache_block,
            llm_config.num_transformer_layers,
        )
        self.model = ToyLLMModel(llm_config, self.block_manager)

        self._worker_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn and start the decode worker thread."""
        self._worker_thread = threading.Thread(
            target=self._run_worker,
            name="DecodeWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def join(self) -> None:
        """Block until the decode worker thread has exited."""
        if self._worker_thread:
            self._worker_thread.join()

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _run_worker(self) -> None:
        """
        Decode worker thread main loop.

        On each iteration:

        1. Block on ``decode_worker_queue`` until at least one sequence
           arrives (avoids busy-waiting).
        2. Greedily collect additional sequences up to
           ``decode_max_batch_size``.
        3. Run one batched decode forward pass.
        4. Seal any KV-cache blocks that became full during this step.
        5. Return each sequence to ``decode_schedule_queue``.

        Exits when a :data:`STOP` sentinel is dequeued.
        """
        while True:
            batch: list[Sequence] = []
            stop = False

            # Block on the first item to avoid spinning on an empty queue.
            first = self.decode_worker_queue.get()
            if isinstance(first, _Sentinel):
                break
            batch.append(first)

            # Greedily collect more items up to the batch-size cap.
            for _ in range(self.llm_config.decode_max_batch_size - 1):
                try:
                    item = self.decode_worker_queue.get_nowait()
                    if isinstance(item, _Sentinel):
                        stop = True
                        break
                    batch.append(item)
                except queue.Empty:
                    break

            # One decode step across the whole batch.
            self.model(batch, is_prefill=False)

            for seq in batch:
                # Seal blocks that filled up during this step so they are
                # available as cached prefixes for future sequences.
                self.block_manager.seal_full_decode_blocks(seq)
                self.decode_schedule_queue.put(seq)

            if stop:
                break
