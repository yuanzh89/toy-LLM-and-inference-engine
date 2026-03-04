import queue
from multiprocessing import Queue

from block_manager import BlockManager
from config import ToyLLMConfig
from layer.toy_llm_model import ToyLLMModel
from sequence import Sequence, SequenceStatus


class ToyLLMModelRunner:
    """
    Orchestrates the prefill and decode forward passes for the :class:`ToyLLMModel`.

    The runner owns the :class:`~block_manager.BlockManager` and the model weights.
    It communicates with the :class:`~scheduler.Scheduler` through three queues:

    * ``prefill_queue`` – incoming sequences waiting for prefill.
    * ``decode_schedule_queue`` – sequences returned to the scheduler after a
      decode step (scheduler decides whether to continue or finish).
    * ``decode_worker_queue`` – sequences dispatched by the scheduler for the
      next decode step.

    Prefill is chunked: ``run_chunk_sequence_process`` reads from ``prefill_queue``
    and fans each sequence out into ``chunk_prefill_queue`` (one entry per query
    chunk).  ``run_prefill_a_chunk_process`` consumes ``chunk_prefill_queue`` and
    runs the model forward pass one chunk at a time.

    Attributes
    ----------
    llm_config : ToyLLMConfig
        Unified model and inference configuration.
    block_manager : BlockManager
        Manages paged physical KV-cache blocks.
    model : ToyLLMModel
        The decoder-only language model.
    """

    SENTINEL = object()

    def __init__(
            self,
            llm_config: ToyLLMConfig,
            prefill_queue: Queue[Sequence],
            decode_schedule_queue: Queue[Sequence],
            decode_worker_queue: Queue[Sequence],
    ):
        self.llm_config = llm_config
        self.prefill_queue = prefill_queue
        self.chunk_prefill_queue: Queue = Queue()
        self.decode_schedule_queue = decode_schedule_queue
        self.decode_worker_queue = decode_worker_queue

        self.block_manager = BlockManager(
            llm_config.max_block_size,
            llm_config.max_token_size_per_kv_cache_block,
            llm_config.num_transformer_layers,
        )
        self.model = ToyLLMModel(llm_config, self.block_manager)

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def run_chunk_sequence_process(self) -> None:
        """
        Consumer loop that reads sequences from ``prefill_queue`` and fans them
        out into ``chunk_prefill_queue`` as (sequence, chunk_index) pairs.

        Stops when it receives the sentinel value.
        """
        while True:
            sequence = self.prefill_queue.get()
            if sequence is ToyLLMModelRunner.SENTINEL:
                break
            self.chunk_sequence(sequence)

    def run_prefill_a_chunk_process(self) -> None:
        """
        Consumer loop that reads ``(sequence, query_chunk_idx)`` pairs from
        ``chunk_prefill_queue`` and runs the prefill forward pass for each chunk.

        Stops when it receives the sentinel pair.
        """
        while True:
            sequence, query_chunk_idx = self.chunk_prefill_queue.get()
            if sequence is ToyLLMModelRunner.SENTINEL and query_chunk_idx is ToyLLMModelRunner.SENTINEL:
                break
            self.model([sequence], [query_chunk_idx], is_prefill=True)

    def chunk_sequence(self, seq: Sequence) -> None:
        """
        Allocate KV-cache blocks for *seq* and enqueue its query chunks.

        If block allocation fails (pool exhausted), the sequence is marked
        ``FAILED`` and its reference counts are released.

        Parameters
        ----------
        seq : Sequence
            The sequence to prepare for chunked prefill.
        """
        if not self.block_manager.allocate_blocks(seq):
            # Not enough KV-cache space — drop the sequence rather than retry.
            seq.status = SequenceStatus.FAILED
            seq.release()
            return

        # Enqueue one (sequence, chunk_index) pair per query chunk.
        token_ids_in_chunks = seq.token_ids_in_chunks()
        for query_chunk_idx in range(len(token_ids_in_chunks)):
            self.chunk_prefill_queue.put((seq, query_chunk_idx))

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def run_in_decode_mode(self) -> None:
        """
        Decode loop: drains up to ``decode_max_batch_size`` sequences from
        ``decode_worker_queue``, runs one decode step for the batch, seals any
        newly filled KV-cache blocks, and returns each sequence to
        ``decode_schedule_queue`` for the scheduler to decide the next action.

        Stops when it receives the sentinel value.
        """
        while True:
            batch: list[Sequence] = []
            stop_listening = False

            for _ in range(self.llm_config.decode_max_batch_size):
                try:
                    sequence = self.decode_worker_queue.get_nowait()
                    if sequence is ToyLLMModelRunner.SENTINEL:
                        stop_listening = True
                        break
                    batch.append(sequence)
                except queue.Empty:
                    break

            if batch:
                # Run one decode step for the entire batch.
                self.model(batch, is_prefill=False)

                for sequence in batch:
                    # Seal any KV-cache blocks that became full during this decode step.
                    self.block_manager.seal_full_decode_blocks(sequence)
                    # Return the sequence to the scheduler for the next step.
                    self.decode_schedule_queue.put(sequence)

            if stop_listening:
                break