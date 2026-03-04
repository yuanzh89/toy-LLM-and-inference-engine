"""
launch.py
---------
Entry point for the ToyLLM disaggregated inference engine.

Thread topology
---------------
All components run as daemon threads inside one OS process, sharing
``queue.Queue`` objects for communication.  The three queues that cross the
conceptual node boundary are called out explicitly; in a real deployment they
would be replaced by network transports.

                             ┌─────────────────────────────────────────────┐
                             │               Prefill Node                  │
                             │                                             │
     ┌──────────────┐        │  ┌─────────────────┐   chunk_prefill_queue  │
     │  Scheduler   │        │  │ PrefillChunker  │──────────────┐        │
     │              │        │  │    Thread        │              ▼        │
     │  (main       │ [A]    │  └─────────────────┘   ┌──────────────────┐ │
     │   thread)    ├────────┼──► prefill_queue        │ PrefillForward  │ │
     │              │        │                         │    Thread       │ │
     │              │ [C]    │                         └────────┬────────┘ │
     │              ◄────────┼─────────────────────────────────┘  [B]     │
     │              │        └─────────────────────────────────────────────┘
     │              │
     │              │        ┌─────────────────────────────────────────────┐
     │              │ [C]    │               Decode Node                  │
     │              ◄────────┼──────────────────────────────┐             │
     │              │        │                              │             │
     │              │ [D]    │  ┌──────────────────────┐    │             │
     │              ├────────┼──► DecodeWorker Thread  │────┘ [B]        │
     └──────────────┘        │  └──────────────────────┘                 │
                             └─────────────────────────────────────────────┘

Queue legend
------------
[A] prefill_queue         scheduler ──► prefill node     (Sequence | STOP)
[B] decode_schedule_queue both nodes ──► scheduler       (Sequence | STOP)
[C] (same queue [B], two producers)
[D] decode_worker_queue   scheduler ──► decode node      (Sequence | STOP)

Shutdown sequence
-----------------
1. The scheduler sends STOP to ``prefill_queue`` once all prompts are
   dispatched and all finished sequences have been collected.
2. The prefill chunker thread exits and forwards STOP to the prefill
   forward thread.
3. The prefill forward thread exits and sends STOP to
   ``decode_schedule_queue``.
4. The scheduler detects STOP on ``decode_schedule_queue`` and sends STOP
   to ``decode_worker_queue``.
5. The decode worker thread exits.
6. ``launch()`` joins all threads and returns.

Usage
-----
    python launch.py
"""

from __future__ import annotations

import logging
import threading
import time
from queue import Queue

from config import ToyLLMConfig
from model_runner import PrefillRunner, DecodeRunner, STOP, _Sentinel
from scheduler import Scheduler, Tokenizer
from sequence import Sequence, SequenceStatus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(threadName)-20s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scheduler (extended to handle the disaggregated shutdown flow)
# ---------------------------------------------------------------------------

class DisaggregatedScheduler(Scheduler):
    """
    Scheduler extended for the disaggregated thread topology.

    Differences from the base :class:`~scheduler.Scheduler`:

    * Tracks the total number of sequences in flight so it knows when all
      work is done and shutdown can begin.
    * Recognises the :data:`~model_runner.STOP` sentinel on
      ``decode_schedule_queue`` and propagates it to ``decode_worker_queue``
      before exiting.
    * Exposes a ``wait_until_done()`` method that blocks until every prompt
      has been fully generated or marked ``FAILED``.

    Parameters
    ----------
    llm_config : ToyLLMConfig
        Unified model and inference configuration.
    tokenizer : Tokenizer
        Converts raw prompt strings to token ID lists.
    prompts : list[str]
        The set of prompts to process in this run.
    decode_schedule_queue : Queue
        Inbound: receives sequences from both the prefill and decode runners.
    prefill_queue : Queue
        Outbound: newly created sequences waiting for prefill.
    decode_worker_queue : Queue
        Outbound: sequences ready for the next decode step.
    """

    def __init__(
            self,
            llm_config: ToyLLMConfig,
            tokenizer: Tokenizer,
            prompts: list[str],
            decode_schedule_queue: Queue,
            prefill_queue: Queue,
            decode_worker_queue: Queue,
    ):
        super().__init__(
            llm_config,
            tokenizer,
            prompts,
            decode_schedule_queue,
            prefill_queue,
            decode_worker_queue,
        )
        # Number of sequences not yet in finished_sequences.
        self._in_flight: int = 0
        self._done_event = threading.Event()

    def run(self) -> None:
        """
        Main scheduler loop (disaggregated version).

        1. Tokenises all prompts and pushes them to ``prefill_queue``.
        2. Reads completed-step sequences from ``decode_schedule_queue``.
           * ``FAILED`` or EOS → collect in ``finished_sequences``.
           * Otherwise → re-dispatch to ``decode_worker_queue``.
        3. When all sequences are accounted for, signals ``STOP`` to the
           prefill queue and waits for the prefill runners to cascade the
           shutdown through to ``decode_schedule_queue``.
        4. On receiving ``STOP`` from ``decode_schedule_queue``, forwards
           ``STOP`` to ``decode_worker_queue`` and exits.
        """
        # ------------------------------------------------------------------ #
        # Phase 1: dispatch all prompts to prefill.                          #
        # ------------------------------------------------------------------ #
        log.info("Scheduler: dispatching %d prompt(s) to prefill.", len(self.prompts))
        for prompt in self.prompts:
            token_ids = self.tokenizer.tokenize(prompt)
            if not token_ids:
                log.warning("Tokenizer returned empty token list for prompt %r; skipping.", prompt)
                continue
            seq = Sequence(
                token_ids,
                self.llm_config.max_token_size_per_kv_cache_block,
                self.llm_config.max_sequence_length,
                self.llm_config.query_chunk_size,
            )
            self.prefill_queue.put(seq)
            self._in_flight += 1

        if self._in_flight == 0:
            log.warning("Scheduler: no valid prompts; shutting down immediately.")
            self._initiate_shutdown()
            return

        # ------------------------------------------------------------------ #
        # Phase 2: route sequences as they complete steps.                   #
        # ------------------------------------------------------------------ #
        while True:
            item = self.decode_schedule_queue.get()

            # Cascade shutdown: prefill runners have all exited.
            if isinstance(item, _Sentinel):
                log.info("Scheduler: received STOP from prefill/decode runners; "
                         "forwarding to decode worker and exiting.")
                self.decode_worker_queue.put(STOP)
                break

            seq: Sequence = item

            # Sequence failed during block allocation.
            if seq.status == SequenceStatus.FAILED:
                log.warning("Scheduler: seq %d FAILED (block allocation).", seq.seq_id)
                self.finished_sequences.append(seq)
                self._in_flight -= 1

            # Sequence has reached EOS.
            elif seq.get_last_token_id() == self.eos_token_id:
                log.info("Scheduler: seq %d finished (%d tokens).",
                         seq.seq_id, len(seq))
                seq.status = SequenceStatus.FINISHED
                seq.release()
                self.finished_sequences.append(seq)
                self._in_flight -= 1

            # Sequence still generating — dispatch for the next decode step.
            else:
                self.decode_worker_queue.put(seq)

            # When all sequences have been resolved, start the shutdown chain.
            if self._in_flight == 0:
                log.info("Scheduler: all sequences finished; initiating shutdown.")
                self._initiate_shutdown()

        self._done_event.set()

    def _initiate_shutdown(self) -> None:
        """
        Begin the graceful shutdown cascade.

        Sending STOP to ``prefill_queue`` causes the prefill chunker thread to
        exit after finishing any in-progress sequence, which in turn sends STOP
        to the prefill forward thread, which sends STOP to
        ``decode_schedule_queue``.  The scheduler picks that up and forwards
        STOP to ``decode_worker_queue``.
        """
        log.info("Scheduler: sending STOP to prefill queue.")
        self.prefill_queue.put(STOP)

    def wait_until_done(self, timeout: float | None = None) -> bool:
        """
        Block until the scheduler's ``run()`` method has returned.

        Parameters
        ----------
        timeout : float | None
            Maximum seconds to wait.  ``None`` means wait indefinitely.

        Returns
        -------
        bool
            ``True`` if the scheduler finished within *timeout*; ``False``
            if it timed out.
        """
        return self._done_event.wait(timeout=timeout)


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def launch(
        llm_config: ToyLLMConfig,
        prompts: list[str],
        tokenizer: Tokenizer | None = None,
) -> list[Sequence]:
    """
    Start the full disaggregated inference engine and run until all prompts
    are processed.

    Spawns four threads:

    ============  ===================  =======================================
    Thread name   Class / method       Role
    ============  ===================  =======================================
    Scheduler     DisaggregatedScheduler.run    Routes sequences between nodes
    PrefillChunker  PrefillRunner._run_chunker  Allocates blocks, fans chunks
    PrefillForward  PrefillRunner._run_forward  Runs prefill forward passes
    DecodeWorker    DecodeRunner._run_worker    Runs batched decode steps
    ============  ===================  =======================================

    Parameters
    ----------
    llm_config : ToyLLMConfig
        Unified model and inference configuration.
    prompts : list[str]
        Input prompts to process.
    tokenizer : Tokenizer | None
        Tokenizer instance.  If ``None``, a default ``Tokenizer`` is
        constructed from *llm_config*.

    Returns
    -------
    list[Sequence]
        Finished sequences in the order they completed.
    """
    if tokenizer is None:
        tokenizer = Tokenizer(llm_config)

    # ------------------------------------------------------------------ #
    # Create all inter-component queues.                                  #
    # ------------------------------------------------------------------ #

    # [A] scheduler ──► prefill node
    prefill_queue: Queue = Queue()

    # [B] both nodes ──► scheduler  (single shared inbound queue)
    decode_schedule_queue: Queue = Queue()

    # [D] scheduler ──► decode node
    decode_worker_queue: Queue = Queue()

    # ------------------------------------------------------------------ #
    # Construct components.                                               #
    # (Models and block managers are created inside each runner so that  #
    # weight initialisation happens in the correct thread context.)       #
    # ------------------------------------------------------------------ #
    prefill_runner = PrefillRunner(
        llm_config=llm_config,
        prefill_queue=prefill_queue,
        decode_schedule_queue=decode_schedule_queue,
    )

    decode_runner = DecodeRunner(
        llm_config=llm_config,
        decode_worker_queue=decode_worker_queue,
        decode_schedule_queue=decode_schedule_queue,
    )

    scheduler = DisaggregatedScheduler(
        llm_config=llm_config,
        tokenizer=tokenizer,
        prompts=prompts,
        decode_schedule_queue=decode_schedule_queue,
        prefill_queue=prefill_queue,
        decode_worker_queue=decode_worker_queue,
    )

    # ------------------------------------------------------------------ #
    # Start runner threads first, then the scheduler.                    #
    # (Runners must be listening before the scheduler starts producing.) #
    # ------------------------------------------------------------------ #
    log.info("Engine: starting prefill runner threads.")
    prefill_runner.start()

    log.info("Engine: starting decode runner thread.")
    decode_runner.start()

    log.info("Engine: starting scheduler thread.")
    scheduler_thread = threading.Thread(
        target=scheduler.run,
        name="Scheduler",
        daemon=True,
    )
    scheduler_thread.start()

    # ------------------------------------------------------------------ #
    # Wait for everything to finish.                                      #
    # ------------------------------------------------------------------ #
    scheduler_thread.join()
    prefill_runner.join()
    decode_runner.join()

    log.info(
        "Engine: all threads stopped.  %d sequence(s) finished, %d failed.",
        sum(1 for s in scheduler.finished_sequences if s.status == SequenceStatus.FINISHED),
        sum(1 for s in scheduler.finished_sequences if s.status == SequenceStatus.FAILED),
    )

    return scheduler.finished_sequences


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # Example configuration — adjust to match your ToyLLMConfig fields.  #
    # ------------------------------------------------------------------ #
    config = ToyLLMConfig(
        # Model architecture
        vocab_size=32000,
        d_model=512,
        d_ff=1024,
        num_transformer_layers=4,
        num_query_heads=8,
        num_kv_heads=2,
        dropout=0.0,
        # KV-cache / paging
        max_block_size=64,  # max blocks in the pool
        max_token_size_per_kv_cache_block=16,
        max_sequence_length=256,
        query_chunk_size=16,
        # Decode batching
        decode_max_batch_size=8,
    )

    sample_prompts = [
        "The capital of France is",
        "Once upon a time in a land far away",
        "The quick brown fox",
    ]

    start = time.perf_counter()
    finished = launch(config, sample_prompts)
    elapsed = time.perf_counter() - start

    log.info("Done in %.2fs.  Generated sequences:", elapsed)
    for seq in finished:
        log.info("  seq %d  status=%-10s  tokens=%d",
                 seq.seq_id, seq.status.name, len(seq))
