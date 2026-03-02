from __future__ import annotations

import queue
from multiprocessing import Queue

from block_manager import BlockManager
from config import ToyLLMConfig
from sequence import Sequence, SequenceStatus
from layer.toy_llm_model import ToyLLMModel


class ToyLLMModelRunner:
    """
    Wraps a :class:`~toy_llm_model.ToyLLMModel` and its configuration, providing
    high-level entry points for running the model in **prefill** or **decode** mode.

    The runner owns a single :class:`~block_manager.BlockManager` shared between
    the model and the scheduling loop.  It reads incoming :class:`~sequence.Sequence`
    objects from a :class:`~multiprocessing.Queue` and drives the model forward.

    Prefill mode
    ------------
    The model computes KV caches from scratch (or from cached prefixes) and
    populates the block manager.  Each call to ``prefill_sequence`` processes one
    sequence end-to-end.

    Decode mode
    -----------
    The model reads KV caches that were populated during prefill (potentially on a
    different node in a disaggregated setup) and generates one token per step.
    Sequences are batched up to ``llm_config.decode_max_batch_size``.

    Attributes
    ----------
    llm_config : ToyLLMConfig
        Unified model/inference configuration.
    queue : Queue
        Inter-process queue delivering :class:`~sequence.Sequence` objects (or the
        sentinel ``ToyLLMModelRunner.SENTINEL`` to signal shutdown).
    model : ToyLLMModel
        The underlying language model.
    block_manager : BlockManager
        Manages the pool of paged KV-cache blocks.
    """

    SENTINEL = object()

    def __init__(self, llm_config: ToyLLMConfig, queue: Queue):
        self.llm_config = llm_config
        self.queue = queue

        self.block_manager = BlockManager(
            llm_config.max_block_size,
            llm_config.max_token_size_per_kv_cache_block,
            llm_config.num_transformer_layers,
        )
        self.model = ToyLLMModel(llm_config, self.block_manager)

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def run_in_prefill_mode(self) -> None:
        """
        Entry point for the prefill worker loop.

        Continuously pops sequences from the queue, runs ``prefill_sequence`` on
        each, and exits when :attr:`SENTINEL` is received.
        """
        while True:
            sequence = self.queue.get()
            if sequence is ToyLLMModelRunner.SENTINEL:
                break
            self.prefill_sequence(sequence)

    def prefill_sequence(self, seq: Sequence) -> None:
        """
        Allocate KV-cache blocks and run the prefill forward pass for *seq*.

        If the block manager cannot allocate enough blocks (pool exhausted), the
        sequence is marked :attr:`~sequence.SequenceStatus.FAILED` and released.
        Otherwise the model is run to populate the KV cache and generate the first
        token, and the sequence status is set to
        :attr:`~sequence.SequenceStatus.RUNNING`.

        Parameters
        ----------
        seq : Sequence
            The sequence to prefill.  Token IDs must already be populated.
        """

        if not self.block_manager.allocate_blocks(seq):
            # Not enough KV-cache space — drop the sequence rather than retry.
            seq.status = SequenceStatus.FAILED
            seq.release()
            return

        seq.status = SequenceStatus.RUNNING

        # Run the prefill forward pass one query chunk at a time.
        token_ids_in_chunks = seq.token_ids_in_chunks()
        for query_chunk_idx in range(len(token_ids_in_chunks)):
            self.model(seq, query_chunk_idx)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def run_in_decode_mode(self) -> None:
        """
        Entry point for the decode worker loop.

        Batches up to ``llm_config.decode_max_batch_size`` sequences from the
        queue per step and runs a batched decode pass.  Exits when
        :attr:`SENTINEL` is received.

        Note: KV caches are assumed to have been transferred from the prefill
        node and already populated in the block manager before this loop starts.
        """
        while True:
            batch: list[Sequence] = []
            stop_listening = False

            for _ in range(self.llm_config.decode_max_batch_size):
                try:
                    sequence = self.queue.get_nowait()
                    if sequence is ToyLLMModelRunner.SENTINEL:
                        stop_listening = True
                        break
                    batch.append(sequence)
                except queue.Empty:
                    break

            if batch:
                # Run one decode step for the entire batch.
                # Each sequence gets exactly one new token appended.
                for seq in batch:
                    # During decode there is only one chunk (the new token).
                    self.model(seq, query_chunk_idx=0)

            if stop_listening:
                break