from __future__ import annotations

import math
from copy import copy
from enum import IntEnum
from itertools import count
from typing import TYPE_CHECKING

import torch

from engine import block_manager

if TYPE_CHECKING:
    from kv_cache_block import Block


class SequenceStatus(IntEnum):
    """
    Lifecycle state of a :class:`Sequence`.

    Attributes
    ----------
    FINISHED : int
        The sequence has completed generation (EOS reached or max length hit).
    RUNNING : int
        The sequence is actively being processed (prefill or decode step in progress).
    WAITING : int
        The sequence is queued and waiting for KV-cache blocks to be allocated.
    FAILED : int
        Allocation or processing failed; the sequence has been dropped.
    """
    START = 1
    INITIALIZED = 2
    PREFILL_PENDING = 3
    DECODE_PENDING = 4
    FINISHED = 5
    FAILED = 6

class Sequence:
    """
    Represents a single request/sequence being processed by the inference engine.

    A :class:`Sequence` owns a contiguous list of token IDs that grows during
    decoding.  It also tracks which physical KV-cache :class:`~kv_cache_block.Block`
    objects hold its cached key/value tensors.

    Chunked processing
    ------------------
    The token IDs are partitioned into fixed-size *query chunks* (each of length
    ``query_chunk_size``) for the attention layers.  Intermediate activations are
    stored in ``chunked_activations`` — one tensor per chunk — so that different
    chunks can be computed independently during chunked prefill.

    Attributes
    ----------
    seq_id : int
        Unique, auto-incremented sequence identifier.
    status : SequenceStatus
        Current lifecycle state of the sequence.
    token_ids : list[int]
        All token IDs seen so far (prompt + generated tokens).
    num_prompt_tokens : int
        Number of tokens in the original prompt (fixed after construction).
    max_token_size_per_kv_cache_block : int
        Block granularity; used when splitting token IDs into KV-cache chunks.
    max_sequence_length : int
        Hard cap on the total number of tokens (prompt + decode).
    query_chunk_size : int
        Number of tokens processed per attention query chunk.
    prefill_chunked_activations : list[torch.Tensor | None]
        Intermediate hidden-state tensors, one per query chunk.
        Shape of each entry: ``[batch_size, chunk_seq_len, d_model]``.
        Populated by the embedding layer and updated by each transformer block.
    kv_cache_blocks : list[Block | None]
        Ordered list of KV-cache blocks aligned with ``token_ids_in_chunks()``.
        Populated by :class:`~block_manager.BlockManager`; entries may be ``None``
        temporarily between ``reset_kv_cache_blocks`` and ``update_kv_cache_blocks``.
    kv_cache_blocks_initialized : bool
        ``True`` once ``update_kv_cache_blocks`` has been called successfully.
    """

    counter = count()

    def __init__(
            self,
            token_ids: list[int],
            max_token_size_per_kv_cache_block: int,
            max_sequence_length: int,
            query_chunk_size: int,
    ):
        self.seq_id: int = next(Sequence.counter)
        self.status: SequenceStatus = SequenceStatus.START
        self.token_ids: list[int] = copy(token_ids)
        self.num_prompt_tokens: int = len(token_ids)
        self.max_token_size_per_kv_cache_block = max_token_size_per_kv_cache_block
        self.max_sequence_length = max_sequence_length
        self.query_chunk_size = query_chunk_size
        self._num_query_chunks = math.ceil(len(token_ids) / self.query_chunk_size)

        # This is query chunk wise activations for prefill only.
        # One activation tensor per query chunk; filled by the embedding layer and
        # updated in-place by each transformer block.  Shape: [B, chunk_len, d_model].
        self.prefill_chunked_activations: list[torch.Tensor | None] = [None] * self._num_query_chunks

        # Decode activation with seq_len == 1
        # Initialized by embedding layer
        # Tensor shape for the current sequence: [batch_size == 1, seq_len == 1, d_model]
        self.decode_activations: torch.Tensor | None = None

        # Populated by BlockManager; one block per KV-cache chunk.
        self.kv_cache_blocks: list[Block | None] = []
        self.kv_cache_blocks_initialized: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_decode_tokens(self) -> int:
        """Number of tokens generated so far beyond the original prompt."""
        return len(self.token_ids) - self.num_prompt_tokens

    def __len__(self) -> int:
        """Total number of tokens in this sequence (prompt + generated)."""
        return len(self.token_ids)

    # ------------------------------------------------------------------
    # Activation helpers
    # ------------------------------------------------------------------

    def get_query_chunk_activations(self, query_chunk_idx: int) -> torch.Tensor | None:
        """
        Return the activation tensor for *query_chunk_idx*.

        Parameters
        ----------
        query_chunk_idx : int
            Zero-based index into ``prefill_chunked_activations``.

        Returns
        -------
        torch.Tensor | None
            The activation tensor for the requested chunk, or ``None`` if the
            embedding layer has not yet been run for that chunk.
        """
        assert 0 <= query_chunk_idx < self._num_query_chunks
        return self.prefill_chunked_activations[query_chunk_idx]

    def get_full_activations(self) -> torch.Tensor:
        """
        Concatenate all query-chunk activation tensors along the sequence dimension.

        Returns
        -------
        torch.Tensor
            Shape ``[batch_size, total_seq_len, d_model]``.

        Raises
        ------
        ValueError
            If any chunk activation is still ``None`` (embedding not yet run).
        """
        if any(a is None for a in self.prefill_chunked_activations):
            raise ValueError(
                "Not all prefill_chunked_activations have been populated; "
                "call the embedding layer before get_full_activations()."
            )
        return torch.cat(self.prefill_chunked_activations, dim=1)

    def get_decode_activations(self) -> torch.Tensor | None:
        return self.decode_activations

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def append_token(self, token_id: int) -> None:
        """
        Append a newly decoded token to the sequence.

        Parameters
        ----------
        token_id : int
            The token ID produced by the model at the current decode step.
        """
        self.token_ids.append(token_id)

    def token_ids_in_chunks(self) -> list[list[int]]:
        """
        Partition ``token_ids`` into fixed-size chunks aligned to KV-cache blocks.

        All chunks except possibly the last will have exactly
        ``max_token_size_per_kv_cache_block`` elements.  The final chunk may be
        shorter if the total token count is not evenly divisible.

        Returns
        -------
        list[list[int]]
            Ordered list of token-ID chunks whose concatenation equals
            ``self.token_ids``.
        """
        return [
            self.token_ids[i: i + self.max_token_size_per_kv_cache_block]
            for i in range(0, len(self.token_ids), self.max_token_size_per_kv_cache_block)
        ]

    def get_last_token_id(self) -> int:
        return self.token_ids[-1]

    # ------------------------------------------------------------------
    # KV-cache block management
    # ------------------------------------------------------------------

    def reset_kv_cache_blocks(self, num_chunks: int) -> None:
        """
        Clear the KV-cache block list and pre-allocate ``num_chunks`` ``None`` slots.

        Called by :class:`~block_manager.BlockManager` at the start of
        ``allocate_blocks`` before performing prefix matching and new-block
        allocation.

        Parameters
        ----------
        num_chunks : int
            Expected number of KV-cache blocks needed for the current token list,
            i.e. ``ceil(len(token_ids) / max_token_size_per_kv_cache_block)``.
        """
        self.kv_cache_blocks = [None] * num_chunks
        self.kv_cache_blocks_initialized = False

    def update_kv_cache_blocks(self, kv_cache_blocks: list[Block]) -> None:
        """
        Replace the KV-cache block list with the fully resolved set of blocks.

        Called by :class:`~block_manager.BlockManager` after prefix matching and
        new-block allocation are both complete.  The length of *kv_cache_blocks*
        must equal the number of token-ID chunks for the current token list.

        Parameters
        ----------
        kv_cache_blocks : list[Block]
            Ordered list of :class:`~kv_cache_block.Block` objects; one per chunk.
            Order must match the order of ``token_ids_in_chunks()`` so that block
            ``i`` holds the KV cache for chunk ``i``.
        """
        expected = math.ceil(len(self.token_ids) / self.max_token_size_per_kv_cache_block)
        assert len(kv_cache_blocks) == expected, (
            f"Expected {expected} KV-cache blocks for {len(self.token_ids)} tokens "
            f"(block size {self.max_token_size_per_kv_cache_block}), "
            f"got {len(kv_cache_blocks)}."
        )
        self.kv_cache_blocks = copy(kv_cache_blocks)
        self.kv_cache_blocks_initialized = True

    def append_kv_cache(self, layer_id: int, k_tensor: torch.Tensor, v_tensor: torch.Tensor) -> None:
        last_block = self.kv_cache_blocks[-1]
        if not last_block.is_full() and last_block.can_append():
            last_block.decode_append_token_ids_and_kv_cache(layer_id, [self.get_last_token_id()],k_tensor, v_tensor)
            return

        # TODO: Allocate new block for appending.
        pass

    def release(self) -> None:
        """
        Release all KV-cache block references held by this sequence.

        Decrements the ``ref_count`` of every non-``None`` block so that the
        :class:`~block_manager.BlockManager` knows these blocks are no longer
        pinned and may be eligible for eviction.

        Must be called by the scheduler when the sequence finishes or is dropped.
        """
        for block in self.kv_cache_blocks:
            if block is not None:
                block.dec_ref_count()