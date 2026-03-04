from __future__ import annotations

import math
from copy import copy
from enum import IntEnum
from itertools import count
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from block_manager import BlockManager
    from kv_cache_block import Block


class SequenceStatus(IntEnum):
    """
    Lifecycle state of a :class:`Sequence`.

    Attributes
    ----------
    PREFILL_PENDING : int
        The sequence is waiting for or undergoing its prefill pass (prompt ingestion).
    DECODE_PENDING : int
        The sequence has completed prefill and is ready for (or currently in) a
        decode step.
    FINISHED : int
        The sequence has completed generation (EOS reached or max length hit).
    FAILED : int
        Allocation or processing failed; the sequence has been dropped.
    """
    PREFILL_PENDING = 1
    DECODE_PENDING = 2
    FINISHED = 3
    FAILED = 4


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
    stored in ``prefill_chunked_activations`` — one tensor per chunk — so that
    different chunks can be computed independently during chunked prefill.

    KV-cache block layout
    ---------------------
    Token IDs are also partitioned into *KV-cache blocks* (each of length
    ``max_token_size_per_kv_cache_block``).  The last block is partial until it
    fills up during decoding.  Full blocks are eventually sealed into the prefix-
    caching trie by :meth:`~block_manager.BlockManager.seal_full_decode_blocks`.

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
        Number of tokens processed per attention query chunk during prefill.
    prefill_chunked_activations : list[torch.Tensor | None]
        Intermediate hidden-state tensors, one per query chunk.
        Shape of each entry: ``[batch_size, chunk_seq_len, d_model]``.
        Populated by the embedding layer and updated by each transformer block.
    decode_activations : torch.Tensor | None
        Hidden-state tensor for the current decode step.
        Shape: ``[1, 1, d_model]`` (batch_size=1, seq_len=1).
        Initialised by the embedding layer at the start of each decode step.
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
        self.status: SequenceStatus = SequenceStatus.PREFILL_PENDING
        self.token_ids: list[int] = copy(token_ids)
        self.num_prompt_tokens: int = len(token_ids)
        self.max_token_size_per_kv_cache_block = max_token_size_per_kv_cache_block
        self.max_sequence_length = max_sequence_length
        self.query_chunk_size = query_chunk_size
        self._num_query_chunks = math.ceil(len(token_ids) / self.query_chunk_size)

        # One activation tensor per query chunk; filled by the embedding layer and
        # updated in-place by each transformer block.  Shape: [1, chunk_len, d_model].
        self.prefill_chunked_activations: list[torch.Tensor | None] = [None] * self._num_query_chunks

        # Decode activation for the current auto-regressive step.
        # Shape: [1, 1, d_model].  Initialised by the embedding layer.
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
        """Return the cached activation tensor for the given query chunk index."""
        assert 0 <= query_chunk_idx < self._num_query_chunks
        return self.prefill_chunked_activations[query_chunk_idx]

    def get_full_activations(self) -> torch.Tensor:
        """
        Concatenate all per-chunk activations into a single ``[1, seq_len, d_model]``
        tensor.

        Raises ``ValueError`` if any chunk has not yet been populated.
        """
        if any(a is None for a in self.prefill_chunked_activations):
            raise ValueError(
                "Not all prefill_chunked_activations have been populated; "
                "call the embedding layer before get_full_activations()."
            )
        return torch.cat(self.prefill_chunked_activations, dim=1)

    def get_decode_activations(self) -> torch.Tensor | None:
        """Return the current decode activation tensor, or ``None`` if not set."""
        return self.decode_activations

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def append_token(self, token_id: int) -> None:
        """Append a newly generated token ID to the sequence."""
        self.token_ids.append(token_id)

    def token_ids_in_chunks(self) -> list[list[int]]:
        """
        Partition ``self.token_ids`` into fixed-size KV-cache block chunks.

        The last chunk may be shorter than ``max_token_size_per_kv_cache_block``
        if the total token count is not evenly divisible.

        Returns
        -------
        list[list[int]]
            Non-overlapping, ordered chunks of token IDs.
        """
        return [
            self.token_ids[i: i + self.max_token_size_per_kv_cache_block]
            for i in range(0, len(self.token_ids), self.max_token_size_per_kv_cache_block)
        ]

    def get_last_token_id(self) -> int:
        """Return the most recently added token ID."""
        return self.token_ids[-1]

    # ------------------------------------------------------------------
    # KV-cache block management
    # ------------------------------------------------------------------

    def reset_kv_cache_blocks(self, num_chunks: int) -> None:
        """Reset the KV-cache block list to *num_chunks* ``None`` entries."""
        self.kv_cache_blocks = [None] * num_chunks
        self.kv_cache_blocks_initialized = False

    def update_kv_cache_blocks(self, kv_cache_blocks: list[Block]) -> None:
        """
        Replace the KV-cache block list with *kv_cache_blocks*.

        Validates that the number of blocks matches the expected count for the
        current token sequence.

        Parameters
        ----------
        kv_cache_blocks : list[Block]
            Ordered list of blocks, one per KV-cache chunk.
        """
        expected = math.ceil(len(self.token_ids) / self.max_token_size_per_kv_cache_block)
        assert len(kv_cache_blocks) == expected, (
            f"Expected {expected} KV-cache blocks for {len(self.token_ids)} tokens "
            f"(block size {self.max_token_size_per_kv_cache_block}), "
            f"got {len(kv_cache_blocks)}."
        )
        self.kv_cache_blocks = copy(kv_cache_blocks)
        self.kv_cache_blocks_initialized = True

    def append_kv_cache(
            self,
            layer_id: int,
            k_tensor: torch.Tensor,
            v_tensor: torch.Tensor,
            block_manager: BlockManager | None = None,
    ) -> None:
        """
        Append the KV tensors for the most recently generated token to the cache.

        Called once per transformer layer during each decode step.  Three cases
        are handled:

        1. **New-block write** (``last_block.is_empty(layer_id)``): the last block
           was just allocated during this decode step (on layer 0) and has no cached
           tensors for *layer_id* yet.  Use ``prefill_write_kv_cache`` to write the
           fresh tensor.
        2. **Extend existing block** (not full, ``can_append()``): the last block
           still has room and is not sealed.  Use
           ``decode_append_token_ids_and_kv_cache``; only update ``token_ids`` on
           ``layer_id == 0`` to avoid duplicating the token ID across layers.
        3. **Overflow** (last block is full): allocate a new block from
           *block_manager* on ``layer_id == 0``, then fall through to case 1 for
           this and all subsequent layers.

        After a full decode step, call
        :meth:`~block_manager.BlockManager.seal_full_decode_blocks` to insert newly
        filled blocks into the prefix-caching trie.

        Parameters
        ----------
        layer_id : int
            Zero-based transformer layer index.
        k_tensor : torch.Tensor
            Key tensor for the new token, shape ``[1, num_kv_heads, 1, head_dim]``.
        v_tensor : torch.Tensor
            Value tensor for the new token, shape ``[1, num_kv_heads, 1, head_dim]``.
        block_manager : BlockManager | None
            Required when the last block is full and a new block must be allocated.
            May be ``None`` when the caller guarantees the block is not full.
        """
        last_block = self.kv_cache_blocks[-1]

        # Case 1: Writing into a fresh block created earlier in this decode step.
        if last_block.is_empty(layer_id):
            last_block.prefill_write_kv_cache(layer_id, k_tensor, v_tensor)
            return

        # Case 2: Existing partial, unsealed block with room to grow.
        if not last_block.is_full() and last_block.can_append():
            last_block.decode_append_token_ids_and_kv_cache(
                layer_id,
                [self.get_last_token_id()],
                k_tensor,
                v_tensor,
                update_token_ids=(layer_id == 0),
            )
            return

        # Case 3: Last block is full (or sealed) — allocate a new block.
        # Only do this on the first layer; subsequent layers detect the empty block
        # via Case 1 above.
        if layer_id == 0:
            if block_manager is None:
                raise RuntimeError(
                    "block_manager is required when the last KV-cache block is full "
                    "and a new decode block must be allocated."
                )
            if not block_manager.evict_blocks(1):
                raise RuntimeError(
                    "KV-cache pool is exhausted; cannot allocate a new decode block."
                )
            # Create the new block outside the trie (trie_tree_depth stays 0) so it
            # remains appendable.  It will be sealed by seal_full_decode_blocks once
            # it is full and all layers have been written.
            new_block = block_manager.allocate_block(
                [self.get_last_token_id()], parent_trie_node=None
            )
            new_block.inc_ref_count()
            self.kv_cache_blocks.append(new_block)

        # Write the first KV entry into the newly allocated block.
        self.kv_cache_blocks[-1].prefill_write_kv_cache(layer_id, k_tensor, v_tensor)

    def release(self) -> None:
        """
        Decrement the reference count on every KV-cache block owned by this sequence.

        Called when the sequence is finished or dropped.  Blocks whose
        ``ref_count`` reaches zero become eligible for eviction.
        """
        for block in self.kv_cache_blocks:
            if block is not None:
                block.dec_ref_count()