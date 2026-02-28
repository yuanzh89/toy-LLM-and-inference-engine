from copy import copy

import torch


class Block:
    """
    A single physical KV-cache block that may be shared by multiple sequences simultaneously.

    Each block stores pre-computed key and value tensors for every transformer layer,
    covering up to ``max_token_size_per_kv_cache_block`` token positions.  Sharing is
    tracked via ``ref_count``; a block with ``ref_count == 0`` is safe to evict.

    Lifecycle
    ---------
    * **Prefill** – ``prefill_write_kv_cache`` writes the full key/value tensors for
      every token in the block at once.
    * **Decode** – ``decode_append_kv_cache`` appends one (or a few, for speculative
      decoding) new key/value slices to an already-populated block.

    Attributes
    ----------
    block_id : int
        Unique identifier assigned by :class:`BlockManager`.
    token_ids : list[int]
        Token IDs whose KV cache is stored in this block.  Grows during decoding.
    num_transformer_layers : int
        Total number of transformer layers; determines the length of the k/v lists.
    max_token_size_per_kv_cache_block : int
        Maximum number of token positions this block can hold.
    ref_count : int
        Number of active sequences currently referencing this block.
    k_cache : list[torch.Tensor | None]
        Per-layer key tensors.  ``None`` means the layer has not been computed yet.
    v_cache : list[torch.Tensor | None]
        Per-layer value tensors.  ``None`` means the layer has not been computed yet.
    trie_tree_depth : int
        Depth of this block's node in the prefix-caching :class:`BlockTrieTree`.
        Used for eviction ordering: deeper blocks are evicted first.
    """

    def __init__(
            self,
            block_id: int,
            token_ids: list[int],
            num_transformer_layers: int,
            max_token_size_per_kv_cache_block: int = 16,
    ):
        self.block_id = block_id
        self.token_ids: list[int] = copy(token_ids)
        self.num_transformer_layers = num_transformer_layers
        self.max_token_size_per_kv_cache_block = max_token_size_per_kv_cache_block
        self.ref_count: int = 0
        self.k_cache: list[torch.Tensor | None] = [None] * self.num_transformer_layers
        self.v_cache: list[torch.Tensor | None] = [None] * self.num_transformer_layers
        # Set by BlockManager when this block is inserted into the trie tree.
        # trie_tree_depth == 0 means this block is pending, not inserted into prefix caching TrieTree yet.
        self.trie_tree_depth: int = 0

    # ------------------------------------------------------------------
    # Comparison / sizing
    # ------------------------------------------------------------------

    def __lt__(self, other: "Block") -> bool:
        """
        Ordering used during eviction candidate selection.

        Rules (highest priority first):
        1. Only blocks with ``ref_count == 0`` are eligible for eviction.
           Blocks with a lower ``ref_count`` compare *less-than* so they sort
           toward the eviction end of the list.
        2. Among equally-eligible blocks, prefer to evict the one with the
           **greatest** trie-tree depth, because deep blocks represent long,
           request-specific suffixes that are unlikely to be reused as a
           common prefix.

        This comparator is designed for use with ``list.sort(reverse=True)``:
        the element at the tail of the sorted list is the best eviction candidate.
        """
        if self.ref_count != other.ref_count:
            return self.ref_count < other.ref_count
        return self.trie_tree_depth > other.trie_tree_depth

    def __len__(self) -> int:
        """Return the number of token IDs currently stored in this block."""
        return len(self.token_ids)

    def can_append(self) -> bool:
        """Returns whether this block can be appended."""
        return self.trie_tree_depth == 0

    # ------------------------------------------------------------------
    # Reference counting
    # ------------------------------------------------------------------

    def inc_ref_count(self) -> None:
        """Increment the reference count when a new sequence starts using this block."""
        self.ref_count += 1

    def dec_ref_count(self) -> None:
        """
        Decrement the reference count when a sequence releases this block.

        The caller is responsible for ensuring ``ref_count`` never goes below zero.
        When ``ref_count`` reaches zero the block becomes eligible for eviction.
        """
        self.ref_count -= 1

    # ------------------------------------------------------------------
    # KV-cache access
    # ------------------------------------------------------------------

    def is_empty(self, layer_id: int) -> bool:
        """
        Return ``True`` if the KV cache for *layer_id* has not been written yet.

        A block is considered empty for a given layer when either the key or the
        value tensor is ``None``.  This signals the attention layer that it must
        compute (rather than read) the KV tensors for this block.

        Parameters
        ----------
        layer_id : int
            Zero-based transformer layer index.
        """
        assert 0 <= layer_id < self.num_transformer_layers
        return self.k_cache[layer_id] is None or self.v_cache[layer_id] is None

    def prefill_write_kv_cache(
            self, layer_id: int, k_cache: torch.Tensor, v_cache: torch.Tensor
    ) -> None:
        """
        Store the full prefill KV tensors for a transformer layer.

        Called once per layer during the prefill pass after computing the key and
        value projections for all tokens in this block.

        Parameters
        ----------
        layer_id : int
            Zero-based transformer layer index.
        k_cache : torch.Tensor
            Key tensor of shape ``[batch_size, num_heads, seq_len, head_dim]``.
        v_cache : torch.Tensor
            Value tensor of shape ``[batch_size, num_heads, seq_len, head_dim]``.
        """
        assert 0 <= layer_id < self.num_transformer_layers
        _, _, k_seq_len, _ = k_cache.shape
        _, _, v_seq_len, _ = v_cache.shape
        assert k_seq_len == v_seq_len, "Sequence length mismatch between K and V"
        assert (
                0 < k_seq_len <= self.max_token_size_per_kv_cache_block
        ), "Sequence length exceeds max_token_size_per_kv_cache_block"

        self.k_cache[layer_id] = k_cache
        self.v_cache[layer_id] = v_cache

    def read_kv_cache(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Read the stored KV tensors for a transformer layer.

        Used during attention computation to retrieve pre-computed key/value
        tensors for cache-hit blocks, avoiding redundant forward passes.

        Parameters
        ----------
        layer_id : int
            Zero-based transformer layer index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(k_cache, v_cache)`` both of shape
            ``[batch_size, num_heads, seq_len, head_dim]``.

        Raises
        ------
        AssertionError
            If the requested layer has not been written yet (i.e. ``is_empty`` is True).
        """
        assert 0 <= layer_id < self.num_transformer_layers
        k_cache = self.k_cache[layer_id]
        v_cache = self.v_cache[layer_id]
        assert k_cache is not None, f"K cache for layer {layer_id} has not been written"
        assert v_cache is not None, f"V cache for layer {layer_id} has not been written"
        return k_cache, v_cache

    def decode_append_token_ids_and_kv_cache(
            self, layer_id: int, new_token_ids: list[int], k_cache: torch.Tensor, v_cache: torch.Tensor,
    ) -> None:
        """
        Append new KV tensors to an existing block during the decode phase.

        During autoregressive decoding each step produces a single new token
        whose key/value vectors are concatenated onto the tensors already stored
        in the block.  For speculative decoding ``seq_len`` may be greater than 1.

        Parameters
        ----------
        layer_id : int
            Zero-based transformer layer index.
        new_token_ids : list[int]
            One or more token IDs produced during the decode step.
        k_cache : torch.Tensor
            New key tensor of shape ``[batch_size, num_heads, seq_len, head_dim]``.
            ``seq_len`` is typically 1 for standard autoregressive decoding.
        v_cache : torch.Tensor
            New value tensor of shape ``[batch_size, num_heads, seq_len, head_dim]``.

        Raises
        ------
        AssertionError
            If the block has not been populated by ``prefill_write_kv_cache`` first,
            or if appending would exceed ``max_token_size_per_kv_cache_block``.
        """
        assert 0 <= layer_id < self.num_transformer_layers
        assert len(new_token_ids) > 0

        assert (
                len(self.token_ids) + len(new_token_ids) <= self.max_token_size_per_kv_cache_block
        ), "Appending token IDs would exceed max_token_size_per_kv_cache_block"

        assert self.k_cache[layer_id] is not None, (
            f"Layer {layer_id} has not been written via prefill_write_kv_cache; "
            "decode_append_kv_cache requires an existing cache to extend."
        )
        _, _, k_seq_len, _ = k_cache.shape
        _, _, v_seq_len, _ = v_cache.shape
        assert len(new_token_ids) == k_seq_len == v_seq_len, "Sequence length mismatch between new_token_ids, K and V"

        existing_kv_seq_len = self.k_cache[layer_id].size(2)
        assert (
                0 < existing_kv_seq_len + k_seq_len <= self.max_token_size_per_kv_cache_block
        ), "Appending would exceed max_token_size_per_kv_cache_block"

        self.token_ids.extend(new_token_ids)

        # Concatenate along the sequence-length dimension (dim=2).
        self.k_cache[layer_id] = torch.cat([self.k_cache[layer_id], k_cache], dim=2)
        self.v_cache[layer_id] = torch.cat([self.v_cache[layer_id], v_cache], dim=2)
