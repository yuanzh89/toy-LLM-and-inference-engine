from copy import copy

import torch


class Block:
    """
    A block of physical KV cache can be referred by multiple sequences at the same time, recorded by _ref_count.
    Serving just one layers
    """

    def __init__(self, block_id: int, token_ids: list[int],
                 num_transformer_layers: int, max_token_size_per_kv_cache_block: int = 16):
        self.block_id = block_id
        self.token_ids: list[int] = copy(token_ids)
        self.num_transformer_layers = num_transformer_layers
        self.max_token_size_per_kv_cache_block = max_token_size_per_kv_cache_block
        self.ref_count = 1
        self.k_cache: list[torch.Tensor | None] = [None] * self.num_transformer_layers
        self.v_cache: list[torch.Tensor | None] = [None] * self.num_transformer_layers
        # self.referred_seqs = set()
        self.trie_tree_depth = 0

    def __lt__(self, other: "Block") -> bool:
        """
        Comparator used to compare two blocks during eviction.
        The winning block will be evicted first follow the following rules:
        1. Evicted blocks must be not actively referred by any active sequences.
        2. When multiple blocks are available for eviction, we evict the block with max depth in trie tree.
        Because the deeper block generally has the lower chance to be reused.
        """
        if self.ref_count != other.ref_count:
            return self.ref_count < other.ref_count
        return self.trie_tree_depth < other.trie_tree_depth

    def token_ids(self):
        return tuple(self.token_ids)

    def is_empty(self, layer_id: int) -> bool:
        """Return whether the KV cache block is empty at a given layer_id"""
        assert 0 <= layer_id < self.num_transformer_layers
        return self.k_cache[layer_id] is None or self.v_cache[layer_id] is None

    def prefill_write_kv_cache(self, layer_id: int, k_cache: torch.Tensor, v_cache: torch.Tensor):
        """
        k_cache shape: [batch_size, num_heads, seq_len, head_dim]
        v_cache shape: [batch_size, num_heads, seq_len, head_dim]
        """
        assert 0 <= layer_id < self.num_transformer_layers
        _, _, k_seq_len, _ = k_cache.shape
        _, _, v_seq_len, _ = v_cache.shape
        assert k_seq_len == v_seq_len, "Sequence length mismatch between K and V"
        assert 0 < k_seq_len <= self.max_token_size_per_kv_cache_block, "Sequence length exceeds max_token_size_per_kv_cache_block"

        self.k_cache[layer_id] = k_cache
        self.v_cache[layer_id] = v_cache

    def read_kv_cache(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert 0 <= layer_id < self.num_transformer_layers
        k_cache = self.k_cache[layer_id]
        v_cache = self.v_cache[layer_id]
        assert k_cache is not None
        assert v_cache is not None

        return k_cache, v_cache

    def inc_ref_count(self):
        # self.referred_seqs.add(seq)
        self.ref_count += 1

    def dec_ref_count(self):
        # self.referred_seqs.remove(seq)
        self.ref_count -= 1

    def __len__(self) -> int:
        return len(self.token_ids)

    def decode_append_token_ids(self, new_token_ids: list[int]) -> None:
        assert len(new_token_ids) > 0
        assert len(self.token_ids) + len(new_token_ids) <= self.max_token_size_per_kv_cache_block
        self.token_ids.extend(new_token_ids)

    def decode_append_kv_cache(self, layer_id: int, k_cache: torch.Tensor, v_cache: torch.Tensor):
        """
        k_tensor shape: [batch_size, num_heads, seq_len == 1, head_dim]
        v_tensor shape: [batch_size, num_heads, seq_len == 1, head_dim]

        Usually, seq_len should always be 1 for regular autogressive decoding.
        Unless using speculative decoding, then the seq_len might be larger than 1
        """

        assert 0 <= layer_id < self.num_transformer_layers
        _, _, k_seq_len, _ = k_cache.shape
        _, _, v_seq_len, _ = v_cache.shape
        assert k_seq_len == v_seq_len, "Sequence length mismatch between K and V"

        existing_kv_seq_len = self.k_cache[layer_id].size(2)
        assert 0 < existing_kv_seq_len + k_seq_len <= self.max_token_size_per_kv_cache_block, "Sequence length exceeds max_token_size_per_kv_cache_block"

        # Concatenate new KV cache tensors to existing KV cache tensors along the seq_len dimension.
        self.k_cache[layer_id] = torch.concat([self.k_cache[layer_id], k_cache], dim=2)
        self.v_cache[layer_id] = torch.concat([self.v_cache[layer_id], v_cache], dim=2)
