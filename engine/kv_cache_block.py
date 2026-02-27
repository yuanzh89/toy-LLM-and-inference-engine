from copy import copy

import torch

from sequence import Sequence


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
        self.referred_seqs = set()

    def token_ids(self):
        return tuple(self.token_ids)

    def is_empty(self, layer_id: int) -> bool:
        """Return whether the KV cache block is empty at a given layer_id"""
        assert 0 <= layer_id < self.num_transformer_layers
        return self.k_cache[layer_id] is None or self.v_cache[layer_id] is None

    def write_kv_cache(self, layer_id: int, k_cache: torch.Tensor, v_cache: torch.Tensor):
        assert 0 <= layer_id < self.num_transformer_layers
        self.k_cache[layer_id] = k_cache
        self.v_cache[layer_id] = v_cache

    def read_kv_cache(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert 0 <= layer_id < self.num_transformer_layers
        k_cache = self.k_cache[layer_id]
        v_cache = self.v_cache[layer_id]
        assert k_cache is not None
        assert v_cache is not None

        return k_cache, v_cache

    def add_reference(self, seq: Sequence):
        self.referred_seqs.add(seq)
        self.ref_count += 1

    def remove_reference(self, seq: Sequence):
        self.referred_seqs.remove(seq)
        self.ref_count -= 1

    def __len__(self) -> int:
        return len(self.token_ids)

    def append(self, token_id: int, k: torch.Tensor, v: torch.Tensor) -> bool:
        """
        k_tensor shape: [batch_size, num_heads, seq_len == 1, head_dim]
        v_tensor shape: [batch_size, num_heads, seq_len == 1, head_dim]

        Return True if the current block can hold this extra token, False otherwise
        """
        if len(self.token_ids) == self.max_token_size_per_kv_cache_block:
            return False

        self.token_ids.append(token_id)
        self.k_cache = torch.cat([self.k_cache, k], dim=0)
        self.v_cache = torch.cat([self.v_cache, v], dim=0)

        return True
