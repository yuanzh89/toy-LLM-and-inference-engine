import torch


class Block:
    """
    A block of physical KV cache can be referred by multiple sequences at the same time, recorded by _ref_count.
    Serving just one layers
    """

    def __init__(self, block_id: int, layer_id: int, token_ids: list[int], k_cache: torch.Tensor = None,
                 v_cache: torch.Tensor = None, max_token_size_per_kv_cache_block: int = 16):
        self.block_id = block_id
        self.layer_id = layer_id
        self.token_ids = token_ids
        self.max_token_size_per_kv_cache_block = max_token_size_per_kv_cache_block
        self.ref_count = 1
        self.k_cache = k_cache
        self.v_cache = v_cache

    def __len__(self) -> int:
        return len(self.token_ids)

    @property
    def hash(self) -> int:
        return hash(tuple(self.token_ids))

    def append(self, token_id: int, k: torch.Tensor, v: torch.Tensor):
        self.token_ids.append(token_id)
        self.k_cache = torch.cat([self.k_cache, k], dim=0)
        self.v_cache = torch.cat([self.v_cache, v], dim=0)
