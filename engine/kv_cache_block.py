import torch


class Block:
    """
    A block of physical KV cache can be referred by multiple sequences at the same time, recorded by ref_count.
    """

    def __init__(self, block_id: int, block_size: int, dtype: torch.dtype) -> None:
        self.block_id = block_id
        self.block_size = block_size
        self.dtype = dtype

        self.k_cache = torch.empty(block_size, dtype=dtype)
        self.v_cache = torch.empty(block_size, dtype=dtype)

        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache[idx], self.v_cache[idx]

    def update(self, hash: int, token_ids: list[int]) -> None:
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
