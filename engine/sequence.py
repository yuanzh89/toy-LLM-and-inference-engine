from copy import copy
from enum import Enum, auto
from itertools import count
from kv_cache_block import Block
import math


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    # Num of tokens per KV cache block
    counter = count()

    def __init__(self, token_ids: list[int], max_token_size_per_kv_cache_block: int, max_sequence_length: int):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.max_token_size_per_kv_cache_block = max_token_size_per_kv_cache_block
        self.max_sequence_length = max_sequence_length
        # List of KV cache blocks this sequence refers to, the size should match the size of chunks
        # block manager should be responsible for populating this by using prefix matching
        self.kv_cache_blocks = []
        # Indicates whether the KV cache blocks are fully initialized
        self.kv_cache_blocks_initialized = False

    @property
    def num_decode_tokens(self):
        return len(self.token_ids) - self.num_prompt_tokens

    def reset_kv_cache_blocks(self, num_chunks: int) -> None:
        self.kv_cache_blocks = [None] * num_chunks
        self.kv_cache_blocks_initialized = False

    def update_kv_cache_blocks(self, kv_cache_blocks: list[Block]) -> None:
        assert len(kv_cache_blocks) == math.ceil(len(self.token_ids) / self.max_token_size_per_kv_cache_block)
        self.kv_cache_blocks = copy(kv_cache_blocks)
        self.kv_cache_blocks_initialized = True

    def __len__(self):
        return len(self.token_ids)

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)

    def token_ids_in_chunks(self) -> list[list[int]]:
        """
        Split token_ids into chunks with size of max_token_size_per_kv_cache_block.
        The size of the last chunk might be smaller.
        """

        chunks = []
        for i in range(0, len(self.token_ids), self.max_token_size_per_kv_cache_block):
            chunks.append(self.token_ids[i: i + min(i + self.max_token_size_per_kv_cache_block, len(self.token_ids))])
        return chunks
