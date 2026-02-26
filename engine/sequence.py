from collections import defaultdict
from copy import copy
from enum import Enum, auto
from itertools import count


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    # Num of tokens per KV cache block
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], max_token_size_per_kv_cache_block: int):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.max_token_size_per_kv_cache_block = max_token_size_per_kv_cache_block

        self.token_ids_chunks = [
            self.token_ids[i: i + min(i + self.max_token_size_per_kv_cache_block, len(self.token_ids))] for i in
            range(0, len(self.token_ids), self.max_token_size_per_kv_cache_block)]

        # Referred KV cache block ids in order by layers
        # layer_1 -> [block_1, block_2, ...]
        # layer_2 -> [block_1, block_2, ...]
        self.kv_cache_blocks = defaultdict(list)

    @property
    def num_decode_tokens(self):
        return len(self.token_ids) - self.num_prompt_tokens

    def __len__(self):
        return len(self.token_ids)

    # TODO: Append KV cache
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        if len(self.token_ids_chunks[-1]) == self.max_token_size_per_kv_cache_block:
            self.token_ids_chunks.append([])
        self.token_ids_chunks[-1].append(token_id)
