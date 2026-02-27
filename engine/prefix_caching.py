from itertools import count
from copy import copy

import torch

from kv_cache_block import Block
from sequence import Sequence


class BlockTrieNode:
    """
    Trie Node holds a few token ids and a physical block of corresponding KV cache.
    """
    def __init__(self, token_ids: list[int], block: Block | None):
        self.token_ids = copy(token_ids)
        self.block = block
        self.children = {}
        self.block = block
        self.children = {}

    @property
    def ref_count(self):
        return self.block.ref_count

    def __id__(self):
        return tuple(self.token_ids)

    def __len__(self) -> int:
        return len(self.token_ids)

    def add_child(self, token_ids: list[int], block: Block) -> "TrieNode":
        child = self.children.get(tuple(token_ids), None)
        if child is not None:
            return child

        child = BlockTrieNode(token_ids, block)
        self.children[tuple(token_ids)] = child
        return child


class BlockTrieTree:
    def __init__(self):
        self.root = BlockTrieNode([], None)

    def lookup_blocks(self, seq: Sequence) -> list[Block]:
        """
        Prefix lookup of the blocks, may return partial matching result.
        For example, if a seq is split up to 4 blocks, and we found 2 prefix block matches, then we will return
        2 matched prefix blocks.
        """
        blocks = []
        node = self.root
        for token_ids_chunk in seq.token_ids_chunks():
            if tuple(token_ids_chunk) not in node.children:
                break
            node = node.children[tuple(token_ids_chunk)]
            blocks.append(node.block)

        return blocks

    def insert_sequence(self, seq: Sequence, k_tensor: torch.Tensor, v_tensor: torch.Tensor) -> None:
        """
        Insert the KV cache into the
        """