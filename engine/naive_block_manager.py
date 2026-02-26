from itertools import count

import torch

from kv_cache_block import Block
from sequence import Sequence


class TrieNode:
    def __init__(self, token_ids: list[int], block: Block | None):
        self.tokens = tuple(token_ids)
        self.block = block
        self.children = {}

    @property
    def ref_count(self):
        return self.block.ref_count

    def add_child(self, token_ids: list[int], block: Block) -> "TrieNode":
        child = self.children.get(tuple(token_ids), None)
        if child is not None:
            return child

        child = TrieNode(token_ids, block)
        self.children[tuple(token_ids)] = child
        return child


class TrieTree:
    def __init__(self):
        self.root = TrieNode([], None)

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


class BlockManager:
    counter = count()

    def __init__(self, max_block_size: int, max_token_size_per_kv_cache_block: int):
        self.blocks = []
        self.max_block_size = max_block_size
        self.max_token_size_per_kv_cache_block = max_token_size_per_kv_cache_block
        self.id_to_block = {}
        self.hash_to_block = {}

    def __len__(self):
        return len(self.blocks)

    def allocate_blocks(self, layer_id: int, seq: Sequence, k_tensor: torch.Tensor, v_tensor: torch.Tensor) -> list[
        Block]:
        """
        k_tensor shape: [batch_size, num_heads, seq_len, head_dim]
        v_tensor shape: [batch_size, num_heads, seq_len, head_dim]
        """
        token_ids_chunks = seq.token_ids_chunks()

        # Current block size + needed block size
        if len(self) + len(token_ids_chunks) > self.max_block_size:
            # LRU eviction
            pass

        # Split KV tensors on the seq_len dimension
        k_tensors = torch.split(k_tensor, self.max_token_size_per_kv_cache_block, dim=2)
        v_tensors = torch.split(v_tensor, self.max_token_size_per_kv_cache_block, dim=2)

        blocks = []
        for token_ids_chunk, k_tensor, v_tensor in zip(token_ids_chunks, k_tensors, v_tensors):
            blocks.append(self.allocate_block(layer_id, token_ids_chunk, k_tensor, v_tensor))

        return blocks

    def allocate_block(self, layer_id: int, token_ids: list[int], k_tensor: torch.Tensor = None,
                       v_tensor: torch.Tensor = None, ) -> Block:
        block_id = next(BlockManager.counter)
        block = Block(block_id, layer_id, token_ids, k_tensor, v_tensor)
        self.blocks.append(block)
        self.id_to_block[block_id] = block
        self.hash_to_block[block_id] = block

        return block

    def lookup_blocks(self, seq: Sequence) -> list[Block]:
        """
        Prefix lookup of the blocks, may return partial matching result.
        For example, if a seq is split up to 4 blocks, and we found 2 prefix block matches, then we will return
        2 matched prefix blocks.
        """
        blocks = []
        for token_ids_chunk in seq.token_ids_chunks():
            block = self.hash_to_block.get(hash(tuple(token_ids_chunk)), None)
            if block is None:
                break
            block.append(block)

        return blocks
