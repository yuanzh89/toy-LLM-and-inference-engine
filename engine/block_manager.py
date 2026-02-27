from itertools import count

from kv_cache_block import Block
from prefix_caching import BlockTrieTree, BlockTrieNode
from sequence import Sequence


class BlockManager:
    """
    Sample usage flow of KV cache block manager:
    Step 1: after the embedding layer, we have a sequence with a list of token_ids.

    Step 2: Before going into the transformer layers, we chunk the token_ids of the sequence
    by the max_token_size_per_kv_cache_block, where the # of token ids in the last chunk might be unfull.
        token_ids_in_chunks = seq.token_ids_in_chunks()

    Step 3: Then we try to lookup by prefix matching of existing KV cache blocks.
        block_manager.lookup_blocks(seq) -> list[Block]
    in our example, the returned value will be [k1, k2]

    The KV cache prefix matching may give us a partial matching result,
    where for example, we have need 4 block for this sequence, we find a match for the first
    2 blocks, but cache miss of the 3rd and 4th blocks.
    Each KV cache block should include the K_cache and V_cache for all layers in the current transformer.
    Then we pass the matching first two blocks of KV cache to transformer for computation.

    Then block_manager should allocate two extra blocks for the input sequence.
    In this allocation process, we potentially deallocate some blocks by the following rule:

    1. We deallocate the blocks with ref_count == 0, if there is all blocks are used by IN_PROGRESS sequences.
    Then this allocation process should fail.

    2. We deallocate the blocks allocated to the longest sequence, because the longest sequence has a lower chance to be
    reused in prefix matching.

    Then we allocate [block3, block4], these two blocks should be tracked by sequence.
    Where we know block3[layer_i].is_empty is True and block4[layer_i].is_empty is True.

    Step 4: Then in each attention layer, we first read the KV cache blocks for the first 2 blocks to avoid recomputation.
    We read [k1, k2] and [v1, v2] from KV cache blocks.

    Given that we know block3[layer_i].is_empty and block4[layer_i].is_empty,
    Then we compute [k3, k4] and [v3, v4] on fly.
    We apply RoPE for [k3, k4]

    Then we update the calculated [k3, k4] and [v3, v4] to [block3, block4].

    block3.update_kv_cache(layer_i, k_tensor, v_tensor)
    block4.update_kv_cache(layer_i, k_tensor, v_tensor)

    Then we concatenate [k1, k2, k3, k4] -> K and concatenate [v1, v2, v3, v4] -> V

    We also compute [q1, q2, q3, q4] -> Q on fly for prefill.
    And compute [q_last_token] for decode

    Then we compute attention using concatenate Q, K and V.


    """
    counter = count()

    def __init__(self, max_block_size: int, max_token_size_per_kv_cache_block: int, num_transformer_layers: int):
        self.max_block_size = max_block_size
        self.max_token_size_per_kv_cache_block = max_token_size_per_kv_cache_block
        self.num_transformer_layers = num_transformer_layers
        self.id_to_block = {}
        self.hash_to_block = {}
        self.block_trie_tree = BlockTrieTree()
        self.blocks = []

    @property
    def num_blocks(self):
        return len(self.blocks)

    def available_blocks(self) -> int:
        return self.max_block_size - self.num_blocks

    def allocate_blocks(self, layer_id: int, seq: Sequence) -> bool:
        """
        This function attempts to lookup and then allocate KV cache blocks if necessary for the input sequence.

        Step 1: lookup the prefix caching to find reusable KV cache blocks.
        Step 2: calculate the extra needed KV cache blocks.
        Step 3: evict existing KV cache blocks if num_blocks is full.
        If we cannot make room for the needed blocks, this function will return False indicate failure to clients.
        Step 4: allocate new blocks,
        Step 5: update reference between KV cache block and sequence.
        Step 6: return True indicates success of the process.

        Return value: boolean, indicates success of the process.
        """
        token_ids_chunks = seq.token_ids_in_chunks()
        num_chunks = len(token_ids_chunks)
        seq.reset_kv_cache_blocks(num_chunks)

        blocks = []
        trie_node = self.block_trie_tree.root
        for idx, token_ids_chunk in enumerate(token_ids_chunks):
            chunk_key = tuple(token_ids_chunk)
            if chunk_key not in trie_node.children:
                break
            trie_node = trie_node.children[chunk_key]
            # Add reference to the sequence to protect cache hit blocks from eviction.
            trie_node.block.inc_ref_count()
            blocks.append(trie_node.block)

        needed_kv_cache_blocks = num_chunks - len(blocks)
        if not self.evict_blocks(needed_kv_cache_blocks):
            # Failed to make space for new allocations
            # Return False to indicate failure of this process
            for block in blocks:
                block.remove_reference(seq)
            return False

        # At this point, we are ready to allocate new blocks for the sequence
        chunk_index = len(blocks)
        while chunk_index < len(token_ids_chunks):
            token_id_chunk = token_ids_chunks[chunk_index]
            block = self.allocate_block(token_id_chunk, trie_node)
            block.inc_ref_count()
            blocks.append(block)
            trie_node = trie_node.children[tuple(token_id_chunk)]
            chunk_index += 1

        # Add blocks reference to Sequence
        seq.update_kv_cache_blocks(blocks)

        return True

    def evict_blocks(self, needed_kv_cache_blocks: int) -> bool:
        """Return whether we were able to evict needed_kv_cache_blocks number of KV cache blocks."""
        # We need to sort this on fly because when we add and remove sequences during inference.
        # The ref_count of each block may dynamically change
        self.blocks.sort(reverse=True)

        while needed_kv_cache_blocks > 0 and self.blocks and self.blocks[-1].ref_count == 0:
            block_to_evict = self.blocks.pop()
            # Remove block from TrieTree
            parent = block_to_evict.parent

            needed_kv_cache_blocks -= 1

        return needed_kv_cache_blocks == 0

    def allocate_block(self, token_ids: list[int], parent_trie_node: BlockTrieNode | None) -> Block:
        """
        Given a full or partial chunk of token_ids,
        this function allocates a new block and returns it.

        This function assume that there are enough block spaces in block manage for new allocations.
        This function handles:
          1. creation of a new empty KV cache block
          2. Update it into the Trie tree for prefix caching
        """
        block_id = next(BlockManager.counter)
        block = Block(block_id, token_ids, self.num_transformer_layers)
        self.blocks.append(block)
        if parent_trie_node is not None:
            parent_trie_node.add_child(token_ids, block)

        return block
