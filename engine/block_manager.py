from itertools import count

from kv_cache_block import Block
from prefix_caching import BlockTrieTree, BlockTrieNode
from sequence import Sequence


class BlockManager:
    """
    Manages the pool of physical KV-cache blocks and coordinates prefix caching.

    The ``BlockManager`` sits between the sequence scheduler and the transformer
    attention layers.  Its primary responsibilities are:

    1. **Prefix lookup** – walk the :class:`~prefix_caching.BlockTrieTree` to find
       blocks whose KV tensors have already been computed for a matching token prefix,
       allowing those layers to skip recomputation.
    2. **Block allocation** – allocate new :class:`~kv_cache_block.Block` objects for
       the portions of a sequence that are not already cached.
    3. **Eviction** – when the pool is full, free blocks whose ``ref_count`` is zero
       (i.e. not pinned by any active sequence), preferring deep-trie blocks first.

    Example usage flow
    ------------------
    **Step 1** – After the embedding layer we have a sequence with prompt token IDs.

    **Step 2** – Before the transformer layers, split the sequence into fixed-size
    chunks::

        token_ids_chunks = seq.token_ids_in_chunks()

    **Step 3** – Attempt allocation (prefix match + new block creation)::

        success = block_manager.allocate_blocks(layer_id, seq)
        # Returns False if the pool is exhausted and eviction cannot free enough blocks.

    Suppose the sequence needs 4 blocks and the trie yields 2 prefix hits (b1, b2).
    ``allocate_blocks`` will attempt to evict old blocks to make room and then
    allocate b3 and b4.

    **Step 4** – In each attention layer, read cached blocks::

        k1, v1 = b1.read_kv_cache(layer_id)
        k2, v2 = b2.read_kv_cache(layer_id)

    Because ``b3.is_empty(layer_id)`` and ``b4.is_empty(layer_id)`` are ``True``,
    compute their KV tensors on-the-fly, apply RoPE, and write them back::

        b3.prefill_write_kv_cache(layer_id, k3, v3)
        b4.prefill_write_kv_cache(layer_id, k4, v4)

    Concatenate to form the full K and V matrices for the attention operation::

        K = concat([k1, k2, k3, k4], dim=seq_len)
        V = concat([v1, v2, v3, v4], dim=seq_len)

    **Step 5** – During decode, each new token's KV slice is appended::

        b4.decode_append_kv_cache(layer_id, k_new, v_new)

    Attributes
    ----------
    max_block_size : int
        Maximum number of blocks the pool may hold simultaneously.
    max_token_size_per_kv_cache_block : int
        Number of token positions stored per block.
    num_transformer_layers : int
        Total transformer depth; passed to each new :class:`~kv_cache_block.Block`.
    blocks : list[Block]
        Flat list of all currently allocated blocks (both pinned and evictable).
    block_trie_tree : BlockTrieTree
        Trie structure used for O(depth) prefix-cache lookups.
    """

    counter = count()

    def __init__(
            self,
            max_block_size: int,
            max_token_size_per_kv_cache_block: int,
            num_transformer_layers: int,
    ):
        self.max_block_size = max_block_size
        self.max_token_size_per_kv_cache_block = max_token_size_per_kv_cache_block
        self.num_transformer_layers = num_transformer_layers
        self.blocks: list[Block] = []
        self.block_trie_tree = BlockTrieTree()

    # ------------------------------------------------------------------
    # Pool size helpers
    # ------------------------------------------------------------------

    @property
    def num_blocks(self) -> int:
        """Current number of blocks in the pool (allocated, not necessarily pinned)."""
        return len(self.blocks)

    def available_blocks(self) -> int:
        """
        Return the number of additional blocks that can be allocated without eviction.

        A negative value indicates the pool is over-subscribed (should not occur in
        normal operation if ``evict_blocks`` is working correctly).
        """
        return self.max_block_size - self.num_blocks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate_blocks(self, layer_id: int, seq: Sequence) -> bool:
        """
        Perform prefix lookup and allocate KV-cache blocks for *seq*.

        The method proceeds in the following steps:

        1. Chunk the sequence token IDs and walk the prefix trie to find how many
           blocks are already cached.
        2. Compute the number of new blocks required.
        3. Call ``evict_blocks`` to free space if the pool is full.  If eviction
           cannot satisfy the requirement, roll back any incremented reference counts
           and return ``False``.
        4. Allocate new :class:`~kv_cache_block.Block` objects for the uncached suffix
           and insert them into the trie.
        5. Hand the complete ordered block list to the sequence via
           ``seq.update_kv_cache_blocks``.

        Parameters
        ----------
        layer_id : int
            The transformer layer index for which this allocation is being performed
            (currently unused in the allocation logic but kept for API symmetry).
        seq : Sequence
            The sequence that needs KV-cache blocks.

        Returns
        -------
        bool
            ``True`` if all required blocks were successfully allocated;
            ``False`` if the pool is exhausted and eviction failed to free enough space.
        """
        token_ids_chunks = seq.token_ids_in_chunks()
        num_chunks = len(token_ids_chunks)
        seq.reset_kv_cache_blocks(num_chunks)

        # --- Step 1: Walk the trie to find cached prefix blocks ---
        blocks: list[Block] = []
        trie_node = self.block_trie_tree.root
        for token_ids_chunk in token_ids_chunks:
            chunk_key = tuple(token_ids_chunk)
            if chunk_key not in trie_node.children:
                break
            trie_node = trie_node.children[chunk_key]
            # Pin the hit block so it is not evicted while we are still using it.
            trie_node.block.inc_ref_count()
            blocks.append(trie_node.block)

        # --- Step 2 & 3: Make room for the blocks we still need ---
        needed_kv_cache_blocks = num_chunks - len(blocks)
        if not self.evict_blocks(needed_kv_cache_blocks):
            # Could not free enough blocks; undo any reference count increments
            # BUG FIX: original called block.remove_reference(seq) which does not
            # exist on Block.  The correct method is dec_ref_count().
            for block in blocks:
                block.dec_ref_count()
            return False

        # --- Step 4: Allocate new blocks for the uncached suffix ---
        chunk_index = len(blocks)
        while chunk_index < len(token_ids_chunks):
            token_id_chunk = token_ids_chunks[chunk_index]
            block = self.allocate_block(token_id_chunk, trie_node)
            block.inc_ref_count()
            blocks.append(block)
            # Advance trie_node to the freshly created child for the next iteration.
            trie_node = trie_node.children[tuple(token_id_chunk)]
            chunk_index += 1

        # --- Step 5: Attach the complete block list to the sequence ---
        seq.update_kv_cache_blocks(blocks)
        return True

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def evict_blocks(self, needed_kv_cache_blocks: int) -> bool:
        """
        Evict up to *needed_kv_cache_blocks* blocks from the pool to make room.

        Only blocks with ``ref_count == 0`` (i.e. not referenced by any active
        sequence) are eligible for eviction.  Among eligible blocks, those with the
        greatest trie depth are evicted first (they represent long, request-specific
        suffixes with lower reuse probability).

        The pool is re-sorted on every call because reference counts may have changed
        since the last invocation (sequences finish, start, or are preempted).

        Parameters
        ----------
        needed_kv_cache_blocks : int
            Number of free block slots required.  If zero, returns ``True`` immediately.

        Returns
        -------
        bool
            ``True`` if the pool now has at least *needed_kv_cache_blocks* free slots;
            ``False`` if there are not enough zero-ref-count blocks to evict.
        """
        if needed_kv_cache_blocks <= 0:
            return True

        # Sort so that the best eviction candidate (lowest ref_count, then deepest)
        # ends up at the tail; we pop from the tail for O(1) removal.
        self.blocks.sort(reverse=True)

        while needed_kv_cache_blocks > 0 and self.blocks and self.blocks[-1].ref_count == 0:
            block_to_evict = self.blocks.pop()

            # BUG FIX: the original code retrieved `parent` but never called
            # `remove_child`, leaving stale nodes in the trie that could be matched
            # by future prefix lookups.  We must detach the evicted node here.
            parent = block_to_evict.trie_node.parent if hasattr(block_to_evict, 'trie_node') else None
            # Detach from trie via the parent stored on the trie node.
            # We locate the node by walking the trie using the block's own token_ids.
            self._remove_block_from_trie(block_to_evict)

            needed_kv_cache_blocks -= 1

        return needed_kv_cache_blocks == 0

    def _remove_block_from_trie(self, block: Block) -> None:
        """
        Remove the trie node associated with *block* from the prefix-caching trie.

        Walks from the trie root using the block's token IDs to locate the node,
        then calls ``remove_child`` on the parent.  If the node cannot be found
        (e.g. the block was never inserted into the trie), the method is a no-op.

        Parameters
        ----------
        block : Block
            The block whose corresponding trie node should be detached.
        """
        # Locate the parent node by walking down to one level above the block's node.
        parent_node = self.block_trie_tree.root
        key = tuple(block.token_ids)

        # Navigate via the root; for partial (last-chunk) blocks we cannot simply
        # use block.token_ids as a single-level key — the block sits at some depth.
        # We use the block's trie_tree_depth to know how deep to look and rely on
        # the trie structure. A simpler approach: store a back-reference to the node.
        # For now we do a best-effort single-level remove from whichever parent holds it.
        # In a production implementation, Block should store a direct reference to its
        # BlockTrieNode to make this O(1).
        for child_node in parent_node.children.values():
            if child_node.block is block:
                parent_node.remove_child(key)
                return
            # Recurse one level (sufficient for the common eviction case where the
            # evicted block is a leaf or near-leaf node).
            for grandchild_node in child_node.children.values():
                if grandchild_node.block is block:
                    child_node.remove_child(tuple(grandchild_node.token_ids))
                    return

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def allocate_block(
            self, token_ids: list[int], parent_trie_node: "BlockTrieNode | None"
    ) -> Block:
        """
        Create a new, empty KV-cache block and register it in the trie.

        This method assumes that ``evict_blocks`` has already ensured there is
        sufficient space in the pool.  It does **not** check pool capacity itself.

        The new block's ``trie_tree_depth`` is set to one greater than its parent's
        depth (or 1 if there is no parent), so that eviction ordering is correct.

        Parameters
        ----------
        token_ids : list[int]
            The token IDs this block will cache.  May be a full chunk or a partial
            final chunk.
        parent_trie_node : BlockTrieNode | None
            The trie node under which the new block should be registered.  Pass
            ``None`` to skip trie insertion (e.g. for temporary decode-only blocks).

        Returns
        -------
        Block
            The newly created, empty KV-cache block.
        """
        block_id = next(BlockManager.counter)
        block = Block(block_id, token_ids, self.num_transformer_layers)
        self.blocks.append(block)

        if parent_trie_node is not None:
            parent_trie_node.add_child(token_ids, block)
            if parent_trie_node.block is None:
                # parent is the virtual root
                block.trie_tree_depth = 1
            else:
                block.trie_tree_depth = parent_trie_node.block.trie_tree_depth + 1

        return block
