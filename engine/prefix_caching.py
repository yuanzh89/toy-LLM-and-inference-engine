from copy import copy
from itertools import count

from kv_cache_block import Block
from sequence import Sequence


class BlockTrieNode:
    """
    A single node in the :class:`BlockTrieTree`.

    Each node represents one KV-cache block's worth of token IDs and holds a
    reference to the corresponding :class:`Block`.  The children dict maps a
    ``tuple[int, ...]`` of token IDs to the next ``BlockTrieNode`` in the trie,
    enabling O(depth) prefix lookups.

    Attributes
    ----------
    token_ids : list[int]
        The token IDs represented by this node (a copy of the input).
    block : Block | None
        The physical KV-cache block associated with these token IDs.
        ``None`` only for the virtual root node.
    parent : BlockTrieNode | None
        Parent node in the trie.  ``None`` only for the root.
    children : dict[tuple[int, ...], BlockTrieNode]
        Mapping from a chunk of token IDs to the corresponding child node.
    """

    def __init__(
        self,
        token_ids: list[int],
        block: "Block | None",
        parent: "BlockTrieNode | None" = None,
    ):
        self.token_ids: list[int] = copy(token_ids)
        self.block = block
        self.parent = parent
        self.children: dict[tuple[int, ...], "BlockTrieNode"] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ref_count(self) -> int:
        """Convenience proxy for the underlying block's reference count."""
        return self.block.ref_count

    def key(self) -> tuple[int, ...]:
        """Return the token IDs of this node as a hashable tuple (used as dict key)."""
        return tuple(self.token_ids)

    def __len__(self) -> int:
        """Return the number of token IDs stored in this node."""
        return len(self.token_ids)

    # ------------------------------------------------------------------
    # Child management
    # ------------------------------------------------------------------

    def add_child(self, token_ids: list[int], block: "Block") -> "BlockTrieNode":
        """
        Insert a child node for the given token IDs if one does not already exist.

        If a child with the same token IDs is already present, the existing node is
        returned unchanged (idempotent insertion).

        Parameters
        ----------
        token_ids : list[int]
            The token IDs that identify the child node.
        block : Block
            The KV-cache block to associate with the new child node.

        Returns
        -------
        BlockTrieNode
            The (possibly newly created) child node.
        """
        key = tuple(token_ids)
        child = self.children.get(key)
        if child is not None:
            return child

        child = BlockTrieNode(token_ids, block, parent=self)
        self.children[key] = child
        return child

    def remove_child(self, block_key: tuple[int, ...]) -> None:
        """
        Remove the child node identified by *block_key* from this node's children.

        Called during block eviction to detach the evicted node from the trie so
        that future prefix lookups do not follow stale references.

        Parameters
        ----------
        block_key : tuple[int, ...]
            The token-ID tuple that was used as the key when the child was added.
        """
        if block_key in self.children:
            self.children.pop(block_key)


class BlockTrieTree:
    """
    A trie (prefix tree) over KV-cache blocks used for prefix-cache matching.

    Each path from the root to a leaf node represents a sequence of token-ID
    chunks whose KV tensors are already cached.  When a new sequence arrives,
    the trie is walked to find the longest matching cached prefix, avoiding
    redundant computation.

    Attributes
    ----------
    root : BlockTrieNode
        The virtual root node.  It holds no token IDs or block and is never
        directly evicted.
    """

    def __init__(self):
        self.root: BlockTrieNode = BlockTrieNode([], None)

    def lookup_blocks(self, seq: Sequence) -> list[Block]:
        """
        Walk the trie to find the longest cached prefix for *seq*.

        Iterates over the token-ID chunks produced by ``seq.token_ids_in_chunks()``
        and follows matching child nodes until the first cache miss.  The result
        may be a full match (all chunks cached), a partial match, or an empty list
        (no cached prefix at all).

        Parameters
        ----------
        seq : Sequence
            The input sequence whose prompt tokens are used for prefix matching.

        Returns
        -------
        list[Block]
            Ordered list of :class:`Block` objects corresponding to the matched
            prefix chunks.  The list length is between 0 and
            ``len(seq.token_ids_in_chunks())``.
        """
        blocks: list[Block] = []
        node = self.root
        for token_ids_chunk in seq.token_ids_in_chunks():
            key = tuple(token_ids_chunk)
            if key not in node.children:
                break
            node = node.children[key]
            blocks.append(node.block)
        return blocks