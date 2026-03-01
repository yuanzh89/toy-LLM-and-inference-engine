import math
from copy import copy
from enum import IntEnum
from itertools import count

from kv_cache_block import Block


class SequenceStatus(IntEnum):
    """
    Lifecycle state of a :class:`Sequence`.

    Attributes
    ----------
    FINISHED : int
        The sequence has completed generation (EOS reached or max length hit).
    RUNNING : int
        The sequence is actively being processed (prefill or decode step in progress).
    WAITING : int
        The sequence is queued and waiting for KV-cache blocks to be allocated.
    """
    FINISHED = 1
    RUNNING = 2
    WAITING = 3
    FAILED = 4


class Sequence:
    """
    Represents a single request/sequence being processed by the inference engine.

    A :class:`Sequence` owns a reference to a contiguous list of token IDs that
    grows during decoding.  It also tracks which physical KV-cache
    :class:`~kv_cache_block.Block` objects hold its cached key/value tensors.

    Attributes
    ----------
    seq_id : int
        Unique, auto-incremented sequence identifier.
    status : SequenceStatus
        Current lifecycle state of the sequence.
    token_ids : list[int]
        All token IDs seen so far (prompt + generated tokens).
    num_prompt_tokens : int
        Number of tokens in the original prompt (fixed after construction).
    max_token_size_per_kv_cache_block : int
        Block granularity; used when splitting token IDs into chunks.
    max_sequence_length : int
        Hard cap on the total number of tokens (prompt + decode).
    kv_cache_blocks : list[Block | None]
        Ordered list of KV-cache blocks aligned with ``token_ids_in_chunks()``.
        Populated by :class:`~block_manager.BlockManager`; entries may be ``None``
        temporarily between ``reset_kv_cache_blocks`` and ``update_kv_cache_blocks``.
    kv_cache_blocks_initialized : bool
        ``True`` once ``update_kv_cache_blocks`` has been called successfully.
    """

    counter = count()

    def __init__(
            self,
            token_ids: list[int],
            max_token_size_per_kv_cache_block: int,
            max_sequence_length: int,
    ):
        self.seq_id: int = next(Sequence.counter)
        self.status: SequenceStatus = SequenceStatus.WAITING
        self.token_ids: list[int] = copy(token_ids)
        self.num_prompt_tokens: int = len(token_ids)
        self.max_token_size_per_kv_cache_block = max_token_size_per_kv_cache_block
        self.max_sequence_length = max_sequence_length
        # Populated by BlockManager via update_kv_cache_blocks().
        # Blocks are sorted by their order in the Trie tree, so that the previous block should be the parent of the next block in prefix caching Trie tree.
        self.kv_cache_blocks: list["Block | None"] = []
        self.kv_cache_blocks_initialized: bool = False
        # Activations of the sequence in the current state.
        # Activations should be initialized by the tokenizer, then updated sequentially by the embedding layer, multiple transformer blocks and finally the LM head layer.
        # [batch_size, seq_len, d_model]
        self.activations = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_decode_tokens(self) -> int:
        """Number of tokens generated so far beyond the original prompt."""
        return len(self.token_ids) - self.num_prompt_tokens

    def __len__(self) -> int:
        """Total number of tokens in this sequence (prompt + generated)."""
        return len(self.token_ids)

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def append_token(self, token_id: int) -> None:
        """
        Append a newly decoded token to the sequence.

        Parameters
        ----------
        token_id : int
            The token ID produced by the model at the current decode step.
        """
        self.token_ids.append(token_id)

    def token_ids_in_chunks(self) -> list[list[int]]:
        """
        Partition ``token_ids`` into fixed-size chunks aligned to KV-cache blocks.

        All chunks except possibly the last will have exactly
        ``max_token_size_per_kv_cache_block`` elements.  The final chunk may be
        shorter if the total token count is not evenly divisible.

        Returns
        -------
        list[list[int]]
            Ordered list of token-ID chunks.  The concatenation of all chunks
            equals ``self.token_ids``.
        """
        chunks = []
        for i in range(0, len(self.token_ids), self.max_token_size_per_kv_cache_block):
            chunks.append(self.token_ids[i: i + self.max_token_size_per_kv_cache_block])
        return chunks

    # ------------------------------------------------------------------
    # KV-cache block management
    # ------------------------------------------------------------------

    def reset_kv_cache_blocks(self, num_chunks: int) -> None:
        """
        Clear the KV-cache block list and pre-allocate ``num_chunks`` ``None`` slots.

        Called by :class:`~block_manager.BlockManager` at the start of
        ``allocate_blocks`` before performing prefix matching and new-block
        allocation.

        Parameters
        ----------
        num_chunks : int
            Expected number of KV-cache blocks needed for the current token list,
            i.e. ``ceil(len(token_ids) / max_token_size_per_kv_cache_block)``.
        """
        self.kv_cache_blocks = [None] * num_chunks
        self.kv_cache_blocks_initialized = False

    def update_kv_cache_blocks(self, kv_cache_blocks: list["Block"]) -> None:
        """
        Replace the KV-cache block list with the fully resolved set of blocks.

        Called by :class:`~block_manager.BlockManager` after prefix matching and
        new-block allocation are both complete.  The length of *kv_cache_blocks*
        must equal the number of token-ID chunks for the current token list.

        Parameters
        ----------
        kv_cache_blocks : list[Block]
            Ordered list of :class:`~kv_cache_block.Block` objects; one per chunk.
        """
        expected = math.ceil(len(self.token_ids) / self.max_token_size_per_kv_cache_block)
        assert len(kv_cache_blocks) == expected, (
            f"Expected {expected} KV-cache blocks for {len(self.token_ids)} tokens "
            f"(block size {self.max_token_size_per_kv_cache_block}), "
            f"got {len(kv_cache_blocks)}."
        )
        self.kv_cache_blocks = copy(kv_cache_blocks)
        self.kv_cache_blocks_initialized = True

    def release(self) -> None:
        """
        Explicitly release all KV-cache block references.
        Must be called by the scheduler.

        Decrements the ``ref_count`` of every non-``None`` block so that the
        :class:`~block_manager.BlockManager` knows these blocks are no longer
        pinned and may be eligible for eviction.
        """
        for block in self.kv_cache_blocks:
            if block is not None:
                block.dec_ref_count()
