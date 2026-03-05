import torch
import torch.nn as nn

from config import ToyLLMConfig
from engine.block_manager import BlockManager
from engine.sequence import Sequence, SequenceStatus
from layer.gqa_with_kv_cache import GQAWithKVCache
from swiglu_ffn import SwiGLUFFNLayer


class TransformerBlock(nn.Module):
    """
    A single decoder-only transformer block combining GQA and a SwiGLU FFN.

    Each block processes sequences in either prefill (one chunk at a time) or
    decode (one token per sequence, batched) mode.  All activation state is
    stored on the :class:`~engine.sequence.Sequence` objects; the block itself
    is stateless between calls.

    Parameters
    ----------
    llm_config : ToyLLMConfig
        Unified model and inference configuration.
    block_manager : BlockManager
        Passed through to :class:`~layer.gqa_with_kv_cache.GQAWithKVCache` so
        that KV-cache block allocation can happen during the decode forward pass.
    layer_id : int
        Zero-based layer index; used by GQA to read/write the correct KV-cache
        slot on each :class:`~engine.sequence.Sequence`.

    Attributes
    ----------
    group_query_attention : GQAWithKVCache
        The grouped query attention sub-layer for this block.
    swiglu_ffn : SwiGLUFFNLayer
        The SwiGLU feed-forward sub-layer for this block.
    """

    def __init__(
            self,
            llm_config: ToyLLMConfig,
            block_manager: BlockManager,
            layer_id: int,
    ):
        super().__init__()

        self.llm_config = llm_config
        self.block_manager = block_manager
        self.layer_id = layer_id

        self.group_query_attention = GQAWithKVCache(
            self.llm_config,
            self.block_manager,
            self.layer_id,
        )
        self.swiglu_ffn = SwiGLUFFNLayer(
            self.llm_config.d_model,
            self.llm_config.d_ff,
            dropout=self.llm_config.dropout,
        )

    @torch.inference_mode()
    def forward(
            self,
            sequences: list[Sequence],
            query_chunk_idxes: list[int] | None = None,
            is_prefill: bool = True,
    ) -> None:
        """
        Dispatch to :meth:`prefill_a_chunk` or :meth:`decode_a_batch`.

        Parameters
        ----------
        sequences : list[Sequence]
            For prefill: a single-element list.  For decode: one entry per sequence
            in the batch.
        query_chunk_idxes : list[int] | None
            Required for prefill; ignored for decode.
        is_prefill : bool
            ``True`` to run the prefill path, ``False`` for decode.
        """
        if is_prefill:
            assert len(sequences) == len(query_chunk_idxes) == 1
            assert sequences[0].status == SequenceStatus.PREFILL_PENDING
        else:
            assert all(
                seq.status == SequenceStatus.DECODE_PENDING for seq in sequences
            ), "All sequences must be in DECODE_PENDING status for batched decoding."

        if is_prefill:
            self.prefill_a_chunk(sequences[0], query_chunk_idxes[0])
        else:
            self.decode_a_batch(sequences)

    @torch.inference_mode()
    def decode_a_batch(self, sequences: list[Sequence]) -> None:
        """
        Run one decode step for a batch of sequences.

        Calls GQA (which appends new K/V to the cache and computes attention)
        followed by the FFN.  Both sub-layers receive ``is_prefill=False``.

        Parameters
        ----------
        sequences : list[Sequence]
            Batch of sequences in ``DECODE_PENDING`` status.
        """
        self.group_query_attention(sequences, is_prefill=False)
        self.swiglu_ffn(sequences, is_prefill=False)

    @torch.inference_mode()
    def prefill_a_chunk(self, seq: Sequence, query_chunk_idx: int) -> None:
        """
        Run one prefill step for a single query chunk.

        Parameters
        ----------
        seq : Sequence
            The sequence being prefilled.
        query_chunk_idx : int
            Zero-based index of the query chunk to process.
        """
        self.group_query_attention([seq], [query_chunk_idx], is_prefill=True)
        self.swiglu_ffn([seq], [query_chunk_idx], is_prefill=True)
