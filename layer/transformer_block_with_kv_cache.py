import torch.nn as nn

from config import ToyLLMConfig
from engine.block_manager import BlockManager
from engine.sequence import Sequence
from layer.gqa_with_kv_cache import GQAWithKVCache
from swiglu_ffn import SwiGLUFFNLayer


class TransformerBlock(nn.Module):
    """
    A single transformer decoder block consisting of:

    1. **Grouped-Query Attention (GQA)** with paged KV cache and chunked prefill.
    2. **SwiGLU Feed-Forward Network** with pre-norm and residual connection.

    Both sub-layers receive the same ``query_chunk_idx`` so they operate on the
    same token-ID slice of the sequence's chunked activations.

    Parameters
    ----------
    llm_config : ToyLLMConfig
        Global model configuration (block size, chunk size, etc.).
    block_manager : BlockManager
        Manages paged physical KV-cache blocks.
    layer_id : int
        Zero-based index of this transformer layer; passed to :class:`GQAWithKVCache`
        to index into the per-layer KV cache slots.
    d_model : int
        Model (embedding) dimension.
    d_ff : int
        Inner feed-forward dimension.
    num_query_heads : int
        Number of query attention heads.
    num_kv_heads : int
        Number of key/value heads (must evenly divide ``num_query_heads``).
    dropout : float, optional
        Dropout probability applied in both GQA and FFN.  Default: 0.1.
    """

    def __init__(
            self,
            llm_config: ToyLLMConfig,
            block_manager: BlockManager,
            layer_id: int,
            d_model: int,
            d_ff: int,
            num_query_heads: int,
            num_kv_heads: int,
            dropout: float = 0.1,
    ):
        super().__init__()

        self.llm_config = llm_config
        self.block_manager = block_manager
        self.layer_id = layer_id
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.dropout = dropout

        self.group_query_attention = GQAWithKVCache(
            self.llm_config,
            self.block_manager,
            self.layer_id,
            self.d_model,
            self.num_query_heads,
            self.num_kv_heads,
            dropout=self.dropout,
        )
        self.swiglu_ffn = SwiGLUFFNLayer(self.d_model, self.d_ff, dropout=self.dropout)

    def forward(self, seq: Sequence, query_chunk_idx: int) -> None:
        """
        Run one transformer block on a single query chunk.

        Parameters
        ----------
        seq : Sequence
            The sequence being processed.  Activations are read from and written
            back to ``seq.chunked_activations[query_chunk_idx]``.
        query_chunk_idx : int
            Zero-based index of the query chunk within the sequence's chunked
            activations.
        """

        self.group_query_attention(seq, query_chunk_idx)
        self.swiglu_ffn(seq, query_chunk_idx)