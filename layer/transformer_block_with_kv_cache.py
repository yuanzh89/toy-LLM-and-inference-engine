import torch.nn as nn

from config import ToyLLMConfig
from engine.block_manager import BlockManager
from engine.sequence import Sequence
from layer.gqa_with_kv_cache import GQAWithKVCache
from swiglu_ffn import SwiGLUFFNLayer


class TransformerBlock(nn.Module):
    def __init__(
            self,
            llm_config: ToyLLMConfig,
            block_manager: BlockManager,
            layer_id: int,
            d_model: int,
            d_ff: int,
            num_query_heads: int,
            num_kv_heads: int,
            dropout: float = 0.1
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

        self.gqa = GQAWithKVCache(self.llm_config, self.block_manager, self.layer_id, self.d_model,
                                  self.num_query_heads, self.num_kv_heads, dropout=self.dropout)
        self.swiglu_ffn = SwiGLUFFNLayer(self.d_model, self.d_ff, dropout=self.dropout)

    def forward(self, seq: Sequence, query_chunk_idx: int) -> None:
        self.group_query_attention(seq, query_chunk_idx)
        self.swiglu_ffn(seq)
