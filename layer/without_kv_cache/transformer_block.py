from layer.without_kv_cache.group_query_attention import *
from layer.swiglu_ffn import FusedSwiGLUFFNLayer


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_query_heads: int, num_kv_heads: int, dropout: float = 0.1):
        super().__init__()

        self.mha = GroupQueryAttention(d_model, num_query_heads, num_kv_heads, dropout=dropout)
        self.ffn = FusedSwiGLUFFNLayer(d_model, d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        output = self.mha(x, mask)
        output = self.ffn(output)
        return output
