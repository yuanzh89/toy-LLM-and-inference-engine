import torch
import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from feed_forward_network import FeedForwardNetwork


class Transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1,
                 device: torch.device = torch.device('cuda')):
        super().__init__()

        self.device = device

        self.mha = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout=dropout)

    def forward(self, input: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        output = self.mha(input, mask, device=self.device)
        output = self.ffn(output, device=self.device)
        return output
