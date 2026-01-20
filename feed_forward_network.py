import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.rms_norm = nn.RMSNorm(d_model)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.gelu = nn.GELU()
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        residual = input
        x = self.rms_norm(input)

        x = self.up_proj(x)
        x = self.gelu(x)
        x = self.down_proj(x)

        x = self.dropout(x)

        return x + residual

