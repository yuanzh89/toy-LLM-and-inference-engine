import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFNLayer(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.rms_norm = nn.RMSNorm(d_model)

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.rms_norm(x)

        x1 = self.w1(x)
        x2 = self.w2(x)
        output = self.w3(x1 * F.silu(x2))

        output = self.dropout(output)

        return output + residual

class FusedSwiGLUFFNLayer(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.rms_norm = nn.RMSNorm(d_model)

        self.w12 = nn.Linear(d_model, hidden_dim * 2, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.rms_norm(x)

        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        output = self.w3(x1 * F.silu(x2))

        output = self.dropout(output)

        return output + residual