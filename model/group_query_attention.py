import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import apply_rope


class GroupQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) module.

    This implements the attention variant used in modern LLMs (e.g., LLaMA, PaLM),
    where multiple query heads share a smaller set of key/value heads to reduce
    memory and computation while preserving model quality.

    Instead of having one K/V head per Q head (as in standard multi-head attention),
    GQA groups query heads so that:

        num_query_heads = num_kv_heads * group_size

    Each K/V head is broadcast across its corresponding group of query heads.

    When:
        num_query_heads == num_kv_heads

    this module becomes mathematically identical to standard Multi-Head Attention
    (each query head has its own independent key/value head).

    Workflow:
        1. Project input into Q, K, V using separate linear layers.
        2. Reshape into head format:
               Q -> [B, Hq, S, D]
               K/V -> [B, Hkv, S, D]
        3. Expand K/V across query groups.
        4. Compute scaled dot-product attention.
        5. Apply optional causal mask.
        6. Softmax + dropout on attention probabilities.
        7. Weighted sum with V.
        8. Merge heads and apply output projection.
        9. Apply output dropout and residual connection.

    Args:
        d_model (int):
            Embedding dimension of the model.

        num_query_heads (int):
            Number of query heads.

        num_kv_heads (int):
            Number of key/value heads (must divide num_query_heads).

        dropout (float, optional):
            Dropout probability applied to attention probabilities and output
            projection. Default: 0.1

    Shape:
        Input:
            (batch_size, seq_len, d_model)

        Output:
            (batch_size, seq_len, d_model)

    Notes:
        - Uses scaled dot-product attention with head_dim = d_model / num_query_heads.
        - Causal masking prevents tokens from attending to future positions.
        - Numerically stabilized softmax via max subtraction.
    """

    def __init__(self, d_model: int, num_query_heads: int, num_kv_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % num_query_heads == 0, "d_model must be divided by num_query_heads"
        assert num_query_heads >= num_kv_heads, "num_query_heads must be greater than or equal to num_kv_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.query_to_kv_heads_ratio = self.num_query_heads // self.num_kv_heads
        self.dropout = dropout

        self.head_dim = self.d_model // self.num_query_heads

        self.rms_norm = nn.RMSNorm(d_model)

        self.q_proj = nn.Linear(d_model, self.num_query_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = None if math.isclose(self.dropout, 0.0) else nn.Dropout(p=self.dropout)
        self.o_dropout = None if math.isclose(self.dropout, 0.0) else nn.Dropout(p=self.dropout)

    def forward(self, x: torch.Tensor, apply_casual_mask: bool) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()

        # [batch_size, seq_len, d_model]
        x = self.rms_norm(x)

        q = self.q_proj(x)  # [batch_size, seq_len, num_query_heads * head_dim]
        k = self.k_proj(x)  # [batch_size, seq_len, num_kv_heads * head_dim]
        v = self.v_proj(x)  # [batch_size, seq_len, num_kv_heads * head_dim]

        # [batch_size, num_query_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_query_heads, self.head_dim).transpose(1, 2)
        # [batch_size, num_kv_heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # [batch_size, num_kv_heads, seq_len, head_dim]
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K projections before attention calculation
        q, k = apply_rope(q, k)

        # Expand KV heads to match Q heads
        # [batch_size, num_query_heads, seq_len, head_dim]
        k = k.repeat_interleave(self.query_to_kv_heads_ratio, dim=1)
        v = v.repeat_interleave(self.query_to_kv_heads_ratio, dim=1)

        # [batch_size, num_query_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if apply_casual_mask:
            # [seq_len, seq_len]
            casual_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device))
            # [1, 1, seq_len, seq_len]
            casual_mask = casual_mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(casual_mask == 0, float('-inf'))

        # Safe softmax
        attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True).values
        attn_scores = F.softmax(attn_scores, dim=-1)

        if self.attn_dropout is not None:
            attn_scores = self.attn_dropout(attn_scores)

        # [batch_size, num_query_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_scores, v)
        # [batch_size, seq_len, num_query_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [batch_size, seq_len, d_model]
        attn_output = attn_output.view(batch_size, seq_len, d_model)

        # [batch_size, seq_len, d_model]
        output = self.o_proj(attn_output)

        if self.o_dropout is not None:
            output = self.o_dropout(output)

        # [batch_size, seq_len, d_model]
        return x + output
