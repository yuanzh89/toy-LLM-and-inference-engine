import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ToyLLMConfig
from engine.block_manager import BlockManager
from engine.sequence import Sequence
from kernel.triton.flash_attention import flash_attention
from layer.rope import apply_rope


class GQAWithKVCache(nn.Module):
    """
    Grouped-Query Attention (GQA) module with KV cache and chunked prefill.

    This implements the attention variant used in modern LLMs (e.g., LLaMA, PaLM),
    where multiple query heads share a smaller set of key/value heads to reduce
    memory and computation while preserving layer quality.

    Instead of having one K/V head per Q head (as in standard multi-head attention),
    GQA groups query heads so that:

        num_query_heads = num_kv_heads * group_size

    Each K/V head is broadcast across its corresponding group of query heads.

    When:
        num_query_heads == num_kv_heads

    this module becomes mathematically identical to standard Multi-Head Attention
    (each query head has its own independent key/value head).

    Args:
        d_model (int):
            Embedding dimension of the layer.

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
    """

    def __init__(self, layer_id: int, d_model: int, num_query_heads: int, num_kv_heads: int, dropout: float = 0.1,
                 llm_config: ToyLLMConfig, block_manager: BlockManager):
        super().__init__()

        assert d_model % num_query_heads == 0, "d_model must be divisible by num_query_heads"
        assert num_query_heads >= num_kv_heads, "num_query_heads must be >= num_kv_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"

        self.layer_id = layer_id
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.query_to_kv_heads_ratio = self.num_query_heads // self.num_kv_heads
        self.dropout = dropout
        self.llm_config = llm_config
        self.block_manager = block_manager

        self.head_dim = self.d_model // self.num_query_heads

        self.rms_norm = nn.RMSNorm(d_model, eps=1e-9)

        # Q, K, V projection matrix
        self.q_proj = nn.Linear(d_model, self.num_query_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, seq: Sequence, is_casual: bool = True) -> None:
        # [batch_size, seq_len, d_model]
        x = seq.activations

        batch_size, seq_len, d_model = x.shape

        residual = x

        # Pre-norm: [batch_size, seq_len, d_model]
        x = self.rms_norm(x)

        # Populate missing KV cache blocks
        block_idx = 0
        while block_idx < len(seq.kv_cache_blocks):
            block = seq.kv_cache_blocks[block_idx]
            if block.is_empty(self.layer_id):
                break

        # Apply RoPE before writing into KV cache, only for newly populated K cache tensors.
        if block_idx < len(seq.kv_cache_blocks):
            # Slice the partial sequence which is missing KV cache
            kv = x[:, block_idx * self.llm_config.max_token_size_per_kv_cache_block:, :]
            k = self.k_proj(kv).view()
            v = self.v_proj(kv)
            # TODO: Apply RoPE to new k tensors with position offset before writing into KV cache
            k_tensors = torch.split(k, self.llm_config.max_token_size_per_kv_cache_block, dim=1)
            v_tensors = torch.split(v, self.llm_config.max_token_size_per_kv_cache_block, dim=1)
            for offset, (k_tensor, v_tensor) in enumerate(zip(k_tensors, v_tensors)):
                seq.kv_cache_blocks[block_idx + offset].prefill_write_kv_cache(self.layer_id, k_tensor, v_tensor)

        # At this point, KV cache blocks are fully populated
        # [batch_size, seq_len, d_model]
        k = torch.concat([block.k_cache[self.layer_id] for block in seq.kv_cache_blocks], dim=1)
        v = torch.concat([block.v_cache[self.layer_id] for block in seq.kv_cache_blocks], dim=1)

        # Read full KV cache from blocks for attention calculation
        kv_tensors = [(block.k_cache[self.layer_id], block.v_cache[self.layer_id]) for block in seq.kv_cache_blocks]

        # Chunked prefill
        # Within the loop, we perform attention calculation per q_chunk and kv_block, then accumulate result in accumulator.
        # As the idea illustrated in flash attention, the computation across different q_chunk could happen in parallel.
        # We use our custom flash attention kernel to compute attention across each q_chink and kv_block.

        # Split q into chunks with size query_chunk_size for chunked prefill along the seq_len dimension.
        q = self.q_proj(x)
        q_chunks = torch.split(q, split_size_or_sections=self.llm_config.query_chunk_size, dim=1)
        accumulator = [torch.zeros_like(q_chunk) for q_chunk in q_chunks]
        for q_idx, q_chunk in enumerate(q_chunks):
            for k_tensor, v_tensor in kv_tensors:
                o_tensor = flash_attention(q_chunk, k_tensor, v_tensor)
                accumulator[q_idx] += o_tensor

        # [batch_size, num_heads,
        o = torch.stack(accumulator, dim=-1)

        # Apply RoPE to Q and K projections before attention calculation
        q, k = apply_rope(q, k)

        # Reshape q, k, v into groups
        # [batch_size, num_kv_heads, query_to_kv_heads_ratio, seq_len, head_dim]
        q = q.view(batch_size, self.num_kv_heads, self.query_to_kv_heads_ratio, seq_len, self.head_dim)
        # [batch_size, num_kv_heads, 1, seq_len, head_dim]
        k = k.unsqueeze(dim=2)
        # [batch_size, num_kv_heads, 1, seq_len, head_dim]
        v = v.unsqueeze(dim=2)

        # [batch_size, num_kv_heads, query_to_kv_heads_ratio, seq_len, seq_len]
        # Broadcast the single Key head across all the Query heads within each group.
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if is_casual:
            # [seq_len, seq_len]
            casual_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))
            # [1, 1, 1, seq_len, seq_len]
            casual_mask = casual_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(~casual_mask, torch.finfo(attn_scores.dtype).min)

        # F.softmax is safe softmax by default
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = self.attn_dropout(attn_scores)

        # [batch_size, num_kv_heads, query_to_kv_heads_ratio, seq_len, head_dim]
        attn_output = torch.matmul(attn_scores, v)
        # [batch_size, num_query_heads, seq_len, head_dim]
        attn_output = attn_output.view(batch_size, self.num_query_heads, seq_len, self.head_dim)
        # [batch_size, seq_len, num_query_heads, head_dim]
        attn_output = attn_output.transpose(1, 2)
        # [batch_size, seq_len, d_model]
        attn_output = attn_output.reshape(batch_size, seq_len, d_model)

        # [batch_size, seq_len, d_model]
        output = self.o_proj(attn_output)
        output = self.o_dropout(output)

        # Eventually write the output back to seq.activations
        # [batch_size, seq_len, d_model]
        seq.activations = residual + output
