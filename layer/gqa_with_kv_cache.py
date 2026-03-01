import math

import torch
import torch.nn as nn

from config import ToyLLMConfig
from engine.block_manager import BlockManager
from engine.sequence import Sequence
from kernel.triton.flash_attention import flash_attention
from layer.rope import apply_rope


class GQAWithKVCache(nn.Module):
    """
    Grouped-Query Attention (GQA) module with KV cache and chunked prefill.

    In GQA, query heads are divided into groups that share a single key/value
    head, reducing KV cache memory and bandwidth proportional to the group size:

        group_size = num_query_heads // num_kv_heads

    Query head i attends to KV head (i // group_size), so the first
    `group_size` query heads share KV head 0, the next `group_size` share
    KV head 1, and so on.  When num_query_heads == num_kv_heads this
    degenerates to standard Multi-Head Attention (MHA).

    KV Cache & Chunked Prefill
    --------------------------
    The module maintains a block-paged KV cache (one entry per
    `BlockManager` block).  On each forward pass it:
      1. Scans blocks to find the first one whose KV cache is empty for
         this layer.
      2. Runs K/V projections only on the uncached suffix of the sequence
         (chunked prefill) and writes the results into the empty blocks.
      3. Reads the full KV cache back from all blocks and expands each KV
         head to match its query-head group before calling flash attention.
      4. Accumulates per-query-chunk attention outputs and concatenates
         them to form the final output.

    Args:
        llm_config (ToyLLMConfig):
            Global model configuration (block size, chunk size, …).
        block_manager (BlockManager):
            Manages paged physical KV-cache blocks.
        layer_id (int):
            Index of this transformer layer; used to index into per-layer
            KV cache slots inside each block.
        d_model (int):
            Model (embedding) dimension.
        num_query_heads (int):
            Number of query attention heads.
        num_kv_heads (int):
            Number of key/value heads.  Must evenly divide
            `num_query_heads`.
        dropout (float, optional):
            Dropout probability applied to the output projection.
            Default: 0.1.

    Shape:
        Input  (via seq.activations): ``(batch_size, seq_len, d_model)``
        Output (via seq.activations): ``(batch_size, seq_len, d_model)``
    """

    def __init__(
            self,
            llm_config: ToyLLMConfig,
            block_manager: BlockManager,
            layer_id: int,
            d_model: int,
            num_query_heads: int,
            num_kv_heads: int,
            dropout: float = 0.1,
    ):
        super().__init__()

        assert d_model % num_query_heads == 0, \
            "d_model must be divisible by num_query_heads"
        assert num_query_heads >= num_kv_heads, \
            "num_query_heads must be >= num_kv_heads"
        assert num_query_heads % num_kv_heads == 0, \
            "num_query_heads must be divisible by num_kv_heads"

        self.llm_config = llm_config
        self.block_manager = block_manager
        self.layer_id = layer_id
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.dropout = dropout
        # How many Q heads share each single KV head.
        self.group_size = num_query_heads // num_kv_heads

        self.head_dim = d_model // num_query_heads

        self.rms_norm = nn.RMSNorm(d_model, eps=1e-9)

        self.q_proj = nn.Linear(d_model, num_query_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.o_dropout = nn.Identity() if math.isclose(self.dropout, 0.0) else nn.Dropout(dropout)

    def forward(self, seq: Sequence, is_causal: bool = True) -> None:
        # x: [batch_size, seq_len, d_model]
        x = seq.activations
        batch_size, seq_len, d_model = x.shape
        residual = x

        # Pre-norm
        x = self.rms_norm(x)  # [batch_size, seq_len, d_model]

        block_size = self.llm_config.max_token_size_per_kv_cache_block

        # ------------------------------------------------------------------ #
        # Step 1 – find the index of the first block whose KV cache is empty  #
        # for this layer (i.e. the start of the uncached suffix).             #
        # ------------------------------------------------------------------ #
        block_idx = 0
        while block_idx < len(seq.kv_cache_blocks):
            if seq.kv_cache_blocks[block_idx].is_empty(self.layer_id):
                break
            block_idx += 1

        # ------------------------------------------------------------------ #
        # Step 2 – compute K/V projections only for the uncached suffix and   #
        # write them into the empty blocks (chunked prefill write).           #
        # ------------------------------------------------------------------ #
        if block_idx < len(seq.kv_cache_blocks):
            # Slice just the tokens whose KV cache has not been written yet.
            partial_start = block_idx * block_size
            kv_input = x[:, partial_start:, :]  # [B, partial_len, d_model]
            partial_len = kv_input.shape[1]

            # K projection → [B, partial_len, num_kv_heads * head_dim]
            k_new = self.k_proj(kv_input)
            # Reshape and transpose → [B, num_kv_heads, partial_len, head_dim]
            k_new = (k_new
                     .view(batch_size, partial_len, self.num_kv_heads, self.head_dim)
                     .transpose(1, 2)
                     .contiguous())
            # Apply RoPE with correct absolute token offset for this suffix.
            apply_rope(k_new, self.head_dim, start_pos=partial_start)

            # V projection → [B, num_kv_heads, partial_len, head_dim]
            v_new = self.v_proj(kv_input)
            v_new = (v_new
                     .view(batch_size, partial_len, self.num_kv_heads, self.head_dim)
                     .transpose(1, 2)
                     .contiguous())

            # Split along the seq_len dimension
            k_blocks = torch.split(k_new, block_size, dim=2)
            v_blocks = torch.split(v_new, block_size, dim=2)

            for offset, (k_blk, v_blk) in enumerate(zip(k_blocks, v_blocks)):
                seq.kv_cache_blocks[block_idx + offset].prefill_write_kv_cache(
                    self.layer_id, k_blk, v_blk
                )

        # ------------------------------------------------------------------ #
        # Step 3 – reassemble the full KV cache from all blocks.              #
        # ------------------------------------------------------------------ #
        # Each block stores k_cache[layer_id] with shape
        # [B, num_kv_heads, block_seq_len, head_dim].
        k_full = torch.cat(
            [blk.k_cache[self.layer_id] for blk in seq.kv_cache_blocks], dim=2
        )  # [B, num_kv_heads, total_seq_len, head_dim]
        v_full = torch.cat(
            [blk.v_cache[self.layer_id] for blk in seq.kv_cache_blocks], dim=2
        )  # [B, num_kv_heads, total_seq_len, head_dim]

        # ------------------------------------------------------------------ #
        # Step 4 – GQA head expansion.                                        #
        # Each KV head is repeated `group_size` times so that query head i    #
        # attends to KV head (i // group_size).                               #
        # repeat_interleave turns [h0, h1] → [h0, h0, h1, h1] (for size=2),  #
        # which aligns KV heads with query heads in natural order.            #
        # ------------------------------------------------------------------ #
        if self.group_size > 1:
            k_full = k_full.repeat_interleave(self.group_size, dim=1)
            # → [B, num_query_heads, total_seq_len, head_dim]
            v_full = v_full.repeat_interleave(self.group_size, dim=1)

        # ------------------------------------------------------------------ #
        # Step 5 – Q projection, RoPE, then chunked-prefill attention.        #
        # ------------------------------------------------------------------ #
        q = self.q_proj(x)  # [B, seq_len, num_query_heads * head_dim]
        q = (q
             .view(batch_size, seq_len, self.num_query_heads, self.head_dim)
             .transpose(1, 2)
             .contiguous())  # [B, num_query_heads, seq_len, head_dim]
        # Q positions always start at 0 for the full sequence.
        apply_rope(q, self.head_dim, start_pos=0)

        q_chunks = torch.split(q, self.llm_config.query_chunk_size, dim=2)
        # Each chunk: [B, num_query_heads, chunk_size, head_dim]

        output_chunks = []
        for q_chunk in q_chunks:
            # flash_attention expects contiguous tensors.
            o_chunk = flash_attention(q_chunk.contiguous(), k_full, v_full)
            output_chunks.append(o_chunk)

        o = torch.cat(output_chunks, dim=2)  # [B, num_query_heads, seq_len, head_dim]

        # Merge heads back → [B, seq_len, d_model]
        o = o.transpose(1, 2).reshape(batch_size, seq_len, d_model)

        output = self.o_proj(o)  # [B, seq_len, d_model]
        output = self.o_dropout(output)

        # Residual connection; result is written back into the sequence object.
        seq.activations = residual + output
