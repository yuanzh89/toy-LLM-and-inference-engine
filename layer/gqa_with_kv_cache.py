from __future__ import annotations

import math

import torch
import torch.nn as nn
from mpmath import residual
from torch.nn import functional as F

from config import ToyLLMConfig
from engine.block_manager import BlockManager
from engine.sequence import Sequence, SequenceStatus
from kernel.triton.flash_attention import flash_attention
from layer.rope import apply_rope


class GQAWithKVCache(nn.Module):
    """
    Grouped-Query Attention (GQA) with paged KV cache and chunked prefill.

    In GQA, query heads are divided into groups that each share a single key/value
    head, reducing KV-cache memory and bandwidth by a factor of ``group_size``::

        group_size = num_query_heads // num_kv_heads

    Query head ``i`` attends to KV head ``i // group_size``, so the first
    ``group_size`` query heads share KV head 0, the next ``group_size`` share KV
    head 1, and so on.  When ``num_query_heads == num_kv_heads`` this degenerates
    to standard Multi-Head Attention (MHA).

    KV Cache & Chunked Prefill
    --------------------------
    The module maintains a block-paged KV cache (one entry per
    :class:`~block_manager.BlockManager` block).  On each forward pass it:

    1. Scans the blocks to find the first one whose KV cache is empty for this
       layer (the start of the uncached suffix).
    2. Runs K/V projections only on the uncached suffix (chunked prefill) and
       writes the results into the empty blocks.
    3. Reads the full KV cache back from all blocks and expands each KV head to
       match its query-head group before calling flash attention.
    4. Writes the attention output back into
       ``seq.prefill_chunked_activations[query_chunk_idx]``.

    Parameters
    ----------
    llm_config : ToyLLMConfig
        Global model configuration (block size, chunk size, etc.).
    block_manager : BlockManager
        Manages paged physical KV-cache blocks.
    layer_id : int
        Index of this transformer layer; used to index into per-layer KV-cache
        slots inside each block.
    d_model : int
        Model (embedding) dimension.
    num_query_heads : int
        Number of query attention heads.
    num_kv_heads : int
        Number of key/value heads.  Must evenly divide ``num_query_heads``.
    dropout : float, optional
        Dropout probability applied to the output projection.  Default: 0.1.

    Shape
    -----
    Input / output (via ``seq.prefill_chunked_activations``): ``[B, chunk_seq_len, d_model]``
    """

    def __init__(
            self,
            llm_config: ToyLLMConfig,
            block_manager: BlockManager,
            layer_id: int,
    ):
        super().__init__()

        assert self.llm_config.d_model % self.llm_config.num_query_heads == 0, \
            "d_model must be divisible by num_query_heads"
        assert self.llm_config.num_query_heads >= self.llm_config.num_kv_heads, \
            "num_query_heads must be >= num_kv_heads"
        assert self.llm_config.num_query_heads % self.llm_config.num_kv_heads == 0, \
            "num_query_heads must be divisible by num_kv_heads"

        self.llm_config = llm_config
        self.block_manager = block_manager
        self.layer_id = layer_id

        # How many Q heads share each single KV head.
        self.group_size = self.llm_config.num_query_heads // self.llm_config.num_kv_heads

        self.head_dim = self.llm_config.d_model // self.llm_config.num_query_heads

        self.rms_norm = nn.RMSNorm(self.llm_config.d_model, eps=1e-9)

        self.q_proj = nn.Linear(self.llm_config.d_model, self.llm_config.num_query_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.llm_config.d_model, self.llm_config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.llm_config.d_model, self.llm_config.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.llm_config.d_model, self.llm_config.d_model, bias=False)

        self.o_dropout = nn.Identity() if math.isclose(self.dropout, 0.0) else nn.Dropout(self.llm_config.dropout)

    def forward(self, sequences: list[Sequence], query_chunk_idxes: list[int] | None, is_prefill: bool = True) -> None:
        """
        Run GQA on a single query chunk and write the result back into the sequence.

        Parameters
        ----------
        sequences : list[Sequence]
            A list of sequences to be processed at each forward step.
            There are two modes of operation:
            1. Prefill mode, where we always guarantee that the len(sequences) == 1
            2. Decode mode, where  1 <= len(sequences) <= decode_max_batch_size

            We could tell the current operation mode by reading the len(sequences) and
            sequence.status == PREFILL_PENDING

        query_chunk_idxes : list[int]
            Zero-based index of the query chunk to process.  Activations are read
            from ``seq.prefill_chunked_activations[query_chunk_idx]`` and written back to
            the same slot after the residual connection.

            This parameter only make sense in chunked prefill mode, and will be None and ignored in decode mode.
        """
        if is_prefill:
            assert len(sequences) == len(query_chunk_idxes) == 1
            assert sequences[0].status == SequenceStatus.PREFILL_PENDING
        else:
            assert all([seq.status == SequenceStatus.DECODE_PENDING for seq in
                        sequences]), "All input sequences should be in DECODE_PENDING status for batched decoding."

        if is_prefill:
            self.prefill_a_chunk(sequences[0], query_chunk_idxes[0])
        else:
            self.decode_a_batch(sequences)

    def prefill_a_chunk(self, seq: Sequence, query_chunk_idx: int) -> None:
        """
        Run GQA on a single query chunk and write the result back into the sequence.

        Parameters
        ----------
        seq : Sequence
            The sequence being processed.
        query_chunk_idx : int
            Zero-based index of the query chunk to process.  Activations are read
            from ``seq.prefill_chunked_activations[query_chunk_idx]`` and written back to
            the same slot after the residual connection.
        """
        # x: [B, chunk_seq_len, d_model]
        x = seq.get_query_chunk_activations(query_chunk_idx)
        batch_size, seq_len, d_model = x.shape
        residual = x

        # Pre-norm before attention projections.
        x = self.rms_norm(x)  # [B, seq_len, d_model]

        block_size = self.llm_config.max_token_size_per_kv_cache_block

        # ------------------------------------------------------------------ #
        # Step 1 – Find the first block whose KV cache is empty for this      #
        # layer (the start of the uncached suffix).                           #
        # ------------------------------------------------------------------ #
        block_idx = 0
        while block_idx < len(seq.kv_cache_blocks):
            if seq.kv_cache_blocks[block_idx].is_empty(self.layer_id):
                break
            block_idx += 1

        # ------------------------------------------------------------------ #
        # Step 2 – Compute K/V projections only for the uncached suffix and   #
        # write them into the empty blocks (chunked-prefill write).           #
        # ------------------------------------------------------------------ #
        if block_idx < len(seq.kv_cache_blocks):
            # Slice the tokens whose KV cache has not been computed yet.
            partial_start = block_idx * block_size
            kv_input = x[:, partial_start:, :]  # [B, partial_len, d_model]
            partial_len = kv_input.shape[1]

            # K projection → reshape → [B, num_kv_heads, partial_len, head_dim]
            k_new = self.k_proj(kv_input)
            k_new = (
                k_new
                .view(batch_size, partial_len, self.num_kv_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )
            # Apply RoPE with the correct absolute offset for this suffix.
            apply_rope(k_new, self.head_dim, start_pos=partial_start)

            # V projection → reshape → [B, num_kv_heads, partial_len, head_dim]
            v_new = self.v_proj(kv_input)
            v_new = (
                v_new
                .view(batch_size, partial_len, self.num_kv_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )

            # Split the partial K/V tensors into per-block slices along seq_len
            # (dim=2) and write each slice into the corresponding empty block.
            k_blocks = torch.split(k_new, block_size, dim=2)
            v_blocks = torch.split(v_new, block_size, dim=2)

            for offset, (k_blk, v_blk) in enumerate(zip(k_blocks, v_blocks)):
                seq.kv_cache_blocks[block_idx + offset].prefill_write_kv_cache(
                    self.layer_id, k_blk, v_blk
                )

        # ------------------------------------------------------------------ #
        # Step 3 – Reassemble the full KV cache from all blocks.              #
        # Each block's k_cache[layer_id] has shape                           #
        # [B, num_kv_heads, block_seq_len, head_dim].                        #
        # ------------------------------------------------------------------ #
        k_full = torch.cat(
            [blk.k_cache[self.layer_id] for blk in seq.kv_cache_blocks], dim=2
        )  # [B, num_kv_heads, total_seq_len, head_dim]
        v_full = torch.cat(
            [blk.v_cache[self.layer_id] for blk in seq.kv_cache_blocks], dim=2
        )  # [B, num_kv_heads, total_seq_len, head_dim]

        # ------------------------------------------------------------------ #
        # Step 4 – GQA head expansion.                                        #
        # Each KV head is repeated ``group_size`` times so that query head i  #
        # attends to KV head ``i // group_size``.                             #
        # repeat_interleave turns [h0, h1] → [h0, h0, h1, h1] (for size=2), #
        # aligning KV heads with query heads in their natural order.          #
        # ------------------------------------------------------------------ #
        if self.group_size > 1:
            k_full = k_full.repeat_interleave(self.group_size, dim=1)
            v_full = v_full.repeat_interleave(self.group_size, dim=1)
            # Both now → [B, num_query_heads, total_seq_len, head_dim]

        # ------------------------------------------------------------------ #
        # Step 5 – Q projection, RoPE, then chunked-prefill attention.        #
        # ------------------------------------------------------------------ #
        q = self.q_proj(x)  # [B, seq_len, num_query_heads * head_dim]
        q = (
            q
            .view(batch_size, seq_len, self.num_query_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )  # [B, num_query_heads, seq_len, head_dim]

        # Q positions always start at absolute position 0 for the full sequence.
        apply_rope(q, self.head_dim, start_pos=0)

        o = flash_attention(q.contiguous(), k_full, v_full)

        # Merge heads back → [B, seq_len, d_model]
        o = o.transpose(1, 2).reshape(batch_size, seq_len, d_model)

        output = self.o_proj(o)  # [B, seq_len, d_model]
        output = self.o_dropout(output)

        # The result must be stored back into the specific chunk slot.
        seq.prefill_chunked_activations[query_chunk_idx] = residual + output

    def decode_a_batch(self, sequences: list[Sequence]) -> None:
        batch_size = len(sequences)

        # Batched decoding starts
        activations = [seq.decode_activations for seq in sequences]
        # [batch_size = num_seq, seq_len == 1, d_model]
        x = torch.cat(activations, dim=0)
        # [batch_size = num_seq, seq_len == 1, d_model]
        x = self.rms_norm(x)

        # [batch_size, num_heads, seq_len == 1, head_dim]
        q = self.q_proj(x).view(
            batch_size,
            1,
            self.llm_config.num_query_heads,
            self.head_dim).transpose(1, 2).contiguous()
        k = self.k_proj(x).view(batch_size, 1, self.llm_config.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(x).view(batch_size, 1, self.llm_config.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        pos_offsets = [len(seq.token_ids) - 1 for seq in sequences]
        # Apply RoPE on Q and K
        apply_rope(q, self.head_dim, start_pos=pos_offsets)
        apply_rope(k, self.head_dim, start_pos=pos_offsets)

        for idx, seq in enumerate(sequences):
            # Append new KV cache into matching sequences
            seq.append_kv_cache(self.layer_id, k[idx, :, :, :], v[idx, :, :, :])

        # Get sequence lengths for each KV cache
        kv_seq_lens = []
        for seq in sequences:
            total_len = sum(blk.k_cache[self.layer_id].shape[2] for blk in seq.kv_cache_blocks)
            kv_seq_lens.append(total_len)

        max_kv_len = max(kv_seq_lens)

        k_full = []
        v_full = []
        for seq in sequences:
            k = torch.cat([blk.k_cache[self.layer_id] for blk in seq.kv_cache_blocks], dim=2)
            v = torch.cat([blk.v_cache[self.layer_id] for blk in seq.kv_cache_blocks], dim=2)

            # Pad to max_kv_len on the seq_len dimension (dim=2)
            pad_len = max_kv_len - k.shape[2]
            if pad_len > 0:
                k = F.pad(k, (0, 0, 0, pad_len))  # pad last two dims: (head_dim, seq_len)
                v = F.pad(v, (0, 0, 0, pad_len))

            k_full.append(k)
            v_full.append(v)

        # [batch_size, num_kv_heads, max_kv_len, head_dim]
        k_full = torch.cat(k_full, dim=0)
        v_full = torch.cat(v_full, dim=0)

        # Build attention mask: True = ignore this position
        # [batch_size, 1, q_len=1, max_kv_len]
        attn_mask = torch.zeros(batch_size, 1, 1, max_kv_len, dtype=torch.bool, device=k_full.device)
        for i, seq_len in enumerate(kv_seq_lens):
            attn_mask[i, 0, 0, seq_len:] = True  # mask out padding

        o = F.scaled_dot_product_attention(
            q, k_full, v_full,
            attn_mask=~attn_mask,
        )

        o = o.transpose(1, 2).reshape(batch_size, 1, self.llm_config.d_model)

        o = self.o_proj(o)
        o = self.o_dropout(o)

        output = o + residual

        for idx, seq in enumerate(sequences):
            # Decode activation shape: [batch_size == 1, seq_len == 1, d_model] per sequence
            seq.decode_activations = output[idx, :, :]
