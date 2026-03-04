from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import ToyLLMConfig
from engine.block_manager import BlockManager
from engine.sequence import Sequence, SequenceStatus
from kernel.triton.flash_attention import flash_attention
from layer.rope import apply_rope


class GQAWithKVCache(nn.Module):
    """
    Grouped Query Attention (GQA) with paged KV-cache support.

    Each instance is responsible for one transformer layer.  During prefill, it
    processes one query chunk at a time and writes newly computed K/V slices into
    the corresponding KV-cache blocks.  During decode, it performs batched single-
    token attention across the full cached KV context for all sequences in the batch.

    Parameters
    ----------
    llm_config : ToyLLMConfig
        Unified model and inference configuration.
    block_manager : BlockManager
        Manages paged physical KV-cache blocks; passed through to
        :meth:`~engine.sequence.Sequence.append_kv_cache` so that new decode
        blocks can be allocated when the current block fills up.
    layer_id : int
        Zero-based index of this attention layer within the model.

    Attributes
    ----------
    group_size : int
        Number of query heads that share each KV head
        (``num_query_heads // num_kv_heads``).
    head_dim : int
        Per-head feature dimension (``d_model // num_query_heads``).
    rms_norm : nn.RMSNorm
        Pre-attention layer normalization.
    q_proj, k_proj, v_proj : nn.Linear
        Query, key, and value projection matrices (no bias).
    o_proj : nn.Linear
        Output projection matrix (no bias).
    o_dropout : nn.Dropout | nn.Identity
        Output dropout (identity when ``dropout == 0``).
    """

    def __init__(
            self,
            llm_config: ToyLLMConfig,
            block_manager: BlockManager,
            layer_id: int,
    ):
        super().__init__()

        # Assign config first; assertions below reference it via local variable.
        self.llm_config = llm_config
        self.block_manager = block_manager
        self.layer_id = layer_id

        assert llm_config.d_model % llm_config.num_query_heads == 0, \
            "d_model must be divisible by num_query_heads"
        assert llm_config.num_query_heads >= llm_config.num_kv_heads, \
            "num_query_heads must be >= num_kv_heads"
        assert llm_config.num_query_heads % llm_config.num_kv_heads == 0, \
            "num_query_heads must be divisible by num_kv_heads"

        # How many Q heads share each single KV head.
        self.group_size = llm_config.num_query_heads // llm_config.num_kv_heads
        self.head_dim = llm_config.d_model // llm_config.num_query_heads

        self.rms_norm = nn.RMSNorm(llm_config.d_model, eps=1e-9)

        self.q_proj = nn.Linear(llm_config.d_model, llm_config.num_query_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(llm_config.d_model, llm_config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(llm_config.d_model, llm_config.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(llm_config.d_model, llm_config.d_model, bias=False)

        self.o_dropout = (
            nn.Identity()
            if math.isclose(llm_config.dropout, 0.0)
            else nn.Dropout(llm_config.dropout)
        )

    @torch.inference_mode()
    def forward(
            self,
            sequences: list[Sequence],
            query_chunk_idxes: list[int] | None = None,
            is_prefill: bool = True,
    ) -> None:
        """
        Dispatch to :meth:`prefill_a_chunk` or :meth:`decode_a_batch`.

        Parameters
        ----------
        sequences : list[Sequence]
            For prefill: a single-element list.  For decode: one entry per sequence
            in the batch.
        query_chunk_idxes : list[int] | None
            Required for prefill; ignored for decode.
        is_prefill : bool
            ``True`` to run the prefill path, ``False`` for decode.
        """
        if is_prefill:
            assert len(sequences) == len(query_chunk_idxes) == 1
            assert sequences[0].status == SequenceStatus.PREFILL_PENDING
        else:
            assert all(
                seq.status == SequenceStatus.DECODE_PENDING for seq in sequences
            ), "All sequences must be in DECODE_PENDING status for batched decoding."

        if is_prefill:
            self.prefill_a_chunk(sequences[0], query_chunk_idxes[0])
        else:
            self.decode_a_batch(sequences)

    @torch.inference_mode()
    def prefill_a_chunk(self, seq: Sequence, query_chunk_idx: int) -> None:
        """
        Run GQA for one query chunk during the prefill pass.

        Only computes K/V projections for blocks that are not yet cached (the
        uncached suffix of the sequence).  Already-cached prefix blocks are read
        directly from the KV store.  The full K/V context (cached + newly computed)
        is then assembled and used to compute attention for the query chunk.

        Activation update is written back to ``seq.prefill_chunked_activations``
        at the query chunk's index.

        Parameters
        ----------
        seq : Sequence
            The sequence being prefilled.
        query_chunk_idx : int
            Zero-based index of the query chunk to process.
        """
        # x: [1, chunk_seq_len, d_model]
        x = seq.get_query_chunk_activations(query_chunk_idx)
        batch_size, seq_len, d_model = x.shape
        residual = x

        # Pre-norm before attention projections.
        x = self.rms_norm(x)  # [1, seq_len, d_model]

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
        # write them into the empty blocks.                                   #
        # ------------------------------------------------------------------ #
        if block_idx < len(seq.kv_cache_blocks):
            # Slice the tokens whose KV cache has not been computed yet.
            partial_start = block_idx * block_size
            kv_input = x[:, partial_start:, :]  # [1, partial_len, d_model]
            partial_len = kv_input.shape[1]

            # K projection → [1, num_kv_heads, partial_len, head_dim]
            k_new = self.k_proj(kv_input)
            k_new = (
                k_new
                .view(batch_size, partial_len, self.llm_config.num_kv_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )
            # Apply RoPE with the correct absolute offset for this suffix.
            apply_rope(k_new, self.head_dim, start_pos=partial_start)

            # V projection → [1, num_kv_heads, partial_len, head_dim]
            v_new = self.v_proj(kv_input)
            v_new = (
                v_new
                .view(batch_size, partial_len, self.llm_config.num_kv_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )

            # Split into per-block slices along the seq_len dimension (dim=2)
            # and write each slice into the corresponding empty block.
            k_blocks = torch.split(k_new, block_size, dim=2)
            v_blocks = torch.split(v_new, block_size, dim=2)

            for offset, (k_blk, v_blk) in enumerate(zip(k_blocks, v_blocks)):
                seq.kv_cache_blocks[block_idx + offset].prefill_write_kv_cache(
                    self.layer_id, k_blk, v_blk
                )

        # ------------------------------------------------------------------ #
        # Step 3 – Reassemble the full KV cache from all blocks.              #
        # Each block's cached tensor has shape                                #
        # [1, num_kv_heads, block_seq_len, head_dim].                        #
        # ------------------------------------------------------------------ #
        k_full = torch.cat(
            [blk.k_cache[self.layer_id] for blk in seq.kv_cache_blocks], dim=2
        )  # [1, num_kv_heads, total_seq_len, head_dim]
        v_full = torch.cat(
            [blk.v_cache[self.layer_id] for blk in seq.kv_cache_blocks], dim=2
        )  # [1, num_kv_heads, total_seq_len, head_dim]

        # ------------------------------------------------------------------ #
        # Step 4 – GQA head expansion.                                        #
        # repeat_interleave turns [h0, h1] → [h0, h0, h1, h1] (group_size=2) #
        # so that each query head aligns with the correct KV head.           #
        # ------------------------------------------------------------------ #
        if self.group_size > 1:
            k_full = k_full.repeat_interleave(self.group_size, dim=1)
            v_full = v_full.repeat_interleave(self.group_size, dim=1)
            # Both: [1, num_query_heads, total_seq_len, head_dim]

        # ------------------------------------------------------------------ #
        # Step 5 – Q projection, RoPE, then chunked-prefill attention.        #
        # ------------------------------------------------------------------ #
        q = self.q_proj(x)  # [1, seq_len, num_query_heads * head_dim]
        q = (
            q
            .view(batch_size, seq_len, self.llm_config.num_query_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )  # [1, num_query_heads, seq_len, head_dim]

        # Q positions start at the absolute offset of this query chunk.
        q_start_pos = query_chunk_idx * block_size
        apply_rope(q, self.head_dim, start_pos=q_start_pos)

        o = flash_attention(q.contiguous(), k_full, v_full)

        # Merge heads back → [1, seq_len, d_model]
        o = o.transpose(1, 2).reshape(batch_size, seq_len, d_model)

        output = self.o_proj(o)  # [1, seq_len, d_model]
        output = self.o_dropout(output)

        seq.prefill_chunked_activations[query_chunk_idx] = residual + output

    @torch.inference_mode()
    def decode_a_batch(self, sequences: list[Sequence]) -> None:
        """
        Run GQA for one decode step across a batch of sequences.

        Each sequence contributes a single query token.  After projecting Q/K/V,
        the new K/V slice is appended to each sequence's KV cache.  The full
        cached K/V (padded to the longest sequence in the batch) is then used
        for scaled dot-product attention with a padding mask.

        Activation updates are written back per-sequence to
        ``seq.decode_activations``.

        Parameters
        ----------
        sequences : list[Sequence]
            Batch of sequences in ``DECODE_PENDING`` status.
        """
        batch_size = len(sequences)

        activations = [seq.decode_activations for seq in sequences]
        # [batch_size, 1, d_model]
        x = torch.cat(activations, dim=0)
        residual = x  # Save pre-norm for residual connection.
        x = self.rms_norm(x)

        # Project Q, K, V for the single new token per sequence.
        # Shapes: [batch_size, num_heads, 1, head_dim]
        q = (
            self.q_proj(x)
            .view(batch_size, 1, self.llm_config.num_query_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        k = (
            self.k_proj(x)
            .view(batch_size, 1, self.llm_config.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.v_proj(x)
            .view(batch_size, 1, self.llm_config.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        # Each sequence's new token sits at its current length - 1.
        pos_offsets = [len(seq.token_ids) - 1 for seq in sequences]
        apply_rope(q, self.head_dim, start_pos=pos_offsets)
        apply_rope(k, self.head_dim, start_pos=pos_offsets)

        # Append the new K/V slice to each sequence's KV-cache blocks.
        # Use idx:idx+1 to preserve the batch dimension: [1, num_kv_heads, 1, head_dim].
        for idx, seq in enumerate(sequences):
            seq.append_kv_cache(
                self.layer_id,
                k[idx:idx + 1, :, :, :],
                v[idx:idx + 1, :, :, :],
                block_manager=self.block_manager,
            )

        # Collect the full KV history per sequence and pad to the longest one.
        kv_seq_lens = [
            sum(blk.k_cache[self.layer_id].shape[2] for blk in seq.kv_cache_blocks)
            for seq in sequences
        ]
        max_kv_len = max(kv_seq_lens)

        k_full_list = []
        v_full_list = []
        for seq in sequences:
            # Concatenate all blocks for this sequence: [1, num_kv_heads, total_len, head_dim]
            k_seq = torch.cat([blk.k_cache[self.layer_id] for blk in seq.kv_cache_blocks], dim=2)
            v_seq = torch.cat([blk.v_cache[self.layer_id] for blk in seq.kv_cache_blocks], dim=2)

            pad_len = max_kv_len - k_seq.shape[2]
            if pad_len > 0:
                k_seq = F.pad(k_seq, (0, 0, 0, pad_len))  # pad seq_len dim
                v_seq = F.pad(v_seq, (0, 0, 0, pad_len))

            k_full_list.append(k_seq)
            v_full_list.append(v_seq)

        # [batch_size, num_kv_heads, max_kv_len, head_dim]
        k_full = torch.cat(k_full_list, dim=0)
        v_full = torch.cat(v_full_list, dim=0)

        # GQA head expansion so query heads align with KV heads.
        if self.group_size > 1:
            k_full = k_full.repeat_interleave(self.group_size, dim=1)
            v_full = v_full.repeat_interleave(self.group_size, dim=1)

        # Build padding mask: True = ignore this position.
        # Shape: [batch_size, 1, 1, max_kv_len]
        attn_mask = torch.zeros(batch_size, 1, 1, max_kv_len, dtype=torch.bool, device=k_full.device)
        for i, kv_len in enumerate(kv_seq_lens):
            attn_mask[i, 0, 0, kv_len:] = True

        o = F.scaled_dot_product_attention(
            q, k_full, v_full,
            attn_mask=~attn_mask,
        )

        # Merge heads: [batch_size, 1, d_model]
        o = o.transpose(1, 2).reshape(batch_size, 1, self.llm_config.d_model)
        o = self.o_proj(o)
        o = self.o_dropout(o)

        output = o + residual  # [batch_size, 1, d_model]

        for idx, seq in enumerate(sequences):
            # Preserve the [1, 1, d_model] shape expected by subsequent layers.
            seq.decode_activations = output[idx:idx + 1, :, :]