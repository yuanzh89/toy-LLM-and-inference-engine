from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ToyLLMConfig
from engine.block_manager import BlockManager
from engine.sequence import Sequence
from transformer_block_with_kv_cache import TransformerBlock


class ToyLLMModel(nn.Module):
    """
    A minimal decoder-only language model for demonstrating paged KV-cache inference.

    Architecture
    ------------
    * Token embedding (with weight tying to the LM head).
    * ``N`` :class:`~transformer_block_with_kv_cache.TransformerBlock` layers,
      each containing GQA and a SwiGLU FFN.
    * Final RMSNorm followed by a linear LM head.

    Weight tying
    ------------
    The embedding matrix and the LM-head projection matrix share the same
    underlying parameters, halving the parameter count at the vocabulary
    boundary and stabilising training (Press & Wolf, 2017).

    Parameters
    ----------
    llm_config : ToyLLMConfig
        Unified model/inference configuration.  All architecture hyper-parameters
        are read from this object; no individual overrides are needed.
    block_manager : BlockManager
        Manages paged physical KV-cache blocks.  Passed to every transformer
        layer so they can read/write the cache without going through the model.

    Attributes
    ----------
    embedding : nn.Embedding
        Token embedding table, shape ``[vocab_size, d_model]``.
    transformer_layers : nn.ModuleList
        Sequence of :class:`~transformer_block_with_kv_cache.TransformerBlock` instances.
    rms_norm : nn.RMSNorm
        Final layer-norm applied before the LM head.
    lm_head : nn.Linear
        Projects from ``d_model`` to ``vocab_size``.  Weights are tied to
        ``embedding``.
    """

    EOS = "<EOS>"

    def __init__(self, llm_config: ToyLLMConfig, block_manager: BlockManager):
        super().__init__()

        self.llm_config = llm_config
        self.block_manager = block_manager

        self.embedding = nn.Embedding(llm_config.vocab_size, llm_config.d_model)

        self.transformer_layers = nn.ModuleList(
            TransformerBlock(
                llm_config,
                block_manager,
                layer_id=layer_id,
            )
            for layer_id in range(llm_config.num_transformer_layers)
        )

        self.rms_norm = nn.RMSNorm(llm_config.d_model, eps=1e-6)

        self.lm_head = nn.Linear(llm_config.d_model, llm_config.vocab_size, bias=False)
        # Weight tying: share parameters between the embedding table and the LM head.
        self.lm_head.weight = self.embedding.weight

    def forward(self, sequences: list[Sequence], query_chunk_idxes: list[int]) -> list[int]:
        """
        Run a single forward pass for one query chunk and return the next token ID.

        During **prefill** this is called once per query chunk (the model reads the
        prompt).  During **decode** it is called once per generated step with a
        single-token chunk.

        Parameters
        ----------
        seq : Sequence
            The sequence being processed.  Token IDs, KV-cache blocks, and chunked
            activations are all stored on the sequence object.
        query_chunk_idx : int
            Zero-based index of the query chunk to process.

        Returns
        -------
        int
            The token ID with the highest probability for the *next* position
            (i.e. greedy decoding of the last token in the chunk).
        """
        token_ids_in_chunks = seq.token_ids_in_chunks()
        assert 0 <= query_chunk_idx < len(token_ids_in_chunks)

        # Embed the token IDs for this chunk.
        # token_ids_chunk: [batch_size == 1, chunk_seq_len]
        token_ids_chunk = torch.tensor(
            token_ids_in_chunks[query_chunk_idx]
        ).unsqueeze(0)

        # Initialise chunk activations from the embedding table.
        # Shape: [B, chunk_seq_len, d_model]
        seq.prefill_chunked_activations[query_chunk_idx] = self.embedding(token_ids_chunk)

        # Run each transformer block on this chunk.
        for transformer_layer in self.transformer_layers:
            transformer_layer(seq, query_chunk_idx)

        # Read the updated activations for this chunk, apply final norm, and project.
        x = seq.prefill_chunked_activations[query_chunk_idx]  # [B, chunk_seq_len, d_model]
        x = self.rms_norm(x)

        # Project to vocabulary logits and compute probabilities.
        # [B, chunk_seq_len, vocab_size]
        logits = self.lm_head(x)
        probs = F.softmax(logits, dim=-1)

        # Greedy decode: take the argmax at the last token position only.
        # [B, vocab_size] → [B]
        next_token_id = torch.argmax(probs[:, -1, :], dim=-1)

        # Append the newly generated token so subsequent steps see it.
        seq.append_token(next_token_id.item())

        return next_token_id.item()
