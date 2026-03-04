from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ToyLLMConfig
from engine.block_manager import BlockManager
from engine.sequence import Sequence, SequenceStatus
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
    underlying parameters, halving the parameter count at the vocabulary boundary
    and stabilising training (Press & Wolf, 2017).

    Parameters
    ----------
    llm_config : ToyLLMConfig
        Unified model/inference configuration.  All architecture hyper-parameters
        are read from this object.
    block_manager : BlockManager
        Manages paged physical KV-cache blocks.  Passed to every transformer layer
        so they can read/write the cache without going through the model.

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

    @torch.inference_mode()
    def forward(
            self,
            sequences: list[Sequence],
            query_chunk_idxes: list[int] | None = None,
            is_prefill: bool = True,
    ) -> None:
        """
        Run a single forward pass — one query chunk (prefill) or one decode step.

        During **prefill** this is called once per query chunk.  During **decode**
        it is called once per generated step for the whole batch.

        Parameters
        ----------
        sequences : list[Sequence]
            For prefill: a single-element list.  For decode: one entry per sequence
            in the batch.
        query_chunk_idxes : list[int] | None
            Required for prefill (zero-based chunk indices); ignored for decode.
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
        Embed and process one query chunk, then greedily sample the next token.

        The sampled token is appended to ``seq.token_ids`` so that subsequent
        chunks and the eventual decode phase see the full history.

        Parameters
        ----------
        seq : Sequence
            The sequence being prefilled.
        query_chunk_idx : int
            Zero-based index of the query chunk to process.
        """
        token_ids_in_chunks = seq.token_ids_in_chunks()
        assert 0 <= query_chunk_idx < len(token_ids_in_chunks)

        # Embed the token IDs for this chunk.
        # Shape: [1, chunk_seq_len]
        token_ids_chunk = torch.tensor(
            token_ids_in_chunks[query_chunk_idx]
        ).unsqueeze(0)

        # Initialise chunk activations from the embedding table.
        # Shape: [1, chunk_seq_len, d_model]
        seq.prefill_chunked_activations[query_chunk_idx] = self.embedding(token_ids_chunk)

        # Run each transformer block on this chunk.
        for transformer_layer in self.transformer_layers:
            transformer_layer([seq], [query_chunk_idx], is_prefill=True)

        # Apply final norm and project to vocabulary logits.
        x = seq.prefill_chunked_activations[query_chunk_idx]  # [1, chunk_seq_len, d_model]
        x = self.rms_norm(x)

        logits = self.lm_head(x)               # [1, chunk_seq_len, vocab_size]
        probs = F.softmax(logits, dim=-1)

        # Greedy decode: argmax at the last token position.
        next_token_id = torch.argmax(probs[:, -1, :], dim=-1)  # [1]
        seq.append_token(next_token_id.item())

    @torch.inference_mode()
    def decode_a_batch(self, sequences: list[Sequence]) -> None:
        """
        Embed the last token for each sequence and run one decode step.

        The next-token prediction is sampled greedily and appended to each
        sequence's ``token_ids``.

        Parameters
        ----------
        sequences : list[Sequence]
            Batch of sequences in ``DECODE_PENDING`` status.
        """
        # Embed the most recently generated token for each sequence.
        # [batch_size, 1]
        token_ids = torch.tensor(
            [seq.get_last_token_id() for seq in sequences]
        ).unsqueeze(1)

        # [batch_size, 1, d_model]
        x = self.embedding(token_ids)

        # Distribute embeddings as the initial decode activation for each sequence.
        # Shape per sequence: [1, 1, d_model].
        for idx, seq in enumerate(sequences):
            seq.decode_activations = x[idx:idx + 1, :, :]

        # Run all transformer blocks in decode mode.
        for transformer_layer in self.transformer_layers:
            transformer_layer(sequences, is_prefill=False)

        # Gather updated activations and apply final norm.
        # [batch_size, 1, d_model]
        x = torch.cat([seq.decode_activations for seq in sequences], dim=0)
        x = self.rms_norm(x)

        logits = self.lm_head(x)               # [batch_size, 1, vocab_size]
        probs = F.softmax(logits, dim=-1)

        # Greedy decode: argmax at the single output position.
        next_token_ids = torch.argmax(probs[:, -1, :], dim=-1)  # [batch_size]

        for idx, seq in enumerate(sequences):
            seq.append_token(next_token_ids[idx].item())