import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ToyLLMConfig
from engine.block_manager import BlockManager
from engine.sequence import Sequence
from transformer_block_with_kv_cache import TransformerBlock


class ToyLLMModel(nn.Module):
    eos = '<EOS>'

    def __init__(
            self,
            llm_config: ToyLLMConfig,
            block_manager: BlockManager,
            vocab_size: int,
            d_model: int,
            num_query_heads: int,
            num_kv_heads: int,
            d_ff: int,
            num_transformer_layers: int,
            dropout: float = 0.1
    ):
        super().__init__()

        self.llm_config = llm_config
        self.block_manager = block_manager
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.d_ff = d_ff
        self.num_transformer_layers = num_transformer_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight Typing (weight sharing) between embedding layer and LM head layer
        # By sharing their parameters/weights in memory
        self.lm_head.weight = self.embedding.weight

        self.transformer_layers = nn.ModuleList()
        for layer_id in range(self.num_transformer_layers):
            transformer_layer = TransformerBlock(
                self.llm_config,
                self.block_manager,
                layer_id,
                self.d_model,
                self.d_ff,
                self.num_query_heads,
                self.num_kv_heads,
                self.dropout,
            )
            self.transformer_layers.append(transformer_layer)

        self.rms_norm = nn.RMSNorm(d_model, eps=1e-6)

    def forward(self, seq: Sequence, query_chunk_idx: int) -> None:
        token_ids_in_chunks = seq.token_ids_in_chunks()
        assert 0 <= query_chunk_idx < len(token_ids_in_chunks)

        # [batch_size == 1, chunk_seq_len]
        token_ids_chunk = torch.tensor(token_ids_in_chunks[query_chunk_idx]).unsqueeze(0)
        # Initialize sequence activations
        # [batch_size == 1, chunk_seq_len, d_model]
        seq.chunked_activations[query_chunk_idx] = self.embedding(token_ids_chunk)

        # output -> (batch_size, seq_len, d_model)
        for transformer_layer in self.transformer_layers:
            transformer_layer(seq)

        x = seq.chunked_activations[query_chunk_idx]
        x = self.rms_norm(x)

        # (batch_size, seq_len, vocab_size)
        x = self.lm_head(x)

        # (batch_size, seq_len, vocab_size)
        x = F.softmax(x, dim=-1)

        # Sampling
        next_token_id = torch.argmax(x, dim=-1)

        # TODO: Append the next_token_id back into Sequence
