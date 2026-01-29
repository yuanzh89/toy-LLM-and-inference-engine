from transformer import Transformer

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyLLMModel(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int, num_heads: int, d_ff: int, num_layers: int,
                 dropout: float = 0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight Typing (weight sharing) between embedding layer and LM head layer
        # By sharing their parameters/weights in memory
        self.lm_head.weight = self.embedding.weight

        self.transformer_layers = nn.ModuleList([
            Transformer(d_model, num_heads, d_ff, dropout=dropout) for _ in range(num_layers)
        ])

        self.rms_norm = nn.RMSNorm(d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(self.device)

        batch_size, seq_len = token_ids.size()

        # (batch_size, seq_len, token_ids) -> (batch_size, seq_len, d_model)
        x = self.embedding(token_ids)

        # output -> (batch_size, seq_len, d_model)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, apply_casual_mask=True)

        x = self.rms_norm(x)

        # (batch_size, seq_len, vocab_size)
        x = self.lm_head(x)

        # (batch_size, seq_len, vocab_size)
        x = F.softmax(x, dim=-1)

        # TODO: Implement sampling process

        return x
