from transformer import Transformer

import torch
import torch.nn as nn


class NanoLLM(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int, num_heads: int, d_ff: int, num_layers: int,
                 dropout: float = 0.1, device: torch.device = torch.device('cpu')):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, device=device)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, device=device)
        # Weight Typing (weight sharing) between embedding layer and LM head layer
        # By sharing their parameters/weights in memory
        self.lm_head.weight = self.embedding.weight

        # TODO: How to generate and apply RoPE matrix?
        self.pos_embedding = nn.Embedding(max_seq_len, d_model, device=device)

        self.transformer_layers = nn.ModuleList([
            Transformer(d_model, num_heads, d_ff, dropout=dropout, device=device) for _ in range(num_layers)
        ])

        self.rms_norm = nn.RMSNorm(d_model, device=device)

        self.device = device

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(self.device)

        batch_size, seq_len = token_ids.size()

        # input -> (batch_size, seq_len, token_id)
        token_embeddings = self.embedding(token_ids)

        # (1, seq_len)
        position_ids = torch.arange(0, seq_len, device=self.device).unsqueeze(0)
        # (1, seq_len, d_model)
        pos_embeddings = self.pos_embedding(position_ids)

        # (batch_size, seq_len, d_model)
        output = token_embeddings + pos_embeddings

        mask = torch.tril(torch.ones((seq_len, seq_len), device=self.device))
        # self.register_buffer('mask', torch.tril(torch.ones((seq_len, seq_len))))

        # output -> (batch_size, seq_len, d_model)
        for transformer_layer in self.transformer_layers:
            output = transformer_layer(output, mask=mask)

        output = self.rms_norm(output)

        # (batch_size, seq_len, vocab_size)
        output = self.lm_head(output)

        return output
