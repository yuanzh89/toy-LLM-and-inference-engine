import torch
import torch.nn as nn
import torch.nn.functional as F

from sequence import Sequence


class SwiGLUFFNLayer(nn.Module):
    """
    Position-wise feed-forward network using the SwiGLU activation function.

    SwiGLU computes:  ``output = W3( W1(x) * silu(W2(x)) )``

    which has been shown to outperform standard ReLU and GELU FFN variants
    across a range of language model benchmarks (Noam Shazeer, 2020).

    A pre-norm (RMSNorm) is applied before the projection and a residual
    connection is added after, matching the LLaMA / Mistral architecture.

    Parameters
    ----------
    d_model : int
        Model (embedding) dimension.
    d_ff : int
        Inner feed-forward dimension (typically 4 × d_model or 8/3 × d_model).
    dropout : float, optional
        Dropout probability applied to the output projection.  Default: 0.1.

    Shape
    -----
    Input / output (via ``seq.chunked_activations``): ``[B, chunk_seq_len, d_model]``
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.rms_norm = nn.RMSNorm(d_model, eps=1e-6)

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = nn.Linear(d_model, d_ff, bias=False)   # gate
        self.w2 = nn.Linear(d_model, d_ff, bias=False)   # value
        self.w3 = nn.Linear(d_ff, d_model, bias=False)   # down-project

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, seq: Sequence, query_chunk_idx: int) -> None:
        """
        Run the SwiGLU FFN on a single query chunk and write the result back.

        Parameters
        ----------
        seq : Sequence
            The sequence whose activations will be read and updated.
        query_chunk_idx : int
            Zero-based index of the query chunk to process.  The layer reads
            ``seq.chunked_activations[query_chunk_idx]`` and writes the result
            back to the same slot.
        """

        x = seq.get_query_chunk_activations(query_chunk_idx)  # [B, chunk_len, d_model]
        residual = x

        x = self.rms_norm(x)

        x1 = self.w1(x)  # [B, chunk_len, d_ff]  — gate branch
        x2 = self.w2(x)  # [B, chunk_len, d_ff]  — value branch
        output = self.w3(x1 * F.silu(x2))  # [B, chunk_len, d_model]

        output = self.dropout(output)

        seq.chunked_activations[query_chunk_idx] = output + residual


class FusedSwiGLUFFNLayer(nn.Module):
    """
    Fused SwiGLU feed-forward layer that combines the two gate/value projections
    into a single matrix multiply for improved hardware efficiency.

    The fused weight ``w12`` has shape ``[d_model, 2 * d_ff]`` and is split in
    half along the output dimension to yield the gate (``x1``) and value (``x2``)
    streams.

    Unlike :class:`SwiGLUFFNLayer`, this variant operates on a bare
    ``torch.Tensor`` rather than a :class:`Sequence` object, making it suitable
    for use in non-chunked or standalone contexts.

    Parameters
    ----------
    d_model : int
        Model (embedding) dimension.
    d_ff : int
        Inner feed-forward dimension.
    dropout : float, optional
        Dropout probability.  Default: 0.1.

    Shape
    -----
    Input / output: ``[B, seq_len, d_model]``
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.rms_norm = nn.RMSNorm(d_model, eps=1e-6)

        # Fused gate + value projection; split into two halves after the matmul.
        self.w12 = nn.Linear(d_model, d_ff * 2, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``[B, seq_len, d_model]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, seq_len, d_model]``.
        """
        residual = x

        x = self.rms_norm(x)

        x12 = self.w12(x)                        # [B, seq_len, 2 * d_ff]
        x1, x2 = x12.split(self.d_ff, dim=-1)   # each [B, seq_len, d_ff]

        out = x1 * F.silu(x2)   # SwiGLU activation
        out = self.w3(out)       # [B, seq_len, d_model]
        out = self.dropout(out)

        return out + residual