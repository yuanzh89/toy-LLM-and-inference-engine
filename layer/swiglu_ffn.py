import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.sequence import Sequence, SequenceStatus


class SwiGLUFFNLayer(nn.Module):
    """
    Feed-forward network with the SwiGLU activation (Shazeer, 2020).

    Architecture::

        output = w3( silu(w1(x)) * w2(x) ) + residual

    where ``w1`` is the gate branch, ``w2`` is the value branch, and ``w3``
    projects back to ``d_model``.  A pre-norm (RMSNorm) is applied before the
    projections and a residual connection is added around the entire block.

    Parameters
    ----------
    d_model : int
        Input and output feature dimension.
    d_ff : int
        Hidden dimension of the gate and value projections.
    dropout : float
        Dropout probability applied to the output before the residual addition.
        Defaults to ``0.1``; set to ``0.0`` to disable.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.rms_norm = nn.RMSNorm(d_model, eps=1e-6)

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate
        self.w2 = nn.Linear(d_model, d_ff, bias=False)  # value
        self.w3 = nn.Linear(d_ff, d_model, bias=False)  # down-project

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

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
        Run the FFN for one query chunk during prefill.

        Reads from ``seq.prefill_chunked_activations[query_chunk_idx]`` and
        writes the result back to the same slot with the residual applied.

        Parameters
        ----------
        seq : Sequence
            The sequence being prefilled.
        query_chunk_idx : int
            Zero-based index of the query chunk to process.
        """
        x = seq.get_query_chunk_activations(query_chunk_idx)  # [1, chunk_len, d_model]
        residual = x

        x = self.rms_norm(x)

        gate = self.w1(x)   # [1, chunk_len, d_ff]
        value = self.w2(x)  # [1, chunk_len, d_ff]
        output = self.w3(F.silu(gate) * value)  # [1, chunk_len, d_model]

        output = self.dropout(output)

        seq.prefill_chunked_activations[query_chunk_idx] = output + residual

    @torch.inference_mode()
    def decode_a_batch(self, sequences: list[Sequence]) -> None:
        """
        Run the FFN for one decode step across a batch of sequences.

        Reads ``decode_activations`` from each sequence, applies the FFN, and
        writes the result (with the residual) back per-sequence.

        Parameters
        ----------
        sequences : list[Sequence]
            Batch of sequences in ``DECODE_PENDING`` status.
        """
        activations = [seq.decode_activations for seq in sequences]
        # [batch_size, 1, d_model]
        x = torch.cat(activations, dim=0)
        residual = x
        x = self.rms_norm(x)

        gate = self.w1(x)   # [batch_size, 1, d_ff]
        value = self.w2(x)  # [batch_size, 1, d_ff]
        # [batch_size, 1, d_model]
        output = self.w3(F.silu(gate) * value)

        output = self.dropout(output)
        output = output + residual  # [batch_size, 1, d_model]

        for idx, seq in enumerate(sequences):
            # Preserve the [1, 1, d_model] shape expected by subsequent layers.
            seq.decode_activations = output[idx:idx + 1, :, :]


class FusedSwiGLUFFNLayer(nn.Module):
    """
    Feed-forward network with SwiGLU activation using a fused gate+value projection.

    Identical in behaviour to :class:`SwiGLUFFNLayer` but uses a single fused
    matrix multiply for the gate and value branches, splitting the output in two.
    This can improve throughput by reducing the number of GEMM kernel launches.

    Architecture::

        x12 = w12(x)                        # [B, T, 2 * d_ff]
        gate, value = split(x12, d_ff)
        output = w3( silu(gate) * value ) + residual

    Parameters
    ----------
    d_model : int
        Input and output feature dimension.
    d_ff : int
        Hidden dimension of each of the two halves of the fused projection.
    dropout : float
        Dropout probability applied to the output before the residual addition.
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
            Input tensor of shape ``[B, T, d_model]``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``[B, T, d_model]``.
        """
        residual = x

        x = self.rms_norm(x)

        x12 = self.w12(x)               # [B, T, 2 * d_ff]
        gate, value = x12.split(self.d_ff, dim=-1)  # each [B, T, d_ff]

        out = F.silu(gate) * value      # SwiGLU activation
        out = self.w3(out)              # [B, T, d_model]
        out = self.dropout(out)

        return out + residual