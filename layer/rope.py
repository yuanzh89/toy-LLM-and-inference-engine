from typing import Union

import torch


def apply_rope(
        x: torch.Tensor,
        rotary_dim: int,
        base: int = 10000,
        start_pos: Union[int, list[int]] = 0,
) -> None:
    """
    Apply Rotary Position Embedding (RoPE) in-place to the first ``rotary_dim``
    dimensions of the head dimension of *x*.

    RoPE encodes position information by rotating pairs of adjacent head dimensions
    by position-dependent angles.  Dimensions beyond ``rotary_dim`` are left
    unchanged.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape ``(B, num_heads, T, head_dim)``, where:

        * ``B``         – batch size
        * ``num_heads`` – number of attention heads
        * ``T``         – sequence length (the tokens being processed)
        * ``head_dim``  – per-head feature dimension

        Modified **in-place**; no tensor is returned.
    rotary_dim : int
        Number of head dimensions to rotate.  Must be even and ≤ ``head_dim``.
        A common setting is ``head_dim`` (full rotation) or ``head_dim // 2``.
    base : int, optional
        Base for the geometric frequency schedule (default: 10 000, matching the
        original RoPE paper).
    start_pos : int or list[int], optional
        Absolute token offset(s) for the first position in *x*.

        * **int** – the same offset is applied to every sequence in the batch
          (e.g. prefill from the start with ``0``).
        * **list[int]** – one offset per sequence in the batch (length must equal
          ``B``).  Use this when different sequences in the batch have different
          numbers of tokens already cached, so each sequence receives the correct
          absolute positions during a decode step.

        Default: 0.

    Notes
    -----
    The rotation is applied to the ``(x_even, x_odd)`` pairs in the last dimension::

        [ cos  -sin ] [ x_even ]   gives the rotated even outputs
        [ sin   cos ] [ x_odd  ]   gives the rotated odd  outputs

    where the angle for head-dimension pair ``k`` at position ``p`` is::

        θ_{p,k} = p / base^(2k / rotary_dim)

    When ``start_pos`` is a list, each batch entry ``b`` uses positions::

        p = start_pos[b], start_pos[b] + 1, ..., start_pos[b] + T - 1

    so that cos/sin tensors are shaped ``(B, 1, T, rotary_dim/2)`` rather than
    ``(1, 1, T, rotary_dim/2)``.
    """

    B, H, T, D = x.shape
    device, dtype = x.device, x.dtype

    assert rotary_dim % 2 == 0, "rotary_dim must be even"
    assert rotary_dim <= D, f"rotary_dim ({rotary_dim}) must be <= head_dim ({D})"

    # Normalise start_pos to a 1-D tensor of shape (B,).
    if isinstance(start_pos, int):
        start_pos_t = torch.full((B,), start_pos, device=device, dtype=dtype)
    else:
        assert len(start_pos) == B, (
            f"len(start_pos) ({len(start_pos)}) must equal batch size B ({B})"
        )
        start_pos_t = torch.tensor(start_pos, device=device, dtype=dtype)  # (B,)

    # ------------------------------------------------------------------
    # 1. Build per-dimension inverse frequencies.
    #    Shape: (rotary_dim / 2,)
    # ------------------------------------------------------------------
    dim_indices = torch.arange(0, rotary_dim, 2, device=device)
    inv_freq = (1.0 / (base ** (dim_indices.float() / rotary_dim))).to(dtype)

    # ------------------------------------------------------------------
    # 2. Compute per-position angles, one per (batch, position, frequency).
    #
    #    offsets   : (B, 1)         – per-sequence starting offset
    #    steps     : (1, T)         – 0 … T-1 relative step within the window
    #    positions : (B, T)         – absolute position of every token per batch
    #    angles    : (B, T, rotary_dim / 2)
    # ------------------------------------------------------------------
    offsets = start_pos_t[:, None]  # (B, 1)
    steps = torch.arange(T, device=device, dtype=dtype)[None, :]  # (1, T)
    positions = offsets + steps  # (B, T)

    angles = positions[:, :, None] * inv_freq[None, None, :]  # (B, T, rotary_dim/2)

    # Broadcast over the head dimension: (B, 1, T, rotary_dim/2)
    cos = angles.cos()[:, None, :, :]
    sin = angles.sin()[:, None, :, :]

    # ------------------------------------------------------------------
    # 3. Rotate only the first ``rotary_dim`` head dims, in-place.
    #    x_rot shape: (B, H, T, rotary_dim)
    # ------------------------------------------------------------------
    x_rot = x[..., :rotary_dim]  # view into x; writes propagate back

    x_even = x_rot[..., 0::2].clone()  # (B, H, T, rotary_dim/2)
    x_odd = x_rot[..., 1::2].clone()

    # Apply 2-D rotation matrix to each (even, odd) pair:
    #   new_even =  x_even * cos - x_odd * sin
    #   new_odd  =  x_even * sin + x_odd * cos
    x_rot[..., 0::2] = x_even * cos - x_odd * sin
    x_rot[..., 1::2] = x_even * sin + x_odd * cos
    # x[..., rotary_dim:] is untouched — no copy needed.
