import torch


def apply_rope(
        x: torch.Tensor,
        rotary_dim: int,
        base: int = 10000,
        start_pos: int = 0,
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
    start_pos : int, optional
        Absolute token offset for the first position in *x*.  Set to the number of
        tokens already in the KV cache so that newly generated tokens receive the
        correct absolute positions (important for decode steps with a KV cache).
        Default: 0 (prefill from the beginning of the sequence).

    Notes
    -----
    The rotation is applied to the ``(x_even, x_odd)`` pairs in the last dimension::

        [ cos  -sin ] [ x_even ]   gives the rotated even outputs
        [ sin   cos ] [ x_odd  ]   gives the rotated odd  outputs

    where the angle for head-dimension pair ``k`` at position ``p`` is::

        θ_{p,k} = p / base^(2k / rotary_dim)
    """

    B, H, T, D = x.shape
    device, dtype = x.device, x.dtype

    assert rotary_dim % 2 == 0, "rotary_dim must be even"
    assert rotary_dim <= D, f"rotary_dim ({rotary_dim}) must be <= head_dim ({D})"

    # ------------------------------------------------------------------
    # 1. Build per-dimension inverse frequencies.
    #    Shape: (rotary_dim / 2,)
    # ------------------------------------------------------------------
    dim_indices = torch.arange(0, rotary_dim, 2, device=device)
    inv_freq = (1.0 / (base ** (dim_indices.float() / rotary_dim))).to(dtype)

    # ------------------------------------------------------------------
    # 2. Compute per-position angles, one per frequency.
    #    positions : (T,)
    #    angles    : (T, rotary_dim / 2)
    # ------------------------------------------------------------------
    positions = torch.arange(start_pos, start_pos + T, device=device, dtype=dtype)
    angles = positions[:, None] * inv_freq[None, :]     # (T, rotary_dim/2)

    # Broadcast over batch and head dimensions: (1, 1, T, rotary_dim/2)
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]

    # ------------------------------------------------------------------
    # 3. Rotate only the first ``rotary_dim`` head dims, in-place.
    #    x_rot shape: (B, H, T, rotary_dim)
    # ------------------------------------------------------------------
    x_rot = x[..., :rotary_dim]   # view into x; writes propagate back

    x_even = x_rot[..., 0::2].clone()   # (B, H, T, rotary_dim/2)
    x_odd  = x_rot[..., 1::2].clone()

    # Apply 2-D rotation matrix to each (even, odd) pair:
    #   new_even =  x_even * cos - x_odd * sin
    #   new_odd  =  x_even * sin + x_odd * cos
    x_rot[..., 0::2] = x_even * cos - x_odd * sin
    x_rot[..., 1::2] = x_even * sin + x_odd * cos
    # x[..., rotary_dim:] is untouched — no copy needed.