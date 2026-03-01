import torch


def apply_rope(
        x: torch.Tensor,
        rotary_dim: int,
        base: int = 10000,
        start_pos: int = 0,
) -> None:
    """
    Applies Rotary Position Embedding (RoPE) in-place to the first `rotary_dim`
    dimensions of the head dimension of `x`.

    RoPE encodes position information by rotating pairs of adjacent head
    dimensions by position-dependent angles. Dimensions beyond `rotary_dim`
    are left unchanged.

    Args:
        x:           Tensor of shape (B, T, num_heads, head_dim).
        rotary_dim:  Number of head dimensions to rotate. Must be even and
                     <= head_dim.
        base:        Base frequency for the sinusoidal schedule (default 10000).
        start_pos:   Token offset used during decoding with a KV cache. Set to
                     the number of tokens already in the cache so that newly
                     generated tokens receive the correct absolute positions.
    """
    B, T, H, D = x.shape
    device, dtype = x.device, x.dtype

    assert rotary_dim % 2 == 0, "rotary_dim must be even"
    assert rotary_dim <= D, "rotary_dim must be <= head_dim"

    # ------------------------------------------------------------------ #
    # 1. Build per-dimension inverse frequencies.                         #
    #    dim_indices shape: (rotary_dim / 2,)                             #
    # ------------------------------------------------------------------ #
    dim_indices = torch.arange(0, rotary_dim, 2, device=device)
    inv_freq = (1.0 / (base ** (dim_indices.float() / rotary_dim))).to(dtype)

    # ------------------------------------------------------------------ #
    # 2. Compute per-position angles, one per frequency.                  #
    #    positions shape : (T,)                                           #
    #    angles    shape : (T, rotary_dim / 2)                            #
    # ------------------------------------------------------------------ #
    positions = torch.arange(start_pos, start_pos + T, device=device, dtype=dtype)
    angles = positions[:, None] * inv_freq[None, :]  # (T, rotary_dim/2)

    # Broadcast over batch and head dimensions: (1, T, 1, rotary_dim/2)
    cos = angles.cos()[None, :, None, :]
    sin = angles.sin()[None, :, None, :]

    # ------------------------------------------------------------------ #
    # 3. Rotate only the first `rotary_dim` head dims, in-place.         #
    #    x_rot shape: (B, T, H, rotary_dim)                              #
    #    Even indices hold one element of each rotation pair;            #
    #    odd indices hold the other.                                      #
    # ------------------------------------------------------------------ #
    x_rot = x[..., :rotary_dim]  # view into x; writes propagate back

    x_even = x_rot[..., 0::2].clone()  # (B, T, H, rotary_dim/2)
    x_odd  = x_rot[..., 1::2].clone()

    # Apply 2-D rotation matrix to each (even, odd) pair:
    #   [ cos  -sin ] [ x_even ]
    #   [ sin   cos ] [ x_odd  ]
    x_rot[..., 0::2] = x_even * cos - x_odd * sin
    x_rot[..., 1::2] = x_even * sin + x_odd * cos
    # x[..., rotary_dim:] is untouched — no copy needed