import torch


def apply_rope(
        q: torch.Tensor,
        k: torch.Tensor,
        rotary_dim: int,
        base: int = 10000,
        start_pos: int = 0,
):
    """
    q: (B, T, num_query_heads, head_dim)
    k: (B, T, num_kv_heads, head_dim)
    rotary_dim: how many dims to apply RoPE to (must be even)
    start_pos: offset for KV cache / decoding
    """

    B, T, Hq, D = q.shape
    _, _, Hk, _ = k.shape
    device = q.device

    assert rotary_dim % 2 == 0
    assert rotary_dim <= D

    # Split rotary / non-rotary parts
    q_rot_part = q[..., :rotary_dim]  # (B, T, Hq, rotary_dim)
    q_pass = q[..., rotary_dim:]  # (B, T, Hq, D - rotary_dim)

    k_rot_part = k[..., :rotary_dim]  # (B, T, Hk, rotary_dim)
    k_pass = k[..., rotary_dim:]  # (B, T, Hk, D - rotary_dim)

    # Build frequencies
    dim = torch.arange(0, rotary_dim, 2, device=device)  # (rotary_dim/2,)
    inv_freq = 1.0 / (base ** (dim.float() / rotary_dim))

    # Positions (with offset for decoding)
    pos = torch.arange(start_pos, start_pos + T, device=device)  # (T,)
    angles = pos[:, None] * inv_freq[None, :]  # (T, rotary_dim/2)

    sin = angles.sin()[None, :, None, :]  # (1, T, 1, rotary_dim/2)
    cos = angles.cos()[None, :, None, :]

    # Rotate Q
    q_even = q_rot_part[..., 0::2]  # (B, T, Hq, rotary_dim/2)
    q_odd = q_rot_part[..., 1::2]

    q_rotated = torch.stack(
        (q_even * cos - q_odd * sin,
         q_even * sin + q_odd * cos),
        dim=-1
    ).flatten(-2)  # (B, T, Hq, rotary_dim)

    # Rotate K
    k_even = k_rot_part[..., 0::2]  # (B, T, Hk, rotary_dim/2)
    k_odd = k_rot_part[..., 1::2]

    k_rotated = torch.stack(
        (k_even * cos - k_odd * sin,
         k_even * sin + k_odd * cos),
        dim=-1
    ).flatten(-2)  # (B, T, Hk, rotary_dim)

    # Concatenate back non-rotary dims
    q_out = torch.cat((q_rotated, q_pass), dim=-1)  # (B, T, Hq, D)
    k_out = torch.cat((k_rotated, k_pass), dim=-1)  # (B, T, Hk, D)

    return q_out, k_out
