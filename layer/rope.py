import torch


def apply_rope(q: torch.Tensor, k: torch.Tensor, base: int = 10000):
    """
    q: (B, T, num_query_heads, head_dim)
    k: (B, T, num_kv_heads, head_dim)
    """
    B, T, Hq, D = q.shape
    _, _, Hk, _ = k.shape
    device = q.device

    assert D % 2 == 0

    dim = torch.arange(0, D, 2, device=device)
    inv_freq = 1.0 / (base ** (dim.float() / D))

    pos = torch.arange(T, device=device)
    angles = pos[:, None] * inv_freq[None, :]  # (T, D/2)

    sin = angles.sin()[None, :, None, :]  # (1, T, 1, D/2)
    cos = angles.cos()[None, :, None, :]

    # Q
    q_even = q[..., 0::2]
    q_odd = q[..., 1::2]
    q_rot = torch.stack(
        (q_even * cos - q_odd * sin,
         q_even * sin + q_odd * cos),
        dim=-1
    ).flatten(-2)

    # K
    k_even = k[..., 0::2]
    k_odd = k[..., 1::2]
    k_rot = torch.stack(
        (k_even * cos - k_odd * sin,
         k_even * sin + k_odd * cos),
        dim=-1
    ).flatten(-2)

    return q_rot, k_rot
