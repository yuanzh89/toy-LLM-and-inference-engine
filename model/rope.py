import torch


def apply_rope(q: torch.Tensor, k: torch.Tensor, base: int = 10000):
    """
        q: (B, T, num_query_heads, head_dim)
        k: (B, T, num_kv_heads, head_dim)
        Returns: q_rot, k_rot with same shape
    """
    batch_size, seq_len, num_query_heads, head_dim = q.shape
    d_model = num_query_heads * head_dim
    device = q.device
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # Build RoPE angles
    pos = torch.arange(seq_len, device=device)
    dim = torch.arange(0, d_model, 2, device=device)
    inv_freq = 1.0 / (base ** (dim.float() / head_dim))
    # [seq_len, head_dim / 2]
    angles = pos[:, None] * inv_freq[None, :]
    # (1, seq_len, 1, head_dim / 2]
    sin = angles.sin()[None, :, None, :]
    cos = angles.cos()[None, :, None, :]

    # Rotate Q
    q_even = q[..., 0::2]
    q_odd = q[..., 1::2]
    q_rot = torch.stack((q_even * cos - q_odd * sin, q_even * cos + q_odd * sin), dim=-1).flatten(-2)

    # Rotate K
    k_even = k[..., 0::2]
    k_odd = k[..., 1::2]
    k_rot = torch.stack((k_even * cos - k_odd * sin, k_even * sin + k_odd * cos), dim=-1).flatten(-2)

    return q_rot, k_rot
