import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Autotune configs
# ---------------------------------------------------------------------------
def _get_autotune_configs():
    """Return a list of configs for triton.autotune."""
    configs = []
    for block_m in [64, 128]:
        for block_n in [32, 64, 128]:
            for num_warps in [2, 4, 8]:
                for num_stages in [1, 2, 3, 4]:
                    # Skip nonsensical combos that waste compile time
                    if block_m * block_n > 128 * 128:
                        continue
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_N": block_n},
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    )
    return configs


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=_get_autotune_configs(),
    key=["SEQ_LEN", "BLOCK_D"],  # re-tune when these change
)
@triton.jit
def _flash_attention_fwd_kernel(
        Q, K, V, sm_scale, Out,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_on,
        H, SEQ_LEN,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
):
    NEG_INF = -1.0e9

    # ------------------------------------------------------------------
    # 1. Program identity
    # ------------------------------------------------------------------
    start_m = tl.program_id(0)  # which Q-row block
    off_hz = tl.program_id(1)  # flattened (batch, head) index
    b = off_hz // H
    h = off_hz % H

    q_offset = b * stride_qb + h * stride_qh
    k_offset = b * stride_kb + h * stride_kh
    v_offset = b * stride_vb + h * stride_vh
    o_offset = b * stride_ob + h * stride_oh

    # ------------------------------------------------------------------
    # 2. Block pointers
    # ------------------------------------------------------------------
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(SEQ_LEN, BLOCK_D),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(SEQ_LEN, BLOCK_D),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(SEQ_LEN, BLOCK_D),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(SEQ_LEN, BLOCK_D),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )

    # ------------------------------------------------------------------
    # 3. Row / col index vectors (needed for causal mask)
    # ------------------------------------------------------------------
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # ------------------------------------------------------------------
    # 4. Load Q once into registers
    # ------------------------------------------------------------------
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # ------------------------------------------------------------------
    # 5. Online-softmax accumulators
    # ------------------------------------------------------------------
    m_i = tl.full([BLOCK_M], NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # ------------------------------------------------------------------
    # 6. Main loop over K / V tiles
    # ------------------------------------------------------------------
    for start_n in range(0, SEQ_LEN, BLOCK_N):
        cols = start_n + offs_n

        # Load K [BLOCK_N, BLOCK_D], transpose → [BLOCK_D, BLOCK_N]
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        k_T = tl.trans(k)

        # Q @ K^T  →  [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k_T)

        # ---- Apply ALL masks BEFORE scaling -------------------------
        # Causal mask: query position must be >= key position
        causal_mask = offs_m[:, None] >= cols[None, :]
        qk = tl.where(causal_mask, qk, NEG_INF)

        # Padding mask: ignore columns beyond SEQ_LEN
        # (boundary_check pads K with 0, giving qk=0 there; must be -inf)
        padding_mask = cols[None, :] < SEQ_LEN
        qk = tl.where(padding_mask, qk, NEG_INF)

        # Scale after masking so NEG_INF positions stay at NEG_INF
        qk = qk * sm_scale

        # ---- Online softmax update ----------------------------------
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)

        # ---- Load V and update accumulator --------------------------
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = v.to(p.dtype)
        acc = acc * alpha[:, None] + tl.dot(p, v)

        # ---- Update statistics --------------------------------------
        m_i = m_new
        l_i = l_new

        # ---- Advance block pointers ---------------------------------
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # ------------------------------------------------------------------
    # 7. Normalise and store
    # ------------------------------------------------------------------
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Causal flash-attention.

    Args:
        q, k, v: float16 tensors of shape [B, H, SEQ_LEN, D],
                 contiguous, on CUDA.
    Returns:
        out: float16 tensor of shape [B, H, SEQ_LEN, D].
    """
    assert q.is_cuda and q.dtype == torch.float16, "q must be float16 on CUDA"
    assert q.shape == k.shape == v.shape, "q / k / v shapes must match"

    B, H, seq_len, D = q.shape
    sm_scale = D ** -0.5
    out = torch.empty_like(q)

    # BLOCK_M / BLOCK_N are tuned by autotune; we only fix BLOCK_D here.
    # Use a placeholder grid; autotune will pick the winning BLOCK_M.
    # We need a grid lambda so autotune can vary BLOCK_M.
    def grid(meta):
        return (triton.cdiv(seq_len, meta["BLOCK_M"]), B * H)

    _flash_attention_fwd_kernel[grid](
        q, k, v, sm_scale, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        H=H,
        SEQ_LEN=seq_len,
        BLOCK_D=D,
    )

    return out


# ---------------------------------------------------------------------------
# Quick smoke-test (run with:  python flash_attention.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, seq_len, D = 1, 4, 1024, 64
    device = "cuda"

    q = torch.randn((B, H, seq_len, D), dtype=torch.float16, device=device)
    k = torch.randn((B, H, seq_len, D), dtype=torch.float16, device=device)
    v = torch.randn((B, H, seq_len, D), dtype=torch.float16, device=device)

    triton_out = flash_attention(q, k, v)

    # PyTorch reference (causal)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (D ** -0.5)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    ref = torch.matmul(probs, v.float()).half()

    max_diff = (triton_out - ref).abs().max().item()
    print(f"Max diff: {max_diff:.6f}")
    assert max_diff < 1e-2, f"Too large: {max_diff}"
    print("OK — Triton matches PyTorch causal attention.")
