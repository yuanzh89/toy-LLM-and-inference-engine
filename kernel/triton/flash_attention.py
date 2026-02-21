import torch
import triton
import triton.language as tl


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

    # ------------------------------------------------------------------ #
    # 1. Program identity                                                #
    # ------------------------------------------------------------------ #
    start_m = tl.program_id(0)  # which Q-row block
    off_hz = tl.program_id(1)  # flattened (batch, head) index
    b = off_hz // H
    h = off_hz % H

    # Base offsets into the (batch, head) slice
    q_offset = b * stride_qb + h * stride_qh
    k_offset = b * stride_kb + h * stride_kh
    v_offset = b * stride_vb + h * stride_vh
    o_offset = b * stride_ob + h * stride_oh

    # ------------------------------------------------------------------ #
    # 2. Block pointers via tl.make_block_ptr                            #
    # ------------------------------------------------------------------ #

    # Q  — load once; shape [BLOCK_M, BLOCK_D]
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(SEQ_LEN, BLOCK_D),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),  # row-major (contiguous along D)
    )

    # K  — stepped inside the loop; shape [BLOCK_N, BLOCK_D]
    #      we load it as [BLOCK_N, BLOCK_D] and then explicitly transpose
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(SEQ_LEN, BLOCK_D),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    # V  — stepped inside the loop; shape [BLOCK_N, BLOCK_D]
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(SEQ_LEN, BLOCK_D),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    # Out — written once at the end; shape [BLOCK_M, BLOCK_D]
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(SEQ_LEN, BLOCK_D),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )

    # ------------------------------------------------------------------ #
    # 3. Row / col index vectors (still needed for the causal mask)      #
    # ------------------------------------------------------------------ #
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  # Q rows
    offs_n = tl.arange(0, BLOCK_N)  # K/V rows

    # ------------------------------------------------------------------ #
    # 4. Load Q once into registers                                      #
    # ------------------------------------------------------------------ #
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # ------------------------------------------------------------------ #
    # 5. Online-softmax accumulators                                     #
    # ------------------------------------------------------------------ #
    m_i = tl.full([BLOCK_M], NEG_INF, dtype=tl.float32)  # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # running sum of exp
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)  # output accumulator

    # ------------------------------------------------------------------ #
    # 6. Main loop over K / V blocks                                       #
    # ------------------------------------------------------------------ #
    for start_n in range(0, SEQ_LEN, BLOCK_N):
        cols = start_n + offs_n

        # ---- Load K as [BLOCK_N, BLOCK_D], then EXPLICITLY transpose ----
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        k_T = tl.trans(k)  # [BLOCK_D, BLOCK_N] — explicit transpose

        # ---- Compute Q @ K^T  →  [BLOCK_M, BLOCK_N] ----------------------
        qk = tl.dot(q, k_T)

        # ---- Causal mask -------------------------------------------------
        causal_mask = offs_m[:, None] >= cols[None, :]
        qk = tl.where(causal_mask, qk, NEG_INF)

        # ---- Scale -------------------------------------------------------
        qk *= sm_scale

        # ---- Padding mask (handled by boundary_check above, but be safe) -
        qk = tl.where(cols[None, :] < SEQ_LEN, qk, NEG_INF)

        # ---- Online softmax update ---------------------------------------
        m_ij = tl.max(qk, axis=1)  # block max
        m_new = tl.maximum(m_i, m_ij)  # updated running max
        alpha = tl.exp(m_i - m_new)  # rescale old acc
        p = tl.exp(qk - m_new[:, None])  # current block probs
        l_new = l_i * alpha + tl.sum(p, axis=1)  # updated running sum

        # ---- Load V and update accumulator -------------------------------
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = v.to(p.dtype)
        acc = acc * alpha[:, None] + tl.dot(p, v)

        # ---- Update statistics -------------------------------------------
        m_i = m_new
        l_i = l_new

        # ---- Advance block pointers to the next K/V tile -----------------
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # ------------------------------------------------------------------ #
    # 7. Normalise and store                                             #
    # ------------------------------------------------------------------ #
    acc = acc / l_i[:, None]

    tl.store(O_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # Input shape of q, k, v: [batch_size, num_heads, seq_len, head_dim]
    B, H, seq_len, D = q.shape
    sm_scale = D ** -0.5
    o = torch.empty_like(q)

    BLOCK_M = 128
    BLOCK_N = 64
    num_stages = 4 if torch.cuda.get_device_capability()[0] >= 9 else 2
    num_warps = 4
    grid = (triton.cdiv(seq_len, BLOCK_M), B * H)

    _flash_attention_fwd_kernel[grid](
        q, k, v, sm_scale, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H=H, SEQ_LEN=seq_len,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=D,
        num_warps=num_warps, num_stages=num_stages,
    )

    return o


# --- Verification ---
if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, seq_len, D = 1, 4, 1024, 64
    dtype = torch.float16
    device = "cuda"

    q = torch.randn((B, H, seq_len, D), dtype=dtype, device=device)
    k = torch.randn((B, H, seq_len, D), dtype=dtype, device=device)
    v = torch.randn((B, H, seq_len, D), dtype=dtype, device=device)

    triton_output = flash_attention(q, k, v)

    # PyTorch reference (causal)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (D ** -0.5)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    torch_output = torch.matmul(probs, v)

    print(f"Max Diff: {(triton_output - torch_output).abs().max():.6f}")
    assert torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0)
    print("Triton implementation matches PyTorch causal attention!")
