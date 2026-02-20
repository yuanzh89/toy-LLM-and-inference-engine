import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attention_fwd_kernel(
        Q, K, V, sm_scale, Out,
        stride_qb, stride_qh, stride_qm, stride_qk,  # Strides for Q
        stride_kb, stride_kh, stride_kn, stride_kk,  # Strides for K
        stride_vb, stride_vh, stride_vn, stride_vk,  # Strides for V
        stride_ob, stride_oh, stride_om, stride_on,  # Strides for Out
        H, SEQ_LEN,  # Constants
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    NEG_INF = -1.0e9

    # 1. Identify the location in the grid
    # Each program instance handles one block of Query (one block of rows in the attention matrix)
    # inside a specific batch and head.
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # Batch and Head indices
    # off_hz is a flattened index of (Batch * Head). We reconstruct them here.
    # Note: In this kernel setup, we treat (Batch * Head) as the Z-dimension.
    b = off_hz // H
    h = off_hz % H

    # Initialize offsets for pointers to Q, K, V
    # We need to jump to the correct Batch and Head for this specific program.
    q_offset = b * stride_qb + h * stride_qh
    k_offset = b * stride_kb + h * stride_kh
    v_offset = b * stride_vb + h * stride_vh
    o_offset = b * stride_ob + h * stride_oh

    # 2. Define block pointers
    # range of offsets for the Query block (rows)
    # We are processing rows [start_m * BLOCK_M : (start_m + 1) * BLOCK_M]
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # range of offsets for the dimension D (cols of Q/K/V)
    offs_d = tl.arange(0, BLOCK_D)

    # range of offsets for the Key/Value block (cols of attention matrix)
    offs_n = tl.arange(0, BLOCK_N)

    # 3. Create pointers
    # Pointers to Q: Base + (Batch/Head offset) + (Row offset * stride) + (Col offset * stride)
    Q_ptr = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk

    # Pointers to K: Base + (Batch/Head offset) + (Col offset * stride) + (Dim offset * stride)
    # Note: We transpose K implicitly by how we load logic later, but here we set up for (Col, Dim)
    K_ptr = K + k_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk

    # Pointers to V: Base + (Batch/Head offset) + (Col offset * stride) + (Dim offset * stride)
    V_ptr = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    # 4. Load Q
    # We load the Q block once and keep it in SRAM (registers)
    q = tl.load(Q_ptr, mask=offs_m[:, None] < SEQ_LEN, other=0.0)

    # 5. Initialize Accumulators for Online Softmax
    # m_i: running maximum (initialized to NEG_INF)
    # l_i: running sum of exponentials (initialized to 1.0)
    # acc: accumulator for the output (initialized to 0.0)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - NEG_INF
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # TODO:
    # 1. what is the shape of each parameters
    # 2. How k transpose is handled in this code
    # 3. how to apply casual mask to this code

    # 6. Inner Loop: Iterate over blocks of K and V
    # We step through the sequence length (SEQ_LEN) in chunks of BLOCK_N
    for start_n in range(0, SEQ_LEN, BLOCK_N):
        # Update current loop offsets
        cols = start_n + offs_n

        # Load K block (transposed for dot product)
        # Note: We load K such that we can do dot(Q, K^T)
        # K shape in memory: [BLOCK_N, BLOCK_D]. We load it to match that.
        # But for dot product q [M, D] @ k.T [D, N], Triton handles the matmul.
        k = tl.load(K_ptr, mask=cols[None, :] < SEQ_LEN, other=0.0)

        # Compute Q * K^T
        qk = tl.dot(q, k)

        # Apply scaling (1 / sqrt(d))
        qk *= sm_scale

        # --- Online Softmax Logic ---

        # Mask out padding if we are near the end of the sequence
        # If cols > SEQ_LEN, set attention score to NEG_INF
        qk = tl.where(cols[None, :] < SEQ_LEN, qk, NEG_INF)

        # 1. Get the current block's max
        m_ij = tl.max(qk, 1)  # Max along the columns (N dimension)

        # 2. Update global running max
        m_new = tl.maximum(m_i, m_ij)

        # 3. Compute scaling factors for the recurrence
        # alpha = exp(old_max - new_max) -> scales the previous accumulator
        alpha = tl.exp(m_i - m_new)
        # beta = exp(current_block_max - new_max) -> scales the current P matrix
        beta = tl.exp(m_ij - m_new)

        # 4. Compute P (probability) for current block
        p = tl.exp(qk - m_new[:, None])

        # 5. Update running sum of exps
        # l_new = alpha * l_old + sum(P_block)
        # Note: In strict math, beta is incorporated into p computation.
        # But practically: l_new = l_old * alpha + rowsum(exp(qk - m_new))
        p_sum = tl.sum(p, axis=1)
        l_new = l_i * alpha + p_sum

        # 6. Load V block
        v = tl.load(V_ptr, mask=cols[:, None] < SEQ_LEN, other=0.0).to(p.dtype)

        # 7. Update Accumulator (Attention Output)
        # acc = acc * alpha + P @ V
        acc = acc * alpha[:, None]
        acc += tl.dot(p, v)

        # 8. Update statistics for next iteration
        m_i = m_new
        l_i = l_new

        # Advance pointers to the next block of K and V
        K_ptr += BLOCK_N * stride_kn
        V_ptr += BLOCK_N * stride_vn

    # 7. Final Normalization
    # Divide accumulated output by the running sum `l_i`
    acc = acc / l_i[:, None]

    # 8. Store Output
    O_ptr = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(O_ptr, acc, mask=(offs_m[:, None] < SEQ_LEN) & (offs_d[None, :] < BLOCK_D))


def flash_attention(q, k, v):
    # Dimensions: B (Batch), H (Heads), seq_len (Sequence Length), D (Head Dim)
    B, H, seq_len, D = q.shape

    # Scaling factor 1/sqrt(D)
    sm_scale = 1.0 / (D ** 0.5)

    # Output buffer
    o = torch.empty_like(q)

    # Block sizes (tuning these is critical for performance)
    # BLOCK_M = 128 (Query chunk size)
    # BLOCK_N = 64 (Key/Value chunk size)
    BLOCK_M = 128
    BLOCK_N = 64

    # Num stages and warps for optimization
    num_stages = 4 if torch.cuda.get_device_capability()[0] >= 9 else 2
    num_warps = 4

    # Grid: (Number of Q blocks, Batch * Heads)
    grid = (triton.cdiv(seq_len, BLOCK_M), B * H)

    _flash_attention_fwd_kernel[grid](
        q, k, v, sm_scale, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H=H, SEQ_LEN=seq_len,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=D,
        num_warps=num_warps,
        num_stages=num_stages,
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

    # Run Triton implementation
    triton_output = flash_attention(q, k, v)

    # Run PyTorch reference
    # (B, H, seq, D) -> (B, H, seq, seq)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (D ** -0.5)
    probs = torch.softmax(scores, dim=-1)
    torch_output = torch.matmul(probs, v)

    print(f"Max Diff: {(triton_output - torch_output).abs().max()}")
    assert torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0)
    print("Triton implementation matches PyTorch standard attention!")
