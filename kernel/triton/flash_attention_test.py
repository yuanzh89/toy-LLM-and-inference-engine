import pytest
import torch

from flash_attention import flash_attention

# ---------------------------------------------------------------------------
# Guard: skip entire module if CUDA is unavailable
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _pytorch_causal_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
) -> torch.Tensor:
    """Reference causal attention in fp32, returned as fp16."""
    seq_len = q.shape[-2]
    D = q.shape[-1]
    scale = D ** -0.5

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device)).bool()
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(torch.float16)


def _make_qkv(B, H, seq_len, D, seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    dtype = torch.float16
    q = torch.randn((B, H, seq_len, D), dtype=dtype, device=device)
    k = torch.randn((B, H, seq_len, D), dtype=dtype, device=device)
    v = torch.randn((B, H, seq_len, D), dtype=dtype, device=device)
    return q, k, v


# ---------------------------------------------------------------------------
# Parametrised correctness tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "B, H, seq_len, D",
    [
        (1, 1, 64, 64),  # tiny
        (1, 4, 128, 64),  # single batch, 4 heads
        (2, 4, 512, 64),  # typical small case
        (1, 4, 1024, 64),  # original smoke-test size
        (2, 8, 256, 128),  # larger head-dim
        (1, 1, 192, 64),  # seq_len not a multiple of BLOCK_N (64)
        (1, 1, 100, 64),  # seq_len not a multiple of BLOCK_M (128) either
    ],
)
def test_correctness(B, H, seq_len, D):
    """Triton output must match PyTorch reference within fp16 tolerance."""
    q, k, v = _make_qkv(B, H, seq_len, D)
    triton_out = flash_attention(q, k, v)
    ref = _pytorch_causal_attention(q, k, v)

    max_diff = (triton_out - ref).abs().max().item()
    assert max_diff < 1e-2, (
        f"B={B} H={H} seq={seq_len} D={D}: max_diff={max_diff:.6f} exceeds 1e-2"
    )


# ---------------------------------------------------------------------------
# Output shape / dtype
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("B, H, seq_len, D", [(2, 4, 256, 64)])
def test_output_shape_and_dtype(B, H, seq_len, D):
    q, k, v = _make_qkv(B, H, seq_len, D)
    out = flash_attention(q, k, v)
    assert out.shape == q.shape, f"Shape mismatch: {out.shape} vs {q.shape}"
    assert out.dtype == torch.float16, f"Expected float16, got {out.dtype}"
    assert out.device == q.device, "Device mismatch"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_deterministic():
    """Two identical calls must return identical tensors."""
    q, k, v = _make_qkv(1, 4, 256, 64)
    out1 = flash_attention(q, k, v)
    out2 = flash_attention(q, k, v)
    assert torch.equal(out1, out2), "Outputs are not deterministic"


# ---------------------------------------------------------------------------
# Causal masking: output for token i must not depend on tokens j > i
# ---------------------------------------------------------------------------
def test_causal_masking():
    """
    Modify v at position seq_len//2 onwards and verify that the output
    for earlier positions is unchanged.
    """
    B, H, seq_len, D = 1, 2, 128, 64
    q, k, v = _make_qkv(B, H, seq_len, D, seed=42)

    out_original = flash_attention(q, k, v)

    # Corrupt the second half of V
    v_corrupted = v.clone()
    mid = seq_len // 2
    v_corrupted[:, :, mid:, :] = 999.0

    out_corrupted = flash_attention(q, k, v_corrupted)

    # Positions 0 … mid-1 should be unaffected
    max_diff = (out_original[:, :, :mid, :] - out_corrupted[:, :, :mid, :]).abs().max().item()
    assert max_diff == 0.0, (
        f"Causal masking violated: positions before {mid} changed by {max_diff}"
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_mismatched_shapes_raises():
    device = "cuda"
    dtype = torch.float16
    q = torch.randn(1, 4, 128, 64, dtype=dtype, device=device)
    k = torch.randn(1, 4, 256, 64, dtype=dtype, device=device)  # different seq_len
    v = torch.randn(1, 4, 128, 64, dtype=dtype, device=device)
    with pytest.raises(AssertionError):
        flash_attention(q, k, v)


def test_wrong_dtype_raises():
    device = "cuda"
    q = torch.randn(1, 4, 128, 64, dtype=torch.float32, device=device)
    k = torch.randn(1, 4, 128, 64, dtype=torch.float32, device=device)
    v = torch.randn(1, 4, 128, 64, dtype=torch.float32, device=device)
    with pytest.raises(AssertionError):
        flash_attention(q, k, v)
