import pytest
import torch

from matmul import matmul

# ---------------------------------------------------------------------------
# Guard: skip entire module if CUDA is unavailable
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU required for Triton kernels",
)

DEVICE = "cuda"
RTOL, ATOL = 1e-2, 1e-2  # fp16 arithmetic tolerances


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_matrices(M: int, K: int, N: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a random (M,K) and (K,N) float16 pair on CUDA."""
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    return a, b


def torch_ref(a: torch.Tensor, b: torch.Tensor, activation: str = "") -> torch.Tensor:
    """Reference implementation using PyTorch (fp32 accumulation → fp16)."""
    c = torch.matmul(a.float(), b.float()).half()
    if activation == "leaky_relu":
        c = torch.where(c >= 0, c, 0.01 * c)
    return c


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

class TestCorrectnessNoActivation:
    """Verify C = A @ B without any activation."""

    @pytest.mark.parametrize("M,K,N", [
        (128, 128, 128),  # square, power-of-2
        (256, 512, 128),  # rectangular
        (64, 64, 64),  # small square
        (512, 256, 512),  # larger
    ])
    def test_square_and_rectangular(self, M, K, N):
        a, b = make_matrices(M, K, N)
        got = matmul(a, b)
        ref = torch_ref(a, b)
        assert got.shape == (M, N), f"Output shape mismatch: {got.shape}"
        assert torch.allclose(got, ref, rtol=RTOL, atol=ATOL), (
            f"Max diff: {(got - ref).abs().max().item():.4f}"
        )

    @pytest.mark.parametrize("M,K,N", [
        (100, 70, 50),  # non-power-of-2
        (17, 31, 23),  # prime-ish
        (1, 128, 128),  # M=1
        (128, 128, 1),  # N=1
    ])
    def test_non_power_of_two(self, M, K, N):
        a, b = make_matrices(M, K, N)
        got = matmul(a, b)
        ref = torch_ref(a, b)
        assert got.shape == (M, N)
        assert torch.allclose(got, ref, rtol=RTOL, atol=ATOL)


class TestCorrectnessLeakyReLU:
    """Verify C = LeakyReLU(A @ B)."""

    @pytest.mark.parametrize("M,K,N", [
        (128, 128, 128),
        (256, 64, 128),
        (100, 70, 50),
    ])
    def test_leaky_relu_activation(self, M, K, N):
        a, b = make_matrices(M, K, N)
        got = matmul(a, b, activation="leaky_relu")
        ref = torch_ref(a, b, activation="leaky_relu")
        assert got.shape == (M, N)
        assert torch.allclose(got, ref, rtol=RTOL, atol=ATOL), (
            f"Max diff: {(got - ref).abs().max().item():.4f}"
        )

    def test_leaky_relu_no_negative_large_values(self):
        """All-positive input → output should equal plain matmul."""
        M, K, N = 128, 64, 128
        a = torch.abs(torch.randn((M, K), device=DEVICE, dtype=torch.float16))
        b = torch.abs(torch.randn((K, N), device=DEVICE, dtype=torch.float16))
        c_relu = matmul(a, b, activation="leaky_relu")
        c_plain = matmul(a, b)
        assert torch.allclose(c_relu, c_plain, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Output property tests
# ---------------------------------------------------------------------------

class TestOutputProperties:

    def test_output_dtype_is_float16(self):
        a, b = make_matrices(64, 64, 64)
        c = matmul(a, b)
        assert c.dtype == torch.float16

    def test_output_device_matches_input(self):
        a, b = make_matrices(64, 64, 64)
        c = matmul(a, b)
        assert c.device.type == a.device.type

    def test_output_shape(self):
        M, K, N = 37, 51, 29
        a, b = make_matrices(M, K, N)
        c = matmul(a, b)
        assert c.shape == (M, N)

    def test_no_nans_or_infs(self):
        a, b = make_matrices(128, 128, 128)
        c = matmul(a, b)
        assert not torch.isnan(c).any(), "Output contains NaNs"
        assert not torch.isinf(c).any(), "Output contains Infs"


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_incompatible_dimensions_raise(self):
        a = torch.randn((64, 32), device=DEVICE, dtype=torch.float16)
        b = torch.randn((16, 64), device=DEVICE, dtype=torch.float16)  # K mismatch
        with pytest.raises(AssertionError, match="Incompatible dimensions"):
            matmul(a, b)

    def test_non_contiguous_a_raises(self):
        a = torch.randn((128, 64), device=DEVICE, dtype=torch.float16).t()  # transposed → non-contiguous
        b = torch.randn((128, 64), device=DEVICE, dtype=torch.float16)
        with pytest.raises(AssertionError, match="contiguous"):
            matmul(a, b)

    def test_non_contiguous_b_raises(self):
        a = torch.randn((64, 128), device=DEVICE, dtype=torch.float16)
        b = torch.randn((64, 128), device=DEVICE, dtype=torch.float16).t()  # transposed → non-contiguous
        with pytest.raises(AssertionError, match="contiguous"):
            matmul(a, b)


# ---------------------------------------------------------------------------
# Numerical edge-case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_identity_matrix(self):
        """A @ I  should return A (within fp16 tolerance)."""
        M, K = 64, 64
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        identity = torch.eye(K, device=DEVICE, dtype=torch.float16)
        c = matmul(a, identity)
        assert torch.allclose(c, a, rtol=RTOL, atol=ATOL)

    def test_zero_matrix(self):
        """A @ 0  should be all zeros."""
        M, K, N = 64, 64, 64
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.zeros((K, N), device=DEVICE, dtype=torch.float16)
        c = matmul(a, b)
        assert torch.all(c == 0), "Expected all-zero output for zero B"

    def test_leaky_relu_negative_slope(self):
        """Confirm negative outputs are scaled by 0.01, not zeroed."""
        # Build matrices guaranteed to produce a negative dot product.
        M, K, N = 32, 32, 32
        a = torch.ones((M, K), device=DEVICE, dtype=torch.float16)
        b = -torch.ones((K, N), device=DEVICE, dtype=torch.float16)
        c = matmul(a, b, activation="leaky_relu")
        # Every element of A@B == -K; leaky_relu(-K) == -0.01*K
        expected_val = -0.01 * K
        assert torch.allclose(
            c, torch.full_like(c, expected_val), rtol=RTOL, atol=ATOL
        ), f"Expected {expected_val:.3f}, got range [{c.min().item():.3f}, {c.max().item():.3f}]"
