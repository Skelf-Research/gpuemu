"""Tests for PyTorch framework integration.

These tests verify the PyTorch adapter functionality without requiring
an actual gpuemu daemon connection. Framework-specific features like
autograd checking and tensor conversion are tested in isolation.
"""

import numpy as np
import pytest

# Skip all tests if PyTorch is not installed
torch = pytest.importorskip("torch")


class TestPyTorchAdapter:
    """Tests for PyTorchAdapter class."""

    @pytest.fixture
    def adapter(self):
        """Create a PyTorchAdapter instance."""
        from gpuemu.frameworks.pytorch import PyTorchAdapter

        return PyTorchAdapter()

    def test_to_numpy_from_tensor(self, adapter):
        """Test converting torch.Tensor to numpy."""
        t = torch.tensor([1.0, 2.0, 3.0])
        arr = adapter.to_numpy(t)

        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_to_numpy_from_numpy(self, adapter):
        """Test that numpy arrays pass through unchanged."""
        arr = np.array([1.0, 2.0, 3.0])
        result = adapter.to_numpy(arr)

        assert result is arr

    def test_to_numpy_with_grad(self, adapter):
        """Test converting tensor with requires_grad."""
        t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        arr = adapter.to_numpy(t)

        # Should not require grad after conversion
        assert isinstance(arr, np.ndarray)

    def test_to_numpy_from_gpu(self, adapter):
        """Test converting GPU tensor to numpy."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        t = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        arr = adapter.to_numpy(t)

        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_from_numpy_basic(self, adapter):
        """Test converting numpy to torch.Tensor."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = adapter.from_numpy(arr)

        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float32
        torch.testing.assert_close(t, torch.tensor([1.0, 2.0, 3.0]))

    def test_from_numpy_with_like(self, adapter):
        """Test converting numpy with template tensor."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        like = torch.zeros(3, dtype=torch.float64)
        t = adapter.from_numpy(arr, like=like)

        assert t.dtype == torch.float64

    def test_get_dtype_name(self, adapter):
        """Test getting dtype name."""
        t32 = torch.tensor([1.0], dtype=torch.float32)
        t16 = torch.tensor([1.0], dtype=torch.float16)
        t64 = torch.tensor([1.0], dtype=torch.float64)

        assert adapter.get_dtype_name(t32) == "float32"
        assert adapter.get_dtype_name(t16) == "float16"
        assert adapter.get_dtype_name(t64) == "float64"

    def test_requires_grad_true(self, adapter):
        """Test requires_grad detection for gradient tensors."""
        t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        assert adapter.requires_grad(t) is True

    def test_requires_grad_false(self, adapter):
        """Test requires_grad detection for non-gradient tensors."""
        t = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)
        assert adapter.requires_grad(t) is False

    def test_compute_gradient(self, adapter):
        """Test gradient computation."""
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x ** 2

        grads = adapter.compute_gradient(y.sum(), {"x": x})

        assert "x" in grads
        # d/dx(x^2) = 2x
        torch.testing.assert_close(grads["x"], 2 * x)

    def test_compute_gradient_multiple_inputs(self, adapter):
        """Test gradient computation with multiple inputs."""
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = torch.tensor([3.0, 4.0], requires_grad=True)
        z = x * y

        grads = adapter.compute_gradient(z.sum(), {"x": x, "y": y})

        assert "x" in grads
        assert "y" in grads
        torch.testing.assert_close(grads["x"], y)
        torch.testing.assert_close(grads["y"], x)

    def test_compute_gradient_non_grad_input(self, adapter):
        """Test gradient computation with non-grad input."""
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = torch.tensor([3.0, 4.0], requires_grad=False)
        z = x * y

        grads = adapter.compute_gradient(z.sum(), {"x": x, "y": y})

        assert "x" in grads
        assert "y" not in grads

    def test_is_available(self, adapter):
        """Test framework availability check."""
        assert adapter.is_available() is True

    def test_get_framework_name(self, adapter):
        """Test framework name."""
        assert adapter.get_framework_name() == "pytorch"


class TestCheckAutograd:
    """Tests for check_autograd function."""

    def test_correct_gradient(self):
        """Test check_autograd with correct gradient."""
        from gpuemu.frameworks.pytorch import check_autograd

        def square(x):
            return x ** 2

        x = torch.randn(10, requires_grad=True)
        assert check_autograd(square, {"x": x}) is True

    def test_simple_ops(self):
        """Test check_autograd with various simple operations."""
        from gpuemu.frameworks.pytorch import check_autograd

        x = torch.randn(10, requires_grad=True)

        # sin
        assert check_autograd(lambda x: torch.sin(x), {"x": x}) is True

        # exp
        assert check_autograd(lambda x: torch.exp(x), {"x": x}) is True

        # matmul
        y = torch.randn(10, 5, requires_grad=True)
        assert check_autograd(lambda x: torch.matmul(x.unsqueeze(0), x.unsqueeze(1)), {"x": x}) is True

    def test_multi_input(self):
        """Test check_autograd with multiple inputs."""
        from gpuemu.frameworks.pytorch import check_autograd

        def op(x, y):
            return x * y + torch.sin(x)

        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10, requires_grad=True)

        assert check_autograd(op, {"x": x, "y": y}) is True

    def test_check_specific_inputs(self):
        """Test check_autograd with specific input selection."""
        from gpuemu.frameworks.pytorch import check_autograd

        def op(x, y):
            return x * y

        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10, requires_grad=True)

        # Only check x
        assert check_autograd(op, {"x": x, "y": y}, check_inputs=["x"]) is True


class TestValidateCustomAutogradFunction:
    """Tests for validate_custom_autograd_function."""

    def test_correct_custom_function(self):
        """Test validation of correct custom autograd function."""
        from gpuemu.frameworks.pytorch import validate_custom_autograd_function

        class DoubleFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x * 2

            @staticmethod
            def backward(ctx, grad):
                return grad * 2

        x = torch.randn(10, requires_grad=True)
        result = validate_custom_autograd_function(DoubleFunc, {"x": x})

        assert result["forward_ok"] is True
        assert result["backward_ok"] is True

    def test_incorrect_gradient(self):
        """Test validation catches incorrect gradient."""
        from gpuemu.frameworks.pytorch import validate_custom_autograd_function

        class WrongGradFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, grad):
                return grad * 3  # Wrong! Should be 2

        x = torch.randn(10, requires_grad=True)
        result = validate_custom_autograd_function(WrongGradFunc, {"x": x})

        assert result["forward_ok"] is True
        assert result["backward_ok"] is False


class TestLazyImport:
    """Tests for lazy import functionality."""

    def test_get_pytorch_adapter(self):
        """Test get_pytorch_adapter returns correct types."""
        from gpuemu import get_pytorch_adapter

        PyTorchAdapter, validate_pytorch, check_autograd = get_pytorch_adapter()

        assert PyTorchAdapter is not None
        assert callable(validate_pytorch)
        assert callable(check_autograd)

    def test_direct_import(self):
        """Test direct import from submodule."""
        from gpuemu.frameworks.pytorch import (
            PyTorchAdapter,
            check_autograd,
            validate_pytorch,
        )

        assert PyTorchAdapter is not None
        assert callable(validate_pytorch)
        assert callable(check_autograd)
