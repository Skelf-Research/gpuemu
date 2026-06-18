"""Tests for JAX framework integration.

These tests verify the JAX adapter functionality without requiring
an actual gpuemu daemon connection. Framework-specific features like
vmap/jit checking and tensor conversion are tested in isolation.
"""

import numpy as np
import pytest

# Skip all tests if JAX is not installed
jax = pytest.importorskip("jax")
jnp = jax.numpy


class TestJAXAdapter:
    """Tests for JAXAdapter class."""

    @pytest.fixture
    def adapter(self):
        """Create a JAXAdapter instance."""
        from gpuemu.frameworks.jax import JAXAdapter

        return JAXAdapter()

    def test_to_numpy_from_array(self, adapter):
        """Test converting jax.Array to numpy."""
        arr = jnp.array([1.0, 2.0, 3.0])
        result = adapter.to_numpy(arr)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_to_numpy_from_numpy(self, adapter):
        """Test that numpy arrays pass through."""
        arr = np.array([1.0, 2.0, 3.0])
        result = adapter.to_numpy(arr)

        # Should be a numpy array (may or may not be same object)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_from_numpy(self, adapter):
        """Test converting numpy to jax.Array."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = adapter.from_numpy(arr)

        assert isinstance(result, jax.Array)
        np.testing.assert_array_almost_equal(np.asarray(result), [1.0, 2.0, 3.0])

    def test_get_dtype_name(self, adapter):
        """Test getting dtype name."""
        arr32 = jnp.array([1.0], dtype=jnp.float32)
        arr16 = jnp.array([1.0], dtype=jnp.float16)
        arr64 = jnp.array([1.0], dtype=jnp.float64)

        assert "float32" in adapter.get_dtype_name(arr32)
        assert "float16" in adapter.get_dtype_name(arr16)
        assert "float64" in adapter.get_dtype_name(arr64)

    def test_requires_grad_float(self, adapter):
        """Test requires_grad for floating point arrays."""
        arr = jnp.array([1.0, 2.0, 3.0])
        assert adapter.requires_grad(arr) is True

    def test_requires_grad_int(self, adapter):
        """Test requires_grad for integer arrays."""
        arr = jnp.array([1, 2, 3])
        assert adapter.requires_grad(arr) is False

    def test_is_available(self, adapter):
        """Test framework availability check."""
        assert adapter.is_available() is True

    def test_get_framework_name(self, adapter):
        """Test framework name."""
        assert adapter.get_framework_name() == "jax"


class TestCheckVmapCompatible:
    """Tests for check_vmap_compatible function."""

    def test_simple_vmap_compatible(self):
        """Test vmap compatibility for simple operations."""
        from gpuemu.frameworks.jax import check_vmap_compatible

        def simple_op(x):
            return jnp.sin(x)

        x = jnp.ones((4, 10))  # batch of 4
        assert check_vmap_compatible(simple_op, {"x": x}) is True

    def test_elementwise_ops(self):
        """Test vmap compatibility for elementwise operations."""
        from gpuemu.frameworks.jax import check_vmap_compatible

        def op(x):
            return x ** 2 + jnp.cos(x)

        x = jnp.ones((4, 10))
        assert check_vmap_compatible(op, {"x": x}) is True

    def test_matmul(self):
        """Test vmap compatibility for matrix operations."""
        from gpuemu.frameworks.jax import check_vmap_compatible

        def matmul_op(x, y):
            return jnp.dot(x, y)

        x = jnp.ones((4, 3, 5))
        y = jnp.ones((4, 5, 2))
        assert check_vmap_compatible(matmul_op, {"x": x, "y": y}) is True

    def test_single_batch(self):
        """Test with batch size 1 (should pass trivially)."""
        from gpuemu.frameworks.jax import check_vmap_compatible

        def op(x):
            return x * 2

        x = jnp.ones((1, 10))
        assert check_vmap_compatible(op, {"x": x}) is True


class TestCheckJitSafe:
    """Tests for check_jit_safe function."""

    def test_simple_jit_safe(self):
        """Test JIT safety for simple operations."""
        from gpuemu.frameworks.jax import check_jit_safe

        def simple_op(x):
            return jnp.sin(x)

        x = jnp.ones(10)
        assert check_jit_safe(simple_op, {"x": x}) is True

    def test_pure_functions(self):
        """Test JIT safety for pure functions."""
        from gpuemu.frameworks.jax import check_jit_safe

        def pure_op(x, y):
            return x * y + jnp.exp(x)

        x = jnp.ones(10)
        y = jnp.ones(10) * 2
        assert check_jit_safe(pure_op, {"x": x, "y": y}) is True


class TestCheckGradSafe:
    """Tests for check_grad_safe function."""

    def test_differentiable_function(self):
        """Test grad safety for differentiable functions."""
        from gpuemu.frameworks.jax import check_grad_safe

        def op(x):
            return jnp.sum(x ** 2)

        x = jnp.ones(10)
        assert check_grad_safe(op, {"x": x}) is True

    def test_multi_input(self):
        """Test grad safety with multiple inputs."""
        from gpuemu.frameworks.jax import check_grad_safe

        def op(x, y):
            return jnp.sum(x * y)

        x = jnp.ones(10)
        y = jnp.ones(10) * 2

        # Check gradient w.r.t first input
        assert check_grad_safe(op, {"x": x, "y": y}, argnums=0) is True


class TestValidateJaxPrimitive:
    """Tests for validate_jax_primitive function."""

    def test_correct_implementation(self):
        """Test validation of correct primitive implementation."""
        from gpuemu.frameworks.jax import validate_jax_primitive

        def my_sin(x):
            return jnp.sin(x)

        x = jnp.array([0.0, 1.0, 2.0])
        expected = jnp.sin(x)

        result = validate_jax_primitive(
            "my_sin",
            my_sin,
            {"x": x},
            expected,
            check_jvp=True,
            check_vmap=True,
        )

        assert result["forward_ok"] is True
        assert result["jvp_ok"] is True
        assert result["vmap_ok"] is True

    def test_incorrect_forward(self):
        """Test validation catches incorrect forward pass."""
        from gpuemu.frameworks.jax import validate_jax_primitive

        def wrong_sin(x):
            return jnp.cos(x)  # Wrong!

        x = jnp.array([0.0, 1.0, 2.0])
        expected = jnp.sin(x)

        result = validate_jax_primitive(
            "wrong_sin",
            wrong_sin,
            {"x": x},
            expected,
        )

        assert result["forward_ok"] is False


class TestCheckPmapCompatible:
    """Tests for check_pmap_compatible function."""

    def test_simple_op(self):
        """Test pmap compatibility for simple operations."""
        from gpuemu.frameworks.jax import check_pmap_compatible

        def simple_op(x):
            return x * 2

        x = jnp.ones(10)
        # Should at least be traceable
        result = check_pmap_compatible(simple_op, {"x": x})
        # Result depends on available devices
        assert isinstance(result, bool)


class TestLazyImport:
    """Tests for lazy import functionality."""

    def test_get_jax_adapter(self):
        """Test get_jax_adapter returns correct types."""
        from gpuemu import get_jax_adapter

        JAXAdapter, validate_jax, check_vmap_compatible, check_jit_safe = (
            get_jax_adapter()
        )

        assert JAXAdapter is not None
        assert callable(validate_jax)
        assert callable(check_vmap_compatible)
        assert callable(check_jit_safe)

    def test_direct_import(self):
        """Test direct import from submodule."""
        from gpuemu.frameworks.jax import (
            JAXAdapter,
            check_jit_safe,
            check_vmap_compatible,
            validate_jax,
        )

        assert JAXAdapter is not None
        assert callable(validate_jax)
        assert callable(check_vmap_compatible)
        assert callable(check_jit_safe)
