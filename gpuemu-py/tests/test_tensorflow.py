"""Tests for TensorFlow framework integration.

These tests verify the TensorFlow adapter functionality without requiring
an actual gpuemu daemon connection. Framework-specific features like
GradientTape and tensor conversion are tested in isolation.
"""

import numpy as np
import pytest

# Skip all tests if TensorFlow is not installed
tf = pytest.importorskip("tensorflow")


class TestTensorFlowAdapter:
    """Tests for TensorFlowAdapter class."""

    @pytest.fixture
    def adapter(self):
        """Create a TensorFlowAdapter instance."""
        from gpuemu_py.frameworks.tensorflow import TensorFlowAdapter

        return TensorFlowAdapter()

    def test_to_numpy_from_tensor(self, adapter):
        """Test converting tf.Tensor to numpy."""
        t = tf.constant([1.0, 2.0, 3.0])
        arr = adapter.to_numpy(t)

        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_to_numpy_from_variable(self, adapter):
        """Test converting tf.Variable to numpy."""
        v = tf.Variable([1.0, 2.0, 3.0])
        arr = adapter.to_numpy(v)

        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_to_numpy_from_numpy(self, adapter):
        """Test that numpy arrays pass through unchanged."""
        arr = np.array([1.0, 2.0, 3.0])
        result = adapter.to_numpy(arr)

        assert result is arr

    def test_from_numpy_basic(self, adapter):
        """Test converting numpy to tf.Tensor."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = adapter.from_numpy(arr)

        assert isinstance(t, tf.Tensor)
        np.testing.assert_array_equal(t.numpy(), [1.0, 2.0, 3.0])

    def test_from_numpy_with_like(self, adapter):
        """Test converting numpy with template tensor."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        like = tf.zeros(3, dtype=tf.float64)
        t = adapter.from_numpy(arr, like=like)

        assert t.dtype == tf.float64

    def test_get_dtype_name(self, adapter):
        """Test getting dtype name."""
        t32 = tf.constant([1.0], dtype=tf.float32)
        t16 = tf.constant([1.0], dtype=tf.float16)
        t64 = tf.constant([1.0], dtype=tf.float64)

        assert adapter.get_dtype_name(t32) == "float32"
        assert adapter.get_dtype_name(t16) == "float16"
        assert adapter.get_dtype_name(t64) == "float64"

    def test_requires_grad_variable(self, adapter):
        """Test requires_grad detection for Variables."""
        v = tf.Variable([1.0, 2.0, 3.0])
        assert adapter.requires_grad(v) is True

    def test_requires_grad_constant(self, adapter):
        """Test requires_grad detection for constants."""
        t = tf.constant([1.0, 2.0, 3.0])
        assert adapter.requires_grad(t) is False

    def test_compute_gradient(self, adapter):
        """Test gradient computation with GradientTape."""
        x = tf.Variable([1.0, 2.0, 3.0])

        with tf.GradientTape() as tape:
            y = x ** 2

        grads = adapter.compute_gradient(tape, tf.reduce_sum(y), {"x": x})

        assert "x" in grads
        # d/dx(x^2) = 2x
        np.testing.assert_array_almost_equal(grads["x"].numpy(), [2.0, 4.0, 6.0])

    def test_compute_gradient_multiple_inputs(self, adapter):
        """Test gradient computation with multiple Variables."""
        x = tf.Variable([1.0, 2.0])
        y = tf.Variable([3.0, 4.0])

        with tf.GradientTape() as tape:
            z = x * y

        grads = adapter.compute_gradient(tape, tf.reduce_sum(z), {"x": x, "y": y})

        assert "x" in grads
        assert "y" in grads
        np.testing.assert_array_almost_equal(grads["x"].numpy(), y.numpy())
        np.testing.assert_array_almost_equal(grads["y"].numpy(), x.numpy())

    def test_is_available(self, adapter):
        """Test framework availability check."""
        assert adapter.is_available() is True

    def test_get_framework_name(self, adapter):
        """Test framework name."""
        assert adapter.get_framework_name() == "tensorflow"


class TestCheckTfFunctionSafe:
    """Tests for check_tf_function_safe function."""

    def test_simple_function(self):
        """Test tf.function safety for simple operations."""
        from gpuemu_py.frameworks.tensorflow import check_tf_function_safe

        def simple_op(x):
            return tf.sin(x)

        x = tf.ones(10)
        assert check_tf_function_safe(simple_op, {"x": x}) is True

    def test_pure_operations(self):
        """Test tf.function safety for pure operations."""
        from gpuemu_py.frameworks.tensorflow import check_tf_function_safe

        def pure_op(x, y):
            return x * y + tf.exp(x)

        x = tf.ones(10)
        y = tf.ones(10) * 2
        assert check_tf_function_safe(pure_op, {"x": x, "y": y}) is True


class TestCheckXlaCompatible:
    """Tests for check_xla_compatible function."""

    def test_simple_xla_compatible(self):
        """Test XLA compatibility for simple operations."""
        from gpuemu_py.frameworks.tensorflow import check_xla_compatible

        def simple_op(x):
            return tf.sin(x)

        x = tf.ones(10)
        result = check_xla_compatible(simple_op, {"x": x})
        # Result depends on XLA availability
        assert isinstance(result, (bool, np.bool_))

    def test_matmul(self):
        """Test XLA compatibility for matrix operations."""
        from gpuemu_py.frameworks.tensorflow import check_xla_compatible

        def matmul_op(x, y):
            return tf.matmul(x, y)

        x = tf.ones((3, 5))
        y = tf.ones((5, 2))
        result = check_xla_compatible(matmul_op, {"x": x, "y": y})
        assert isinstance(result, (bool, np.bool_))


class TestValidateCustomGradient:
    """Tests for validate_custom_gradient function."""

    def test_correct_custom_gradient(self):
        """Test validation of correct custom gradient."""
        from gpuemu_py.frameworks.tensorflow import validate_custom_gradient

        def func(x):
            return x ** 2

        def gradient_func(x, dy):
            return dy * 2 * x

        x = tf.Variable(tf.ones(10))
        result = validate_custom_gradient(func, gradient_func, {"x": x})

        assert result["gradient_ok"] is True

    def test_incorrect_custom_gradient(self):
        """Test validation catches incorrect custom gradient."""
        from gpuemu_py.frameworks.tensorflow import validate_custom_gradient

        def func(x):
            return x ** 2

        def wrong_gradient(x, dy):
            return dy * 3 * x  # Wrong! Should be 2x

        x = tf.Variable(tf.ones(10))
        result = validate_custom_gradient(func, wrong_gradient, {"x": x})

        assert result["gradient_ok"] is False


class TestLazyImport:
    """Tests for lazy import functionality."""

    def test_get_tensorflow_adapter(self):
        """Test get_tensorflow_adapter returns correct types."""
        from gpuemu_py import get_tensorflow_adapter

        TensorFlowAdapter, validate_tensorflow, check_keras_layer = (
            get_tensorflow_adapter()
        )

        assert TensorFlowAdapter is not None
        assert callable(validate_tensorflow)
        assert callable(check_keras_layer)

    def test_direct_import(self):
        """Test direct import from submodule."""
        from gpuemu_py.frameworks.tensorflow import (
            TensorFlowAdapter,
            check_keras_layer,
            validate_tensorflow,
        )

        assert TensorFlowAdapter is not None
        assert callable(validate_tensorflow)
        assert callable(check_keras_layer)


class TestIntegrationWithKeras:
    """Integration tests with Keras layers."""

    def test_dense_layer_forward(self):
        """Test forward pass of a Dense layer."""
        from gpuemu_py.frameworks.tensorflow import TensorFlowAdapter

        adapter = TensorFlowAdapter()
        layer = tf.keras.layers.Dense(10)

        x = tf.random.normal((4, 5))
        output = layer(x)

        np_output = adapter.to_numpy(output)
        assert np_output.shape == (4, 10)

    def test_dense_layer_gradient(self):
        """Test gradient computation through Dense layer."""
        layer = tf.keras.layers.Dense(10)
        x = tf.Variable(tf.random.normal((4, 5)))

        with tf.GradientTape() as tape:
            output = layer(x)
            loss = tf.reduce_sum(output)

        grad = tape.gradient(loss, x)
        assert grad is not None
        assert grad.shape == x.shape
