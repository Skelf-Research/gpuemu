"""Framework-specific adapters for gpuemu validation.

This module provides adapters for PyTorch, JAX, and TensorFlow to enable
validation of custom ops with framework-specific features like autograd,
vmap, and GradientTape.

Each adapter is lazily imported to avoid import errors when the framework
is not installed.

Example:
    >>> from gpuemu.frameworks.pytorch import validate_pytorch, check_autograd, fuzz_pytorch_op
    >>> from gpuemu.frameworks.jax import validate_jax, check_vmap_compatible, fuzz_jax_op
    >>> from gpuemu.frameworks.tensorflow import validate_tensorflow, fuzz_tensorflow_op
"""

from gpuemu.frameworks.base import FrameworkAdapter

__all__ = ["FrameworkAdapter"]


def get_pytorch_adapter():
    """Get PyTorch adapter and utilities.

    Returns:
        Tuple of (PyTorchAdapter, validate_pytorch, check_autograd, fuzz_pytorch_op)

    Raises:
        ImportError: If PyTorch is not installed.
    """
    from gpuemu.frameworks.pytorch import (
        PyTorchAdapter,
        check_autograd,
        fuzz_pytorch_op,
        validate_pytorch,
    )

    return PyTorchAdapter, validate_pytorch, check_autograd, fuzz_pytorch_op


def get_jax_adapter():
    """Get JAX adapter and utilities.

    Returns:
        Tuple of (JAXAdapter, validate_jax, check_vmap_compatible, check_jit_safe, fuzz_jax_op)

    Raises:
        ImportError: If JAX is not installed.
    """
    from gpuemu.frameworks.jax import (
        JAXAdapter,
        check_jit_safe,
        check_vmap_compatible,
        fuzz_jax_op,
        validate_jax,
    )

    return JAXAdapter, validate_jax, check_vmap_compatible, check_jit_safe, fuzz_jax_op


def get_tensorflow_adapter():
    """Get TensorFlow adapter and utilities.

    Returns:
        Tuple of (TensorFlowAdapter, validate_tensorflow, check_keras_layer, fuzz_tensorflow_op)

    Raises:
        ImportError: If TensorFlow is not installed.
    """
    from gpuemu.frameworks.tensorflow import (
        TensorFlowAdapter,
        check_keras_layer,
        fuzz_tensorflow_op,
        validate_tensorflow,
    )

    return TensorFlowAdapter, validate_tensorflow, check_keras_layer, fuzz_tensorflow_op
