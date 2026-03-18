"""Framework-specific adapters for gpuemu validation.

This module provides adapters for PyTorch, JAX, and TensorFlow to enable
validation of custom ops with framework-specific features like autograd,
vmap, and GradientTape.

Each adapter is lazily imported to avoid import errors when the framework
is not installed.

Example:
    >>> from gpuemu_py.frameworks.pytorch import validate_pytorch, check_autograd
    >>> from gpuemu_py.frameworks.jax import validate_jax, check_vmap_compatible
    >>> from gpuemu_py.frameworks.tensorflow import validate_tensorflow
"""

from gpuemu_py.frameworks.base import FrameworkAdapter

__all__ = ["FrameworkAdapter"]


def get_pytorch_adapter():
    """Get PyTorch adapter and utilities.

    Returns:
        Tuple of (PyTorchAdapter, validate_pytorch, check_autograd)

    Raises:
        ImportError: If PyTorch is not installed.
    """
    from gpuemu_py.frameworks.pytorch import (
        PyTorchAdapter,
        check_autograd,
        validate_pytorch,
    )

    return PyTorchAdapter, validate_pytorch, check_autograd


def get_jax_adapter():
    """Get JAX adapter and utilities.

    Returns:
        Tuple of (JAXAdapter, validate_jax, check_vmap_compatible, check_jit_safe)

    Raises:
        ImportError: If JAX is not installed.
    """
    from gpuemu_py.frameworks.jax import (
        JAXAdapter,
        check_jit_safe,
        check_vmap_compatible,
        validate_jax,
    )

    return JAXAdapter, validate_jax, check_vmap_compatible, check_jit_safe


def get_tensorflow_adapter():
    """Get TensorFlow adapter and utilities.

    Returns:
        Tuple of (TensorFlowAdapter, validate_tensorflow, check_keras_layer)

    Raises:
        ImportError: If TensorFlow is not installed.
    """
    from gpuemu_py.frameworks.tensorflow import (
        TensorFlowAdapter,
        check_keras_layer,
        validate_tensorflow,
    )

    return TensorFlowAdapter, validate_tensorflow, check_keras_layer
