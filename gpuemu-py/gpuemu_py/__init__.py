"""gpuemu-py: Python client for GPU-less validation of deep learning kernels.

This package provides:
- Client for communicating with the gpuemu daemon
- Validation utilities for custom ops
- Fuzzing infrastructure for testing
- Framework-specific adapters for PyTorch, JAX, and TensorFlow
- Cross-framework tolerance calibration

Example:
    >>> from gpuemu_py import Client
    >>> client = Client()
    >>> client.ping()
    {'version': '0.1.0', 'uptime_secs': 123}

For framework-specific validation:
    >>> from gpuemu_py.frameworks.pytorch import validate_pytorch
    >>> with validate_pytorch(client, "my_op", {"x": x}) as ctx:
    ...     ctx["output"] = my_custom_op(x)
"""

from gpuemu_py.client import (
    Client,
    ClientError,
    FuzzResults,
    MinimizeResult,
    ReproduceResult,
    ReproductionInfo,
    ValidationResult,
)
from gpuemu_py.rng import SeededRng, derive_seed, generate_seed
from gpuemu_py.tolerances import (
    ToleranceConfig,
    ToleranceProfile,
    calibrate_tolerance,
    get_recommended_tolerance,
)
from gpuemu_py.validate import (
    FuzzConfig,
    SeededFuzzer,
    ValidationError,
    fuzz_dtypes,
    fuzz_dtypes_seeded,
    fuzz_layouts,
    fuzz_layouts_seeded,
    fuzz_shapes,
    fuzz_shapes_seeded,
    generate_random_tensor,
    validate,
    validate_op,
)

__version__ = "0.1.0"


def get_pytorch_adapter():
    """Get PyTorch adapter and utilities.

    Returns:
        Tuple of (PyTorchAdapter, validate_pytorch, check_autograd)

    Raises:
        ImportError: If PyTorch is not installed.

    Example:
        >>> PyTorchAdapter, validate_pytorch, check_autograd = get_pytorch_adapter()
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

    Example:
        >>> JAXAdapter, validate_jax, check_vmap_compatible, check_jit_safe = get_jax_adapter()
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

    Example:
        >>> TensorFlowAdapter, validate_tensorflow, check_keras_layer = get_tensorflow_adapter()
    """
    from gpuemu_py.frameworks.tensorflow import (
        TensorFlowAdapter,
        check_keras_layer,
        validate_tensorflow,
    )

    return TensorFlowAdapter, validate_tensorflow, check_keras_layer


__all__ = [
    # Client
    "Client",
    "ClientError",
    "ValidationResult",
    "ReproductionInfo",
    "FuzzResults",
    "ReproduceResult",
    "MinimizeResult",
    # RNG
    "SeededRng",
    "derive_seed",
    "generate_seed",
    # Validation
    "validate",
    "validate_op",
    "ValidationError",
    # Fuzzing (exhaustive)
    "fuzz_shapes",
    "fuzz_dtypes",
    "fuzz_layouts",
    # Fuzzing (seeded/reproducible)
    "FuzzConfig",
    "SeededFuzzer",
    "fuzz_shapes_seeded",
    "fuzz_dtypes_seeded",
    "fuzz_layouts_seeded",
    "generate_random_tensor",
    # Tolerances
    "ToleranceConfig",
    "ToleranceProfile",
    "calibrate_tolerance",
    "get_recommended_tolerance",
    # Framework accessors (lazy imports)
    "get_pytorch_adapter",
    "get_jax_adapter",
    "get_tensorflow_adapter",
]
