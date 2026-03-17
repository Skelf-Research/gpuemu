"""Validation context managers and utilities for gpuemu."""

import itertools
from contextlib import contextmanager
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import numpy as np

from gpuemu_py.client import Client, ValidationResult


class ValidationError(Exception):
    """Raised when validation fails."""

    def __init__(self, result: ValidationResult):
        self.result = result
        failures = result.failures[:3]  # Show first 3 failures
        failure_msgs = [f["message"] for f in failures]
        msg = f"Validation failed for '{result.op_name}': {', '.join(failure_msgs)}"
        if len(result.failures) > 3:
            msg += f" (and {len(result.failures) - 3} more failures)"
        super().__init__(msg)


@contextmanager
def validate(
    client: Client,
    model: Any,
    reference: str = "cpu",
    dtype: Optional[str] = None,
    reference_dtype: Optional[str] = None,
    raise_on_failure: bool = True,
) -> Generator[None, None, None]:
    """Context manager for validating model outputs.

    This is a high-level validation context that captures model inputs/outputs
    and validates them against a reference implementation.

    Args:
        client: gpuemu Client instance.
        model: The model/function to validate.
        reference: Reference type ("cpu" for CPU reference).
        dtype: Expected dtype for validation.
        reference_dtype: Dtype to use for reference computation.
        raise_on_failure: Whether to raise ValidationError on failure.

    Yields:
        None. Validation happens after the context exits.

    Example:
        >>> with validate(client, model):
        ...     output = model(input_tensor)
    """
    # For MVP, this is a placeholder that yields without validation
    # Full implementation would hook into the model's forward pass
    yield


@contextmanager
def validate_op(
    client: Client,
    op_name: str,
    inputs: Optional[Dict[str, np.ndarray]] = None,
    raise_on_failure: bool = True,
    **kwargs,
) -> Generator[Dict[str, Any], None, None]:
    """Context manager for validating a specific op.

    Args:
        client: gpuemu Client instance.
        op_name: Name of the op (must be registered in gpuemu.toml).
        inputs: Input tensors for the op.
        raise_on_failure: Whether to raise ValidationError on failure.
        **kwargs: Additional kwargs to pass to the reference script.

    Yields:
        Dict to store the output in. Set result["output"] with the op output.

    Example:
        >>> with validate_op(client, "flash_attn", inputs={"q": q, "k": k, "v": v}) as ctx:
        ...     ctx["output"] = flash_attn_func(q, k, v)
    """
    inputs = inputs or {}
    context: Dict[str, Any] = {"inputs": inputs, "kwargs": kwargs}

    yield context

    # After context exits, validate the output
    if "output" not in context:
        raise ValueError("No output captured. Set ctx['output'] = <your_output>")

    output = context["output"]

    # Convert to numpy if needed
    if hasattr(output, "numpy"):
        output = output.numpy()
    elif hasattr(output, "detach"):
        output = output.detach().cpu().numpy()

    # Convert inputs to numpy
    np_inputs = {}
    for name, tensor in inputs.items():
        if hasattr(tensor, "numpy"):
            np_inputs[name] = tensor.numpy()
        elif hasattr(tensor, "detach"):
            np_inputs[name] = tensor.detach().cpu().numpy()
        else:
            np_inputs[name] = np.asarray(tensor)

    # Send to daemon for validation
    result = client.validate_op(op_name, np_inputs, output, **kwargs)

    context["result"] = result

    if not result.passed and raise_on_failure:
        raise ValidationError(result)


def fuzz_shapes(
    **dimensions: List[int],
) -> Iterator[Tuple[int, ...]]:
    """Generate all combinations of shape dimensions for fuzzing.

    Args:
        **dimensions: Named dimension lists. E.g., batch=[1, 2], seq=[64, 128].

    Yields:
        Tuples of dimension values in the order they were specified.

    Example:
        >>> for batch, seq in fuzz_shapes(batch=[1, 2], seq=[64, 128]):
        ...     x = torch.randn(batch, seq, 512)
        ...     test_model(x)
    """
    if not dimensions:
        return

    names = list(dimensions.keys())
    values = [dimensions[name] for name in names]

    for combo in itertools.product(*values):
        yield combo


def fuzz_dtypes(
    dtypes: Optional[List[str]] = None,
) -> Iterator[str]:
    """Generate dtypes for fuzzing.

    Args:
        dtypes: List of dtype strings. Defaults to ["float32", "float16"].

    Yields:
        Dtype strings.

    Example:
        >>> for dtype in fuzz_dtypes():
        ...     x = torch.randn(2, 3, dtype=getattr(torch, dtype))
        ...     test_model(x)
    """
    if dtypes is None:
        dtypes = ["float32", "float16"]

    yield from dtypes


def fuzz_layouts(
    shape: Tuple[int, ...],
    include_contiguous: bool = True,
    include_strided: bool = True,
    include_transposed: bool = True,
) -> Iterator[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Generate different memory layouts for fuzzing.

    Args:
        shape: Base shape of the tensor.
        include_contiguous: Include contiguous layout.
        include_strided: Include strided views.
        include_transposed: Include transposed layouts.

    Yields:
        Tuples of (shape, strides) representing different layouts.

    Example:
        >>> base_shape = (2, 3, 4)
        >>> for shape, strides in fuzz_layouts(base_shape):
        ...     # Create tensor with specific strides
        ...     pass
    """
    import numpy as np

    # Contiguous layout (row-major)
    if include_contiguous:
        strides = []
        stride = 1
        for dim in reversed(shape):
            strides.insert(0, stride)
            stride *= dim
        yield shape, tuple(strides)

    # Transposed layouts (swap adjacent dimensions)
    if include_transposed and len(shape) >= 2:
        for i in range(len(shape) - 1):
            new_shape = list(shape)
            new_shape[i], new_shape[i + 1] = new_shape[i + 1], new_shape[i]

            strides = []
            stride = 1
            for dim in reversed(new_shape):
                strides.insert(0, stride)
                stride *= dim
            yield tuple(new_shape), tuple(strides)

    # Strided layouts (add gaps)
    if include_strided and len(shape) >= 1:
        # Double the stride of the last dimension
        strides = []
        stride = 2  # Start with stride 2 instead of 1
        for dim in reversed(shape):
            strides.insert(0, stride)
            stride *= dim
        yield shape, tuple(strides)
