"""Validation context managers and utilities for gpuemu."""

import itertools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import numpy as np

from gpuemu_py.client import Client, ValidationResult
from gpuemu_py.rng import SeededRng, generate_seed


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


# =============================================================================
# Seeded Fuzzing Functions
# =============================================================================


@dataclass
class FuzzConfig:
    """Configuration for fuzz testing.

    Attributes:
        seed: Master seed for reproducibility.
        batch_sizes: List of batch sizes to fuzz.
        seq_lengths: List of sequence lengths to fuzz.
        hidden_dims: List of hidden dimensions to fuzz.
        dtypes: List of dtype strings to fuzz.
        layouts: List of layout types to fuzz.
        edge_cases: List of edge case shapes to include.
    """

    seed: int
    batch_sizes: List[int] = None
    seq_lengths: List[int] = None
    hidden_dims: List[int] = None
    dtypes: List[str] = None
    layouts: List[str] = None
    edge_cases: List[List[int]] = None

    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8, 16, 32]
        if self.seq_lengths is None:
            self.seq_lengths = [64, 128, 256, 512, 1024]
        if self.hidden_dims is None:
            self.hidden_dims = [256, 512, 768, 1024]
        if self.dtypes is None:
            self.dtypes = ["float32", "float16"]
        if self.layouts is None:
            self.layouts = ["contiguous", "strided", "transposed"]
        if self.edge_cases is None:
            self.edge_cases = [[1], [1, 1], [1, 1, 1], [0]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for protocol."""
        return {
            "seed": self.seed,
            "shape_options": {
                "batch_sizes": self.batch_sizes,
                "seq_lengths": self.seq_lengths,
                "hidden_dims": self.hidden_dims,
                "edge_cases": self.edge_cases,
            },
            "dtypes": self.dtypes,
            "layouts": self.layouts,
        }


def fuzz_shapes_seeded(
    seed: int,
    batch_sizes: Optional[List[int]] = None,
    seq_lengths: Optional[List[int]] = None,
    hidden_dims: Optional[List[int]] = None,
    edge_cases: Optional[List[List[int]]] = None,
    edge_case_probability: float = 0.1,
) -> Iterator[Tuple[int, Tuple[int, ...]]]:
    """Generate random shapes deterministically from a seed.

    Unlike fuzz_shapes which exhaustively yields all combinations,
    this function generates random shapes that can be reproduced
    from the seed.

    Args:
        seed: Master seed for reproducibility.
        batch_sizes: List of batch sizes. Defaults to [1, 2, 4, 8, 16, 32].
        seq_lengths: List of sequence lengths. Defaults to [64, 128, 256, 512, 1024].
        hidden_dims: List of hidden dimensions. Defaults to [256, 512, 768, 1024].
        edge_cases: List of edge case shapes to include with some probability.
        edge_case_probability: Probability of generating an edge case (default 0.1).

    Yields:
        Tuples of (iteration_seed, shape) where iteration_seed can be used
        to reproduce this specific shape.

    Example:
        >>> for iter_seed, shape in fuzz_shapes_seeded(12345):
        ...     x = torch.randn(*shape)
        ...     # If this fails, record iter_seed for reproduction
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32]
    if seq_lengths is None:
        seq_lengths = [64, 128, 256, 512, 1024]
    if hidden_dims is None:
        hidden_dims = [256, 512, 768, 1024]
    if edge_cases is None:
        edge_cases = [[1], [1, 1], [1, 1, 1]]

    rng = SeededRng(seed)
    iteration = 0

    while True:
        iter_rng = rng.derive(f"iter_{iteration}")
        iter_seed = iter_rng.seed
        iteration += 1

        shape_rng = iter_rng.derive("shape")

        # Maybe use an edge case
        if edge_cases and shape_rng.gen_bool(edge_case_probability):
            shape = tuple(shape_rng.choice(edge_cases))
        else:
            batch = shape_rng.choice(batch_sizes)
            seq = shape_rng.choice(seq_lengths)
            hidden = shape_rng.choice(hidden_dims)
            shape = (batch, seq, hidden)

        yield iter_seed, shape


def fuzz_dtypes_seeded(
    seed: int,
    dtypes: Optional[List[str]] = None,
) -> Iterator[Tuple[int, str]]:
    """Generate random dtypes deterministically from a seed.

    Args:
        seed: Master seed for reproducibility.
        dtypes: List of dtype strings. Defaults to ["float32", "float16"].

    Yields:
        Tuples of (iteration_seed, dtype).

    Example:
        >>> for iter_seed, dtype in fuzz_dtypes_seeded(12345):
        ...     x = torch.randn(2, 3, dtype=getattr(torch, dtype))
    """
    if dtypes is None:
        dtypes = ["float32", "float16"]

    rng = SeededRng(seed)
    iteration = 0

    while True:
        iter_rng = rng.derive(f"iter_{iteration}")
        iter_seed = iter_rng.seed
        iteration += 1

        dtype = iter_rng.derive("dtype").choice(dtypes)
        yield iter_seed, dtype


def fuzz_layouts_seeded(
    seed: int,
    shape: Tuple[int, ...],
    layouts: Optional[List[str]] = None,
) -> Iterator[Tuple[int, str, Tuple[int, ...], Tuple[int, ...]]]:
    """Generate random memory layouts deterministically from a seed.

    Args:
        seed: Master seed for reproducibility.
        shape: Base shape of the tensor.
        layouts: List of layout types. Defaults to ["contiguous", "strided", "transposed"].

    Yields:
        Tuples of (iteration_seed, layout_type, shape, strides).

    Example:
        >>> for iter_seed, layout, shape, strides in fuzz_layouts_seeded(12345, (2, 3, 4)):
        ...     print(f"Layout {layout}: shape={shape}, strides={strides}")
    """
    if layouts is None:
        layouts = ["contiguous", "strided", "transposed"]

    rng = SeededRng(seed)
    iteration = 0

    while True:
        iter_rng = rng.derive(f"iter_{iteration}")
        iter_seed = iter_rng.seed
        iteration += 1

        layout = iter_rng.derive("layout").choice(layouts)

        if layout == "contiguous":
            strides = _compute_contiguous_strides(shape)
            yield iter_seed, layout, shape, strides
        elif layout == "transposed":
            if len(shape) >= 2:
                # Transpose last two dimensions
                new_shape = list(shape)
                new_shape[-1], new_shape[-2] = new_shape[-2], new_shape[-1]
                strides = _compute_contiguous_strides(tuple(new_shape))
                yield iter_seed, layout, shape, strides
            else:
                strides = _compute_contiguous_strides(shape)
                yield iter_seed, layout, shape, strides
        elif layout == "strided":
            # Add gaps by multiplying strides by 2
            contiguous_strides = _compute_contiguous_strides(shape)
            strides = tuple(s * 2 for s in contiguous_strides)
            yield iter_seed, layout, shape, strides
        else:
            strides = _compute_contiguous_strides(shape)
            yield iter_seed, layout, shape, strides


def _compute_contiguous_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute contiguous (row-major) strides for a shape."""
    if not shape:
        return ()
    strides = []
    stride = 1
    for dim in reversed(shape):
        strides.insert(0, stride)
        stride *= dim
    return tuple(strides)


def generate_random_tensor(
    seed: int,
    shape: Tuple[int, ...],
    dtype: str = "float32",
    domain: str = "data",
) -> np.ndarray:
    """Generate a random tensor deterministically from a seed.

    Args:
        seed: Seed for reproducibility.
        shape: Shape of the tensor.
        dtype: Data type string ("float32", "float16", "float64", "int32", "int64").
        domain: Domain string for seed derivation. Allows generating
                multiple different tensors from the same seed.

    Returns:
        Random numpy array with the specified shape and dtype.

    Example:
        >>> t1 = generate_random_tensor(12345, (2, 3), "float32", "input")
        >>> t2 = generate_random_tensor(12345, (2, 3), "float32", "input")
        >>> assert np.array_equal(t1, t2)  # Same seed = same tensor
    """
    rng = SeededRng(seed).derive(domain)

    np_dtype = np.dtype(dtype)

    if np_dtype in (np.float32, np.float64):
        # Generate random values in [-10, 10]
        data = rng.randn(*shape).astype(np_dtype) * 10.0
    elif np_dtype == np.float16:
        data = rng.randn(*shape).astype(np.float16) * 10.0
    elif np_dtype in (np.int32, np.int64):
        data = rng._rng.integers(-100, 100, size=shape, dtype=np_dtype)
    else:
        # For other types, generate zeros
        data = np.zeros(shape, dtype=np_dtype)

    return data


class SeededFuzzer:
    """Stateful fuzzer for generating reproducible test cases.

    Maintains iteration state for generating sequences of test cases.
    Each test case has a unique seed that can be used for reproduction.

    Example:
        >>> config = FuzzConfig(seed=12345)
        >>> fuzzer = SeededFuzzer(config)
        >>> for i in range(10):
        ...     test_case = fuzzer.next()
        ...     # Test with test_case.shape, test_case.dtype, etc.
        ...     # On failure, record test_case.seed
    """

    @dataclass
    class TestCase:
        """A single test case from the fuzzer."""

        seed: int
        shape: Tuple[int, ...]
        dtype: str
        layout: str
        strides: Tuple[int, ...]

        def generate_input(self, name: str = "input") -> np.ndarray:
            """Generate a deterministic input tensor for this test case."""
            return generate_random_tensor(
                self.seed, self.shape, self.dtype, domain=name
            )

    def __init__(self, config: FuzzConfig):
        """Create a new fuzzer from configuration.

        Args:
            config: FuzzConfig with seed and options.
        """
        self.config = config
        self._rng = SeededRng(config.seed)
        self._iteration = 0

    def next(self) -> "SeededFuzzer.TestCase":
        """Generate the next test case.

        Returns:
            A TestCase with shape, dtype, layout, and a reproducible seed.
        """
        iter_rng = self._rng.derive(f"iter_{self._iteration}")
        iter_seed = iter_rng.seed
        self._iteration += 1

        shape_rng = iter_rng.derive("shape")
        dtype_rng = iter_rng.derive("dtype")
        layout_rng = iter_rng.derive("layout")

        if self.config.edge_cases and shape_rng.gen_bool(0.1):
            shape = tuple(shape_rng.choice(self.config.edge_cases))
        else:
            batch = shape_rng.choice(self.config.batch_sizes)
            seq = shape_rng.choice(self.config.seq_lengths)
            hidden = shape_rng.choice(self.config.hidden_dims)
            shape = (batch, seq, hidden)

        dtype = dtype_rng.choice(self.config.dtypes)

        layout = layout_rng.choice(self.config.layouts)

        if layout == "contiguous":
            strides = _compute_contiguous_strides(shape)
        elif layout == "transposed" and len(shape) >= 2:
            new_shape = list(shape)
            new_shape[-1], new_shape[-2] = new_shape[-2], new_shape[-1]
            strides = _compute_contiguous_strides(tuple(new_shape))
        elif layout == "strided":
            contiguous = _compute_contiguous_strides(shape)
            strides = tuple(s * 2 for s in contiguous)
        else:
            strides = _compute_contiguous_strides(shape)

        return SeededFuzzer.TestCase(
            seed=iter_seed,
            shape=shape,
            dtype=dtype,
            layout=layout,
            strides=strides,
        )

    def iterator(self, max_iterations: int = 1000) -> Iterator["SeededFuzzer.TestCase"]:
        """Generate test cases up to max_iterations.

        This is a safer alternative to calling next() in an infinite loop.

        Args:
            max_iterations: Maximum number of test cases to generate.

        Yields:
            TestCase objects with shape, dtype, layout, and a reproducible seed.
        """
        for _ in range(max_iterations):
            yield self.next()

    def reset(self):
        """Reset the fuzzer to iteration 0."""
        self._rng = SeededRng(self.config.seed)
        self._iteration = 0
