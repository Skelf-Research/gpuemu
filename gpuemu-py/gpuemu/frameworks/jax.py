"""JAX-specific tensor handling and transformation support."""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from gpuemu.frameworks.base import FrameworkAdapter

if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp

    from gpuemu.client import Client


class ValidationError(Exception):
    """Error raised when validation fails."""

    pass


class JAXAdapter(FrameworkAdapter):
    """JAX-specific tensor handling and transformation support.

    JAX uses a functional programming paradigm, so gradients are computed
    differently than in PyTorch or TensorFlow. This adapter handles:

    - Converting between jax.Array and numpy
    - Functional gradient computation with jax.grad
    - Compatibility with JAX transformations (vmap, jit, pmap)

    Example:
        >>> adapter = JAXAdapter()
        >>> np_arr = adapter.to_numpy(jax_array)
        >>> jax_arr = adapter.from_numpy(np_arr)
    """

    def __init__(self):
        """Initialize the adapter by importing JAX."""
        try:
            import jax
            import jax.numpy as jnp

            self.jax = jax
            self.jnp = jnp
        except ImportError as e:
            raise ImportError(
                "JAX is required for JAXAdapter. Install with: pip install jax jaxlib"
            ) from e

    def to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert jax.Array to numpy.

        Args:
            tensor: jax.Array or numpy array.

        Returns:
            Numpy array.
        """
        if isinstance(tensor, np.ndarray):
            return tensor
        return np.asarray(tensor)

    def from_numpy(self, arr: np.ndarray, like: Optional[Any] = None) -> Any:
        """Convert numpy to jax.Array.

        Args:
            arr: Numpy array to convert.
            like: Optional template (unused for JAX as device placement is automatic).

        Returns:
            jax.Array with data from arr.
        """
        return self.jnp.array(arr)

    def get_dtype_name(self, tensor: Any) -> str:
        """Get dtype name for tolerance lookup.

        Args:
            tensor: jax.Array.

        Returns:
            String like "float32", "float16".
        """
        return str(tensor.dtype)

    def requires_grad(self, tensor: Any) -> bool:
        """Check if tensor can be differentiated.

        JAX uses functional gradients, so any floating-point tensor
        can be differentiated.

        Args:
            tensor: jax.Array.

        Returns:
            True if tensor has a floating-point dtype.
        """
        return self.jnp.issubdtype(tensor.dtype, self.jnp.floating)

    def compute_gradient(
        self,
        output_fn: Callable[..., Any],
        inputs: Dict[str, Any],
        argnums: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, Any]:
        """Compute gradients using jax.grad.

        Unlike PyTorch/TensorFlow, JAX requires a function rather than
        a tensor to compute gradients.

        Args:
            output_fn: Function that takes inputs and returns output.
            inputs: Dictionary of input tensors.
            argnums: Which arguments to differentiate. If None, all floating inputs.

        Returns:
            Dictionary mapping input names to their gradients.
        """
        # Identify floating-point inputs
        float_inputs = [k for k, v in inputs.items() if self.requires_grad(v)]

        if argnums is None:
            argnums = tuple(i for i, k in enumerate(inputs.keys()) if k in float_inputs)

        if not argnums:
            return {}

        def fn(*args):
            kwargs = dict(zip(inputs.keys(), args))
            return output_fn(**kwargs).sum()

        grads = self.jax.grad(fn, argnums=argnums)(*inputs.values())

        if not isinstance(grads, tuple):
            grads = (grads,)

        # Map back to input names
        input_names = list(inputs.keys())
        result = {}
        for i, argnum in enumerate(argnums):
            result[input_names[argnum]] = grads[i]

        return result

    def is_available(self) -> bool:
        """Check if JAX is installed."""
        try:
            import jax

            return True
        except ImportError:
            return False

    def get_framework_name(self) -> str:
        """Get framework name for tolerance lookup."""
        return "jax"


@contextmanager
def validate_jax(
    client: "Client",
    op_name: str,
    inputs: Dict[str, Any],
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    **kwargs,
):
    """Validate a JAX operation against a reference implementation.

    This context manager captures the output of your operation and validates
    it against the reference registered with gpuemu.

    Args:
        client: gpuemu Client instance.
        op_name: Name of the op (must be registered in gpuemu.toml).
        inputs: Dictionary of input arrays.
        atol: Absolute tolerance for comparison.
        rtol: Relative tolerance for comparison.
        **kwargs: Additional kwargs passed to the reference.

    Yields:
        Context dict. Set ctx["output"] to your operation's output.

    Raises:
        ValidationError: If validation fails.

    Example:
        >>> with validate_jax(client, "my_op", {"x": x}) as ctx:
        ...     ctx["output"] = my_custom_op(x)
    """
    adapter = JAXAdapter()
    ctx: Dict[str, Any] = {"output": None, "grads": None}

    # Convert inputs to numpy for validation
    np_inputs = {k: adapter.to_numpy(v) for k, v in inputs.items()}

    yield ctx

    if ctx["output"] is None:
        raise ValueError("Output not set in context. Set ctx['output'] = your_result")

    # Validate forward pass
    np_output = adapter.to_numpy(ctx["output"])

    validation_kwargs = {**kwargs}
    if atol is not None:
        validation_kwargs["atol"] = atol
    if rtol is not None:
        validation_kwargs["rtol"] = rtol

    result = client.validate_op(op_name, np_inputs, np_output, **validation_kwargs)

    if not result.passed:
        failure_msgs = [f.get("message", str(f)) for f in result.failures[:3]]
        raise ValidationError(
            f"Forward validation failed for {op_name}: {'; '.join(failure_msgs)}"
        )


def check_vmap_compatible(
    op: Callable[..., Any],
    inputs: Dict[str, Any],
    batch_axis: int = 0,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> bool:
    """Check if operation is vmap-compatible.

    Tests that applying vmap to the operation produces the same result
    as executing the operation individually on each batch element.

    Args:
        op: Function that takes **inputs and returns an array.
        inputs: Dictionary of input arrays (must have a batch dimension).
        batch_axis: Axis to batch over.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Returns:
        True if vmapped output matches individual execution.

    Example:
        >>> def my_op(x):
        ...     return jnp.sin(x)
        >>> x = jnp.ones((4, 10))  # batch of 4
        >>> assert check_vmap_compatible(my_op, {"x": x})
    """
    import jax
    import jax.numpy as jnp

    # Get batch size
    first_input = list(inputs.values())[0]
    batch_size = first_input.shape[batch_axis]

    if batch_size < 2:
        # Need at least 2 elements to test batching
        return True

    # Build in_axes spec for vmap
    in_axes = {k: batch_axis for k in inputs.keys()}

    # Execute with vmap
    def fn(**kw):
        return op(**kw)

    try:
        vmapped = jax.vmap(
            lambda *args: fn(**dict(zip(inputs.keys(), args))), in_axes=batch_axis
        )
        vmap_output = vmapped(*inputs.values())
    except Exception:
        # vmap failed - operation is not compatible
        return False

    # Execute individually and stack
    individual_outputs = []
    for i in range(batch_size):
        sliced = {k: jnp.take(v, i, axis=batch_axis) for k, v in inputs.items()}
        individual_outputs.append(op(**sliced))

    stacked = jnp.stack(individual_outputs, axis=batch_axis)

    return jnp.allclose(vmap_output, stacked, rtol=rtol, atol=atol)


def check_jit_safe(
    op: Callable[..., Any],
    inputs: Dict[str, Any],
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> bool:
    """Check if operation produces same results with and without JIT.

    Tests that JIT compilation doesn't change the numerical results,
    which can happen if the operation uses Python control flow that
    isn't JIT-compatible.

    Args:
        op: Function that takes **inputs and returns an array.
        inputs: Dictionary of input arrays.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Returns:
        True if JIT and eager results match.

    Example:
        >>> def my_op(x):
        ...     return jnp.sin(x)
        >>> assert check_jit_safe(my_op, {"x": jnp.ones(10)})
    """
    import jax
    import jax.numpy as jnp

    # Eager execution
    eager_output = op(**inputs)

    # JIT execution
    try:
        jitted_op = jax.jit(op)
        jitted_output = jitted_op(**inputs)
    except Exception:
        # JIT failed - operation is not safe to JIT
        return False

    return jnp.allclose(eager_output, jitted_output, rtol=rtol, atol=atol)


def check_pmap_compatible(
    op: Callable[..., Any],
    inputs: Dict[str, Any],
    axis_name: str = "batch",
) -> bool:
    """Check if operation is pmap-compatible for multi-device execution.

    Tests that the operation can be parallelized across devices using pmap.
    This is useful for validating operations that should work in distributed
    training scenarios.

    Note: This test requires multiple devices (CPUs or GPUs) to be available.
    On a single-device system, it will return True if the operation can at
    least be traced.

    Args:
        op: Function that takes **inputs and returns an array.
        inputs: Dictionary of input arrays.
        axis_name: Axis name for collective operations.

    Returns:
        True if operation can be pmapped.
    """
    import jax

    n_devices = jax.local_device_count()

    if n_devices < 2:
        # Can't properly test pmap with single device, just check if it traces
        try:
            pmapped = jax.pmap(
                lambda *args: op(**dict(zip(inputs.keys(), args))), axis_name=axis_name
            )
            # Reshape inputs to have device dimension
            reshaped = {k: v.reshape((1,) + v.shape) for k, v in inputs.items()}
            _ = pmapped(*reshaped.values())
            return True
        except Exception:
            return False

    # With multiple devices, do a full test
    try:
        pmapped = jax.pmap(
            lambda *args: op(**dict(zip(inputs.keys(), args))), axis_name=axis_name
        )

        # Replicate inputs across devices
        replicated = {k: jax.numpy.stack([v] * n_devices) for k, v in inputs.items()}

        output = pmapped(*replicated.values())

        # Check all device outputs are the same
        return jax.numpy.allclose(output[0], output[1])
    except Exception:
        return False


def check_grad_safe(
    op: Callable[..., Any],
    inputs: Dict[str, Any],
    argnums: Union[int, Tuple[int, ...]] = 0,
) -> bool:
    """Check if operation can be differentiated with jax.grad.

    Tests that the operation is differentiable and doesn't produce
    NaN or Inf gradients.

    Args:
        op: Function that takes **inputs and returns a scalar.
        inputs: Dictionary of input arrays.
        argnums: Which positional arguments to differentiate.

    Returns:
        True if gradients can be computed without errors or NaN/Inf.

    Example:
        >>> def my_op(x):
        ...     return jnp.sum(x ** 2)
        >>> assert check_grad_safe(my_op, {"x": jnp.ones(10)}, argnums=0)
    """
    import jax
    import jax.numpy as jnp

    def fn(*args):
        kwargs = dict(zip(inputs.keys(), args))
        result = op(**kwargs)
        # Ensure scalar output for grad
        return result.sum() if result.ndim > 0 else result

    try:
        grad_fn = jax.grad(fn, argnums=argnums)
        grads = grad_fn(*inputs.values())

        # Check for NaN/Inf
        if isinstance(grads, tuple):
            return all(jnp.isfinite(g).all() for g in grads)
        return jnp.isfinite(grads).all()
    except Exception:
        return False


def validate_jax_primitive(
    primitive_name: str,
    impl: Callable,
    inputs: Dict[str, Any],
    expected_output: Any,
    check_jvp: bool = True,
    check_vmap: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> Dict[str, Any]:
    """Validate a JAX primitive implementation.

    Comprehensive validation of a custom JAX primitive, including:
    - Forward pass correctness
    - JVP (forward-mode autodiff) rules
    - Batching rules (vmap compatibility)

    Args:
        primitive_name: Name of the primitive for reporting.
        impl: Implementation function to test.
        inputs: Dictionary of input arrays.
        expected_output: Expected output for comparison.
        check_jvp: Whether to test JVP rules.
        check_vmap: Whether to test batching rules.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        Dictionary with 'forward_ok', 'jvp_ok', 'vmap_ok', and 'details'.
    """
    import jax
    import jax.numpy as jnp

    result = {
        "forward_ok": True,
        "jvp_ok": True,
        "vmap_ok": True,
        "details": [],
    }

    # Forward pass
    try:
        output = impl(**inputs)
        if not jnp.allclose(output, expected_output, rtol=rtol, atol=atol):
            result["forward_ok"] = False
            result["details"].append(
                f"Forward mismatch: max diff = {jnp.abs(output - expected_output).max()}"
            )
    except Exception as e:
        result["forward_ok"] = False
        result["details"].append(f"Forward failed: {e}")

    # JVP check
    if check_jvp:
        try:
            # Create tangent vectors
            primals = tuple(inputs.values())
            tangents = tuple(jnp.ones_like(v) for v in inputs.values())

            def fn(*args):
                return impl(**dict(zip(inputs.keys(), args)))

            _, jvp_result = jax.jvp(fn, primals, tangents)

            if not jnp.isfinite(jvp_result).all():
                result["jvp_ok"] = False
                result["details"].append("JVP produced non-finite values")
        except Exception as e:
            result["jvp_ok"] = False
            result["details"].append(f"JVP failed: {e}")

    # vmap check
    if check_vmap:
        result["vmap_ok"] = check_vmap_compatible(impl, inputs, rtol=rtol, atol=atol)
        if not result["vmap_ok"]:
            result["details"].append("vmap compatibility check failed")

    return result


def fuzz_jax_op(
    client: "Client",
    op_name: str,
    run_op: Callable[..., Any],
    iterations: int = 100,
    seed: Optional[int] = None,
    fail_fast: bool = False,
    check_vmap: bool = False,
    check_jit: bool = False,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
) -> Dict[str, Any]:
    """Fuzz a JAX op with client-side execution.

    The daemon generates random inputs; the client runs the JAX op
    and submits the output for validation. Optionally checks vmap
    and jit compatibility.

    Args:
        client: gpuemu Client instance.
        op_name: Name of the op (must be registered in gpuemu.toml).
        run_op: Callable that takes **inputs (as jax.Arrays) and returns output.
        iterations: Number of fuzz iterations.
        seed: Master seed. Auto-generated if None.
        fail_fast: Stop on first failure.
        check_vmap: Also test vmap compatibility for each case.
        check_jit: Also test jit safety for each case.
        atol: Absolute tolerance override.
        rtol: Relative tolerance override.

    Returns:
        Dict with 'total', 'passed', 'failed', 'forward_failures',
        'vmap_failures', 'jit_failures'.

    Example:
        >>> client = Client()
        >>> result = fuzz_jax_op(
        ...     client,
        ...     "custom_attention",
        ...     run_op=lambda q, k, v: jnp.dot(q, k.T),
        ...     iterations=50,
        ...     check_jit=True,
        ... )
        >>> print(f"Passed: {result['passed']}/{result['total']}")
    """
    adapter = JAXAdapter()

    cases = client.get_test_batch(op_name, count=iterations, seed=seed)

    total = 0
    passed = 0
    failed = 0
    forward_failures = []
    vmap_failures = []
    jit_failures = []

    for case in cases:
        total += 1

        # Convert numpy inputs to JAX arrays
        jax_inputs = {k: adapter.from_numpy(v) for k, v in case["inputs"].items()}

        try:
            output = run_op(**jax_inputs)
            np_output = adapter.to_numpy(output)

            kwargs = {}
            if atol is not None:
                kwargs["atol"] = atol
            if rtol is not None:
                kwargs["rtol"] = rtol

            result = client.submit_output(
                op_name,
                case["inputs"],
                np_output,
                case["seed"],
                **kwargs,
            )

            if result.passed:
                passed += 1
            else:
                failed += 1
                forward_failures.append(result)
                if fail_fast:
                    break

            if check_vmap:
                try:
                    vmapped = __import__("jax").vmap(
                        lambda *args: run_op(**dict(zip(jax_inputs.keys(), args)))
                    )
                    _ = vmapped(*jax_inputs.values())
                except Exception as e:
                    vmap_failures.append({"seed": case["seed"], "message": str(e)})

            if check_jit:
                try:
                    jitted = __import__("jax").jit(run_op)
                    jitted_output = jitted(**jax_inputs)
                    jnp = __import__("jax.numpy")
                    if not jnp.allclose(output, jitted_output, rtol=1e-6, atol=1e-6):
                        jit_failures.append(
                            {
                                "seed": case["seed"],
                                "message": "JIT output differs from eager output",
                            }
                        )
                except Exception as e:
                    jit_failures.append({"seed": case["seed"], "message": str(e)})

        except Exception as e:
            failed += 1
            forward_failures.append(
                {
                    "seed": case["seed"],
                    "message": f"Op execution failed: {e}",
                }
            )
            if fail_fast:
                break

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "forward_failures": forward_failures,
        "vmap_failures": vmap_failures,
        "jit_failures": jit_failures,
    }
