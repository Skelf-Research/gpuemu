"""TensorFlow-specific tensor handling and GradientTape support."""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from gpuemu_py.frameworks.base import FrameworkAdapter

if TYPE_CHECKING:
    import tensorflow as tf

    from gpuemu_py.client import Client


class ValidationError(Exception):
    """Error raised when validation fails."""

    pass


class TensorFlowAdapter(FrameworkAdapter):
    """TensorFlow-specific tensor handling and GradientTape support.

    Handles conversion between tf.Tensor and numpy arrays, including:
    - Eager vs graph mode handling
    - tf.Variable tracking for gradients
    - GradientTape integration

    Example:
        >>> adapter = TensorFlowAdapter()
        >>> np_arr = adapter.to_numpy(tf_tensor)
        >>> tf_tensor = adapter.from_numpy(np_arr)
    """

    def __init__(self):
        """Initialize the adapter by importing TensorFlow."""
        try:
            import tensorflow as tf

            self.tf = tf
        except ImportError as e:
            raise ImportError(
                "TensorFlow is required for TensorFlowAdapter. "
                "Install with: pip install tensorflow"
            ) from e

    def to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert tf.Tensor to numpy.

        Args:
            tensor: tf.Tensor, tf.Variable, or numpy array.

        Returns:
            Numpy array.
        """
        if isinstance(tensor, np.ndarray):
            return tensor
        return tensor.numpy()

    def from_numpy(self, arr: np.ndarray, like: Optional[Any] = None) -> "tf.Tensor":
        """Convert numpy to tf.Tensor.

        Args:
            arr: Numpy array to convert.
            like: Optional template tensor to match dtype.

        Returns:
            tf.Tensor with data from arr.
        """
        tensor = self.tf.constant(arr)
        if like is not None:
            tensor = self.tf.cast(tensor, like.dtype)
        return tensor

    def get_dtype_name(self, tensor: Any) -> str:
        """Get dtype name for tolerance lookup.

        Args:
            tensor: tf.Tensor.

        Returns:
            String like "float32", "float16".
        """
        return tensor.dtype.name

    def requires_grad(self, tensor: Any) -> bool:
        """Check if tensor is a Variable (trainable).

        Only tf.Variable instances are tracked by GradientTape by default.
        Regular tensors can be watched explicitly but aren't by default.

        Args:
            tensor: tf.Tensor or tf.Variable.

        Returns:
            True if tensor is a tf.Variable.
        """
        return isinstance(tensor, self.tf.Variable)

    def compute_gradient(
        self,
        tape: "tf.GradientTape",
        output: "tf.Tensor",
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute gradients using GradientTape.

        Args:
            tape: Active GradientTape that recorded the forward pass.
            output: The output tensor to differentiate.
            inputs: Dictionary of input tensors.

        Returns:
            Dictionary mapping input names to their gradients.
        """
        variables = [v for v in inputs.values() if self.requires_grad(v)]
        if not variables:
            return {}

        grads = tape.gradient(output, variables)

        var_names = [k for k, v in inputs.items() if self.requires_grad(v)]
        return {
            name: grad for name, grad in zip(var_names, grads) if grad is not None
        }

    def is_available(self) -> bool:
        """Check if TensorFlow is installed."""
        try:
            import tensorflow

            return True
        except ImportError:
            return False

    def get_framework_name(self) -> str:
        """Get framework name for tolerance lookup."""
        return "tensorflow"


@contextmanager
def validate_tensorflow(
    client: "Client",
    op_name: str,
    inputs: Dict[str, Any],
    check_gradient: bool = False,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    **kwargs,
):
    """Validate a TensorFlow operation against a reference implementation.

    This context manager captures the output of your operation and validates
    it against the reference registered with gpuemu. Optionally validates
    gradients using GradientTape.

    Args:
        client: gpuemu Client instance.
        op_name: Name of the op (must be registered in gpuemu.toml).
        inputs: Dictionary of input tensors.
        check_gradient: If True, creates a GradientTape and validates gradients.
        atol: Absolute tolerance for comparison.
        rtol: Relative tolerance for comparison.
        **kwargs: Additional kwargs passed to the reference.

    Yields:
        Context dict with:
        - Set ctx["output"] to your operation's output.
        - ctx["tape"] is the GradientTape if check_gradient=True.

    Raises:
        ValidationError: If validation fails.

    Example:
        >>> # Simple forward validation
        >>> with validate_tensorflow(client, "my_op", {"x": x}) as ctx:
        ...     ctx["output"] = my_custom_op(x)

        >>> # With gradient check
        >>> x = tf.Variable(tf.random.normal((32, 128)))
        >>> with validate_tensorflow(client, "my_op", {"x": x}, check_gradient=True) as ctx:
        ...     with ctx["tape"]:
        ...         ctx["output"] = my_custom_op(x)
    """
    import tensorflow as tf

    adapter = TensorFlowAdapter()
    tape = tf.GradientTape(persistent=True) if check_gradient else None
    ctx: Dict[str, Any] = {"output": None, "tape": tape, "grads": None}

    # Convert inputs to numpy for validation
    np_inputs = {k: adapter.to_numpy(v) for k, v in inputs.items()}

    if tape:
        tape.__enter__()

    try:
        yield ctx
    finally:
        if tape:
            tape.__exit__(None, None, None)

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

    # Validate gradients if requested
    if check_gradient and tape:
        grads = adapter.compute_gradient(tape, ctx["output"], inputs)
        ctx["grads"] = grads

        for name, grad in grads.items():
            if grad is not None:
                grad_result = client.validate_op(
                    f"{op_name}_grad_{name}",
                    np_inputs,
                    adapter.to_numpy(grad),
                    **validation_kwargs,
                )
                if not grad_result.passed:
                    failure_msgs = [
                        f.get("message", str(f)) for f in grad_result.failures[:3]
                    ]
                    raise ValidationError(
                        f"Gradient validation for {name} failed: {'; '.join(failure_msgs)}"
                    )


def check_keras_layer(
    layer: "tf.keras.layers.Layer",
    input_shape: Tuple[int, ...],
    client: "Client",
    op_name: str,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    **kwargs,
) -> bool:
    """Validate a Keras layer's forward and backward pass.

    Tests that a Keras layer produces correct outputs and gradients
    when compared to the gpuemu reference.

    Args:
        layer: Keras layer to test.
        input_shape: Shape of input tensor (including batch dimension).
        client: gpuemu Client instance.
        op_name: Name of the op (must be registered in gpuemu.toml).
        atol: Absolute tolerance for comparison.
        rtol: Relative tolerance for comparison.
        **kwargs: Additional kwargs passed to the reference.

    Returns:
        True if both forward and backward pass validation succeeds.

    Example:
        >>> layer = MyCustomLayer(units=64)
        >>> assert check_keras_layer(layer, (32, 128), client, "my_custom_layer")
    """
    import tensorflow as tf

    adapter = TensorFlowAdapter()

    # Create test input
    x = tf.Variable(tf.random.normal(input_shape))

    # Forward pass with gradient recording
    with tf.GradientTape() as tape:
        output = layer(x)
        loss = tf.reduce_sum(output)

    grad = tape.gradient(loss, x)

    # Validate forward
    np_input = adapter.to_numpy(x)
    np_output = adapter.to_numpy(output)

    validation_kwargs = {**kwargs}
    if atol is not None:
        validation_kwargs["atol"] = atol
    if rtol is not None:
        validation_kwargs["rtol"] = rtol

    result = client.validate_op(op_name, {"x": np_input}, np_output, **validation_kwargs)
    if not result.passed:
        return False

    # Validate gradient
    if grad is not None:
        grad_result = client.validate_op(
            f"{op_name}_grad",
            {"x": np_input},
            adapter.to_numpy(grad),
            **validation_kwargs,
        )
        if not grad_result.passed:
            return False

    return True


def check_tf_function_safe(
    op: Callable[..., Any],
    inputs: Dict[str, Any],
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> bool:
    """Check if operation produces same results with and without @tf.function.

    Tests that tf.function tracing doesn't change the numerical results,
    which can happen if the operation uses Python-only features.

    Args:
        op: Function that takes **inputs and returns a tensor.
        inputs: Dictionary of input tensors.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Returns:
        True if tf.function and eager results match.

    Example:
        >>> @tf.function
        ... def my_op(x):
        ...     return tf.sin(x)
        >>> assert check_tf_function_safe(my_op, {"x": tf.ones(10)})
    """
    import tensorflow as tf

    # Eager execution
    eager_output = op(**inputs)

    # tf.function execution
    try:
        tf_func = tf.function(op)
        traced_output = tf_func(**inputs)
    except Exception:
        # Tracing failed
        return False

    return tf.reduce_all(
        tf.abs(eager_output - traced_output) <= atol + rtol * tf.abs(traced_output)
    ).numpy()


def check_xla_compatible(
    op: Callable[..., Any],
    inputs: Dict[str, Any],
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> bool:
    """Check if operation is compatible with XLA compilation.

    Tests that the operation can be compiled with XLA and produces
    correct numerical results. XLA compilation can improve performance
    but requires operations to be XLA-compatible.

    Args:
        op: Function that takes **inputs and returns a tensor.
        inputs: Dictionary of input tensors.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Returns:
        True if XLA compilation succeeds and results match.

    Example:
        >>> def my_op(x):
        ...     return tf.sin(x)
        >>> assert check_xla_compatible(my_op, {"x": tf.ones(10)})
    """
    import tensorflow as tf

    # Eager execution
    eager_output = op(**inputs)

    # XLA execution
    try:
        xla_func = tf.function(op, jit_compile=True)
        xla_output = xla_func(**inputs)
    except Exception:
        # XLA compilation failed
        return False

    return tf.reduce_all(
        tf.abs(eager_output - xla_output) <= atol + rtol * tf.abs(xla_output)
    ).numpy()


def validate_custom_gradient(
    func: Callable,
    gradient_func: Callable,
    inputs: Dict[str, Any],
    eps: float = 1e-4,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> Dict[str, Any]:
    """Validate a custom gradient implementation.

    Tests a custom gradient function against numerical finite differences.
    Useful for validating @tf.custom_gradient implementations.

    Args:
        func: Forward function.
        gradient_func: Custom gradient function.
        inputs: Dictionary of input tensors.
        eps: Epsilon for finite differences.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Returns:
        Dictionary with 'gradient_ok' and 'details'.

    Example:
        >>> @tf.custom_gradient
        ... def my_func(x):
        ...     def grad(dy):
        ...         return dy * 2
        ...     return x * 2, grad
        >>> result = validate_custom_gradient(
        ...     lambda x: my_func(x)[0],
        ...     lambda x, dy: my_func(x)[1](dy),
        ...     {"x": tf.ones(10)}
        ... )
        >>> assert result["gradient_ok"]
    """
    import tensorflow as tf

    result = {"gradient_ok": True, "details": [], "max_diff": 0.0}

    # Compute custom gradients
    with tf.GradientTape() as tape:
        for v in inputs.values():
            if isinstance(v, tf.Variable):
                tape.watch(v)
            else:
                tape.watch(v)
        output = func(**inputs)
        loss = tf.reduce_sum(output)

    custom_grads = tape.gradient(loss, list(inputs.values()))

    # Compute numerical gradients
    numerical_grads = []
    for i, (name, inp) in enumerate(inputs.items()):
        inp_np = inp.numpy().flatten()
        grad_np = np.zeros_like(inp_np)

        for j in range(min(len(inp_np), 100)):  # Limit for large tensors
            orig = inp_np[j]

            # f(x + eps)
            inp_np[j] = orig + eps
            inputs_copy = {k: tf.constant(v.numpy()) for k, v in inputs.items()}
            inputs_copy[name] = tf.constant(inp_np.reshape(inp.shape))
            out_plus = tf.reduce_sum(func(**inputs_copy)).numpy()

            # f(x - eps)
            inp_np[j] = orig - eps
            inputs_copy[name] = tf.constant(inp_np.reshape(inp.shape))
            out_minus = tf.reduce_sum(func(**inputs_copy)).numpy()

            grad_np[j] = (out_plus - out_minus) / (2 * eps)
            inp_np[j] = orig

        numerical_grads.append(grad_np.reshape(inp.shape))

    # Compare gradients
    max_diff = 0.0
    for custom, numerical in zip(custom_grads, numerical_grads):
        if custom is None:
            continue
        custom_np = custom.numpy().flatten()[:100]
        numerical_np = numerical.flatten()[:100]

        diff = np.abs(custom_np - numerical_np)
        threshold = atol + rtol * np.abs(numerical_np)

        if not np.all(diff <= threshold):
            result["gradient_ok"] = False
            max_diff = max(max_diff, np.max(diff))

    result["max_diff"] = float(max_diff)
    if not result["gradient_ok"]:
        result["details"].append(
            f"Gradient mismatch: max diff = {max_diff:.6e}"
        )

    return result
