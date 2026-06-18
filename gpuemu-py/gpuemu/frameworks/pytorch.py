"""PyTorch-specific tensor handling and autograd support."""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np

from gpuemu.frameworks.base import FrameworkAdapter

if TYPE_CHECKING:
    import torch

    from gpuemu.client import Client, ValidationResult


class ValidationError(Exception):
    """Error raised when validation fails."""

    pass


class PyTorchAdapter(FrameworkAdapter):
    """PyTorch-specific tensor handling and autograd support.

    Handles conversion between torch.Tensor and numpy arrays, including:
    - GPU to CPU transfer
    - Gradient detachment
    - Device/dtype matching

    Example:
        >>> adapter = PyTorchAdapter()
        >>> np_arr = adapter.to_numpy(torch_tensor)
        >>> torch_tensor = adapter.from_numpy(np_arr, like=original_tensor)
    """

    def __init__(self):
        """Initialize the adapter by importing torch."""
        try:
            import torch

            self.torch = torch
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for PyTorchAdapter. "
                "Install with: pip install torch"
            ) from e

    def to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert torch.Tensor to numpy, handling device and grad.

        Args:
            tensor: torch.Tensor or numpy array.

        Returns:
            Numpy array (on CPU, detached from autograd graph).
        """
        if isinstance(tensor, np.ndarray):
            return tensor
        return tensor.detach().cpu().numpy()

    def from_numpy(
        self, arr: np.ndarray, like: Optional["torch.Tensor"] = None
    ) -> "torch.Tensor":
        """Convert numpy to torch.Tensor with matching device/dtype.

        Args:
            arr: Numpy array to convert.
            like: Optional template tensor to match device/dtype.

        Returns:
            torch.Tensor with data from arr.
        """
        tensor = self.torch.from_numpy(arr.copy())
        if like is not None:
            tensor = tensor.to(device=like.device, dtype=like.dtype)
        return tensor

    def get_dtype_name(self, tensor: "torch.Tensor") -> str:
        """Get dtype name for tolerance lookup.

        Args:
            tensor: torch.Tensor.

        Returns:
            String like "float32", "float16".
        """
        dtype = tensor.dtype
        dtype_map = {
            self.torch.float16: "float16",
            self.torch.float32: "float32",
            self.torch.float64: "float64",
            self.torch.bfloat16: "bfloat16",
        }
        return dtype_map.get(dtype, str(dtype).split(".")[-1])

    def requires_grad(self, tensor: Any) -> bool:
        """Check if tensor requires gradient tracking.

        Args:
            tensor: torch.Tensor.

        Returns:
            True if tensor.requires_grad is True.
        """
        return hasattr(tensor, "requires_grad") and tensor.requires_grad

    def compute_gradient(
        self,
        output: "torch.Tensor",
        inputs: Dict[str, "torch.Tensor"],
        grad_output: Optional["torch.Tensor"] = None,
    ) -> Dict[str, "torch.Tensor"]:
        """Compute gradients using torch.autograd.grad.

        Args:
            output: The output tensor to differentiate.
            inputs: Dictionary of input tensors.
            grad_output: Optional upstream gradient. If None, uses ones.

        Returns:
            Dictionary mapping input names to their gradients.
        """
        if grad_output is None:
            grad_output = self.torch.ones_like(output)

        # Filter to inputs that require grad
        grad_inputs = {k: v for k, v in inputs.items() if self.requires_grad(v)}

        if not grad_inputs:
            return {}

        grads = self.torch.autograd.grad(
            output,
            list(grad_inputs.values()),
            grad_outputs=grad_output,
            retain_graph=True,
            allow_unused=True,
        )

        return {
            name: grad
            for name, grad in zip(grad_inputs.keys(), grads)
            if grad is not None
        }

    def is_available(self) -> bool:
        """Check if PyTorch is installed."""
        try:
            import torch

            return True
        except ImportError:
            return False

    def get_framework_name(self) -> str:
        """Get framework name for tolerance lookup."""
        return "pytorch"


@contextmanager
def validate_pytorch(
    client: "Client",
    op_name: str,
    inputs: Dict[str, "torch.Tensor"],
    check_backward: bool = False,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    **kwargs,
):
    """Validate a PyTorch operation against a reference implementation.

    This context manager captures the output of your operation and validates
    it against the reference registered with gpuemu. Optionally validates
    gradients as well.

    Args:
        client: gpuemu Client instance.
        op_name: Name of the op (must be registered in gpuemu.toml).
        inputs: Dictionary of input tensors.
        check_backward: If True, also validate gradients.
        atol: Absolute tolerance for comparison.
        rtol: Relative tolerance for comparison.
        **kwargs: Additional kwargs passed to the reference.

    Yields:
        Context dict. Set ctx["output"] to your operation's output.

    Raises:
        ValidationError: If validation fails.

    Example:
        >>> with validate_pytorch(client, "my_op", {"x": x}) as ctx:
        ...     ctx["output"] = my_custom_op(x)

        >>> # With gradient check:
        >>> x = torch.randn(32, 128, requires_grad=True)
        >>> with validate_pytorch(client, "my_op", {"x": x}, check_backward=True) as ctx:
        ...     ctx["output"] = my_custom_op(x)
    """
    adapter = PyTorchAdapter()
    ctx: Dict[str, Any] = {"output": None, "grad_inputs": None}

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

    # Validate backward pass if requested
    if check_backward and any(adapter.requires_grad(t) for t in inputs.values()):
        grad_inputs = adapter.compute_gradient(ctx["output"], inputs)
        ctx["grad_inputs"] = grad_inputs

        # Validate gradients against reference
        for name, grad in grad_inputs.items():
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


def check_autograd(
    op: Callable[..., "torch.Tensor"],
    inputs: Dict[str, "torch.Tensor"],
    eps: float = 1e-4,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    check_inputs: Optional[List[str]] = None,
) -> bool:
    """Check autograd correctness using finite differences.

    Compares analytical gradients from torch.autograd with numerical gradients
    computed via finite differences. This is useful for validating custom
    autograd functions.

    Args:
        op: Function that takes **inputs and returns a tensor.
        inputs: Dictionary of input tensors.
        eps: Epsilon for finite differences.
        atol: Absolute tolerance for comparison.
        rtol: Relative tolerance for comparison.
        check_inputs: Optional list of input names to check. If None, checks all.

    Returns:
        True if gradients match numerical approximation.

    Example:
        >>> def my_op(x):
        ...     return x ** 2
        >>> x = torch.randn(10, requires_grad=True)
        >>> assert check_autograd(my_op, {"x": x})
    """
    import torch

    # Determine which inputs to check
    if check_inputs is None:
        check_inputs = [
            k for k, v in inputs.items() if v.is_floating_point() and v.requires_grad
        ]

    # Ensure inputs require grad
    inputs_with_grad = {}
    for k, v in inputs.items():
        if k in check_inputs:
            inputs_with_grad[k] = v.clone().detach().requires_grad_(True)
        else:
            inputs_with_grad[k] = v.clone().detach()

    # Forward pass
    output = op(**inputs_with_grad)
    loss = output.sum()

    # Analytical gradients
    loss.backward()
    analytical_grads = {
        k: inputs_with_grad[k].grad.clone()
        for k in check_inputs
        if inputs_with_grad[k].grad is not None
    }

    # Numerical gradients via finite differences
    numerical_grads = {}
    for name in check_inputs:
        inp = inputs_with_grad[name].detach().clone()
        grad = torch.zeros_like(inp)

        # Flatten for iteration
        inp_flat = inp.flatten()
        grad_flat = grad.flatten()

        for i in range(min(inp_flat.numel(), 1000)):  # Limit for large tensors
            orig = inp_flat[i].item()

            # f(x + eps)
            inp_flat[i] = orig + eps
            inputs_copy = {k: v.clone() for k, v in inputs.items()}
            inputs_copy[name] = inp.view_as(inputs[name])
            out_plus = op(**inputs_copy).sum().item()

            # f(x - eps)
            inp_flat[i] = orig - eps
            inputs_copy[name] = inp.view_as(inputs[name])
            out_minus = op(**inputs_copy).sum().item()

            # Numerical gradient
            grad_flat[i] = (out_plus - out_minus) / (2 * eps)

            # Reset
            inp_flat[i] = orig

        numerical_grads[name] = grad

    # Compare
    for name in check_inputs:
        if name not in analytical_grads:
            continue

        analytical = analytical_grads[name]
        numerical = numerical_grads[name]

        # Only compare first elements for large tensors
        if analytical.numel() > 1000:
            analytical = analytical.flatten()[:1000]
            numerical = numerical.flatten()[:1000]

        if not torch.allclose(analytical, numerical, atol=atol, rtol=rtol):
            return False

    return True


def validate_custom_autograd_function(
    func_class: type,
    inputs: Dict[str, "torch.Tensor"],
    eps: float = 1e-4,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> Dict[str, Any]:
    """Validate a custom autograd.Function implementation.

    Tests both the forward pass behavior and gradient correctness for
    a torch.autograd.Function subclass.

    Args:
        func_class: The autograd.Function class to test.
        inputs: Dictionary of input tensors.
        eps: Epsilon for finite differences.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Returns:
        Dictionary with 'forward_ok', 'backward_ok', and 'details'.

    Example:
        >>> class MyFunc(torch.autograd.Function):
        ...     @staticmethod
        ...     def forward(ctx, x):
        ...         ctx.save_for_backward(x)
        ...         return x * 2
        ...     @staticmethod
        ...     def backward(ctx, grad):
        ...         return grad * 2
        >>> result = validate_custom_autograd_function(MyFunc, {"x": x})
        >>> assert result["backward_ok"]
    """
    import torch

    result = {"forward_ok": True, "backward_ok": True, "details": []}

    # Prepare inputs with gradients
    grad_inputs = {}
    for k, v in inputs.items():
        if v.is_floating_point():
            grad_inputs[k] = v.clone().detach().requires_grad_(True)
        else:
            grad_inputs[k] = v.clone().detach()

    # Forward pass
    try:
        output = func_class.apply(*grad_inputs.values())
    except Exception as e:
        result["forward_ok"] = False
        result["details"].append(f"Forward failed: {e}")
        return result

    # Check gradients
    try:
        grad_ok = check_autograd(
            lambda **kw: func_class.apply(*kw.values()),
            grad_inputs,
            eps=eps,
            atol=atol,
            rtol=rtol,
        )
        result["backward_ok"] = grad_ok
        if not grad_ok:
            result["details"].append(
                "Gradient mismatch: analytical gradients don't match numerical"
            )
    except Exception as e:
        result["backward_ok"] = False
        result["details"].append(f"Gradient check failed: {e}")

    return result


def fuzz_pytorch_op(
    client: "Client",
    op_name: str,
    run_op: "Callable[[Dict[str, torch.Tensor]], torch.Tensor]",
    iterations: int = 100,
    seed: Optional[int] = None,
    fail_fast: bool = False,
    check_backward: bool = False,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
) -> Dict[str, Any]:
    """Fuzz a PyTorch op with client-side GPU execution.

    This is the primary drop-in method for PyTorch developers. The daemon
    generates random inputs, you run your GPU op, and gpuemu validates
    the output against the reference. Optionally validates gradients too.

    Args:
        client: gpuemu Client instance.
        op_name: Name of the op (must be registered in gpuemu.toml).
        run_op: Callable that takes a dict of torch.Tensor inputs and
                returns a torch.Tensor output. This is YOUR GPU kernel.
        iterations: Number of fuzz iterations.
        seed: Master seed. Auto-generated if None.
        fail_fast: Stop on first failure.
        check_backward: Also validate gradients via finite differences.
        atol: Absolute tolerance override.
        rtol: Relative tolerance override.

    Returns:
        Dict with 'total', 'passed', 'failed', 'forward_failures', 'backward_failures'.

    Example:
        >>> client = Client()
        >>> result = fuzz_pytorch_op(
        ...     client,
        ...     "flash_attention",
        ...     run_op=lambda inputs: flash_attn_func(inputs["q"], inputs["k"], inputs["v"]),
        ...     iterations=50,
        ...     check_backward=True,
        ... )
        >>> print(f"Passed: {result['passed']}/{result['total']}")
    """
    import torch

    adapter = PyTorchAdapter()

    cases = client.get_test_batch(op_name, count=iterations, seed=seed)

    total = 0
    passed = 0
    failed = 0
    forward_failures = []
    backward_failures = []

    for case in cases:
        total += 1

        # Convert numpy inputs to torch tensors on GPU
        torch_inputs = {
            k: adapter.from_numpy(v).cuda() for k, v in case["inputs"].items()
        }

        try:
            # Run the actual GPU op
            output = run_op(torch_inputs)

            # Validate forward pass
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

            # Optional backward pass check
            if check_backward and any(
                adapter.requires_grad(t) for t in torch_inputs.values()
            ):
                try:
                    grad_ok = check_autograd(
                        lambda **kw: run_op(kw),
                        {
                            k: v.clone().detach().requires_grad_(True)
                            if adapter.requires_grad(v)
                            else v.clone().detach()
                            for k, v in torch_inputs.items()
                        },
                    )
                    if not grad_ok:
                        backward_failures.append(
                            {
                                "seed": case["seed"],
                                "message": "Gradient check failed (analytical vs numerical)",
                            }
                        )
                except Exception as e:
                    backward_failures.append(
                        {
                            "seed": case["seed"],
                            "message": f"Gradient check error: {e}",
                        }
                    )

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
        "backward_failures": backward_failures,
    }
