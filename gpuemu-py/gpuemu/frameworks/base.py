"""Base framework adapter for tensor conversion and gradient computation."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import numpy as np


class FrameworkAdapter(ABC):
    """Base class for framework-specific adapters.

    Each ML framework (PyTorch, JAX, TensorFlow) has its own tensor type
    and gradient computation mechanism. This adapter provides a uniform
    interface for:

    1. Converting between framework tensors and numpy arrays
    2. Checking if tensors require gradient tracking
    3. Computing gradients of outputs with respect to inputs

    Subclasses should implement all abstract methods for their specific
    framework.

    Example:
        >>> class MyFrameworkAdapter(FrameworkAdapter):
        ...     def to_numpy(self, tensor):
        ...         return tensor.numpy()
        ...
        >>> adapter = MyFrameworkAdapter()
        >>> np_arr = adapter.to_numpy(framework_tensor)
    """

    @abstractmethod
    def to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert framework tensor to numpy array.

        Args:
            tensor: Framework-specific tensor (torch.Tensor, jax.Array, tf.Tensor).

        Returns:
            Numpy array with the same data. Should handle:
            - Device transfer (GPU to CPU)
            - Gradient detachment if applicable
            - Copy semantics (return contiguous array)
        """
        pass

    @abstractmethod
    def from_numpy(self, arr: np.ndarray, like: Optional[Any] = None) -> Any:
        """Convert numpy array to framework tensor.

        Args:
            arr: Numpy array to convert.
            like: Optional template tensor to match device/dtype from.

        Returns:
            Framework tensor with the same data.
        """
        pass

    @abstractmethod
    def get_dtype_name(self, tensor: Any) -> str:
        """Get dtype name for tolerance lookup.

        Args:
            tensor: Framework tensor.

        Returns:
            String dtype name like "float32", "float16", etc.
        """
        pass

    @abstractmethod
    def requires_grad(self, tensor: Any) -> bool:
        """Check if tensor requires gradient tracking.

        Args:
            tensor: Framework tensor.

        Returns:
            True if tensor is set up for gradient computation.
        """
        pass

    @abstractmethod
    def compute_gradient(
        self,
        output: Any,
        inputs: Dict[str, Any],
        grad_output: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Compute gradients of output w.r.t. inputs.

        Args:
            output: The output tensor to differentiate.
            inputs: Dictionary of input tensors.
            grad_output: Optional upstream gradient. If None, uses ones.

        Returns:
            Dictionary mapping input names to their gradients.
            Inputs that don't require gradients should be omitted.
        """
        pass

    def is_available(self) -> bool:
        """Check if the framework is installed and available.

        Returns:
            True if the framework can be imported.
        """
        return True

    def get_framework_name(self) -> str:
        """Get the framework name for tolerance lookup.

        Returns:
            String like "pytorch", "jax", or "tensorflow".
        """
        return "unknown"


class GradientChecker:
    """Utility for checking gradient correctness via finite differences.

    This class provides framework-agnostic gradient checking by comparing
    analytical gradients (from autograd) with numerical gradients computed
    via finite differences.
    """

    def __init__(
        self,
        adapter: FrameworkAdapter,
        eps: float = 1e-4,
        atol: float = 1e-5,
        rtol: float = 1e-3,
    ):
        """Initialize the gradient checker.

        Args:
            adapter: Framework adapter for tensor operations.
            eps: Epsilon for finite differences.
            atol: Absolute tolerance for comparison.
            rtol: Relative tolerance for comparison.
        """
        self.adapter = adapter
        self.eps = eps
        self.atol = atol
        self.rtol = rtol

    def check(
        self,
        func: Callable[..., Any],
        inputs: Dict[str, Any],
        check_inputs: Optional[list] = None,
    ) -> bool:
        """Check gradient correctness for a function.

        Args:
            func: Function that takes **inputs and returns a tensor.
            inputs: Dictionary of input tensors.
            check_inputs: Optional list of input names to check. If None,
                checks all inputs that require gradients.

        Returns:
            True if analytical and numerical gradients match.
        """
        # This is implemented in framework-specific adapters
        raise NotImplementedError(
            "Use the framework-specific check_autograd or similar function"
        )
