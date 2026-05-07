"""Cross-framework tolerance calibration for numerical validation.

Different ML frameworks have different numerical behaviors:
- PyTorch: Deterministic by default on CPU, may vary on GPU
- JAX: May use different BLAS implementations, XLA optimizations
- TensorFlow: XLA compilation may affect precision

This module provides:
1. Default tolerances for common dtypes
2. Framework-specific tolerance multipliers
3. Automatic tolerance calibration between implementations
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ToleranceConfig:
    """Tolerance thresholds for numerical validation.

    Attributes:
        atol: Absolute tolerance - maximum allowed absolute difference.
        rtol: Relative tolerance - maximum allowed relative difference.

    The comparison passes if: |a - b| <= atol + rtol * |b|
    """

    atol: float
    rtol: float

    def __post_init__(self):
        """Validate tolerance values."""
        if self.atol < 0:
            raise ValueError(f"atol must be non-negative, got {self.atol}")
        if self.rtol < 0:
            raise ValueError(f"rtol must be non-negative, got {self.rtol}")

    @classmethod
    def for_dtype(cls, dtype: str, framework: str = "numpy") -> "ToleranceConfig":
        """Get calibrated tolerances for dtype/framework combination.

        This returns pre-calibrated tolerances that account for typical
        numerical differences between implementations.

        Args:
            dtype: Data type string like "float32", "float16".
            framework: Framework name like "pytorch", "jax", "tensorflow".

        Returns:
            ToleranceConfig with appropriate thresholds.

        Example:
            >>> tol = ToleranceConfig.for_dtype("float16", "jax")
            >>> print(f"atol={tol.atol}, rtol={tol.rtol}")
        """
        base = DEFAULT_TOLERANCES.get(dtype, ToleranceConfig(1e-5, 1e-5))

        # Apply framework-specific adjustments
        multiplier = FRAMEWORK_MULTIPLIERS.get(framework, 1.0)

        return ToleranceConfig(
            atol=base.atol * multiplier,
            rtol=base.rtol * multiplier,
        )

    @classmethod
    def strict(cls, dtype: str = "float32") -> "ToleranceConfig":
        """Get strict tolerances for exact comparison.

        Use when you expect near-exact matches (e.g., comparing
        the same implementation on the same hardware).

        Args:
            dtype: Data type string.

        Returns:
            ToleranceConfig with tight thresholds.
        """
        strict_tolerances = {
            "float16": ToleranceConfig(atol=1e-4, rtol=1e-3),
            "bfloat16": ToleranceConfig(atol=1e-3, rtol=1e-2),
            "float32": ToleranceConfig(atol=1e-6, rtol=1e-5),
            "float64": ToleranceConfig(atol=1e-12, rtol=1e-11),
        }
        return strict_tolerances.get(dtype, ToleranceConfig(1e-6, 1e-5))

    @classmethod
    def relaxed(cls, dtype: str = "float32") -> "ToleranceConfig":
        """Get relaxed tolerances for cross-framework comparison.

        Use when comparing across different frameworks or hardware
        where larger numerical differences are expected.

        Args:
            dtype: Data type string.

        Returns:
            ToleranceConfig with loose thresholds.
        """
        relaxed_tolerances = {
            "float16": ToleranceConfig(atol=1e-2, rtol=1e-1),
            "bfloat16": ToleranceConfig(atol=5e-2, rtol=5e-1),
            "float32": ToleranceConfig(atol=1e-4, rtol=1e-3),
            "float64": ToleranceConfig(atol=1e-8, rtol=1e-7),
        }
        return relaxed_tolerances.get(dtype, ToleranceConfig(1e-4, 1e-3))

    def scale(self, factor: float) -> "ToleranceConfig":
        """Scale tolerances by a factor.

        Args:
            factor: Multiplier for both atol and rtol.

        Returns:
            New ToleranceConfig with scaled values.
        """
        return ToleranceConfig(
            atol=self.atol * factor,
            rtol=self.rtol * factor,
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {"atol": self.atol, "rtol": self.rtol}


# Default tolerances by dtype
# These are calibrated for typical numerical precision
DEFAULT_TOLERANCES: Dict[str, ToleranceConfig] = {
    "float16": ToleranceConfig(atol=1e-3, rtol=1e-2),
    "bfloat16": ToleranceConfig(atol=1e-2, rtol=1e-1),
    "float32": ToleranceConfig(atol=1e-5, rtol=1e-4),
    "float64": ToleranceConfig(atol=1e-10, rtol=1e-9),
}

# Framework-specific multipliers (account for implementation differences)
# Higher values = more lenient
FRAMEWORK_MULTIPLIERS: Dict[str, float] = {
    "numpy": 1.0,
    "pytorch": 1.0,
    "jax": 1.5,  # JAX XLA may have slightly different results
    "tensorflow": 1.5,  # TF XLA similar to JAX
}


def calibrate_tolerance(
    reference_fn: Callable,
    test_fn: Callable,
    input_shapes: List[Tuple[int, ...]],
    dtype: str = "float32",
    n_samples: int = 100,
    seed: int = 42,
    percentile: float = 99.0,
    safety_margin: float = 2.0,
) -> ToleranceConfig:
    """Empirically calibrate tolerances between reference and test implementations.

    Runs both functions on random inputs and measures the maximum differences.
    Returns tolerances that would pass a specified percentile of samples.

    This is useful when you have a known-correct reference implementation
    and want to determine appropriate tolerances for a new implementation.

    Args:
        reference_fn: Reference implementation (ground truth).
        test_fn: Implementation to calibrate against.
        input_shapes: List of input shapes to use.
        dtype: NumPy dtype string for inputs.
        n_samples: Number of random samples to test.
        seed: Random seed for reproducibility.
        percentile: Use this percentile of differences as base tolerance.
        safety_margin: Multiply base tolerance by this factor.

    Returns:
        ToleranceConfig that would pass the specified percentile of samples.

    Example:
        >>> def ref(x):
        ...     return np.sin(x)
        >>> def test(x):
        ...     return custom_sin(x)
        >>> tol = calibrate_tolerance(ref, test, [(100,)], "float32")
        >>> print(f"Suggested: atol={tol.atol:.2e}, rtol={tol.rtol:.2e}")
    """
    rng = np.random.default_rng(seed)
    np_dtype = np.dtype(dtype)

    max_abs_diffs: List[float] = []
    max_rel_diffs: List[float] = []

    for _ in range(n_samples):
        # Generate random inputs
        inputs = [rng.standard_normal(shape).astype(np_dtype) for shape in input_shapes]

        # Run both implementations
        ref_output = reference_fn(*inputs)
        test_output = test_fn(*inputs)

        # Compute differences
        abs_diff = np.abs(ref_output - test_output)
        rel_diff = abs_diff / (np.abs(ref_output) + 1e-10)

        max_abs_diffs.append(float(np.max(abs_diff)))
        max_rel_diffs.append(float(np.max(rel_diff)))

    # Use specified percentile as tolerance
    atol = float(np.percentile(max_abs_diffs, percentile)) * safety_margin
    rtol = float(np.percentile(max_rel_diffs, percentile)) * safety_margin

    return ToleranceConfig(atol=atol, rtol=rtol)


def compare_tolerances(
    a: ToleranceConfig,
    b: ToleranceConfig,
) -> Dict[str, str]:
    """Compare two tolerance configurations.

    Args:
        a: First tolerance config.
        b: Second tolerance config.

    Returns:
        Dictionary with comparison details.
    """
    return {
        "atol_ratio": f"{a.atol / b.atol:.2f}x" if b.atol > 0 else "inf",
        "rtol_ratio": f"{a.rtol / b.rtol:.2f}x" if b.rtol > 0 else "inf",
        "a_stricter": a.atol < b.atol and a.rtol < b.rtol,
        "b_stricter": b.atol < a.atol and b.rtol < a.rtol,
    }


class ToleranceProfile:
    """Named collection of tolerances for common scenarios.

    Provides pre-configured tolerance settings for different use cases.

    Example:
        >>> profile = ToleranceProfile.for_testing()
        >>> tol = profile.get("float32")
    """

    def __init__(self, name: str, tolerances: Dict[str, ToleranceConfig]):
        """Initialize a tolerance profile.

        Args:
            name: Profile name for identification.
            tolerances: Mapping of dtype to ToleranceConfig.
        """
        self.name = name
        self.tolerances = tolerances

    def get(self, dtype: str) -> ToleranceConfig:
        """Get tolerance for a dtype.

        Args:
            dtype: Data type string.

        Returns:
            ToleranceConfig for the dtype, or default if not found.
        """
        return self.tolerances.get(dtype, ToleranceConfig(1e-5, 1e-5))

    @classmethod
    def for_testing(cls) -> "ToleranceProfile":
        """Get tolerances suitable for unit testing.

        These are moderately strict tolerances suitable for
        comparing implementations in tests.
        """
        return cls(
            "testing",
            {
                "float16": ToleranceConfig(atol=1e-3, rtol=1e-2),
                "bfloat16": ToleranceConfig(atol=1e-2, rtol=1e-1),
                "float32": ToleranceConfig(atol=1e-5, rtol=1e-4),
                "float64": ToleranceConfig(atol=1e-10, rtol=1e-9),
            },
        )

    @classmethod
    def for_production(cls) -> "ToleranceProfile":
        """Get tolerances suitable for production validation.

        These are stricter tolerances for production use where
        numerical accuracy is critical.
        """
        return cls(
            "production",
            {
                "float16": ToleranceConfig(atol=5e-4, rtol=5e-3),
                "bfloat16": ToleranceConfig(atol=5e-3, rtol=5e-2),
                "float32": ToleranceConfig(atol=1e-6, rtol=1e-5),
                "float64": ToleranceConfig(atol=1e-12, rtol=1e-11),
            },
        )

    @classmethod
    def for_cross_framework(cls, source: str, target: str) -> "ToleranceProfile":
        """Get tolerances for comparing across frameworks.

        Args:
            source: Source framework name.
            target: Target framework name.

        Returns:
            ToleranceProfile calibrated for the framework pair.
        """
        # Get multipliers for both frameworks
        source_mult = FRAMEWORK_MULTIPLIERS.get(source, 1.0)
        target_mult = FRAMEWORK_MULTIPLIERS.get(target, 1.0)
        combined_mult = max(source_mult, target_mult) * 1.5

        return cls(
            f"{source}_to_{target}",
            {
                dtype: ToleranceConfig(
                    atol=tol.atol * combined_mult,
                    rtol=tol.rtol * combined_mult,
                )
                for dtype, tol in DEFAULT_TOLERANCES.items()
            },
        )


def get_recommended_tolerance(
    dtype: str,
    framework: Optional[str] = None,
    operation: Optional[str] = None,
) -> ToleranceConfig:
    """Get recommended tolerance for a specific scenario.

    This function provides intelligent tolerance recommendations
    based on dtype, framework, and operation type.

    Args:
        dtype: Data type string.
        framework: Optional framework name.
        operation: Optional operation name (e.g., "matmul", "conv2d").

    Returns:
        Recommended ToleranceConfig.

    Example:
        >>> tol = get_recommended_tolerance("float16", "pytorch", "matmul")
    """
    # Start with base tolerance for dtype
    tol = DEFAULT_TOLERANCES.get(dtype, ToleranceConfig(1e-5, 1e-5))

    # Apply framework multiplier
    if framework:
        mult = FRAMEWORK_MULTIPLIERS.get(framework, 1.0)
        tol = tol.scale(mult)

    # Apply operation-specific adjustments
    # Some operations have inherently larger numerical errors
    OPERATION_MULTIPLIERS = {
        "matmul": 1.5,  # Matrix multiply accumulates errors
        "conv2d": 2.0,  # Convolution has many FMAs
        "softmax": 1.5,  # Exp can amplify differences
        "batchnorm": 2.0,  # Normalization sensitive to variance
        "layernorm": 2.0,
        "attention": 2.5,  # Multiple softmax + matmul
    }

    if operation:
        op_mult = OPERATION_MULTIPLIERS.get(operation.lower(), 1.0)
        tol = tol.scale(op_mult)

    return tol
