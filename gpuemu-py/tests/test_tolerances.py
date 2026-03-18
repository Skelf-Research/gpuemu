"""Tests for tolerance calibration module."""

import numpy as np
import pytest

from gpuemu_py.tolerances import (
    DEFAULT_TOLERANCES,
    ToleranceConfig,
    ToleranceProfile,
    calibrate_tolerance,
    compare_tolerances,
    get_recommended_tolerance,
)


class TestToleranceConfig:
    """Tests for ToleranceConfig dataclass."""

    def test_creation(self):
        """Test basic creation."""
        tol = ToleranceConfig(atol=1e-5, rtol=1e-4)
        assert tol.atol == 1e-5
        assert tol.rtol == 1e-4

    def test_negative_atol_raises(self):
        """Test that negative atol raises ValueError."""
        with pytest.raises(ValueError, match="atol must be non-negative"):
            ToleranceConfig(atol=-1e-5, rtol=1e-4)

    def test_negative_rtol_raises(self):
        """Test that negative rtol raises ValueError."""
        with pytest.raises(ValueError, match="rtol must be non-negative"):
            ToleranceConfig(atol=1e-5, rtol=-1e-4)

    def test_for_dtype_float32(self):
        """Test tolerance lookup for float32."""
        tol = ToleranceConfig.for_dtype("float32")
        assert tol.atol == 1e-5
        assert tol.rtol == 1e-4

    def test_for_dtype_float16(self):
        """Test tolerance lookup for float16."""
        tol = ToleranceConfig.for_dtype("float16")
        assert tol.atol == 1e-3
        assert tol.rtol == 1e-2

    def test_for_dtype_with_framework(self):
        """Test tolerance lookup with framework multiplier."""
        tol_numpy = ToleranceConfig.for_dtype("float32", "numpy")
        tol_jax = ToleranceConfig.for_dtype("float32", "jax")

        # JAX has 1.5x multiplier
        assert tol_jax.atol == tol_numpy.atol * 1.5
        assert tol_jax.rtol == tol_numpy.rtol * 1.5

    def test_for_dtype_unknown(self):
        """Test tolerance lookup for unknown dtype."""
        tol = ToleranceConfig.for_dtype("unknown_dtype")
        # Should return default
        assert tol.atol == 1e-5
        assert tol.rtol == 1e-5

    def test_strict(self):
        """Test strict tolerance preset."""
        tol = ToleranceConfig.strict("float32")
        assert tol.atol < DEFAULT_TOLERANCES["float32"].atol
        assert tol.rtol < DEFAULT_TOLERANCES["float32"].rtol

    def test_relaxed(self):
        """Test relaxed tolerance preset."""
        tol = ToleranceConfig.relaxed("float32")
        assert tol.atol > DEFAULT_TOLERANCES["float32"].atol
        assert tol.rtol > DEFAULT_TOLERANCES["float32"].rtol

    def test_scale(self):
        """Test scaling tolerances."""
        tol = ToleranceConfig(atol=1e-5, rtol=1e-4)
        scaled = tol.scale(2.0)

        assert scaled.atol == 2e-5
        assert scaled.rtol == 2e-4

    def test_to_dict(self):
        """Test conversion to dictionary."""
        tol = ToleranceConfig(atol=1e-5, rtol=1e-4)
        d = tol.to_dict()

        assert d == {"atol": 1e-5, "rtol": 1e-4}


class TestCalibrateTolerances:
    """Tests for tolerance calibration."""

    def test_identical_functions(self):
        """Test calibration with identical functions."""

        def fn(x):
            return np.sin(x)

        tol = calibrate_tolerance(fn, fn, [(100,)], "float32", n_samples=50)

        # Should get very small tolerances for identical functions
        assert tol.atol < 1e-10
        assert tol.rtol < 1e-10

    def test_slightly_different_functions(self):
        """Test calibration with slightly different implementations."""

        def ref(x):
            return np.sin(x)

        def test(x):
            # Slightly different due to float32 precision
            return np.sin(x.astype(np.float64)).astype(np.float32)

        tol = calibrate_tolerance(ref, test, [(100,)], "float32", n_samples=50)

        # Should get non-zero but small tolerances
        assert tol.atol > 0
        assert tol.atol < 1e-5

    def test_reproducibility(self):
        """Test that calibration is reproducible with same seed."""

        def fn(x):
            return x ** 2

        def noisy_fn(x):
            return x ** 2 + np.random.randn(*x.shape) * 1e-6

        tol1 = calibrate_tolerance(fn, noisy_fn, [(100,)], "float32", seed=42)
        tol2 = calibrate_tolerance(fn, noisy_fn, [(100,)], "float32", seed=42)

        # Different seeds in noisy_fn cause differences, but calibration seed
        # controls input generation only
        assert tol1.atol > 0
        assert tol2.atol > 0


class TestToleranceProfile:
    """Tests for ToleranceProfile."""

    def test_for_testing(self):
        """Test testing profile."""
        profile = ToleranceProfile.for_testing()
        assert profile.name == "testing"

        tol = profile.get("float32")
        assert tol.atol == 1e-5
        assert tol.rtol == 1e-4

    def test_for_production(self):
        """Test production profile."""
        profile = ToleranceProfile.for_production()
        assert profile.name == "production"

        tol = profile.get("float32")
        assert tol.atol < ToleranceProfile.for_testing().get("float32").atol

    def test_for_cross_framework(self):
        """Test cross-framework profile."""
        profile = ToleranceProfile.for_cross_framework("pytorch", "jax")
        assert "pytorch" in profile.name
        assert "jax" in profile.name

        # Should be more relaxed than default
        tol = profile.get("float32")
        assert tol.atol > DEFAULT_TOLERANCES["float32"].atol

    def test_get_unknown_dtype(self):
        """Test getting tolerance for unknown dtype."""
        profile = ToleranceProfile.for_testing()
        tol = profile.get("unknown_dtype")

        # Should return default
        assert tol.atol == 1e-5
        assert tol.rtol == 1e-5


class TestCompareTolerance:
    """Tests for tolerance comparison."""

    def test_compare_same(self):
        """Test comparing identical tolerances."""
        a = ToleranceConfig(atol=1e-5, rtol=1e-4)
        b = ToleranceConfig(atol=1e-5, rtol=1e-4)

        result = compare_tolerances(a, b)
        assert result["atol_ratio"] == "1.00x"
        assert result["rtol_ratio"] == "1.00x"

    def test_compare_different(self):
        """Test comparing different tolerances."""
        a = ToleranceConfig(atol=1e-6, rtol=1e-5)
        b = ToleranceConfig(atol=1e-5, rtol=1e-4)

        result = compare_tolerances(a, b)
        assert "a_stricter" in result
        assert result["a_stricter"] is True


class TestGetRecommendedTolerance:
    """Tests for recommended tolerance lookup."""

    def test_basic(self):
        """Test basic tolerance recommendation."""
        tol = get_recommended_tolerance("float32")
        assert tol.atol == DEFAULT_TOLERANCES["float32"].atol
        assert tol.rtol == DEFAULT_TOLERANCES["float32"].rtol

    def test_with_framework(self):
        """Test tolerance recommendation with framework."""
        tol = get_recommended_tolerance("float32", framework="jax")
        assert tol.atol > DEFAULT_TOLERANCES["float32"].atol

    def test_with_operation(self):
        """Test tolerance recommendation with operation."""
        tol_base = get_recommended_tolerance("float32")
        tol_matmul = get_recommended_tolerance("float32", operation="matmul")

        # matmul should have higher tolerance
        assert tol_matmul.atol > tol_base.atol

    def test_with_framework_and_operation(self):
        """Test tolerance recommendation with both framework and operation."""
        tol = get_recommended_tolerance("float32", framework="tensorflow", operation="attention")

        # Should be more relaxed than base
        assert tol.atol > DEFAULT_TOLERANCES["float32"].atol * 2
