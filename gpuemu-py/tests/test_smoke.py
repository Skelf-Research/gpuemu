"""Smoke tests for gpuemu package import and basic functionality."""

import base64
import json

import numpy as np
import pytest


def test_import_version():
    """Package version is accessible."""
    from gpuemu import __version__

    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"


def test_import_client():
    """Client class is importable."""
    from gpuemu import Client, ClientError

    assert Client is not None
    assert ClientError is not None


def test_import_rng():
    """RNG module is importable."""
    from gpuemu import SeededRng, derive_seed, generate_seed

    assert SeededRng is not None


def test_import_validate():
    """Validation utilities are importable."""
    from gpuemu import validate, validate_op, FuzzConfig, SeededFuzzer

    assert validate is not None


def test_import_tolerances():
    """Tolerance utilities are importable."""
    from gpuemu import ToleranceConfig, get_recommended_tolerance

    assert ToleranceConfig is not None


def test_rng_cross_language_compatibility():
    """RNG output matches the Rust xorshift128+ implementation."""
    from gpuemu.rng import SeededRng, _splitmix64

    assert _splitmix64(42) == 13679457532755275413
    assert _splitmix64(43) == 13432527470776545160

    rng = SeededRng(42)
    values = [rng._next_u64() for _ in range(5)]
    expected = [
        14921385303349856026,
        836881716698787820,
        2679325795615653720,
        15230690181549778750,
        17419016854217654406,
    ]
    assert values == expected, f"RNG mismatch: {values} != {expected}"


def test_client_encode_decode_tensor():
    """Tensor encoding/decoding round-trips correctly."""
    from gpuemu import Client

    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    encoded = Client._encode_tensor(arr)
    decoded = Client._decode_tensor(encoded)

    assert decoded.shape == arr.shape
    np.testing.assert_array_equal(decoded, arr)


def test_client_dtype_mapping():
    """Dtype mapping covers all common types."""
    from gpuemu import Client

    for np_dtype, proto_str in [
        (np.float16, "float16"),
        (np.float32, "float32"),
        (np.float64, "float64"),
        (np.int8, "int8"),
        (np.int16, "int16"),
        (np.int32, "int32"),
        (np.int64, "int64"),
        (np.uint8, "uint8"),
        (np.uint16, "uint16"),
        (np.uint32, "uint32"),
        (np.uint64, "uint64"),
        (np.bool_, "bool"),
    ]:
        result = Client._numpy_dtype_to_protocol(np.dtype(np_dtype))
        assert result == proto_str, (
            f"Mapping for {np_dtype} returned {result}, expected {proto_str}"
        )


def test_client_bfloat16_mapping():
    """BFloat16 maps to 'bfloat16' protocol string."""
    from gpuemu import Client

    result = Client._numpy_dtype_to_protocol("bfloat16")
    assert result == "bfloat16"


def test_seeded_fuzzer_iterator():
    """SeededFuzzer iterator produces deterministic test cases."""
    from gpuemu import FuzzConfig, SeededFuzzer

    config = FuzzConfig(
        seed=12345,
        batch_sizes=[1],
        seq_lengths=[4],
        hidden_dims=[8],
        dtypes=["float32"],
        layouts=["Contiguous"],
    )
    fuzzer = SeededFuzzer(config)

    cases = list(fuzzer.iterator(max_iterations=3))
    assert len(cases) == 3
    for case in cases:
        assert hasattr(case, "seed")
        assert hasattr(case, "shape")
        assert hasattr(case, "dtype")
        assert case.dtype == "float32"
        arr = case.generate_input("input")
        assert isinstance(arr, np.ndarray)
        assert arr.shape == case.shape


def test_protocol_version_constant():
    """PROTOCOL_VERSION is defined and is an integer."""
    from gpuemu.client import PROTOCOL_VERSION

    assert isinstance(PROTOCOL_VERSION, int)
    assert PROTOCOL_VERSION >= 1
