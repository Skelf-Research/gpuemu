"""Tests for the built-in fp64 reference oracles."""

import base64
import io
import json

import numpy as np
import pytest

from gpuemu.references import ops


def test_silu_matches_definition():
    x = np.array([-2.0, 0.0, 1.0, 3.0])
    expected = x / (1.0 + np.exp(-x))
    np.testing.assert_allclose(ops.silu(x), expected, rtol=0, atol=1e-12)


def test_gelu_matches_scipy_free_definition():
    from math import erf

    x = np.array([-1.5, 0.0, 0.7, 2.0])
    expected = np.array([0.5 * v * (1.0 + erf(v / np.sqrt(2.0))) for v in x])
    np.testing.assert_allclose(ops.gelu(x), expected, rtol=0, atol=1e-12)


def test_softmax_rows_sum_to_one():
    x = np.random.RandomState(0).randn(4, 7)
    out = ops.softmax(x)
    np.testing.assert_allclose(out.sum(axis=-1), np.ones(4), atol=1e-12)
    assert (out >= 0).all()


def test_rms_norm_is_not_layernorm():
    # RMSNorm does NOT subtract the mean: a constant-shifted input normalises
    # differently than it would under LayerNorm.
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    w = np.ones(4)
    out = ops.rms_norm(x, w)
    ms = np.mean(x * x)
    np.testing.assert_allclose(out, x / np.sqrt(ms + 1e-6), atol=1e-10)


def test_rope_is_norm_preserving():
    # Rotations preserve the L2 norm of each (x1, x2) pair, so the per-vector
    # norm is unchanged by RoPE.
    rng = np.random.RandomState(1)
    x = rng.randn(1, 2, 5, 8)
    out = ops.rope(x)
    np.testing.assert_allclose(
        np.linalg.norm(out, axis=-1), np.linalg.norm(x, axis=-1), atol=1e-10
    )


def test_rope_rejects_odd_head_dim():
    with pytest.raises(ValueError):
        ops.rope(np.zeros((1, 1, 2, 3)))


def test_matmul_silu_is_fused_composition():
    a = np.random.RandomState(2).randn(3, 4)
    b = np.random.RandomState(3).randn(4, 5)
    np.testing.assert_allclose(ops.matmul_silu(a, b), ops.silu(a @ b), atol=1e-12)


def test_kv_cache_attention_shapes_and_decouples_seqlen():
    # Sq=2 query positions attend to Sk=5 cached positions.
    rng = np.random.RandomState(4)
    q = rng.randn(1, 3, 2, 8)
    k = rng.randn(1, 3, 5, 8)
    v = rng.randn(1, 3, 5, 8)
    out = ops.kv_cache_attention(q, k, v)
    assert out.shape == (1, 3, 2, 8)


def test_run_protocol_roundtrip():
    # Drive ops.run via the daemon's stdin/stdout JSON protocol.
    x = np.arange(6, dtype=np.float64).reshape(2, 3)
    payload = {
        "inputs": {
            "input": {
                "shape": [2, 3],
                "dtype": "float64",
                "data": base64.b64encode(x.tobytes()).decode(),
            }
        },
        "kwargs": {},
    }
    out_buf = io.StringIO()
    ops.run(ops.softmax, stdin=io.StringIO(json.dumps(payload)), stdout=out_buf)
    result = json.loads(out_buf.getvalue())
    assert result["dtype"] == "float64"
    arr = np.frombuffer(base64.b64decode(result["data"]), dtype=np.float64).reshape(
        result["shape"]
    )
    np.testing.assert_allclose(arr, ops.softmax(x), atol=1e-12)


def test_registry_covers_all_builtin_ops():
    # Mirror of the Rust OpSchema::builtin names that have a reference here.
    expected = {
        "silu",
        "gelu",
        "softmax",
        "rmsnorm",
        "rope",
        "matmul_silu",
        "matmul_gelu",
        "attention",
        "kv_cache_attention",
    }
    assert expected <= set(ops.REGISTRY)
