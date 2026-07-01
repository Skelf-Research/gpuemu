"""Canonical fp64 reference implementations for the built-in op schemas.

These are the high-precision ground-truth oracles that pair with the
``OpSchema::builtin`` shapes in the Rust daemon (matmul, attention, rmsnorm,
rope, silu/gelu/softmax, fused matmul+epilogue, kv-cache attention).

Every function computes in float64 regardless of the input dtype, so they are
correct ground truth when the daemon promotes inputs to fp64
(``[validation] oracle_fp64 = true``). They are intentionally simple and
vectorised with numpy — readability over speed, since correctness is the point.

Each function takes named numpy arrays and returns a numpy array, matching the
``reference(**inputs)`` contract. The :func:`run` helper wraps a function in the
daemon's stdin/stdout JSON protocol so a one-line script can serve as a
registered reference (see the sibling per-op scripts).
"""

from __future__ import annotations

import base64
import json
import sys
from math import erf

import numpy as np

_ERF = np.vectorize(erf)


def _f64(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


# --- elementwise activations -------------------------------------------------


def silu(input: np.ndarray) -> np.ndarray:
    """SiLU / swish: x * sigmoid(x)."""
    x = _f64(input)
    return x / (1.0 + np.exp(-x))


def gelu(input: np.ndarray) -> np.ndarray:
    """Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))."""
    x = _f64(input)
    return 0.5 * x * (1.0 + _ERF(x / np.sqrt(2.0)))


def softmax(input: np.ndarray) -> np.ndarray:
    """Row-softmax over the last axis (numerically stabilised)."""
    x = _f64(input)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


# --- normalisation -----------------------------------------------------------


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm over the last dim: x / sqrt(mean(x^2) + eps) * weight.

    Note this is RMSNorm, *not* LayerNorm — there is no mean subtraction.
    """
    xf = _f64(x)
    w = _f64(weight)
    ms = np.mean(xf * xf, axis=-1, keepdims=True)
    return xf / np.sqrt(ms + eps) * w


# --- rotary position embedding ----------------------------------------------


def rope(x: np.ndarray, theta: float = 10000.0) -> np.ndarray:
    """Rotary position embedding over x[..., S, D] (D even, GPT-NeoX halves).

    Positions are 0..S-1 along the second-to-last axis. The last dim is split
    into two halves that are rotated against each other.
    """
    xf = _f64(x)
    *_, s, d = xf.shape
    if d % 2 != 0:
        raise ValueError(f"rope requires an even head dim, got D={d}")
    half = d // 2
    inv_freq = theta ** (-np.arange(0, half, dtype=np.float64) / half)  # [half]
    pos = np.arange(s, dtype=np.float64)  # [S]
    ang = np.outer(pos, inv_freq)  # [S, half]
    cos = np.cos(ang)
    sin = np.sin(ang)
    # broadcast [S, half] to xf's leading dims
    lead = (1,) * (xf.ndim - 2)
    cos = cos.reshape(*lead, s, half)
    sin = sin.reshape(*lead, s, half)
    x1 = xf[..., :half]
    x2 = xf[..., half:]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return np.concatenate([out1, out2], axis=-1)


# --- fused matmul + activation epilogue -------------------------------------


def matmul_silu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fused MatMul + SiLU: silu(a @ b)."""
    return silu(_f64(a) @ _f64(b))


def matmul_gelu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fused MatMul + GELU: gelu(a @ b)."""
    return gelu(_f64(a) @ _f64(b))


# --- attention ---------------------------------------------------------------


def _sdpa(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    qf, kf, vf = _f64(q), _f64(k), _f64(v)
    d = qf.shape[-1]
    scores = qf @ np.swapaxes(kf, -1, -2) / np.sqrt(d)  # [..., Sq, Sk]
    return softmax(scores) @ vf  # [..., Sq, D]


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Scaled dot-product attention with equal-length Q/K/V."""
    return _sdpa(q, k, v)


def kv_cache_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """SDPA where the query seq-len (Sq) differs from the cached K/V seq-len (Sk).

    No causal mask — at decode the query attends to the whole cache.
    """
    return _sdpa(q, k, v)


# --- registry + daemon protocol runner --------------------------------------

REGISTRY = {
    "silu": silu,
    "gelu": gelu,
    "softmax": softmax,
    "rmsnorm": rms_norm,
    "rope": rope,
    "matmul_silu": matmul_silu,
    "matmul_gelu": matmul_gelu,
    "attention": attention,
    "kv_cache_attention": kv_cache_attention,
}


def _decode_tensor(t: dict) -> np.ndarray:
    arr = np.frombuffer(base64.b64decode(t["data"]), dtype=np.dtype(t["dtype"]))
    return arr.reshape(t["shape"])


def _encode_tensor(arr: np.ndarray) -> dict:
    arr = np.ascontiguousarray(arr)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data": base64.b64encode(arr.tobytes()).decode("utf-8"),
    }


def run(fn, stdin=None, stdout=None) -> None:
    """Serve ``fn`` over the daemon's stdin/stdout JSON protocol.

    Reads ``{"inputs": {...}, "kwargs": {...}}``, decodes inputs to numpy,
    calls ``fn(**inputs)`` and writes the encoded result. ``kwargs`` are not
    forwarded (the reference defaults — eps, theta — are the calibrated ones).
    """
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    payload = json.load(stdin)
    inputs = {name: _decode_tensor(t) for name, t in payload["inputs"].items()}
    result = fn(**inputs)
    json.dump(_encode_tensor(np.asarray(result)), stdout)
