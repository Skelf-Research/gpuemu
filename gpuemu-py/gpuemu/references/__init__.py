"""Canonical fp64 reference oracles for gpuemu's built-in op schemas.

Use these as the ``reference`` for an op in ``gpuemu.toml``. Two ways:

1. Point at a shipped per-op script (no code to write)::

       [[ops]]
       name = "rmsnorm"
       reference = ".../site-packages/gpuemu/references/scripts/rmsnorm.py"

   (find the path with ``python -c "import gpuemu.references, os;
   print(os.path.dirname(gpuemu.references.__file__))"``)

2. Import the function in your own reference script::

       from gpuemu.references.ops import run, rms_norm
       run(rms_norm)

The fp64 math lives in :mod:`gpuemu.references.ops`; pair it with
``[validation] oracle_fp64 = true`` so inputs reach the reference in fp64.
"""

from .ops import (  # noqa: F401
    REGISTRY,
    attention,
    gelu,
    kv_cache_attention,
    matmul_gelu,
    matmul_silu,
    rms_norm,
    rope,
    run,
    silu,
    softmax,
)

__all__ = [
    "REGISTRY",
    "run",
    "silu",
    "gelu",
    "softmax",
    "rms_norm",
    "rope",
    "matmul_silu",
    "matmul_gelu",
    "attention",
    "kv_cache_attention",
]
