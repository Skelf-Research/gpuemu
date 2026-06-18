# JAX Validation Tutorial

Validate JAX custom ops and primitives end-to-end using gpuemu -- including JIT safety, vmap compatibility, and gradient correctness.

---

## Prerequisites

Before you begin, make sure the following are in place:

- [x] **gpuemu CLI** installed and on your `PATH` ([Installation](../getting-started/installation.md))
- [x] **gpuemu daemon** running (`gpuemu daemon start --background`)
- [x] **Python 3.9+** with a virtual environment activated
- [ ] **gpuemu-py with JAX adapter** installed:

```bash
pip install ./gpuemu-py[jax]
```

!!! tip "Verify your setup"

    ```bash
    gpuemu daemon status          # Should show "running"
    python -c "import jax; import gpuemu; print('ready')"
    ```

---

## Setup

Initialize a new gpuemu project configured for JAX:

```bash
gpuemu init --name my-jax-ops --framework jax
```

This generates the following project structure:

```
my-jax-ops/
├── gpuemu.toml
└── scripts/
    └── .gitkeep
```

The generated `gpuemu.toml` includes JAX-specific defaults:

```toml title="gpuemu.toml"
[project]
name = "my-jax-ops"
version = "0.1.0"
framework = "jax"

[validation]
dtypes = ["float32", "float16", "bfloat16"]
check_nan = true
check_inf = true

[validation.tolerances]
float32 = { atol = 1.5e-5, rtol = 1.5e-5 }
float16 = { atol = 1.5e-2, rtol = 1.5e-2 }
bfloat16 = { atol = 1.5e-2, rtol = 1.5e-2 }

[[ops]]
name = "my_op"
module = "my_jax_ops.ops"
reference = "scripts/my_op_ref.py"
execution_mode = "script_based"
```

!!! info "Why 1.5x tolerance for JAX?"

    JAX compiles operations through XLA, which applies aggressive optimizations such as operator fusion and layout transformations. These optimizations can change the order of floating-point operations, introducing small numerical differences compared to a plain NumPy reference. The default tolerances are set to 1.5x the PyTorch defaults to account for this.

---

## Reference Script

Write a NumPy-based reference implementation. The protocol is the same as for other frameworks -- JSON+base64 over stdin/stdout.

```python title="scripts/softmax_ref.py"
"""Reference implementation for softmax."""
import json
import base64
import sys

import numpy as np


def decode_tensor(encoded: dict) -> np.ndarray:
    """Decode a base64-encoded tensor from the input payload."""
    data = base64.b64decode(encoded["data"])
    dtype = np.dtype(encoded["dtype"])
    shape = tuple(encoded["shape"])
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def encode_tensor(arr: np.ndarray) -> dict:
    """Encode a numpy array as a base64 JSON-serializable dict."""
    return {
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def main():
    request = json.loads(sys.stdin.read())

    x = decode_tensor(request["inputs"]["x"])
    axis = request["inputs"].get("axis", -1)

    result = softmax(x, axis=axis)

    response = {"outputs": {"result": encode_tensor(result)}}
    json.dump(response, sys.stdout)


if __name__ == "__main__":
    main()
```

Wire up the reference in `gpuemu.toml`:

```toml title="gpuemu.toml (ops section)"
[[ops]]
name = "softmax"
module = "my_jax_ops.ops.softmax"
reference = "scripts/softmax_ref.py"
execution_mode = "script_based"

[ops.tolerances]
float32 = { atol = 1.5e-5, rtol = 1.5e-5 }
float16 = { atol = 1.5e-2, rtol = 1.5e-2 }
```

---

## Single-Shot Validation

The `validate_jax()` context manager validates a single JAX op invocation against its reference.

```python title="validate_single.py"
import jax
import jax.numpy as jnp
from gpuemu import Client
from gpuemu.frameworks.jax import validate_jax

client = Client()

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (4, 64))

with validate_jax(client, "softmax", {"x": x}) as ctx:
    ctx["output"] = jax.nn.softmax(x, axis=-1)
```

The context manager:

1. Converts JAX arrays to the JSON+base64 wire format.
2. Sends inputs to the daemon, which runs the reference script.
3. Compares `ctx["output"]` against the reference using configured tolerances.

!!! warning "Use `jnp` arrays, not NumPy arrays"

    Always pass `jax.numpy` arrays (not plain `numpy` arrays) to `validate_jax()`. The adapter relies on JAX array metadata for dtype handling and device transfer. Passing NumPy arrays may produce incorrect dtype conversions.

=== "Basic validation"

    ```python
    with validate_jax(client, "softmax", {"x": x}) as ctx:
        ctx["output"] = jax.nn.softmax(x)
    ```

=== "Custom tolerances"

    ```python
    with validate_jax(
        client,
        "softmax",
        {"x": x},
        atol=1e-4,
        rtol=1e-4,
    ) as ctx:
        ctx["output"] = jax.nn.softmax(x)
    ```

=== "With scalar parameters"

    ```python
    with validate_jax(
        client,
        "softmax",
        {"x": x, "axis": -1},
    ) as ctx:
        ctx["output"] = jax.nn.softmax(x, axis=-1)
    ```

---

## JAX-Specific Checks

JAX programs are expected to work correctly under several transformations: `jit`, `vmap`, `pmap`, and `grad`. gpuemu provides dedicated checks for each.

### JIT Safety

`check_jit_safe()` verifies that your op produces the same results when run eagerly versus under `jax.jit`. Differences indicate reliance on Python-level side effects or tracing-time values.

```python title="check_jit.py"
from gpuemu.frameworks.jax import check_jit_safe

result = check_jit_safe(
    func=my_custom_softmax,
    sample_inputs={"x": x},
    atol=1e-6,
)

assert result.passed, f"JIT safety check failed: {result.message}"
```

### vmap Compatibility

`check_vmap_compatible()` verifies that your op can be batched with `jax.vmap` and produces correct results across the batch dimension.

```python title="check_vmap.py"
from gpuemu.frameworks.jax import check_vmap_compatible

result = check_vmap_compatible(
    func=my_custom_softmax,
    sample_inputs={"x": x},       # x has shape (4, 64)
    vmap_axis=0,                   # Batch over the first dimension
)

assert result.passed, f"vmap check failed: {result.message}"
```

### pmap Compatibility

`check_pmap_compatible()` verifies that your op works correctly under `jax.pmap` for multi-device execution. On a single-device machine, this simulates multi-device behavior.

```python title="check_pmap.py"
from gpuemu.frameworks.jax import check_pmap_compatible

result = check_pmap_compatible(
    func=my_custom_softmax,
    sample_inputs={"x": x},
    num_devices=2,                 # Simulated device count
)

assert result.passed, f"pmap check failed: {result.message}"
```

### Gradient Safety

`check_grad_safe()` verifies that `jax.grad` can be computed for your op and that the resulting gradients are finite and consistent with a numerical approximation.

```python title="check_grad.py"
from gpuemu.frameworks.jax import check_grad_safe

# Use a scalar-output wrapper for grad (JAX requires scalar output)
def scalar_softmax(x):
    return jnp.sum(my_custom_softmax(x))

result = check_grad_safe(
    func=scalar_softmax,
    sample_inputs=(x,),
    eps=1e-4,
)

assert result.passed, f"Gradient check failed: {result.message}"
```

!!! note "All four checks at a glance"

    | Check | What it validates | Common failure causes |
    |-------|-------------------|-----------------------|
    | `check_jit_safe()` | Eager vs JIT produce same results | Python side effects during tracing |
    | `check_vmap_compatible()` | Correct batching under vmap | Hardcoded shapes, non-batched indexing |
    | `check_pmap_compatible()` | Multi-device correctness under pmap | Device-specific state, collective ops |
    | `check_grad_safe()` | Gradient correctness | Non-differentiable ops, numerical instability |

---

## Primitive Validation

If you have implemented a custom JAX primitive (using `jax.core.Primitive`), use `validate_jax_primitive()` for a comprehensive validation that covers the forward pass, JVP rule, and vmap rule in one call.

```python title="validate_primitive.py"
from gpuemu.frameworks.jax import validate_jax_primitive

result = validate_jax_primitive(
    client,
    primitive=my_custom_primitive,
    sample_inputs={"x": x},
    op_name="my_primitive",
    check_jvp=True,    # Validate the JVP (forward-mode AD) rule
    check_vmap=True,   # Validate the vmap batching rule
)

assert result.passed, f"Primitive validation failed: {result.message}"
```

This validates:

- [x] Forward evaluation matches the reference implementation
- [x] JVP rule produces correct tangent outputs (if `check_jvp=True`)
- [x] vmap batching rule correctly handles the batch dimension (if `check_vmap=True`)
- [x] Composed transformations (e.g., `jit(vmap(grad(...)))`) are consistent

---

## Fuzzing

Use `fuzz_jax_op()` to stress-test your op with randomized inputs across many shapes and dtypes, optionally including vmap and JIT checks on every iteration.

```python title="fuzz_softmax.py"
from gpuemu import Client
from gpuemu.frameworks.jax import fuzz_jax_op

client = Client()


def my_softmax(inputs):
    """The op under test."""
    import jax.numpy as jnp
    x = inputs["x"]
    x_max = jnp.max(x, axis=-1, keepdims=True)
    exp_x = jnp.exp(x - x_max)
    return {"result": exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)}


results = fuzz_jax_op(
    client,
    op_name="softmax",
    op_fn=my_softmax,
    iterations=100,
    check_vmap=True,   # Run vmap check on each iteration
    check_jit=True,    # Run JIT safety check on each iteration
)

print(f"Passed: {results.passed}, Failed: {results.failed}")
for failure in results.failures:
    print(f"  Seed {failure.seed}: {failure.message}")
```

!!! tip "Fuzzing flags"

    | Flag | Effect |
    |------|--------|
    | `check_vmap=True` | Verifies vmap compatibility on each fuzz iteration |
    | `check_jit=True` | Verifies JIT safety on each fuzz iteration |
    | Both enabled | Catches transformation-related bugs alongside numerical errors |

    Enabling both flags increases the time per iteration but provides much stronger coverage.

---

## Tips

!!! tip "JAX tolerance is 1.5x the PyTorch default"

    Due to XLA optimizations (operator fusion, layout changes, constant folding), JAX outputs can differ slightly more from a NumPy reference than PyTorch outputs do. The default gpuemu tolerances for JAX are set 1.5x higher to account for this. If you are seeing false positives, consider whether XLA is legitimately reordering operations.

!!! warning "Always use `jnp` arrays"

    The JAX adapter expects `jax.numpy` arrays. Passing plain NumPy arrays will bypass JAX's type system and may cause subtle dtype mismatches, especially with `bfloat16` which NumPy does not natively support.

!!! info "JAX tracing and side effects"

    JAX traces functions to build computation graphs. Operations that depend on concrete Python values at trace time (such as `if x > 0:` where `x` is a traced value) will fail under `jit`. The `check_jit_safe()` function specifically catches these issues by comparing eager and JIT-compiled results.

!!! tip "Reproducible keys"

    When writing tests, always use a fixed `jax.random.PRNGKey` to ensure reproducibility:

    ```python
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (4, 64))
    ```

    gpuemu's seed-based reproduction is separate from JAX's PRNG, but using fixed keys in your test scripts makes manual debugging easier.

---

## Next Steps

- [PyTorch Validation Tutorial](pytorch-validation.md) -- Validate PyTorch custom ops.
- [TensorFlow Validation Tutorial](tensorflow-validation.md) -- Validate TensorFlow custom ops.
- [CI Integration](ci-integration.md) -- Run gpuemu validations in your CI pipeline.
- [Configuration](../getting-started/configuration.md) -- Fine-tune tolerances, dtypes, and policies.
