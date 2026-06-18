# PyTorch Validation Tutorial

Validate PyTorch custom ops end-to-end using gpuemu -- from single-shot validation to automated fuzzing, gradient checking, and failure reproduction.

---

## Prerequisites

Before you begin, make sure the following are in place:

- [x] **gpuemu CLI** installed and on your `PATH` ([Installation](../getting-started/installation.md))
- [x] **gpuemu daemon** running (`gpuemu daemon start --background`)
- [x] **Python 3.9+** with a virtual environment activated
- [ ] **gpuemu-py with PyTorch adapter** installed:

```bash
pip install ./gpuemu-py[torch]
```

!!! tip "Verify your setup"

    ```bash
    gpuemu daemon status          # Should show "running"
    python -c "import torch; import gpuemu; print('ready')"
    ```

---

## Setup

Initialize a new gpuemu project configured for PyTorch:

```bash
gpuemu init --name my-pytorch-ops --framework pytorch
```

This generates the following project structure:

```
my-pytorch-ops/
├── gpuemu.toml
└── scripts/
    └── .gitkeep
```

The generated `gpuemu.toml` includes PyTorch-specific defaults:

```toml title="gpuemu.toml"
[project]
name = "my-pytorch-ops"
version = "0.1.0"
framework = "pytorch"

[validation]
dtypes = ["float32", "float16", "bfloat16"]
check_nan = true
check_inf = true

[validation.tolerances]
float32 = { atol = 1e-5, rtol = 1e-5 }
float16 = { atol = 1e-2, rtol = 1e-2 }
bfloat16 = { atol = 1e-2, rtol = 1e-2 }

[[ops]]
name = "my_op"
module = "my_pytorch_ops.ops"
reference = "scripts/my_op_ref.py"
execution_mode = "script_based"
```

!!! info "PyTorch tolerance defaults"

    PyTorch uses CUDA internally, which can introduce small numerical differences compared to CPU execution. The default tolerances above are calibrated for typical PyTorch ops. You can tighten or loosen them per-op in the `[[ops]]` section.

---

## Write a Reference Script

A reference script computes the expected output using only NumPy. It communicates with the daemon via the JSON+base64 protocol over stdin/stdout.

```python title="scripts/matmul_ref.py"
"""Reference implementation for matrix multiplication."""
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


def main():
    request = json.loads(sys.stdin.read())

    a = decode_tensor(request["inputs"]["a"])
    b = decode_tensor(request["inputs"]["b"])

    result = np.matmul(a, b)

    response = {"outputs": {"result": encode_tensor(result)}}
    json.dump(response, sys.stdout)


if __name__ == "__main__":
    main()
```

!!! tip "Keep reference scripts pure"

    Reference scripts should be deterministic and side-effect-free. No GPU libraries, no network calls, no file I/O beyond stdin/stdout. This ensures they are portable and safe to run in any environment, including CPU-only CI runners.

Wire up the reference script in `gpuemu.toml`:

```toml title="gpuemu.toml (ops section)"
[[ops]]
name = "matmul"
module = "my_pytorch_ops.ops.matmul"
reference = "scripts/matmul_ref.py"
execution_mode = "script_based"

[ops.tolerances]
float32 = { atol = 1e-5, rtol = 1e-5 }
float16 = { atol = 1e-2, rtol = 1e-2 }
```

---

## Single-Shot Validation

The `validate_pytorch()` context manager is the primary way to validate a single PyTorch op invocation. It captures inputs, runs the reference, and compares outputs automatically.

```python title="validate_single.py"
import torch
from gpuemu import Client
from gpuemu.frameworks.pytorch import validate_pytorch

client = Client()

x = torch.randn(4, 64, requires_grad=True)

with validate_pytorch(client, "my_op", {"x": x}, check_backward=True) as ctx:
    ctx["output"] = my_op(x)
```

The context manager handles the following steps:

1. Converts `x` from a PyTorch tensor to the JSON+base64 wire format.
2. Sends the inputs to the daemon, which runs the reference script.
3. Compares `ctx["output"]` against the reference output using the configured tolerances.
4. When `check_backward=True`, also validates gradients by running backward on both the op output and the reference output.

!!! note "GPU to CPU transfer"

    The PyTorch adapter automatically handles GPU-to-CPU transfer. If your tensors are on a CUDA device, they are moved to CPU transparently before comparison. You do not need to call `.cpu()` manually.

=== "Basic validation"

    ```python
    with validate_pytorch(client, "matmul", {"a": a, "b": b}) as ctx:
        ctx["output"] = torch.matmul(a, b)
    ```

=== "With backward pass"

    ```python
    with validate_pytorch(client, "matmul", {"a": a, "b": b}, check_backward=True) as ctx:
        ctx["output"] = torch.matmul(a, b)
    ```

=== "Custom tolerances"

    ```python
    with validate_pytorch(
        client,
        "matmul",
        {"a": a, "b": b},
        atol=1e-4,
        rtol=1e-4,
    ) as ctx:
        ctx["output"] = torch.matmul(a, b)
    ```

---

## Client-Side Fuzzing

Use `fuzz_pytorch_op()` to automatically generate randomized inputs and stress-test your op across many shapes, dtypes, and value ranges.

```python title="fuzz_matmul.py"
from gpuemu import Client
from gpuemu.frameworks.pytorch import fuzz_pytorch_op

client = Client()


def my_matmul(inputs):
    """The op under test. Receives a dict of tensors, returns a dict of tensors."""
    import torch
    return {"result": torch.matmul(inputs["a"], inputs["b"])}


results = fuzz_pytorch_op(
    client,
    op_name="matmul",
    op_fn=my_matmul,
    iterations=100,
    check_backward=True,  # Also validate gradients on each iteration
)

print(f"Passed: {results.passed}, Failed: {results.failed}")
for failure in results.failures:
    print(f"  Seed {failure.seed}: {failure.message}")
```

Setting `check_backward=True` enables gradient validation on every fuzz iteration. This verifies that the backward pass of your op produces gradients consistent with the reference implementation.

---

## Drop-In Fuzzing

For the simplest possible fuzzing workflow, use `client.fuzz_op_client_side()`. This requires no separate op function -- it reads the op configuration directly from `gpuemu.toml`.

```python title="fuzz_drop_in.py"
from gpuemu import Client

client = Client()

results = client.fuzz_op_client_side(
    op_name="matmul",
    iterations=100,
)

print(f"Passed: {results.passed}, Failed: {results.failed}")
```

!!! tip "When to use which fuzzing method"

    | Method | Best for |
    |--------|----------|
    | `fuzz_pytorch_op()` | Full control over the op function and input generation |
    | `client.fuzz_op_client_side()` | Quick smoke testing using the config-defined op |

---

## Gradient Validation

gpuemu provides two specialized tools for validating gradients in PyTorch ops.

### Finite-Difference Gradient Checking

`check_autograd()` compares PyTorch's autograd gradients against a finite-difference approximation. This catches errors in custom backward implementations.

```python title="check_gradients.py"
from gpuemu.frameworks.pytorch import check_autograd

x = torch.randn(4, 64, requires_grad=True, dtype=torch.float64)

result = check_autograd(
    func=my_op,
    inputs=(x,),
    eps=1e-6,       # Finite-difference step size
    atol=1e-5,
    rtol=1e-3,
)

assert result.passed, f"Gradient check failed: {result.message}"
```

!!! warning "Use float64 for gradient checking"

    Finite-difference gradient checking requires high numerical precision. Always use `dtype=torch.float64` for the input tensors. Using `float32` will produce noisy results and false failures.

### Custom `autograd.Function` Validation

If your op implements a custom `torch.autograd.Function`, use `validate_custom_autograd_function()` for a comprehensive check of both the forward and backward passes.

```python title="validate_autograd_function.py"
from gpuemu.frameworks.pytorch import validate_custom_autograd_function


class MyCustomOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        sig = torch.sigmoid(x)
        return grad_output * (sig + x * sig * (1 - sig))


result = validate_custom_autograd_function(
    client,
    func_class=MyCustomOp,
    sample_inputs=(torch.randn(4, 64, requires_grad=True, dtype=torch.float64),),
    op_name="my_custom_op",
)

assert result.passed, f"Custom autograd validation failed: {result.message}"
```

This validates:

- [x] Forward pass matches the reference implementation
- [x] Backward pass matches finite-difference gradients
- [x] Saved tensors are correct
- [x] Double-backward (if applicable) is consistent

---

## Reproducing Failures

When a fuzz run discovers a failure, gpuemu records the seed that produced it. You can reproduce any failure deterministically.

```python title="reproduce_failure.py"
from gpuemu import Client

client = Client()

# Reproduce the exact inputs and result for a given seed
reproduction = client.reproduce(seed=123456)

print(f"Inputs: {reproduction.inputs}")
print(f"Expected: {reproduction.expected}")
print(f"Actual: {reproduction.actual}")
print(f"Max diff: {reproduction.max_diff}")
```

From the CLI:

```bash
gpuemu test --seed 123456
```

!!! info "Cross-language RNG"

    gpuemu uses a bit-for-bit identical xorshift128+ PRNG in both Rust and Python. This means a seed recorded by the CLI is reproducible in Python, and vice versa.

---

## Minimizing Failures

Once you have a failing seed, use `client.minimize()` to find a smaller input that still triggers the failure. This makes debugging significantly easier.

```python title="minimize_failure.py"
from gpuemu import Client

client = Client()

minimized = client.minimize(
    seed=123456,
    strategy="binary-search-dims",
)

print(f"Minimized shape: {minimized.inputs['x'].shape}")
print(f"Still fails: {not minimized.passed}")
print(f"Original shape: {minimized.original_shape}")
```

Available minimization strategies:

| Strategy | Description |
|----------|-------------|
| `binary-search-dims` | Binary search on each dimension independently to find the smallest failing shape |
| `shrink-values` | Attempt to simplify tensor values while preserving the failure |
| `shrink-all` | Combine dimension and value shrinking (slowest but most thorough) |

From the CLI:

```bash
gpuemu minimize --seed 123456 --strategy binary-search-dims
```

---

## Tips

!!! tip "PyTorch tolerance defaults"

    PyTorch's default `atol` and `rtol` for `torch.allclose()` are `1e-8` and `1e-5` respectively. gpuemu uses slightly looser defaults (`1e-5` / `1e-5` for float32) to account for differences between CPU reference execution and the actual op implementation. You can override these per-op in `gpuemu.toml` or per-call in the `validate_pytorch()` context manager.

!!! tip "GPU to CPU transfer is automatic"

    The PyTorch adapter detects whether tensors are on GPU and moves them to CPU transparently for comparison. You never need to manually call `.cpu()` or `.detach()` on tensors passed to gpuemu.

!!! warning "Gradient checking caveats"

    - **Non-differentiable ops**: Ops that use `torch.argmax`, `torch.where` (with discrete conditions), or other non-differentiable operations will fail gradient checks. Use `check_backward=False` for these.
    - **In-place operations**: In-place modifications to tensors (`x.add_(1)`) can break autograd tracking. Avoid in-place ops in code under gradient validation.
    - **Stochastic ops**: Ops involving dropout or other random behavior produce non-deterministic gradients. Seed the RNG or disable stochasticity during validation.

---

## Next Steps

- [JAX Validation Tutorial](jax-validation.md) -- Validate JAX custom primitives.
- [TensorFlow Validation Tutorial](tensorflow-validation.md) -- Validate TensorFlow custom ops.
- [CI Integration](ci-integration.md) -- Run gpuemu validations in your CI pipeline.
- [Configuration](../getting-started/configuration.md) -- Fine-tune tolerances, dtypes, and policies.
