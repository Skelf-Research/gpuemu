# TensorFlow Validation Tutorial

Validate TensorFlow custom ops end-to-end using gpuemu -- including GradientTape support, Keras layer validation, `@tf.function` compatibility, and XLA compilation checks.

---

## Prerequisites

Before you begin, make sure the following are in place:

- [x] **gpuemu CLI** installed and on your `PATH` ([Installation](../getting-started/installation.md))
- [x] **gpuemu daemon** running (`gpuemu daemon start --background`)
- [x] **Python 3.9+** with a virtual environment activated
- [ ] **gpuemu-py with TensorFlow adapter** installed:

```bash
pip install ./gpuemu-py[tensorflow]
```

!!! tip "Verify your setup"

    ```bash
    gpuemu daemon status          # Should show "running"
    python -c "import tensorflow as tf; import gpuemu_py; print('ready')"
    ```

---

## Setup

Initialize a new gpuemu project configured for TensorFlow:

```bash
gpuemu init --name my-tf-ops --framework tensorflow
```

This generates the following project structure:

```
my-tf-ops/
├── gpuemu.toml
└── scripts/
    └── .gitkeep
```

The generated `gpuemu.toml` includes TensorFlow-specific defaults:

```toml title="gpuemu.toml"
[project]
name = "my-tf-ops"
version = "0.1.0"
framework = "tensorflow"

[validation]
dtypes = ["float32", "float16"]
check_nan = true
check_inf = true

[validation.tolerances]
float32 = { atol = 1.5e-5, rtol = 1.5e-5 }
float16 = { atol = 1.5e-2, rtol = 1.5e-2 }

[[ops]]
name = "my_op"
module = "my_tf_ops.ops"
reference = "scripts/my_op_ref.py"
execution_mode = "script_based"
```

!!! info "Why 1.5x tolerance for TensorFlow?"

    TensorFlow can compile operations through XLA, which applies operator fusion and layout transformations that change the order of floating-point operations. The default tolerances are set to 1.5x the baseline to account for these differences compared to a plain NumPy reference.

---

## Reference Script

Write a NumPy-based reference implementation using the standard JSON+base64 protocol over stdin/stdout.

```python title="scripts/gelu_ref.py"
"""Reference implementation for GELU activation."""
import json
import base64
import sys

import numpy as np
from scipy.special import erf


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


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))."""
    return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def main():
    request = json.loads(sys.stdin.read())

    x = decode_tensor(request["inputs"]["x"])
    result = gelu(x)

    response = {"outputs": {"result": encode_tensor(result)}}
    json.dump(response, sys.stdout)


if __name__ == "__main__":
    main()
```

Wire up the reference in `gpuemu.toml`:

```toml title="gpuemu.toml (ops section)"
[[ops]]
name = "gelu"
module = "my_tf_ops.ops.gelu"
reference = "scripts/gelu_ref.py"
execution_mode = "script_based"

[ops.tolerances]
float32 = { atol = 1.5e-5, rtol = 1.5e-5 }
float16 = { atol = 1.5e-2, rtol = 1.5e-2 }
```

---

## Single-Shot Validation

The `validate_tensorflow()` context manager validates a single TensorFlow op invocation against its reference. It integrates with `tf.GradientTape` for gradient validation.

```python title="validate_single.py"
import tensorflow as tf
from gpuemu_py import Client
from gpuemu_py.frameworks.tensorflow import validate_tensorflow

client = Client()

x = tf.Variable(tf.random.normal([4, 64]))

with validate_tensorflow(client, "my_op", {"x": x}, check_gradient=True) as ctx:
    with ctx["tape"]:
        ctx["output"] = my_op(x)
```

The context manager handles the following steps:

1. Creates a `tf.GradientTape` (accessible via `ctx["tape"]`) when `check_gradient=True`.
2. Converts TensorFlow tensors to the JSON+base64 wire format.
3. Sends inputs to the daemon, which runs the reference script.
4. Compares `ctx["output"]` against the reference using configured tolerances.
5. When `check_gradient=True`, computes and compares gradients from both the op and the reference.

!!! warning "Use `tf.Variable` for gradient tracking"

    TensorFlow's `GradientTape` only tracks gradients for `tf.Variable` objects by default. If you pass a plain `tf.Tensor`, gradients will not be computed and `check_gradient=True` will report an error. Always wrap inputs in `tf.Variable` when gradient validation is needed.

=== "Basic validation"

    ```python
    x = tf.constant(tf.random.normal([4, 64]))

    with validate_tensorflow(client, "gelu", {"x": x}) as ctx:
        ctx["output"] = tf.nn.gelu(x)
    ```

=== "With gradient checking"

    ```python
    x = tf.Variable(tf.random.normal([4, 64]))

    with validate_tensorflow(client, "gelu", {"x": x}, check_gradient=True) as ctx:
        with ctx["tape"]:
            ctx["output"] = tf.nn.gelu(x)
    ```

=== "Custom tolerances"

    ```python
    x = tf.constant(tf.random.normal([4, 64]))

    with validate_tensorflow(
        client,
        "gelu",
        {"x": x},
        atol=1e-4,
        rtol=1e-4,
    ) as ctx:
        ctx["output"] = tf.nn.gelu(x)
    ```

---

## TensorFlow-Specific Checks

TensorFlow operations need to work correctly across several execution contexts: eager mode, `@tf.function`, XLA compilation, and within Keras layers. gpuemu provides dedicated checks for each.

### Keras Layer Validation

`check_keras_layer()` validates that a Keras layer produces correct outputs across different input shapes and batch sizes, including during training and inference modes.

```python title="check_keras.py"
from gpuemu_py.frameworks.tensorflow import check_keras_layer


class MyGeluLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.nn.gelu(inputs)


result = check_keras_layer(
    client,
    layer=MyGeluLayer(),
    sample_input=tf.random.normal([4, 64]),
    op_name="gelu",
    check_training_mode=True,   # Validate both training=True and training=False
)

assert result.passed, f"Keras layer check failed: {result.message}"
```

### `@tf.function` Compatibility

`check_tf_function_safe()` verifies that your op produces the same results when run eagerly versus under `@tf.function`. Differences indicate reliance on Python-level side effects or tracing-time values.

```python title="check_tf_function.py"
from gpuemu_py.frameworks.tensorflow import check_tf_function_safe

result = check_tf_function_safe(
    func=my_custom_gelu,
    sample_inputs={"x": tf.random.normal([4, 64])},
    atol=1e-6,
)

assert result.passed, f"@tf.function check failed: {result.message}"
```

### XLA Compilation Compatibility

`check_xla_compatible()` verifies that your op compiles and runs correctly under XLA via `tf.function(jit_compile=True)`.

```python title="check_xla.py"
from gpuemu_py.frameworks.tensorflow import check_xla_compatible

result = check_xla_compatible(
    func=my_custom_gelu,
    sample_inputs={"x": tf.random.normal([4, 64])},
    atol=1.5e-5,
)

assert result.passed, f"XLA compatibility check failed: {result.message}"
```

### Custom Gradient Validation

`validate_custom_gradient()` validates ops that use `@tf.custom_gradient` to define custom gradient functions.

```python title="validate_custom_grad.py"
from gpuemu_py.frameworks.tensorflow import validate_custom_gradient


@tf.custom_gradient
def my_gelu_with_custom_grad(x):
    output = x * 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))

    def grad(dy):
        cdf = 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))
        pdf = tf.exp(-0.5 * x ** 2) / tf.sqrt(2.0 * np.pi)
        return dy * (cdf + x * pdf)

    return output, grad


result = validate_custom_gradient(
    client,
    func=my_gelu_with_custom_grad,
    sample_inputs=(tf.Variable(tf.random.normal([4, 64], dtype=tf.float64)),),
    op_name="gelu_custom_grad",
)

assert result.passed, f"Custom gradient validation failed: {result.message}"
```

This validates:

- [x] Forward pass matches the reference implementation
- [x] Custom gradient matches finite-difference approximation
- [x] Gradient is correct under `@tf.function`
- [x] No NaN or Inf values in gradients

!!! note "All four checks at a glance"

    | Check | What it validates | Common failure causes |
    |-------|-------------------|-----------------------|
    | `check_keras_layer()` | Correct output in Keras layer context | Training/inference mode differences, state issues |
    | `check_tf_function_safe()` | Eager vs `@tf.function` produce same results | Python side effects during tracing |
    | `check_xla_compatible()` | Correct output under XLA compilation | Unsupported ops, dynamic shapes |
    | `validate_custom_gradient()` | Custom gradient correctness | Incorrect gradient formula, numerical instability |

---

## Fuzzing

Use `fuzz_tensorflow_op()` to stress-test your op with randomized inputs, optionally including gradient and XLA checks on every iteration.

```python title="fuzz_gelu.py"
from gpuemu_py import Client
from gpuemu_py.frameworks.tensorflow import fuzz_tensorflow_op

client = Client()


def my_gelu(inputs):
    """The op under test."""
    import tensorflow as tf
    x = inputs["x"]
    return {"result": tf.nn.gelu(x)}


results = fuzz_tensorflow_op(
    client,
    op_name="gelu",
    op_fn=my_gelu,
    iterations=100,
    check_gradient=True,   # Validate gradients on each iteration
    check_xla=True,        # Validate XLA compatibility on each iteration
)

print(f"Passed: {results.passed}, Failed: {results.failed}")
for failure in results.failures:
    print(f"  Seed {failure.seed}: {failure.message}")
```

!!! tip "Fuzzing flags"

    | Flag | Effect |
    |------|--------|
    | `check_gradient=True` | Verifies gradient correctness on each fuzz iteration |
    | `check_xla=True` | Verifies XLA compilation compatibility on each iteration |
    | Both enabled | Catches both numerical and compilation-related bugs |

    Enabling both flags increases the time per iteration but provides much stronger coverage.

---

## Tips

!!! tip "Use `tf.Variable` for gradient tracking"

    TensorFlow's `GradientTape` only watches `tf.Variable` by default. When using `check_gradient=True`, always wrap your inputs in `tf.Variable`:

    ```python
    # Correct - gradients will be tracked
    x = tf.Variable(tf.random.normal([4, 64]))

    # Incorrect - gradients will NOT be tracked
    x = tf.random.normal([4, 64])
    ```

    You can also manually watch a tensor with `tape.watch(x)`, but using `tf.Variable` is the recommended approach with gpuemu.

!!! tip "TensorFlow tolerance is 1.5x due to XLA"

    Like JAX, TensorFlow can compile through XLA, which reorders floating-point operations. The default gpuemu tolerances for TensorFlow are set 1.5x higher than the baseline to account for this. If you are not using XLA (`jit_compile=False`), you may be able to tighten the tolerances.

!!! warning "Dynamic shapes and XLA"

    XLA requires static shapes at compile time. If your op uses dynamic shapes (e.g., `tf.boolean_mask`, ragged tensors), the `check_xla_compatible()` check will fail. This is expected behavior -- mark these ops with `xla_compatible = false` in `gpuemu.toml` to skip the XLA check:

    ```toml
    [[ops]]
    name = "my_dynamic_op"
    xla_compatible = false
    ```

!!! info "TensorFlow eager vs graph mode"

    gpuemu runs all validations in eager mode by default. The `check_tf_function_safe()` function explicitly tests graph mode by wrapping your op in `@tf.function` and comparing results. If you need to validate graph-only behavior, use this check.

---

## Next Steps

- [PyTorch Validation Tutorial](pytorch-validation.md) -- Validate PyTorch custom ops.
- [JAX Validation Tutorial](jax-validation.md) -- Validate JAX custom primitives.
- [CI Integration](ci-integration.md) -- Run gpuemu validations in your CI pipeline.
- [Configuration](../getting-started/configuration.md) -- Fine-tune tolerances, dtypes, and policies.
