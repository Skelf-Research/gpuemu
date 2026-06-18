# Model Developer Guide

A guide for ML engineers who write PyTorch, JAX, or TensorFlow training code and want to validate custom ops without GPU access.

---

## Who This Is For

!!! info "Target audience"

    You are a **model developer** who:

    - Writes training pipelines using PyTorch, JAX, or TensorFlow
    - Uses custom ops (fused kernels, custom autograd functions, third-party extensions)
    - Wants to validate that those ops produce correct results **without needing a GPU**
    - Works in a team where GPU machines are shared or limited

    If you are integrating third-party kernels you did not write (FlashAttention, xFormers, etc.), see the [Custom Op Integrator Guide](custom-op-integrator.md). If you are writing CUDA kernels directly, see the [Kernel Author Guide](kernel-author.md).

---

## Day 1 Setup

### 1. Initialize your project

Run the init command with your framework of choice:

=== "PyTorch"

    ```bash
    gpuemu init --framework pytorch
    ```

=== "JAX"

    ```bash
    gpuemu init --framework jax
    ```

=== "TensorFlow"

    ```bash
    gpuemu init --framework tensorflow
    ```

This creates a `gpuemu.toml` configuration file in your project root with sensible defaults for your framework.

### 2. Start the daemon

The daemon is the background process that handles validation, fuzzing, and result storage:

```bash
gpuemu daemon start
```

!!! tip "Auto-start in VS Code"

    If you use the [VS Code extension](vscode-extension.md), the daemon starts automatically when your workspace contains a `gpuemu.toml` file. You can control this with the `gpuemu.autoStartDaemon` setting.

### 3. Register your first op

Open `gpuemu.toml` and add an op entry:

```toml
[[ops]]
name = "my_custom_op"
module = "my_module"
reference = "scripts/ref_my_custom_op.py"

[ops.tolerances]
float32 = 1e-5
float16 = 1e-3

[ops.invariants]
no_nan = true
no_inf = true
```

### 4. Write the reference script

Create `scripts/ref_my_custom_op.py`. This is a pure NumPy implementation of your op that gpuemu uses as ground truth:

```python
# scripts/ref_my_custom_op.py
import numpy as np

def reference(inputs: dict, **kwargs) -> np.ndarray:
    """Reference implementation of my_custom_op.

    Args:
        inputs: Dict mapping input names to numpy arrays.

    Returns:
        The expected output as a numpy array.
    """
    x = inputs["x"]
    # Your op's math in pure numpy
    return np.square(x) + 2.0 * x + 1.0
```

### 5. Install the Python client

```bash
pip install gpuemu[torch]   # or [jax] or [tensorflow]
```

---

## Daily Workflow

The core development loop with gpuemu is: **Write op, write reference, validate, fix, repeat.**

### The `validate_op()` context manager

The primary way to validate an op is the `validate_op()` context manager. It captures your op's output and sends it to the daemon for comparison against the reference:

=== "PyTorch"

    ```python
    import torch
    from gpuemu.client import Client
    from gpuemu.frameworks.pytorch import validate_pytorch

    client = Client()
    x = torch.randn(32, 128)

    with validate_pytorch(client, "my_custom_op", {"x": x}) as ctx:
        ctx["output"] = my_custom_op(x)

    # If the output does not match the reference, ValidationError is raised.
    # If it passes, execution continues normally.
    ```

=== "JAX"

    ```python
    import jax.numpy as jnp
    from gpuemu.client import Client
    from gpuemu.frameworks.jax import validate_jax

    client = Client()
    x = jnp.ones((32, 128))

    with validate_jax(client, "my_custom_op", {"x": x}) as ctx:
        ctx["output"] = my_custom_op(x)
    ```

=== "TensorFlow"

    ```python
    import tensorflow as tf
    from gpuemu.client import Client
    from gpuemu.frameworks.tensorflow import validate_tensorflow

    client = Client()
    x = tf.random.normal((32, 128))

    with validate_tensorflow(client, "my_custom_op", {"x": x}) as ctx:
        ctx["output"] = my_custom_op(x)
    ```

### The low-level `validate_op()` helper

For framework-agnostic usage or more control, use the generic `validate_op` context manager from `gpuemu.validate`:

```python
from gpuemu.client import Client
from gpuemu.validate import validate_op
import numpy as np

client = Client()
inputs = {"x": np.random.randn(32, 128).astype(np.float32)}

with validate_op(client, "my_custom_op", inputs=inputs) as ctx:
    # Run your op and store the result
    ctx["output"] = my_custom_op_numpy(inputs["x"])

# Access the result object after the context exits
result = ctx["result"]
print(f"Passed: {result.passed}, max_diff: {result.max_diff:.2e}")
```

### Validating gradients

For PyTorch ops with custom backward passes, pass `check_backward=True`:

```python
x = torch.randn(32, 128, requires_grad=True)

with validate_pytorch(client, "my_custom_op", {"x": x}, check_backward=True) as ctx:
    ctx["output"] = my_custom_op(x)
```

This validates both the forward output and the analytical gradients against numerical finite differences.

---

## Using Fuzzing

Manual tests with a single shape and dtype are not enough. Use fuzzing to test across diverse configurations and catch edge cases.

### Shape fuzzing with `fuzz_shapes()`

The `fuzz_shapes()` generator produces all combinations of the dimension values you provide:

```python
from gpuemu.validate import fuzz_shapes

for batch, seq in fuzz_shapes(batch=[1, 2, 4, 8, 16], seq=[64, 128, 256, 512]):
    x = torch.randn(batch, seq, 512)
    with validate_pytorch(client, "my_custom_op", {"x": x}) as ctx:
        ctx["output"] = my_custom_op(x)
```

### Dtype fuzzing with `fuzz_dtypes()`

The `fuzz_dtypes()` generator iterates over dtype strings:

```python
from gpuemu.validate import fuzz_dtypes

for dtype in fuzz_dtypes(["float32", "float16", "bfloat16"]):
    x = torch.randn(8, 256, dtype=getattr(torch, dtype))
    with validate_pytorch(client, "my_custom_op", {"x": x}) as ctx:
        ctx["output"] = my_custom_op(x)
```

### Combined shape + dtype fuzzing

```python
from gpuemu.validate import fuzz_shapes, fuzz_dtypes

for batch, seq in fuzz_shapes(batch=[1, 4, 16], seq=[64, 256]):
    for dtype in fuzz_dtypes(["float32", "float16"]):
        x = torch.randn(batch, seq, 512, dtype=getattr(torch, dtype))
        with validate_pytorch(client, "my_custom_op", {"x": x}) as ctx:
            ctx["output"] = my_custom_op(x)
```

### Drop-in fuzzing with `fuzz_op_client_side()`

For the simplest possible fuzzing integration, use `client.fuzz_op_client_side()`. The daemon generates random inputs, you provide a callable that runs your op, and gpuemu validates every output:

```python
from gpuemu.client import Client

client = Client()

results = client.fuzz_op_client_side(
    "my_custom_op",
    run_op=lambda inputs: my_custom_op(torch.from_numpy(inputs["x"])).numpy(),
    iterations=100,
    seed=42,
)

print(f"Passed: {results.passed}/{results.total}")
for failure in results.failures:
    print(f"  Seed {failure.seed}: {failure.failures[0]['message']}")
```

!!! tip "Deterministic seeds"

    Every fuzz iteration has a unique seed. If a test fails, record the seed. You can reproduce the exact same inputs later with `client.reproduce(seed)`.

### Daemon-side fuzzing

If you do not need client-side GPU execution (e.g., your op can run on CPU), you can let the daemon handle everything:

```python
results = client.fuzz_op(
    "my_custom_op",
    seed=42,
    iterations=200,
    fail_fast=False,
    dtypes=["float32", "float16"],
    layouts=["Contiguous", "Strided", "Transposed"],
)

print(f"Passed: {results.passed}/{results.total}, Failed: {results.failed}")
```

---

## Reproducing Failures

When a fuzz run or validation fails, every failure has a **seed** that fully determines the inputs, shape, dtype, and layout that triggered the failure.

### CLI reproduction

```bash
gpuemu reproduce 8374629105
```

This regenerates the exact inputs and re-runs validation, printing a detailed report of what went wrong.

### Python API reproduction

```python
repro = client.reproduce(8374629105)

print(f"Op: {repro.result.op_name}")
print(f"Shape: {repro.inputs['x'].shape}")
print(f"Dtype: {repro.inputs['x'].dtype}")
print(f"Max diff: {repro.result.max_diff:.2e}")

# You now have the exact input tensors to debug with
x = torch.from_numpy(repro.inputs["x"])
output = my_custom_op(x)
```

!!! note "Cross-language RNG"

    gpuemu uses a bit-for-bit identical xorshift128+ PRNG in both Rust and Python. Seeds are fully reproducible across the CLI, daemon, and Python client regardless of which component generated the original test case.

---

## CI Integration

For full CI pipeline setup, see the [CI Integration Tutorial](../tutorials/ci-integration.md). Here is the quick version.

### Add quick validation to PR checks

Add this to your CI script (GitHub Actions, GitLab CI, etc.):

```bash
gpuemu daemon start
gpuemu ci --quick
```

The `--quick` flag runs a reduced set of iterations per op for fast feedback. A typical `--quick` run completes in under 30 seconds.

### Example GitHub Actions step

```yaml
- name: Validate ops
  run: |
    gpuemu daemon start
    gpuemu ci --quick --output junit
```

!!! warning "Full fuzzing before merge"

    `--quick` is designed for fast PR feedback. Always run full fuzzing (`gpuemu fuzz --iterations 500`) before merging to main. Consider running the full suite as a nightly CI job.

---

## Tips and Best Practices

- [x] **Use `--quick` for fast iteration.** During development, run `gpuemu test --quick` or `gpuemu ci --quick` to get feedback in seconds. Save full fuzzing for pre-merge and nightly runs.

- [x] **Run full fuzzing before merge.** Shape and dtype edge cases (batch=1, float16, strided layouts) are where most bugs hide. A 500-iteration fuzz run catches issues that a handful of manual tests will miss.

- [x] **Store baselines for regression detection.** Before a major refactor or library upgrade, snapshot your current results:

    ```bash
    gpuemu baseline v1.0
    ```

    After the change, compare:

    ```bash
    gpuemu diff --baseline v1.0 --fail-on-regression
    ```

- [x] **Keep reference scripts simple.** Your `ref_*.py` scripts should be the simplest possible NumPy implementation of the operation. Avoid optimizations -- clarity is more important than speed in a reference.

- [x] **Use per-dtype tolerances.** `float16` and `bfloat16` need much wider tolerances than `float32`. Configure them separately in `gpuemu.toml`:

    ```toml
    [ops.tolerances]
    float32 = 1e-5
    float16 = 1e-3
    bfloat16 = 1e-2
    ```

- [x] **Enable invariant checks.** Beyond numerical tolerances, invariants catch structural bugs:

    ```toml
    [ops.invariants]
    no_nan = true
    no_inf = true
    shape_preserved = true
    ```

---

## Next Steps

- [Custom Op Integrator Guide](custom-op-integrator.md) -- Validating third-party kernels
- [Kernel Author Guide](kernel-author.md) -- Writing and testing CUDA kernels
- [VS Code Extension](vscode-extension.md) -- Editor integration with live diagnostics
- [CI Integration Tutorial](../tutorials/ci-integration.md) -- Full CI pipeline setup
- [Python API Reference](../reference/python-api.md) -- Complete client API documentation
