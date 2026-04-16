# Quick Start

Get your first gpuemu validation running in under 5 minutes. This guide assumes you have already completed the [Installation](installation.md) steps.

---

## 1. Initialize a Project

Create a new gpuemu project using the CLI:

```bash
gpuemu init --name my-project --framework pytorch
```

This creates the following structure in your current directory:

```
my-project/
├── gpuemu.toml          # Project configuration
└── scripts/
    └── .gitkeep         # Directory for reference scripts
```

!!! info "What is `gpuemu.toml`?"

    The `gpuemu.toml` file is the central configuration for your project. It defines which ops to validate, their reference implementations, tolerance thresholds, and more. See the [Configuration](configuration.md) guide for full details.

The `--framework` flag pre-populates the config with sensible defaults for the specified framework (PyTorch, JAX, or TensorFlow).

---

## 2. Start the Daemon

The gpuemu daemon is a background process that handles validation, fuzzing, storage, and IPC:

```bash
gpuemu daemon start --background
```

!!! note "Daemon socket"

    The daemon listens on a Unix domain socket at `~/.gpuemu/gpuemu.sock`. All communication between the CLI, the Python client, and the VS Code extension goes through this socket. You can check daemon status at any time with:

    ```bash
    gpuemu daemon status
    ```

!!! tip "Auto-start in development"

    If the daemon is not running when you issue a CLI command, gpuemu will prompt you to start it. For a smoother workflow, consider adding `gpuemu daemon start --background` to your shell startup or project setup script.

---

## 3. Write a Reference Script

A **reference script** is a standalone Python program that computes the expected output for an operation. It reads JSON with base64-encoded tensor data from stdin and writes JSON with base64-encoded results to stdout.

Create a file at `scripts/matmul_ref.py`:

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

    Reference scripts should be deterministic and side-effect-free. They receive inputs via stdin and return outputs via stdout. No GPU libraries, no network calls, no file I/O beyond stdin/stdout. This makes them portable, reproducible, and safe to run anywhere.

---

## 4. Configure an Op

Open `gpuemu.toml` and add (or edit) an `[[ops]]` entry to wire up your reference script:

```toml title="gpuemu.toml"
[project]
name = "my-project"
version = "0.1.0"
framework = "pytorch"

[validation]
dtypes = ["float32", "float16"]
check_nan = true
check_inf = true

[[ops]]
name = "matmul"
module = "my_project.ops.matmul"
reference = "scripts/matmul_ref.py"
execution_mode = "script_based"

[ops.tolerances]
float32 = { atol = 1e-5, rtol = 1e-5 }
float16 = { atol = 1e-2, rtol = 1e-2 }
```

!!! info "Execution modes"

    - **`script_based`** -- The daemon spawns the reference script as a subprocess. Best for getting started.
    - **`client_side`** -- The Python client computes the reference inline. Best for tight integration with framework code.
    - **`daemon_orchestrated`** -- The daemon manages both execution and comparison. Best for CI and batch runs.

    See [Execution Modes](../concepts/execution-modes.md) for a full comparison.

---

## 5. Run Validation

Run the full test suite:

```bash
gpuemu test
```

You should see output similar to:

```
 Running 2 validations for op "matmul"...
  PASS  matmul (float32) — max diff: 2.38e-07
  PASS  matmul (float16) — max diff: 4.88e-04
 2 passed, 0 failed
```

For a faster feedback loop during development, use the quick mode which tests only the most common dtype:

```bash
gpuemu test --quick
```

!!! tip "Seed-based reproduction"

    Every test run uses a deterministic seed. If a test fails, the output includes the seed so you can reproduce it exactly:

    ```bash
    gpuemu test --seed 42
    ```

---

## 6. Try Fuzzing

Fuzzing automatically generates randomized tensor shapes, batch sizes, and values to stress-test your op:

```bash
gpuemu fuzz --op matmul --iterations 50
```

Example output:

```
 Fuzzing op "matmul" with 50 iterations...
 [██████████████████████████████████████████████████] 50/50

Results:
  50 passed, 0 failed
  Shapes tested: (1,1)x(1,1) to (128,256)x(256,64)
  Max absolute error: 3.81e-06 (float32, iteration 37)
```

!!! warning "Fuzzing and tolerances"

    Fuzzing may exercise edge cases (very large or very small values, unusual shapes) that push beyond your configured tolerances. This is by design -- it helps you find the right tolerance boundaries. Adjust `[ops.tolerances]` in `gpuemu.toml` if you encounter false positives.

---

## 7. Check Results

If any tests or fuzz iterations fail, you can review them:

```bash
gpuemu failures
```

This displays a summary of all recorded failures, including the seed, input shapes, dtype, and the magnitude of the deviation. You can reproduce any specific failure:

```bash
gpuemu test --seed <seed-from-failure>
```

!!! note "Failure storage"

    Failures are stored in the daemon's embedded database at `~/.gpuemu/data/`. They persist across daemon restarts so you can investigate later.

---

## 8. Stop the Daemon

When you are done, stop the background daemon:

```bash
gpuemu daemon stop
```

To confirm it has stopped:

```bash
gpuemu daemon status
```

---

## What's Next?

Now that you have run your first validation, here are the recommended next steps:

- **[Configuration](configuration.md)** -- Learn how to fine-tune tolerances, add invariants, and configure CI policies.
- **[PyTorch Validation Tutorial](../tutorials/pytorch-validation.md)** -- Validate a real PyTorch custom op end-to-end.
- **[JAX Validation Tutorial](../tutorials/jax-validation.md)** -- Integrate gpuemu with JAX custom primitives.
- **[TensorFlow Validation Tutorial](../tutorials/tensorflow-validation.md)** -- Validate TensorFlow custom ops.
- **[Execution Modes](../concepts/execution-modes.md)** -- Understand the three ways to run reference implementations.
- **[CI Integration Tutorial](../tutorials/ci-integration.md)** -- Set up gpuemu in GitHub Actions or GitLab CI.
