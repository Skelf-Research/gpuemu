# Developer Guide

This guide covers the gpuemu developer lifecycle for three personas, from high-level ML code to low-level kernel development.

## Quick Start

```bash
# Install gpuemu (Rust binary)
cargo install gpuemu

# Or download pre-built binary
curl -sSL https://gpuemu.dev/install.sh | sh

# Install Python client
pip install gpuemu-py

# Start the daemon
gpuemu daemon start

# Initialize project
gpuemu init
```

## Personas

| Persona | What they do | What gpuemu provides |
|---------|--------------|---------------------|
| **Model Developer** | Write PyTorch/JAX/TF training code | CPU validation, numerical checks, GPU-less CI |
| **Custom Op Integrator** | Use third-party kernels (FlashAttention, xFormers) | Op validation, equivalence testing, regression detection |
| **Kernel Author** | Write CUDA/HIP kernels | CPU mirrors, artifact inspection, correctness fuzzing |

---

## Model Developer Lifecycle

You write PyTorch/JAX/TF code and want to validate it works correctly before GPU deployment.

### The Problem

```python
# You wrote this model
model = TransformerBlock(dim=512, heads=8)
output = model(input_tensor)

# Questions you can't answer without a GPU:
# - Does this produce correct outputs?
# - Will fp16 training be numerically stable?
# - Do all shapes work, or just the ones I tested?
```

### Day 1: Setup

```bash
# Start gpuemu daemon
gpuemu daemon start

# Initialize in your project
gpuemu init
```

This creates `gpuemu.toml`:

```toml
[project]
name = "my-model"
framework = "pytorch"

[validation]
dtypes = ["float32", "float16", "bfloat16"]
check_nan = true
check_inf = true

[validation.tolerances]
float32 = 1e-5
float16 = 1e-3
bfloat16 = 1e-3
```

### Daily Loop: Write → Validate → Fix

```python
# test_model.py
import torch
from gpuemu_py import Client, validate, fuzz_shapes

# Connect to daemon
client = Client()

def test_transformer_correctness():
    model = TransformerBlock(dim=512, heads=8)

    # Validate against CPU reference
    with validate(client, model, reference="cpu"):
        input = torch.randn(2, 128, 512)
        output = model(input)
        # gpuemu checks: no NaN, no Inf, shapes match, dtypes correct

def test_transformer_shapes():
    model = TransformerBlock(dim=512, heads=8)

    # Fuzz with random shapes
    for batch, seq in fuzz_shapes(batch=[1, 2, 7, 32], seq=[1, 64, 127, 512]):
        with validate(client, model):
            input = torch.randn(batch, seq, 512)
            output = model(input)
            assert output.shape == (batch, seq, 512)

def test_transformer_precision():
    model = TransformerBlock(dim=512, heads=8)

    # Test fp16 numerical stability
    with validate(client, model, dtype="float16", reference_dtype="float32"):
        input = torch.randn(2, 128, 512, dtype=torch.float16)
        output = model(input)
        # gpuemu compares fp16 output against fp32 reference within tolerance
```

### Run locally

```bash
# Fast check (single dtype, small shapes)
gpuemu test --quick

# Full validation (all dtypes, fuzzed shapes)
gpuemu test

# Check daemon status
gpuemu status
```

### CI Integration

```yaml
# .github/workflows/validate.yml
name: GPU-less Validation
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest  # No GPU needed
    steps:
      - uses: actions/checkout@v4

      - name: Install gpuemu
        run: |
          curl -sSL https://gpuemu.dev/install.sh | sh
          pip install gpuemu-py

      - name: Start daemon
        run: gpuemu daemon start --background

      - name: Run validation
        run: gpuemu ci

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: gpuemu-results
          path: .gpuemu/results/
```

### When a test fails

```
FAILED test_transformer_precision
  Input shape: (7, 127, 512)
  Dtype: float16
  Issue: NaN detected in output
  Seed: 42

  To reproduce:
    gpuemu reproduce --seed 42 --test test_transformer_precision
```

```bash
# Reproduce locally
gpuemu reproduce --seed 42 --test test_transformer_precision

# View stored result details
gpuemu show --seed 42

# Debug interactively
gpuemu debug --seed 42
```

---

## Custom Op Integrator Lifecycle

You're adding FlashAttention, xFormers, or other custom ops to your stack.

### The Problem

```python
# You want to use FlashAttention
from flash_attn import flash_attn_func

# But you can't test it without a GPU
# And you don't know if it handles your edge cases
output = flash_attn_func(q, k, v)
```

### Setup: Register the custom op in TOML

```toml
# gpuemu.toml
[[ops]]
name = "flash_attn"
module = "flash_attn.flash_attn_func"
reference = "scripts/ref_flash_attn.py"

[ops.tolerances]
float16 = 1e-2
bfloat16 = 1e-2
float32 = 1e-4
```

Create the reference implementation:

```python
# scripts/ref_flash_attn.py
"""CPU reference for FlashAttention validation."""
import sys
import torch
import numpy as np

def reference(q, k, v, causal=False):
    """Standard scaled dot-product attention."""
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)

if __name__ == "__main__":
    # gpuemu invokes this script with serialized inputs
    import pickle
    inputs = pickle.load(sys.stdin.buffer)
    result = reference(**inputs)
    pickle.dump(result, sys.stdout.buffer)
```

### Daily Loop: Validate custom ops

```python
# test_flash_attn.py
import torch
from gpuemu_py import Client, validate_op, fuzz_shapes

client = Client()

def test_flash_attn_correctness():
    """Compare FlashAttention against CPU reference."""
    from flash_attn import flash_attn_func

    for batch, heads, seq, dim in fuzz_shapes(
        batch=[1, 4], heads=[8, 16], seq=[64, 128, 513], dim=[64]
    ):
        q = torch.randn(batch, seq, heads, dim)
        k = torch.randn(batch, seq, heads, dim)
        v = torch.randn(batch, seq, heads, dim)

        with validate_op(client, "flash_attn", inputs={"q": q, "k": k, "v": v}):
            output = flash_attn_func(q, k, v)
            # gpuemu runs reference script and compares

def test_flash_attn_causal():
    """Validate causal masking behavior."""
    q = torch.randn(2, 128, 8, 64)
    k = torch.randn(2, 128, 8, 64)
    v = torch.randn(2, 128, 8, 64)

    with validate_op(client, "flash_attn", inputs={"q": q, "k": k, "v": v, "causal": True}):
        output = flash_attn_func(q, k, v, causal=True)
```

### Regression detection

```bash
# Baseline current behavior
gpuemu baseline --tag v1.0

# After updating flash-attn
pip install flash-attn==2.5.0

# Check for regressions (compares against sled-stored baseline)
gpuemu diff --baseline v1.0
```

Output:
```
flash_attn:
  Numerical diff: max 1.2e-3 -> 2.1e-3 (WARN: increased)
  New failure: seq=513, causal=True (was passing)

Action: Review flash-attn 2.5.0 changes or pin to 2.4.x
```

---

## Kernel Author Lifecycle

You write CUDA kernels and need to validate correctness without a GPU.

### The Problem

```cpp
// You wrote this kernel
__global__ void fused_add_relu(float* out, float* a, float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = max(0.0f, a[idx] + b[idx]);
    }
}

// Questions:
// - Does this handle non-aligned sizes?
// - What about strided tensors?
// - Is the boundary condition correct?
```

### Kernel Contract Pattern

Structure your kernel so the math is testable on CPU:

```cpp
// kernel_math.cuh - The testable core
template<typename T>
__host__ __device__ inline T fused_add_relu_elem(T a, T b) {
    return max(T(0), a + b);
}

// kernel_launch.cu - GPU-specific launch
__global__ void fused_add_relu_kernel(float* out, float* a, float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fused_add_relu_elem(a[idx], b[idx]);
    }
}

// kernel_cpu.cpp - CPU mirror for testing
void fused_add_relu_cpu(float* out, float* a, float* b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = fused_add_relu_elem(a[i], b[i]);
    }
}
```

### Register kernel in TOML

```toml
# gpuemu.toml
[[kernels]]
name = "fused_add_relu"
source = "kernels/fused_add_relu.cu"
reference = "scripts/ref_fused_add_relu.py"

[kernels.tolerances]
float32 = 1e-5
float16 = 1e-3

[kernels.invariants]
non_negative = true
shape_preserved = true

[kernels.artifact_checks]
max_registers = 32
max_spills = 0
required_patterns = ["FADD", "FMAX"]
```

Reference implementation:

```python
# scripts/ref_fused_add_relu.py
import numpy as np
import sys
import pickle

def reference(a, b):
    return np.maximum(0, a + b)

if __name__ == "__main__":
    inputs = pickle.load(sys.stdin.buffer)
    result = reference(**inputs)
    pickle.dump(result, sys.stdout.buffer)
```

### Daily Loop: Test → Inspect → Fix

```bash
# Run CPU mirror tests
gpuemu test --kernel fused_add_relu

# Fuzz with edge-case shapes
gpuemu fuzz --kernel fused_add_relu \
  --shapes "1,7,127,1024,1048577" \
  --layouts "contiguous,strided,transposed"

# Inspect compiled artifacts (requires CUDA toolkit, not GPU)
gpuemu lint --kernel fused_add_relu
```

Lint output:
```
fused_add_relu.ptx:
  Registers: 16 (OK, threshold: 32)
  Spills: 0 (OK)
  Patterns: FADD found, FMAX found (OK)

  WARN: No bounds check for idx >= n in some paths
  INFO: Consider using __ldg for read-only inputs
```

### Debugging a failure

```
FAILED test_fused_add_relu_strided
  Shape: (128, 256)
  Layout: strided (strides=[512, 1])
  Issue: Output mismatch at index [7, 128]
  Expected: 1.234
  Got: 0.0
  Seed: 12345
```

```bash
# Reproduce with exact inputs
gpuemu reproduce --seed 12345 --kernel fused_add_relu

# View stored failure details
gpuemu show --seed 12345

# Get a minimal reproducer
gpuemu minimize --seed 12345 --kernel fused_add_relu
# Output: Minimal failing case is shape=(8, 129), strides=(512, 1)
```

---

## Daemon Management

```bash
# Start daemon (foreground)
gpuemu daemon start

# Start daemon (background, for CI)
gpuemu daemon start --background

# Check status
gpuemu daemon status

# View logs
gpuemu daemon logs

# Stop daemon
gpuemu daemon stop

# Data location
ls ~/.gpuemu/
#   db/           # sled database (results, baselines, artifacts)
#   gpuemu.sock   # nng socket
#   logs/         # daemon logs
```

## Configuration Reference

See `docs/CONFIGURATION.md` for full TOML schema.

## Common Patterns

### Quick vs Full Validation

```bash
# Quick (seconds): Single dtype, small shapes, no fuzzing
gpuemu test --quick

# Standard (minutes): All dtypes, representative shapes
gpuemu test

# Full (longer): Exhaustive fuzzing, all edge cases
gpuemu test --thorough

# CI default
gpuemu ci  # Equivalent to: build + test + lint + report
```

### Seed management

```bash
# Run with specific seed (deterministic)
gpuemu test --seed 42

# View all stored seeds
gpuemu seeds list

# Export seeds for sharing
gpuemu seeds export > seeds.json

# Import seeds
gpuemu seeds import < seeds.json
```

---

## Transitioning to GPU

When you finally have GPU access:

```bash
# Run same tests on GPU
gpuemu test --device cuda

# Compare CPU vs GPU outputs
gpuemu compare --cpu --gpu

# Full equivalence check
gpuemu equivalence --reference cpu --target cuda
```

Output:
```
CPU vs GPU Equivalence Report:

  test_transformer_correctness: PASS
    Max diff: 2.3e-6 (within float32 tolerance)

  test_flash_attn: PASS
    Max diff: 1.1e-3 (within float16 tolerance)

  test_fused_add_relu: FAIL
    Shape (1048577,): GPU output differs
    This may indicate a kernel bug with non-aligned sizes
```
