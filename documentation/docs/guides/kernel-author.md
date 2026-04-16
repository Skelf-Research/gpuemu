# Kernel Author Guide

A guide for engineers writing custom CUDA or HIP kernels who want to validate correctness, inspect compiled artifacts, and detect regressions.

---

## Who This Is For

!!! info "Target audience"

    You are a **kernel author** who:

    - Writes custom CUDA or HIP kernels (`.cu`, `.cuh` files)
    - Needs to validate that your kernel math is correct across diverse shapes, dtypes, and memory layouts
    - Wants to inspect compiled artifacts (PTX/SASS) for register pressure, spills, and forbidden instruction patterns
    - Wants to detect numerical and artifact regressions when modifying kernels

    If you use pre-built third-party kernels, see the [Custom Op Integrator Guide](custom-op-integrator.md). If you write model training code, see the [Model Developer Guide](model-developer.md).

---

## Kernel Contract Pattern

The key insight for testable kernels is to **separate kernel math from launch configuration**. The same math can run on CPU (via a NumPy reference) and GPU (via your CUDA kernel).

### Structure

```
my_kernel/
  kernel.cu        # CUDA kernel: launch config + device code
  kernel_math.h    # Pure math: the computation, no CUDA-specific code
  ref_my_kernel.py # NumPy reference: same math in Python
```

### Example: a fused GELU kernel

The math (shared between CPU reference and GPU kernel):

```cpp
// kernel_math.h -- Pure computation, no __global__ or threadIdx
inline float gelu_forward(float x) {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float c = 0.7978845608f;  // sqrt(2/pi)
    const float k = 0.044715f;
    float inner = c * (x + k * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}
```

The CUDA kernel (launch configuration wrapping the math):

```cuda
// kernel.cu
#include "kernel_math.h"

__global__ void gelu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = gelu_forward(input[idx]);
    }
}
```

The NumPy reference (same math):

```python
# scripts/ref_my_gelu.py
import numpy as np

def reference(inputs: dict, **kwargs) -> np.ndarray:
    x = inputs["x"]
    c = np.sqrt(2.0 / np.pi)
    k = 0.044715
    inner = c * (x + k * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(inner))
```

!!! tip "Why separate the math?"

    When the math is isolated, you can test it independently of CUDA launch parameters, shared memory tiling, or warp-level primitives. Bugs in the math are caught by gpuemu. Bugs in the launch configuration show up as shape mismatches or crashes.

---

## Registering Kernels

Kernels are registered in the `[[kernels]]` section of `gpuemu.toml`. This is distinct from `[[ops]]` because kernels have additional properties for source paths, artifact checks, and compilation settings.

```toml
[[kernels]]
name = "fused_gelu"
source = "kernels/gelu_kernel.cu"
reference = "scripts/ref_my_gelu.py"

[kernels.tolerances]
float32 = 1e-5
float16 = 1e-3

[kernels.invariants]
no_nan = true
no_inf = true
non_negative = false
shape_preserved = true

[kernels.artifact_checks]
max_registers = 32
max_spills = 0
max_local_memory = 0
required_patterns = ["gelu_forward"]
forbidden_patterns = ["__syncthreads"]
```

### Configuration fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Unique kernel identifier |
| `source` | string | Path to the `.cu` or `.cuh` source file |
| `reference` | string | Path to the NumPy reference script |
| `tolerances` | table | Per-dtype tolerance overrides |
| `invariants` | table | Structural correctness checks |
| `artifact_checks` | table | PTX/SASS artifact inspection rules |

---

## Artifact Inspection

gpuemu can inspect compiled PTX or SASS artifacts to catch performance and correctness issues at the binary level, without executing on GPU hardware.

### Running artifact linting

```bash
gpuemu lint --kernel fused_gelu --ptx kernels/gelu_kernel.ptx
```

This analyzes the compiled artifact and reports:

- **Register pressure**: How many registers the kernel uses per thread
- **Spills**: Whether registers spill to local memory (a major performance issue)
- **Local memory usage**: Bytes of per-thread local memory allocated
- **Pattern matching**: Presence of required instructions, absence of forbidden ones

### Example lint output

```
Kernel: fused_gelu
  Registers:     28 / 32 (max)     OK
  Spills:         0 / 0  (max)     OK
  Local memory:   0 / 0  (max)     OK
  Required:       gelu_forward      FOUND
  Forbidden:      __syncthreads     NOT FOUND (good)

Result: PASS
```

---

## Artifact Checks Configuration

The `[kernels.artifact_checks]` table controls what gpuemu looks for in compiled artifacts.

### `max_registers`

Maximum number of registers per thread. Exceeding this limit reduces occupancy.

```toml
[kernels.artifact_checks]
max_registers = 64  # Default: 64
```

!!! note "Register pressure vs. occupancy"

    Lower register counts allow more thread blocks to run concurrently. The optimal value depends on your GPU architecture. For most kernels, 32-64 registers is a reasonable target. Compute-heavy kernels may need more.

### `max_spills`

Maximum number of register spills to local memory. Any spills indicate the kernel uses more registers than available, causing slow memory accesses.

```toml
[kernels.artifact_checks]
max_spills = 0  # Default: 0 (no spills allowed)
```

### `max_local_memory`

Maximum bytes of per-thread local memory. Local memory lives in global memory and is much slower than registers.

```toml
[kernels.artifact_checks]
max_local_memory = 0  # Default: 0
```

### `required_patterns`

List of strings that **must** appear in the PTX/SASS. Useful for verifying that specific instructions or function calls are present:

```toml
[kernels.artifact_checks]
required_patterns = [
    "fma.rn.f32",     # Fused multiply-add instruction
    "gelu_forward",   # Inlined math function
]
```

### `forbidden_patterns`

List of strings that **must not** appear in the PTX/SASS. Useful for catching unintended synchronization, slow instructions, or debug code:

```toml
[kernels.artifact_checks]
forbidden_patterns = [
    "__syncthreads",   # No synchronization needed for elementwise kernel
    "printf",          # Debug output left in production code
    "div.full",        # Slow full-precision division (use fast math)
]
```

### Full example

```toml
[[kernels]]
name = "fused_attention"
source = "kernels/attention.cu"
reference = "scripts/ref_attention.py"

[kernels.tolerances]
float32 = 1e-4
float16 = 1e-2

[kernels.invariants]
no_nan = true
no_inf = true
shape_preserved = true

[kernels.artifact_checks]
max_registers = 48
max_spills = 0
max_local_memory = 0
required_patterns = [
    "fma.rn.f32",
    "shfl.sync",
]
forbidden_patterns = [
    "printf",
    "assert",
]
```

---

## Fuzzing Kernels

Kernel bugs often hide in specific shape and layout combinations. Use fuzzing to test systematically.

### CLI fuzzing

```bash
gpuemu fuzz --op fused_gelu --iterations 500
```

### Python fuzzing with diverse shapes

```python
from gpuemu_py.validate import fuzz_shapes, fuzz_dtypes, fuzz_layouts
from gpuemu_py.frameworks.pytorch import validate_pytorch
from gpuemu_py.client import Client

client = Client()

for batch, seq, hidden in fuzz_shapes(
    batch=[1, 2, 4, 8, 16, 32],
    seq=[1, 64, 128, 256, 512, 1024],
    hidden=[64, 128, 256, 512, 768, 1024],
):
    for dtype in fuzz_dtypes(["float32", "float16"]):
        x = torch.randn(batch, seq, hidden, dtype=getattr(torch, dtype))
        with validate_pytorch(client, "fused_gelu", {"x": x}) as ctx:
            ctx["output"] = fused_gelu_cuda(x)
```

### Layout fuzzing

Memory layout bugs are a common source of kernel failures. Test with contiguous, strided, and transposed inputs:

```python
from gpuemu_py.validate import fuzz_layouts

base_shape = (8, 256, 512)

for shape, strides in fuzz_layouts(base_shape):
    # Create a tensor with specific strides to test non-contiguous access
    x = torch.randn(base_shape).as_strided(shape, strides)
    with validate_pytorch(client, "fused_gelu", {"x": x}) as ctx:
        ctx["output"] = fused_gelu_cuda(x)
```

### Drop-in fuzzing

For the simplest approach, use `fuzz_op_client_side()`:

```python
results = client.fuzz_op_client_side(
    "fused_gelu",
    run_op=lambda inputs: fused_gelu_cuda(
        torch.from_numpy(inputs["x"]).cuda()
    ).cpu().numpy(),
    iterations=500,
    seed=42,
)

print(f"Results: {results.passed}/{results.total} passed")
for f in results.failures:
    print(f"  Seed {f.seed}: {f.failures[0]['message']}")
```

### Edge cases to test

!!! warning "Common kernel edge cases"

    Pay special attention to these shapes and configurations:

    - **Batch size 1**: Single-element batches can expose indexing bugs
    - **Sequence length 1**: Degenerate reductions
    - **Non-power-of-2 dimensions**: Tiling logic often assumes powers of 2
    - **Dimensions smaller than warp size (32)**: Partial warps
    - **Very large dimensions**: Integer overflow in index calculations
    - **Transposed inputs**: Non-contiguous memory access patterns
    - **Mixed dtypes**: float16 inputs with float32 accumulation

---

## Failure Minimization

When a fuzz run finds a failure, the failing shape might be large and hard to debug. Use minimization to find the **smallest input** that still triggers the bug.

### CLI minimization

```bash
gpuemu minimize 8374629105
```

### Minimization strategies

gpuemu supports two minimization strategies:

=== "binary-search-dims"

    Reduces each dimension of the input shape via binary search until the failure disappears. This is the default strategy and works well for shape-dependent bugs.

    ```bash
    gpuemu minimize 8374629105 --strategy binary-search-dims
    ```

    Example: A failure at shape `(32, 1024, 512)` might minimize to `(1, 3, 512)`, revealing that the bug is in the sequence dimension handling when `seq < warp_size`.

=== "binary-search-values"

    Reduces the range of input values via binary search. This is useful for numerical stability bugs that only trigger with large or small values.

    ```bash
    gpuemu minimize 8374629105 --strategy binary-search-values
    ```

    Example: A failure with values in `[-10, 10]` might minimize to values in `[5.2, 5.3]`, revealing a precision issue in a specific range.

### Python API minimization

```python
result = client.minimize(8374629105, strategy="binary-search-dims", max_iters=100)

print(f"Original shape: (unknown)")
print(f"Minimized shape: {result.minimized_shape}")
print(f"Minimized seed: {result.minimized_seed}")
print(f"Still fails: {not result.result.passed}")
```

---

## Regression Detection

When you modify a kernel, you need to verify that the change did not introduce numerical regressions or degrade artifact quality.

### Step 1: Store a baseline before changes

```bash
gpuemu test
gpuemu baseline v1
```

### Step 2: Modify the kernel

Edit your `.cu` file, optimize the math, change tiling, etc.

### Step 3: Re-validate and diff

```bash
gpuemu test
gpuemu lint --kernel fused_gelu --ptx kernels/gelu_kernel.ptx
gpuemu diff --baseline v1 --fail-on-regression
```

### What the diff reports

The diff command compares both **numerical results** and **artifact metrics**:

**Numerical diff:**

| Metric | Baseline (v1) | Current | Delta |
|--------|--------------|---------|-------|
| `fused_gelu` max_diff (float32) | 2.3e-6 | 2.1e-6 | -0.2e-6 (improved) |
| `fused_gelu` max_diff (float16) | 4.1e-3 | 8.7e-3 | +4.6e-3 (regression) |

**Artifact diff:**

| Metric | Baseline (v1) | Current | Delta |
|--------|--------------|---------|-------|
| registers | 28 | 32 | +4 (`register_delta`) |
| spills | 0 | 0 | 0 (`spill_delta`) |
| local_memory | 0 | 0 | 0 (`local_memory_delta`) |

!!! danger "Artifact regression example"

    If your kernel goes from 28 to 48 registers after an optimization, that may reduce occupancy. The diff report surfaces this as a `register_delta` of +20 so you can evaluate the trade-off.

### CI integration for kernel regression detection

```yaml
- name: Kernel regression check
  run: |
    gpuemu daemon start
    gpuemu test
    gpuemu lint --kernel fused_gelu --ptx build/gelu_kernel.ptx
    gpuemu diff --baseline main --fail-on-regression
```

---

## Putting It All Together

A typical kernel development session:

1. **Write the math** in `kernel_math.h` and `ref_my_kernel.py`
2. **Register** the kernel in `gpuemu.toml` with tolerances and artifact checks
3. **Validate** with `gpuemu test` to confirm the reference matches
4. **Fuzz** with `gpuemu fuzz --op my_kernel --iterations 500` to catch edge cases
5. **Minimize** any failures with `gpuemu minimize <seed>`
6. **Inspect artifacts** with `gpuemu lint --kernel my_kernel --ptx my_kernel.ptx`
7. **Baseline** with `gpuemu baseline v1`
8. **Iterate**: modify kernel, re-test, diff against baseline
9. **Ship** when all tests pass and no regressions are detected

---

## Next Steps

- [Model Developer Guide](model-developer.md) -- Validating ops from training code
- [Custom Op Integrator Guide](custom-op-integrator.md) -- Validating third-party kernels
- [VS Code Extension](vscode-extension.md) -- On-save linting for `.cu` files
- [CLI Reference](../reference/cli.md) -- Full command reference for `lint`, `fuzz`, `minimize`, `diff`
- [Config Schema Reference](../reference/config-schema.md) -- Complete `gpuemu.toml` documentation
