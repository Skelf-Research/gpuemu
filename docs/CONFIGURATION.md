# Configuration Reference

gpuemu uses TOML configuration files for project setup, validation policies, and op/kernel registration.

## File Locations

| File | Purpose |
|------|---------|
| `gpuemu.toml` | Project configuration (per-project) |
| `~/.config/gpuemu/config.toml` | User defaults |
| `~/.gpuemu/daemon.toml` | Daemon configuration |

## Project Configuration (`gpuemu.toml`)

### Minimal Example

```toml
[project]
name = "my-project"
framework = "pytorch"
```

### Full Example

```toml
[project]
name = "my-ml-project"
version = "0.1.0"
framework = "pytorch"  # pytorch | jax | tensorflow

[validation]
dtypes = ["float32", "float16", "bfloat16"]
check_nan = true
check_inf = true
seed = 42  # optional: fixed seed for reproducibility

[validation.tolerances]
float32 = 1e-5
float16 = 1e-3
bfloat16 = 1e-3

[validation.shapes]
# Shape fuzzing configuration
batch_sizes = [1, 2, 4, 8, 16, 32]
sequence_lengths = [1, 64, 128, 256, 512, 1024]
edge_cases = [1, 7, 127, 513]  # primes, off-by-one

[validation.layouts]
contiguous = true
strided = true
transposed = true

# Custom ops registration
[[ops]]
name = "flash_attention"
module = "flash_attn.flash_attn_func"
reference = "scripts/ref_flash_attn.py"

[ops.tolerances]
float16 = 1e-2
bfloat16 = 1e-2
float32 = 1e-4

[ops.invariants]
non_negative = false
shape_preserved = true
no_nan = true
no_inf = true

# Kernel registration (for kernel authors)
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
max_local_memory = 0
required_patterns = ["FADD", "FMAX"]
forbidden_patterns = ["IMUL"]  # optional

[kernels.build]
arch = ["sm_80", "sm_90"]
flags = ["-O3", "--use_fast_math"]

# Artifact inspection policies
[policies]
fail_on_regression = true
warn_threshold = 0.1  # 10% tolerance for warnings

[policies.registers]
max = 64
warn_above = 48

[policies.spills]
max = 0
warn_above = 0

[policies.local_memory]
max_bytes = 0
warn_above = 0

# CI configuration
[ci]
quick_dtypes = ["float32"]
quick_shapes = [[2, 128, 512]]
thorough_timeout = 3600  # seconds
parallel_jobs = 4
```

## Schema Reference

### `[project]`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Project name |
| `version` | string | no | Project version |
| `framework` | string | no | ML framework: `pytorch`, `jax`, `tensorflow` |

### `[validation]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dtypes` | array | `["float32"]` | Dtypes to test |
| `check_nan` | bool | `true` | Fail on NaN outputs |
| `check_inf` | bool | `true` | Fail on Inf outputs |
| `seed` | int | random | Fixed RNG seed |

### `[validation.tolerances]`

Numeric tolerances per dtype for approximate comparison.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `float32` | float | `1e-5` | Tolerance for float32 |
| `float16` | float | `1e-3` | Tolerance for float16 |
| `bfloat16` | float | `1e-3` | Tolerance for bfloat16 |
| `float64` | float | `1e-10` | Tolerance for float64 |

### `[validation.shapes]`

Shape fuzzing configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_sizes` | array | `[1, 2, 4]` | Batch sizes to test |
| `sequence_lengths` | array | `[64, 128]` | Sequence lengths to test |
| `edge_cases` | array | `[1, 7]` | Edge case dimensions |

### `[validation.layouts]`

Layout fuzzing configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `contiguous` | bool | `true` | Test contiguous tensors |
| `strided` | bool | `true` | Test strided views |
| `transposed` | bool | `true` | Test transposed tensors |

### `[[ops]]`

Custom op registration (repeatable).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Op identifier |
| `module` | string | no | Python module path |
| `reference` | string | yes | Path to reference script |
| `frameworks` | array | no | Frameworks this op supports |

### `[[kernels]]`

Kernel registration (repeatable).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Kernel identifier |
| `source` | string | no | Path to CUDA source |
| `reference` | string | yes | Path to reference script |

### `[kernels.artifact_checks]`

PTX/SASS inspection rules.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_registers` | int | 64 | Max register count |
| `max_spills` | int | 0 | Max spill count |
| `max_local_memory` | int | 0 | Max local memory bytes |
| `required_patterns` | array | `[]` | Required instruction patterns |
| `forbidden_patterns` | array | `[]` | Forbidden instruction patterns |

### `[policies]`

Global policy configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `fail_on_regression` | bool | `true` | Fail CI on regressions |
| `warn_threshold` | float | `0.1` | Warning threshold (10%) |

### `[ci]`

CI-specific configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `quick_dtypes` | array | `["float32"]` | Dtypes for `--quick` |
| `quick_shapes` | array | `[[2, 128]]` | Shapes for `--quick` |
| `thorough_timeout` | int | `3600` | Timeout for `--thorough` |
| `parallel_jobs` | int | `4` | Parallel validation jobs |

## Daemon Configuration (`~/.gpuemu/daemon.toml`)

```toml
[daemon]
socket_path = "~/.gpuemu/gpuemu.sock"
log_level = "info"  # debug | info | warn | error
max_connections = 16

[storage]
db_path = "~/.gpuemu/db"
max_results = 10000
max_baselines = 100
retention_days = 30

[execution]
reference_timeout = 60  # seconds per reference script
max_parallel_refs = 4
python_path = "python3"
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GPUEMU_CONFIG` | Override config file path |
| `GPUEMU_SOCKET` | Override daemon socket path |
| `GPUEMU_LOG_LEVEL` | Override log level |
| `GPUEMU_NO_COLOR` | Disable colored output |

## Reference Script Protocol

Reference scripts receive inputs via stdin (pickle) and return outputs via stdout (pickle):

```python
#!/usr/bin/env python3
import sys
import pickle

def reference(**inputs):
    # Your implementation
    return output

if __name__ == "__main__":
    inputs = pickle.load(sys.stdin.buffer)
    result = reference(**inputs)
    pickle.dump(result, sys.stdout.buffer)
```

The daemon invokes scripts with:
- Working directory: project root
- Environment: inherits from daemon
- Timeout: configurable per-op or global default
