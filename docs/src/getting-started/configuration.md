# Configuration

gpuemu is configured via `gpuemu.toml` in your project root.

## Project Section

```toml
[project]
name = "my-project"
framework = "pytorch"  # pytorch, jax, tensorflow
version = "0.1.0"
```

## Validation Section

```toml
[validation]
dtypes = ["float32", "float16", "bfloat16"]
check_nan = true
check_inf = true
seed = 12345  # Optional: fixed seed for deterministic tests

[validation.tolerances]
float32 = 1e-5
float16 = 1e-3
bfloat16 = 1e-3
```

## Ops Configuration

```toml
[[ops]]
name = "my_op"
module = "my_module.my_op"
reference = "scripts/ref_my_op.py"

[ops.tolerances]
float32 = 1e-5
float16 = 1e-3

[ops.invariants]
no_nan = true       # Output must not contain NaN
no_inf = true       # Output must not contain Inf
non_negative = false # Output must be >= 0
shape_preserved = true # Output shape equals input shape
```

## Kernels Configuration

```toml
[[kernels]]
name = "my_kernel"
source = "kernels/my_kernel.cu"
reference = "scripts/ref_my_kernel.py"

[kernels.tolerances]
float32 = 1e-5

[kernels.artifact_checks]
max_registers = 64
max_spills = 0
max_local_memory = 0
required_patterns = ["ld.global", "st.global"]
forbidden_patterns = ["div.rn.f32"]
```

## CI Section

```toml
[ci]
quick_dtypes = ["float32"]  # Dtypes for --quick mode
thorough_timeout = 3600     # Timeout for thorough tests (seconds)
parallel_jobs = 4           # Number of parallel validation jobs
```

## Policies Section

```toml
[policies]
fail_on_regression = true   # Fail CI if artifact metrics regress
warn_threshold = 0.1        # Warn if results differ by more than 10%
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GPUEMU_SOCKET` | Path to daemon socket | `~/.gpuemu/daemon.sock` |
| `GPUEMU_CONFIG` | Path to config file | `./gpuemu.toml` |
| `GPUEMU_LOG_LEVEL` | Log level (error, warn, info, debug) | `info` |
| `GPUEMU_INSTALL_DIR` | Installation directory | `~/.gpuemu` |

## Complete Example

```toml
[project]
name = "transformer-ops"
framework = "pytorch"
version = "0.1.0"

[validation]
dtypes = ["float32", "float16"]
check_nan = true
check_inf = true

[validation.tolerances]
float32 = 1e-5
float16 = 1e-3

[[ops]]
name = "flash_attention"
module = "transformer.attention"
reference = "scripts/ref_flash_attention.py"

[ops.tolerances]
float32 = 1e-4
float16 = 1e-2

[ops.invariants]
no_nan = true
no_inf = true

[[kernels]]
name = "fused_softmax"
source = "kernels/fused_softmax.cu"
reference = "scripts/ref_fused_softmax.py"

[kernels.artifact_checks]
max_registers = 48
max_spills = 0

[ci]
quick_dtypes = ["float32"]
parallel_jobs = 4

[policies]
fail_on_regression = true
```
