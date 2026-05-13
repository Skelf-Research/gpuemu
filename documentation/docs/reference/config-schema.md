# Configuration Schema Reference

Complete schema reference for the `gpuemu.toml` configuration file. This file defines
your project, operations, kernels, validation policies, and CI settings.

!!! tip "Config Discovery"
    gpuemu searches for `gpuemu.toml` starting from the current directory and walking
    up the directory tree. Override this with the `GPUEMU_CONFIG` environment variable.

---

## `[project]`

Top-level project metadata.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `String` | `"unnamed"` | Project name used in reports and baselines |
| `version` | `Option<String>` | `None` | Optional project version string |
| `framework` | `Option<String>` | `None` | Default framework: `"pytorch"`, `"jax"`, or `"tensorflow"` |

```toml
[project]
name = "my-kernels"
version = "0.2.1"
framework = "pytorch"
```

---

## `[validation]`

Global validation settings that apply to all ops and kernels unless overridden.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dtypes` | `Vec<String>` | `["float32"]` | Data types to validate by default |
| `check_nan` | `bool` | `true` | Check for NaN values in outputs |
| `check_inf` | `bool` | `true` | Check for Inf values in outputs |
| `seed` | `Option<u64>` | `None` | Global RNG seed for reproducible runs |
| `tolerances` | `HashMap<String, f64>` | *(see below)* | Per-dtype absolute tolerance thresholds |

**Default Tolerances**

| Dtype | Default Tolerance |
|-------|------------------|
| `float32` | `1e-5` |
| `float16` | `1e-3` |
| `bfloat16` | `1e-3` |

```toml
[validation]
dtypes = ["float32", "float16"]
check_nan = true
check_inf = true
seed = 42

[validation.tolerances]
float32 = 1e-5
float16 = 1e-3
bfloat16 = 1e-3
```

!!! note
    The `tolerances` map uses dtype names as keys and absolute tolerance values as
    floating-point numbers. Any dtype not listed falls back to the `float32` default.

---

## `[[ops]]`

Define operations to validate. Each `[[ops]]` entry describes a single op with its
reference implementation and validation parameters.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `String` | *(required)* | Unique name for this op |
| `module` | `Option<String>` | `None` | Python module path for the op |
| `reference` | `String` | *(required)* | Path to the reference script |
| `op_script` | `Option<String>` | `None` | Path to an optional op-under-test script |
| `input_names` | `Vec<String>` | `[]` | Names of the input tensors |
| `execution_mode` | `String` | `"client_side"` | Execution mode: `"client_side"` or `"daemon_side"` |
| `frameworks` | `Vec<String>` | `[]` | Frameworks this op supports |
| `tolerances` | `HashMap<String, f64>` | `{}` | Per-dtype tolerances overriding the global defaults |
| `invariants` | `InvariantConfig` | *(see below)* | Output invariant checks |

```toml
[[ops]]
name = "softmax"
reference = "scripts/softmax_ref.py"
op_script = "scripts/softmax_op.py"
input_names = ["logits"]
execution_mode = "client_side"
frameworks = ["pytorch", "jax"]

[ops.tolerances]
float32 = 1e-6
float16 = 5e-4

[ops.invariants]
non_negative = true
no_nan = true
```

---

### `[[ops]].invariants`

Invariant checks applied to op outputs after numerical comparison.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `non_negative` | `bool` | `false` | Assert all output values are >= 0 |
| `shape_preserved` | `bool` | `false` | Assert output shape matches input shape |
| `no_nan` | `bool` | `false` | Assert no NaN values in output |
| `no_inf` | `bool` | `false` | Assert no Inf values in output |

```toml
[ops.invariants]
non_negative = true
shape_preserved = true
no_nan = true
no_inf = true
```

!!! info
    Invariant checks run independently of numerical tolerance checks.
    An op can pass the tolerance check but fail an invariant check.

---

## `[[kernels]]`

Define compiled GPU kernels to validate and lint. Each `[[kernels]]` entry describes
a kernel with its source, reference, and artifact-level checks.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `String` | *(required)* | Unique name for this kernel |
| `source` | `Option<String>` | `None` | Path to the kernel source file (e.g., `.cu`) |
| `reference` | `String` | *(required)* | Path to the reference script |
| `tolerances` | `HashMap<String, f64>` | `{}` | Per-dtype tolerances overriding the global defaults |
| `invariants` | `InvariantConfig` | *(defaults)* | Output invariant checks (same schema as `[[ops]].invariants`) |
| `artifact_checks` | `ArtifactCheckConfig` | *(see below)* | Artifact-level resource and pattern checks |

```toml
[[kernels]]
name = "fused_softmax"
source = "kernels/fused_softmax.cu"
reference = "scripts/softmax_ref.py"

[kernels.tolerances]
float32 = 1e-5

[kernels.invariants]
non_negative = true
no_nan = true
```

---

### `[[kernels]].artifact_checks`

Resource usage and pattern checks applied to compiled kernel artifacts (PTX assembly).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_registers` | `u32` | `64` | Maximum number of registers per thread |
| `max_spills` | `u32` | `0` | Maximum number of register spills allowed |
| `max_local_memory` | `u32` | `0` | Maximum local memory usage in bytes |
| `required_patterns` | `Vec<String>` | `[]` | PTX patterns that must appear in the artifact |
| `forbidden_patterns` | `Vec<String>` | `[]` | PTX patterns that must not appear in the artifact |

```toml
[kernels.artifact_checks]
max_registers = 48
max_spills = 0
max_local_memory = 0
required_patterns = ["shared.f32"]
forbidden_patterns = ["spill", "local.f32"]
```

!!! warning
    Setting `max_spills = 0` is strict and will fail if the compiler introduces any
    register spills. Increase this value if your kernel legitimately requires spills.

---

## `[policies]`

Global policies that govern how gpuemu treats validation results.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `fail_on_regression` | `bool` | `true` | Treat numerical regressions as failures |
| `warn_threshold` | `f64` | `0.1` | Tolerance delta above which a warning is emitted |

```toml
[policies]
fail_on_regression = true
warn_threshold = 0.1
```

---

## `[ci]`

Settings specific to CI pipeline execution (`gpuemu ci`).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `quick_dtypes` | `Vec<String>` | `["float32"]` | Dtypes used in `--quick` CI mode |
| `thorough_timeout` | `u64` | `3600` | Timeout in seconds for thorough CI runs |
| `parallel_jobs` | `u32` | `4` | Default number of parallel validation jobs |

```toml
[ci]
quick_dtypes = ["float32"]
thorough_timeout = 3600
parallel_jobs = 4
```

---

## Complete Example

A full `gpuemu.toml` demonstrating all sections:

```toml
[project]
name = "my-gpu-kernels"
version = "1.0.0"
framework = "pytorch"

[validation]
dtypes = ["float32", "float16", "bfloat16"]
check_nan = true
check_inf = true
seed = 42

[validation.tolerances]
float32 = 1e-5
float16 = 1e-3
bfloat16 = 1e-3

[[ops]]
name = "softmax"
reference = "scripts/softmax_ref.py"
op_script = "scripts/softmax_op.py"
input_names = ["logits"]
execution_mode = "client_side"
frameworks = ["pytorch", "jax"]

[ops.tolerances]
float32 = 1e-6
float16 = 5e-4

[ops.invariants]
non_negative = true
shape_preserved = true
no_nan = true
no_inf = false

[[ops]]
name = "layernorm"
reference = "scripts/layernorm_ref.py"
input_names = ["x", "weight", "bias"]
execution_mode = "daemon_side"
frameworks = ["pytorch"]

[ops.invariants]
no_nan = true

[[kernels]]
name = "fused_softmax"
source = "kernels/fused_softmax.cu"
reference = "scripts/softmax_ref.py"

[kernels.tolerances]
float32 = 1e-5

[kernels.invariants]
non_negative = true
no_nan = true

[kernels.artifact_checks]
max_registers = 48
max_spills = 0
max_local_memory = 0
required_patterns = ["shared.f32"]
forbidden_patterns = ["spill"]

[[kernels]]
name = "fused_layernorm"
source = "kernels/fused_layernorm.cu"
reference = "scripts/layernorm_ref.py"

[kernels.artifact_checks]
max_registers = 64
max_spills = 2
max_local_memory = 512

[policies]
fail_on_regression = true
warn_threshold = 0.05

[ci]
quick_dtypes = ["float32"]
thorough_timeout = 1800
parallel_jobs = 8
```
