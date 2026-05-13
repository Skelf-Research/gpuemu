# Configuration

gpuemu uses TOML configuration files to control project settings, validation behavior, op definitions, kernel analysis, policies, and CI integration. This guide covers every configuration option in detail.

---

## File Locations

gpuemu reads configuration from three locations, in order of precedence (highest first):

| File | Purpose | Typical Location |
|------|---------|-----------------|
| **Project config** | Per-project settings checked into version control | `gpuemu.toml` (project root) |
| **User defaults** | Personal defaults applied to all projects | `~/.config/gpuemu/config.toml` |
| **Daemon config** | Daemon-specific runtime settings | `~/.gpuemu/daemon.toml` |

!!! info "Merge behavior"

    Settings in the project-level `gpuemu.toml` override user defaults, which in turn override daemon defaults. Array fields like `[[ops]]` and `[[kernels]]` are **not** merged -- they are taken entirely from whichever file defines them (usually the project config).

---

## `[project]`

Top-level metadata about your project.

```toml
[project]
name = "my-project"
version = "0.1.0"
framework = "pytorch"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `string` | **required** | Project name. Used in reports and log output. |
| `version` | `string` | `"0.0.0"` | Project version. Informational only. |
| `framework` | `string` | `"pytorch"` | Default framework. One of `"pytorch"`, `"jax"`, or `"tensorflow"`. Determines which adapter is loaded when none is specified. |

---

## `[validation]`

Global validation settings that apply to all ops unless overridden at the op level.

```toml
[validation]
dtypes = ["float32", "float16", "bfloat16"]
check_nan = true
check_inf = true
seed = 12345

[validation.tolerances]
float32 = { atol = 1e-5, rtol = 1e-5 }
float16 = { atol = 1e-2, rtol = 1e-2 }
bfloat16 = { atol = 1e-2, rtol = 1e-2 }
float64 = { atol = 1e-10, rtol = 1e-10 }
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dtypes` | `array[string]` | `["float32"]` | Data types to test. Supported values: `"float16"`, `"bfloat16"`, `"float32"`, `"float64"`. |
| `check_nan` | `bool` | `true` | Fail validation if any output element is NaN. |
| `check_inf` | `bool` | `true` | Fail validation if any output element is Inf. |
| `seed` | `int` | `0` (random) | Global RNG seed for deterministic test generation. Set to `0` for a random seed per run. |

### `[validation.tolerances]`

Per-dtype tolerance thresholds. Each entry is a table with `atol` (absolute tolerance) and `rtol` (relative tolerance).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `atol` | `float` | `1e-5` | Maximum allowed absolute difference between actual and reference outputs. |
| `rtol` | `float` | `1e-5` | Maximum allowed relative difference between actual and reference outputs. |

!!! tip "Choosing tolerances"

    - **float32**: `atol=1e-5, rtol=1e-5` works for most operations.
    - **float16**: Reduced precision requires looser tolerances; start with `atol=1e-2, rtol=1e-2`.
    - **bfloat16**: Similar to float16 in practice; `atol=1e-2, rtol=1e-2` is a good starting point.
    - **float64**: If you test in double precision, use tight tolerances like `atol=1e-10, rtol=1e-10`.

    When in doubt, run `gpuemu fuzz` and look at the maximum errors to calibrate.

---

## `[[ops]]`

An array of operation definitions. Each `[[ops]]` entry defines one operation to validate.

```toml
[[ops]]
name = "matmul"
module = "my_project.ops.matmul"
reference = "scripts/matmul_ref.py"
execution_mode = "script_based"
invariants = ["shape_preserved"]

[ops.tolerances]
float32 = { atol = 1e-5, rtol = 1e-5 }
float16 = { atol = 1e-2, rtol = 1e-2 }
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `string` | **required** | Unique name for this op. Used in CLI commands and reports. |
| `module` | `string` | **required** | Dotted Python module path to the op implementation. |
| `reference` | `string` | **required** | Path to the reference script (relative to project root) or a Python callable in `module:function` format. |
| `execution_mode` | `string` | `"script_based"` | How the reference is executed. See below. |
| `tolerances` | `table` | Inherits from `[validation.tolerances]` | Per-dtype tolerances that override the global defaults for this op. |
| `invariants` | `array[string]` | `[]` | Invariant checks to enforce on outputs. |

### Execution Modes

!!! abstract "Understanding execution modes"

    The `execution_mode` field controls how gpuemu runs the reference implementation and compares results. Choosing the right mode affects performance, isolation, and debugging ergonomics.

=== "script_based"

    The daemon spawns the reference script as a subprocess. Input is passed via stdin as JSON with base64-encoded tensors; output is read from stdout in the same format.

    **Best for**: Getting started, simple ops, language-agnostic references.

    ```toml
    [[ops]]
    name = "matmul"
    reference = "scripts/matmul_ref.py"
    execution_mode = "script_based"
    ```

=== "client_side"

    The Python client computes the reference inline within the same process. The reference is a Python callable specified as `module:function`.

    **Best for**: Tight integration with framework code, fast iteration, debugging.

    ```toml
    [[ops]]
    name = "matmul"
    reference = "my_project.refs:matmul_reference"
    execution_mode = "client_side"
    ```

=== "daemon_orchestrated"

    The daemon manages both execution and comparison. The reference script is pre-loaded by the daemon and kept warm for repeated invocations.

    **Best for**: CI pipelines, batch runs, high-throughput fuzzing.

    ```toml
    [[ops]]
    name = "matmul"
    reference = "scripts/matmul_ref.py"
    execution_mode = "daemon_orchestrated"
    ```

### Supported Invariants

| Invariant | Description |
|-----------|-------------|
| `shape_preserved` | Output tensors must have the same shape as the reference output. |
| `non_negative` | All output elements must be >= 0. |
| `finite` | All output elements must be finite (no NaN or Inf). Equivalent to `check_nan + check_inf` per op. |
| `symmetric` | Output matrix must be symmetric (for square matrix outputs). |
| `normalized` | Output values must sum to 1 along the last axis (e.g., softmax outputs). |

---

## `[[kernels]]`

An array of kernel definitions for artifact-level analysis. Each `[[kernels]]` entry defines one compiled kernel to inspect.

```toml
[[kernels]]
name = "matmul_kernel"
source = "kernels/matmul.cu"
reference = "scripts/matmul_ref.py"

[kernels.artifact_checks]
max_registers = 64
max_spills = 0
max_local_memory_bytes = 0
forbidden_instructions = ["LDG.E.SYS", "STG.E.SYS"]
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `string` | **required** | Unique name for this kernel. |
| `source` | `string` | **required** | Path to the kernel source file (relative to project root). |
| `reference` | `string` | `""` | Path to a reference script for correctness validation of the kernel. |

### `[kernels.artifact_checks]`

Static analysis checks applied to the compiled PTX/SASS artifact.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_registers` | `int` | `128` | Maximum number of registers the kernel may use. Exceeding this triggers a warning or failure. |
| `max_spills` | `int` | `0` | Maximum allowed register spills to local memory. |
| `max_local_memory_bytes` | `int` | `0` | Maximum allowed local (stack) memory usage in bytes. |
| `forbidden_instructions` | `array[string]` | `[]` | List of PTX/SASS instruction patterns that must not appear in the compiled artifact. |

!!! warning "Artifact inspection requires Linux"

    Artifact checks (PTX/SASS analysis) require Linux with CUDA toolkit installed. On macOS these checks are skipped with a warning. See the [Platform Support](../index.md#platform-support) table for details.

---

## `[policies]`

Policies control how validation results are interpreted in CI and reporting contexts.

```toml
[policies]
fail_on_regression = true
warn_threshold = 0.8
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `fail_on_regression` | `bool` | `true` | If `true`, the CLI exits with a non-zero status when a previously passing op starts failing. Useful for CI gates. |
| `warn_threshold` | `float` | `0.8` | Fraction of tolerance consumed (0.0 to 1.0) at which a warning is emitted. For example, `0.8` means a warning fires when the error exceeds 80% of the allowed tolerance, even if the test still passes. |

!!! note "Regression detection"

    Regression detection compares the current run against the most recent stored results in the daemon's database. On the first run for an op there is no baseline, so `fail_on_regression` has no effect.

---

## `[ci]`

Settings specific to Continuous Integration environments.

```toml
[ci]
quick_dtypes = ["float32"]
thorough_timeout = 600
parallel_jobs = 4
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `quick_dtypes` | `array[string]` | `["float32"]` | Subset of dtypes to test during quick CI runs (`gpuemu test --quick`). Keeps fast checks under a time budget. |
| `thorough_timeout` | `int` | `300` | Maximum time in seconds for a thorough (non-quick) test run before it is terminated. |
| `parallel_jobs` | `int` | `1` | Number of ops to validate in parallel. Set to the number of available CPU cores for best throughput. |

!!! tip "CI strategy"

    A common pattern is to run `gpuemu test --quick` on every push (fast, tests only `quick_dtypes`) and run the full `gpuemu test` on merge to main or on a nightly schedule. Fuzzing (`gpuemu fuzz`) is typically run nightly with a high iteration count.

---

## Complete Example

Below is a full `gpuemu.toml` file demonstrating all sections together:

```toml title="gpuemu.toml"
# ─── Project ──────────────────────────────────────────────
[project]
name = "my-ml-project"
version = "0.2.0"
framework = "pytorch"

# ─── Global Validation ────────────────────────────────────
[validation]
dtypes = ["float32", "float16", "bfloat16"]
check_nan = true
check_inf = true
seed = 42

[validation.tolerances]
float32  = { atol = 1e-5,  rtol = 1e-5  }
float16  = { atol = 1e-2,  rtol = 1e-2  }
bfloat16 = { atol = 1e-2,  rtol = 1e-2  }
float64  = { atol = 1e-10, rtol = 1e-10 }

# ─── Ops ──────────────────────────────────────────────────
[[ops]]
name = "matmul"
module = "my_project.ops.matmul"
reference = "scripts/matmul_ref.py"
execution_mode = "script_based"
invariants = ["shape_preserved"]

[ops.tolerances]
float32 = { atol = 1e-5, rtol = 1e-5 }
float16 = { atol = 1e-2, rtol = 1e-2 }

[[ops]]
name = "softmax"
module = "my_project.ops.softmax"
reference = "scripts/softmax_ref.py"
execution_mode = "daemon_orchestrated"
invariants = ["shape_preserved", "non_negative", "normalized"]

[[ops]]
name = "layernorm"
module = "my_project.ops.layernorm"
reference = "my_project.refs:layernorm_reference"
execution_mode = "client_side"
invariants = ["shape_preserved"]

# ─── Kernels ──────────────────────────────────────────────
[[kernels]]
name = "matmul_kernel"
source = "kernels/matmul.cu"
reference = "scripts/matmul_ref.py"

[kernels.artifact_checks]
max_registers = 64
max_spills = 0
max_local_memory_bytes = 0
forbidden_instructions = ["LDG.E.SYS"]

[[kernels]]
name = "softmax_kernel"
source = "kernels/softmax.cu"
reference = "scripts/softmax_ref.py"

[kernels.artifact_checks]
max_registers = 32
max_spills = 0
max_local_memory_bytes = 512

# ─── Policies ────────────────────────────────────────────
[policies]
fail_on_regression = true
warn_threshold = 0.8

# ─── CI ───────────────────────────────────────────────────
[ci]
quick_dtypes = ["float32"]
thorough_timeout = 600
parallel_jobs = 4
```

---

## Next Steps

- **[Quick Start](quickstart.md)** -- Use this configuration to run your first validation.
- **[Config Schema Reference](../reference/config-schema.md)** -- Machine-readable JSON schema for `gpuemu.toml`.
- **[Execution Modes](../concepts/execution-modes.md)** -- Deep dive into `script_based`, `client_side`, and `daemon_orchestrated`.
- **[CI Integration Tutorial](../tutorials/ci-integration.md)** -- Set up gpuemu with GitHub Actions or GitLab CI.
