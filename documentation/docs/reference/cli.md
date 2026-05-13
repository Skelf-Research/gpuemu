# CLI Command Reference

Complete reference for the `gpuemu` command-line interface. All commands are invoked as
subcommands of the `gpuemu` binary.

## Global Flags

| Flag | Short | Description |
|------|-------|-------------|
| `--verbose` | `-v` | Enable verbose output for any command |

```bash
gpuemu -v <command> [flags]
```

---

## Daemon Management

### `daemon start`

Start the gpuemu daemon process.

**Synopsis**

```bash
gpuemu daemon start [--background]
```

**Flags**

| Flag | Description |
|------|-------------|
| `--background` | Run the daemon as a background process (daemonize) |

**Examples**

```bash
# Start the daemon in the foreground
gpuemu daemon start

# Start the daemon in the background
gpuemu daemon start --background
```

!!! tip
    Running with `--background` is recommended for normal development workflows.
    Foreground mode is useful for debugging daemon startup issues.

---

### `daemon stop`

Stop a running gpuemu daemon.

**Synopsis**

```bash
gpuemu daemon stop
```

**Examples**

```bash
gpuemu daemon stop
```

---

### `daemon status`

Check whether the daemon is currently running and responsive.

**Synopsis**

```bash
gpuemu daemon status
```

**Examples**

```bash
gpuemu daemon status
```

---

### `daemon logs`

Display recent daemon log output.

**Synopsis**

```bash
gpuemu daemon logs [--lines <N>]
```

**Flags**

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--lines` | `-l` | `50` | Number of log lines to display |

**Examples**

```bash
# Show the last 50 lines (default)
gpuemu daemon logs

# Show the last 200 lines
gpuemu daemon logs --lines 200
```

---

## Project Initialization

### `init`

Initialize a new gpuemu project. Creates a `gpuemu.toml` configuration file and
optional scaffolding for framework-specific validation and CI pipelines.

**Synopsis**

```bash
gpuemu init [--name <NAME>] [--framework <FRAMEWORK>] [--with-examples] [--ci <PROVIDER>] [--target_dir <DIR>]
```

**Flags**

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--name` | | | Project name written to `gpuemu.toml` |
| `--framework` | | | Target framework: `pytorch`, `jax`, or `tensorflow` |
| `--with-examples` | | | Scaffold example op definitions and reference scripts |
| `--ci` | | | Generate CI configuration: `github` or `gitlab` |
| `--target_dir` | `-t` | `.` | Directory in which to create the project |

**Examples**

=== "PyTorch project"

    ```bash
    gpuemu init --name my-kernels --framework pytorch --with-examples
    ```

=== "JAX project with GitHub CI"

    ```bash
    gpuemu init --name jax-ops --framework jax --ci github
    ```

=== "TensorFlow project in a custom directory"

    ```bash
    gpuemu init --name tf-custom --framework tensorflow -t ./projects/tf-custom
    ```

!!! note
    If `--name` is omitted, the project name defaults to the target directory name.

---

## Validation

### `test`

Run the validation test suite defined in `gpuemu.toml`.

**Synopsis**

```bash
gpuemu test [--quick] [--thorough] [--seed <SEED>]
```

**Flags**

| Flag | Description |
|------|-------------|
| `--quick` | Run a reduced set of tests for fast feedback |
| `--thorough` | Run the full, exhaustive test suite |
| `--seed` | Fixed RNG seed for reproducible test runs |

**Examples**

```bash
# Run the default test suite
gpuemu test

# Quick smoke test
gpuemu test --quick

# Reproducible thorough run
gpuemu test --thorough --seed 42
```

---

### `status`

Check the current daemon status. This is a convenience alias for `daemon status`.

**Synopsis**

```bash
gpuemu status
```

---

### `version`

Print the gpuemu version string and exit.

**Synopsis**

```bash
gpuemu version
```

---

## Fuzz Testing

### `fuzz`

Run fuzz testing on one or more operations. Generates random inputs across shapes,
dtypes, and layouts to find numerical discrepancies.

**Synopsis**

```bash
gpuemu fuzz [--op <OP>] [--iterations <N>] [--seed <SEED>] [--fail-fast]
```

**Flags**

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--op` | `-o` | | Target a specific op by name. If omitted, all ops are fuzzed. |
| `--iterations` | `-i` | `100` | Number of fuzz iterations per op |
| `--seed` | | | Fixed RNG seed for reproducibility |
| `--fail-fast` | | | Stop on the first failure |

**Examples**

```bash
# Fuzz all ops with default iterations
gpuemu fuzz

# Fuzz a single op with more iterations
gpuemu fuzz --op softmax --iterations 500

# Reproducible fuzz run that stops on first failure
gpuemu fuzz --seed 12345 --fail-fast
```

!!! warning
    Large iteration counts with `--thorough` dtypes can take significant time.
    Use `--fail-fast` during development to catch issues early.

---

### `reproduce`

Reproduce a specific fuzz failure using its seed.

**Synopsis**

```bash
gpuemu reproduce <seed> [--verbose]
```

**Arguments**

| Argument | Description |
|----------|-------------|
| `seed` | **(positional, required)** The seed of the failure to reproduce |

**Flags**

| Flag | Short | Description |
|------|-------|-------------|
| `--verbose` | `-v` | Show detailed comparison output |

**Examples**

```bash
# Reproduce a failure
gpuemu reproduce 8837462910

# Reproduce with detailed output
gpuemu reproduce 8837462910 --verbose
```

---

### `minimize`

Minimize a failing test case to the smallest input that still triggers the failure.

**Synopsis**

```bash
gpuemu minimize <seed> [--strategy <STRATEGY>] [--max_iters <N>]
```

**Arguments**

| Argument | Description |
|----------|-------------|
| `seed` | **(positional, required)** The seed of the failure to minimize |

**Flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--strategy` | | Minimization strategy: `binary-search-dims` or `binary-search-values` |
| `--max_iters` | `100` | Maximum number of minimization iterations |

**Examples**

```bash
# Minimize using default settings
gpuemu minimize 8837462910

# Minimize with a specific strategy
gpuemu minimize 8837462910 --strategy binary-search-dims

# Limit minimization iterations
gpuemu minimize 8837462910 --strategy binary-search-values --max_iters 50
```

!!! info "Minimization Strategies"
    - **`binary-search-dims`** -- Reduces tensor dimensions to find the smallest shape that fails.
    - **`binary-search-values`** -- Narrows the value range of inputs to isolate problematic values.

---

### `failures`

List stored fuzz failures.

**Synopsis**

```bash
gpuemu failures [--limit <N>]
```

**Flags**

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--limit` | `-l` | `20` | Maximum number of failures to display |

**Examples**

```bash
# List recent failures
gpuemu failures

# Show up to 100 failures
gpuemu failures --limit 100
```

---

## Kernel Linting

### `lint`

Lint kernel artifacts such as PTX assembly for correctness, performance patterns, and
resource usage.

**Synopsis**

```bash
gpuemu lint [--kernel <NAME>] [--ptx <PATH>] [--format <FORMAT>]
```

**Flags**

| Flag | Short | Description |
|------|-------|-------------|
| `--kernel` | `-k` | Name of the kernel to lint (from `gpuemu.toml`) |
| `--ptx` | `-p` | Path to a raw PTX file to lint directly |
| `--format` | | Output format: `text` (default) or `json` |

**Examples**

```bash
# Lint a kernel defined in config
gpuemu lint --kernel my_softmax

# Lint a PTX file directly
gpuemu lint --ptx ./kernels/softmax.ptx

# JSON output for CI integration
gpuemu lint --kernel my_softmax --format json
```

---

## Baselines and Diffing

### `baseline`

Store the current validation results as a named baseline for future comparison.

**Synopsis**

```bash
gpuemu baseline <tag>
```

**Arguments**

| Argument | Description |
|----------|-------------|
| `tag` | **(positional, required)** A name/tag for this baseline snapshot |

**Examples**

```bash
# Store a baseline before making changes
gpuemu baseline v1.0

# Store a baseline with a descriptive name
gpuemu baseline pre-refactor
```

---

### `diff`

Compare current results against a stored baseline to detect regressions.

**Synopsis**

```bash
gpuemu diff [--baseline <TAG>] [--fail-on-regression] [--format <FORMAT>]
```

**Flags**

| Flag | Description |
|------|-------------|
| `--baseline` | Tag of the baseline to compare against |
| `--fail-on-regression` | Exit with a non-zero code if a regression is detected |
| `--format` | Output format: `text` (default) or `json` |

**Examples**

```bash
# Compare against a named baseline
gpuemu diff --baseline v1.0

# Fail in CI if regressions are found
gpuemu diff --baseline pre-refactor --fail-on-regression

# Machine-readable output
gpuemu diff --baseline v1.0 --format json
```

---

### `artifacts`

Display artifact-level metrics for compiled kernels (register usage, spills, local
memory, etc.).

**Synopsis**

```bash
gpuemu artifacts [--kernel <NAME>]
```

**Flags**

| Flag | Short | Description |
|------|-------|-------------|
| `--kernel` | `-k` | Show metrics for a specific kernel. If omitted, all kernels are shown. |

**Examples**

```bash
# Show all kernel artifacts
gpuemu artifacts

# Show artifacts for a specific kernel
gpuemu artifacts --kernel my_softmax
```

---

## CI Integration

### `ci`

Run the full CI validation suite. This combines testing, linting, baseline comparison,
and report generation into a single command suitable for CI pipelines.

**Synopsis**

```bash
gpuemu ci [--quick] [--baseline <TAG>] [--parallel <N>] [--format <FORMAT>] [--output <PATH>]
```

**Flags**

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--quick` | | | Run only the quick validation subset |
| `--baseline` | | | Compare against this baseline tag |
| `--parallel` | | `0` (auto) | Number of parallel jobs. `0` means auto-detect CPU count. |
| `--format` | | `text` | Output format: `text`, `json`, or `junit` |
| `--output` | `-o` | | Write output to a file instead of stdout |

**Examples**

=== "GitHub Actions"

    ```bash
    gpuemu ci --baseline main --fail-on-regression --format junit --output results.xml
    ```

=== "GitLab CI"

    ```bash
    gpuemu ci --quick --format json --output report.json
    ```

=== "Local pre-push check"

    ```bash
    gpuemu ci --quick --parallel 4
    ```

!!! tip
    Use `--format junit` to integrate with CI systems that support JUnit XML test reports.

---

## Reporting

### `report`

Generate a comprehensive validation report.

**Synopsis**

```bash
gpuemu report [--format <FORMAT>] [--output <PATH>] [--since_hours <N>] [--include_lint] [--include_artifacts]
```

**Flags**

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--format` | | `text` | Output format: `text`, `json`, or `junit` |
| `--output` | `-o` | | Write the report to a file instead of stdout |
| `--since_hours` | | | Only include results from the last N hours |
| `--include_lint` | | | Include kernel lint results in the report |
| `--include_artifacts` | | | Include artifact metrics in the report |

**Examples**

```bash
# Generate a full text report
gpuemu report

# JSON report for the last 24 hours
gpuemu report --format json --since_hours 24

# Comprehensive report with lint and artifacts
gpuemu report --include_lint --include_artifacts --output full-report.txt
```

---

## Debugging

### `debug`

Launch an interactive debugging session for inspecting op behavior, stepping through
computations, and comparing intermediate results.

**Synopsis**

```bash
gpuemu debug [--seed <SEED>] [--repl] [--op <OP>]
```

**Flags**

| Flag | Description |
|------|-------------|
| `--seed` | Start debugging a specific fuzz failure by seed |
| `--repl` | Launch an interactive REPL for exploration |
| `--op` | Target a specific op for debugging |

**Examples**

```bash
# Debug a specific failure
gpuemu debug --seed 8837462910

# Interactive REPL for an op
gpuemu debug --op softmax --repl

# Verbose interactive session
gpuemu -v debug --op layernorm --repl
```

!!! info
    The REPL provides commands for inspecting tensor values, comparing outputs,
    and stepping through reference vs. emulated execution paths.

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `gpuemu daemon start` | Start the daemon |
| `gpuemu daemon stop` | Stop the daemon |
| `gpuemu daemon status` | Check daemon status |
| `gpuemu daemon logs` | Show daemon logs |
| `gpuemu init` | Initialize a new project |
| `gpuemu test` | Run validation tests |
| `gpuemu status` | Check daemon status (alias) |
| `gpuemu version` | Show version |
| `gpuemu fuzz` | Fuzz test operations |
| `gpuemu reproduce` | Reproduce a fuzz failure |
| `gpuemu minimize` | Minimize a failing test case |
| `gpuemu failures` | List stored failures |
| `gpuemu lint` | Lint kernel artifacts |
| `gpuemu baseline` | Store a baseline |
| `gpuemu diff` | Compare against a baseline |
| `gpuemu artifacts` | Show artifact metrics |
| `gpuemu ci` | Run CI validation suite |
| `gpuemu report` | Generate a report |
| `gpuemu debug` | Interactive debugging |
