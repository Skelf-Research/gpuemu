# CI Integration

Set up gpuemu in your continuous integration pipeline to catch numerical regressions automatically. gpuemu runs entirely on CPU, so no GPU runners are required.

---

## Overview

gpuemu is designed for CPU-only CI runners. It validates kernel correctness without GPU hardware, making it practical to run on every pull request and nightly build.

Two modes are available:

| Mode | Use case | What it tests |
|------|----------|---------------|
| **Quick** (`--quick`) | PR checks, fast feedback | `float32` only, default shapes |
| **Thorough** (default) | Nightly builds, release gates | All configured dtypes, extended shapes |

!!! info "No GPU required"

    gpuemu executes your ops on CPU using reference implementations. This means your CI runners do not need GPU hardware, CUDA drivers, or any GPU-specific configuration. Standard GitHub Actions runners and GitLab shared runners work out of the box.

---

## GitHub Actions

The following workflow runs quick checks on every pull request and thorough checks on a nightly schedule.

```yaml title=".github/workflows/gpuemu.yml"
name: gpuemu validation

on:
  pull_request:
  schedule:
    - cron: "0 3 * * *"  # Nightly at 03:00 UTC

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Build gpuemu
        run: |
          cargo build --release
          echo "$PWD/target/release" >> $GITHUB_PATH

      - name: Install Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install gpuemu-py
        run: pip install ./gpuemu-py[torch]

      - name: Start daemon
        run: gpuemu daemon start --background

      - name: Run validation (PR - quick)
        if: github.event_name == 'pull_request'
        run: >
          gpuemu ci
          --quick
          --format junit
          --output results.xml

      - name: Run validation (nightly - thorough)
        if: github.event_name == 'schedule'
        run: >
          gpuemu ci
          --format junit
          --output results.xml

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: gpuemu-results
          path: results.xml

      - name: Publish test report
        if: always()
        uses: mikepenz/action-junit-report@v4
        with:
          report_paths: results.xml

      - name: Stop daemon
        if: always()
        run: gpuemu daemon stop
```

!!! tip "Caching the Rust build"

    To speed up CI runs, cache the Rust build artifacts:

    ```yaml
    - name: Cache Rust build
      uses: actions/cache@v4
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
    ```

---

## GitLab CI

The equivalent configuration for GitLab CI:

```yaml title=".gitlab-ci.yml"
stages:
  - validate

variables:
  CARGO_HOME: "$CI_PROJECT_DIR/.cargo"

.gpuemu-base:
  image: rust:1.75-bookworm
  before_script:
    - cargo build --release
    - export PATH="$PWD/target/release:$PATH"
    - apt-get update && apt-get install -y python3 python3-pip python3-venv
    - python3 -m venv .venv
    - source .venv/bin/activate
    - pip install ./gpuemu-py[torch]
    - gpuemu daemon start --background
  after_script:
    - gpuemu daemon stop
  cache:
    key: cargo-$CI_COMMIT_REF_SLUG
    paths:
      - .cargo/registry
      - .cargo/git
      - target

gpuemu-quick:
  extends: .gpuemu-base
  stage: validate
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
  script:
    - >
      gpuemu ci
      --quick
      --format junit
      --output results.xml
  artifacts:
    when: always
    reports:
      junit: results.xml

gpuemu-thorough:
  extends: .gpuemu-base
  stage: validate
  rules:
    - if: $CI_PIPELINE_SOURCE == "schedule"
  script:
    - >
      gpuemu ci
      --format junit
      --output results.xml
  artifacts:
    when: always
    reports:
      junit: results.xml
```

---

## CI Command Options

The `gpuemu ci` command runs all configured ops and produces structured output.

```bash
gpuemu ci [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--quick` | Quick mode: test only `float32` with default shapes | Off (thorough) |
| `--baseline <tag>` | Compare results against a stored baseline tag | None |
| `--parallel <n>` | Number of parallel validation jobs. `0` = auto-detect CPU count | `0` |
| `--format text\|json\|junit` | Output format | `text` |
| `--output <file>` | Write results to a file instead of stdout | stdout |
| `--fail-on-regression` | Exit with code 1 if any metric regresses compared to `--baseline` | Off |

=== "Quick PR check"

    ```bash
    gpuemu ci --quick --format junit --output results.xml
    ```

=== "Thorough nightly"

    ```bash
    gpuemu ci --format junit --output results.xml
    ```

=== "With baseline comparison"

    ```bash
    gpuemu ci --baseline v1.2.0 --fail-on-regression --format junit --output results.xml
    ```

=== "Parallel execution"

    ```bash
    gpuemu ci --parallel 4 --format json --output results.json
    ```

---

## Quick vs Thorough Mode

The two modes differ in dtype coverage and shape diversity:

| Aspect | Quick (`--quick`) | Thorough (default) |
|--------|-------------------|--------------------|
| **Dtypes** | `float32` only | All configured dtypes (`float32`, `float16`, `bfloat16`, etc.) |
| **Shapes** | Default shapes from `gpuemu.toml` | Extended shapes including edge cases |
| **Fuzz iterations** | Minimal (configured by `[ci.quick]`) | Full count (configured by `[ci]`) |
| **Typical duration** | 30 seconds -- 2 minutes | 5 -- 30 minutes |
| **Use case** | PR checks, fast feedback | Nightly builds, release gates |

Configure both modes in `gpuemu.toml`:

```toml title="gpuemu.toml"
[ci]
fuzz_iterations = 200
parallel = 0              # 0 = auto-detect

[ci.quick]
fuzz_iterations = 10
dtypes = ["float32"]
```

!!! tip "Balancing speed and coverage"

    For most projects, running `--quick` on every PR and thorough mode nightly provides a good balance. Quick mode catches obvious regressions in seconds, while thorough mode exercises edge cases overnight.

---

## Baseline Regression Detection

gpuemu can compare current results against a stored baseline to detect numerical regressions.

### Store a baseline

After a release or at any known-good state, store a baseline:

```bash
gpuemu ci --format json --output baseline.json
```

Commit `baseline.json` to your repository or store it as a CI artifact.

### Compare against the baseline

On pull requests, compare the current run against the stored baseline:

```bash
gpuemu ci --baseline baseline.json --fail-on-regression --format junit --output results.xml
```

!!! warning "What counts as a regression?"

    A regression is detected when any op's maximum numerical error **increases** compared to the baseline beyond the configured tolerance. Specifically:

    - If `max_error_current > max_error_baseline * (1 + regression_threshold)`, it is flagged.
    - The default `regression_threshold` is `0.1` (10% increase).

    Configure the threshold in `gpuemu.toml`:

    ```toml
    [ci]
    regression_threshold = 0.1   # Flag if error increases by more than 10%
    ```

### GitHub Actions workflow with baseline

```yaml title=".github/workflows/gpuemu.yml (baseline section)"
      - name: Download baseline
        if: github.event_name == 'pull_request'
        uses: actions/download-artifact@v4
        with:
          name: gpuemu-baseline
          path: .
        continue-on-error: true  # First run won't have a baseline

      - name: Run validation with baseline
        if: github.event_name == 'pull_request'
        run: >
          gpuemu ci
          --quick
          --baseline baseline.json
          --fail-on-regression
          --format junit
          --output results.xml

      - name: Store baseline (nightly)
        if: github.event_name == 'schedule'
        run: gpuemu ci --format json --output baseline.json

      - name: Upload baseline artifact
        if: github.event_name == 'schedule'
        uses: actions/upload-artifact@v4
        with:
          name: gpuemu-baseline
          path: baseline.json
```

---

## Report Command

The `gpuemu report` command generates a standalone report from the daemon's stored results. This is useful for custom reporting workflows outside of `gpuemu ci`.

```bash
gpuemu report [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--format text\|json\|junit` | Output format | `text` |
| `--output <file>` | Write report to a file instead of stdout | stdout |
| `--since_hours <n>` | Include only results from the last N hours | All results |
| `--include_lint` | Include artifact lint results in the report | Off |
| `--include_artifacts` | Include artifact analysis details | Off |

=== "JUnit report for CI"

    ```bash
    gpuemu report --format junit --output report.xml
    ```

=== "JSON report from last 24 hours"

    ```bash
    gpuemu report --format json --since_hours 24 --output report.json
    ```

=== "Full report with lint and artifacts"

    ```bash
    gpuemu report \
      --format json \
      --include_lint \
      --include_artifacts \
      --output full-report.json
    ```

!!! info "Difference between `gpuemu ci` and `gpuemu report`"

    - `gpuemu ci` **runs** validations and produces output in one step. Use this in your CI pipeline.
    - `gpuemu report` **reads** previously stored results from the daemon database and formats them. Use this for ad-hoc reporting or when you want to generate reports separately from the validation run.

---

## Exit Codes

Both `gpuemu ci` and `gpuemu report` use consistent exit codes:

| Exit code | Meaning |
|-----------|---------|
| `0` | All validations passed, no regressions detected |
| `1` | One or more validations failed, or a regression was detected (with `--fail-on-regression`) |

!!! tip "Using exit codes in CI"

    CI systems like GitHub Actions and GitLab CI automatically treat a non-zero exit code as a job failure. No additional configuration is needed -- `gpuemu ci` will fail the pipeline if any validation fails.

    ```yaml
    # This step will fail the job if gpuemu reports failures
    - name: Run validation
      run: gpuemu ci --quick --format junit --output results.xml
    ```

---

## Next Steps

- [PyTorch Validation Tutorial](pytorch-validation.md) -- Validate PyTorch custom ops.
- [JAX Validation Tutorial](jax-validation.md) -- Validate JAX custom primitives.
- [TensorFlow Validation Tutorial](tensorflow-validation.md) -- Validate TensorFlow custom ops.
- [Configuration](../getting-started/configuration.md) -- Fine-tune tolerances, dtypes, and CI policies.
- [CLI Reference](../reference/cli.md) -- Full reference for all CLI commands.
