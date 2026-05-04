# Workflow

This workflow is designed for developers without GPUs and for CI pipelines that need deterministic validation.

## Prerequisites

```bash
# Install gpuemu daemon (Rust)
cargo install gpuemu

# Install Python client
pip install gpuemu-py

# Start daemon
gpuemu daemon start
```

## Local Development Loop

```
┌─────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌─────────┐
│  Build  │───▶│  Mirror │───▶│ Validate │───▶│ Inspect │───▶│ Report  │
└─────────┘    └─────────┘    └──────────┘    └─────────┘    └─────────┘
     │              │               │              │              │
     ▼              ▼               ▼              ▼              ▼
  Compile       Execute          Compare        Check PTX     Store in
  kernels,      CPU refs,        against        artifacts,    sled DB,
  extract PTX   fuzz shapes      tolerances     apply rules   show summary
```

### 1. Build

```bash
gpuemu build
```

- Compile kernel sources for target toolchain (if available)
- Extract PTX/IR artifacts for inspection
- Store artifact metadata in sled database

### 2. Mirror

```bash
gpuemu test --quick  # fast feedback
gpuemu test          # full validation
```

- Execute reference implementations on CPU
- Fuzz shapes and layouts with seeded RNG
- Daemon invokes reference scripts and captures outputs

### 3. Validate

Automatic during `gpuemu test`:

- Compare outputs against reference implementations
- Apply numeric tolerance checks per dtype
- Enforce NaN/Inf rules and domain-specific invariants
- Check shape preservation and other invariants

### 4. Inspect

```bash
gpuemu lint
```

- Parse compiled PTX/IR artifacts
- Apply policy checks: register pressure, spills, patterns
- Diff against stored baselines for regression detection

### 5. Report

```bash
gpuemu report
```

- Summarize validation results with pass/fail status
- Record failing seeds for reproduction
- Store results in sled for history and diffing

## CI Pipeline Loop

```yaml
# .github/workflows/gpuemu.yml
name: GPU-less Validation
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest  # No GPU required
    steps:
      - uses: actions/checkout@v4

      - name: Install gpuemu
        run: |
          curl -sSL https://gpuemu.dev/install.sh | sh
          pip install gpuemu-py

      - name: Start daemon
        run: gpuemu daemon start --background

      - name: Build
        run: gpuemu build

      - name: Validate
        run: gpuemu test

      - name: Inspect
        run: gpuemu lint

      - name: Report
        run: gpuemu report --format json > results.json

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: gpuemu-results
          path: |
            results.json
            .gpuemu/results/
```

### CI Stages

1. **Build** - Deterministic builds with cached toolchains
2. **Mirror + Validate** - CPU correctness tests with fuzz seeds
3. **Inspect** - Policy-driven artifact checks with diff history
4. **Report** - Summary with reproducible seeds for failures

## CLI Reference

| Command | Description |
|---------|-------------|
| `gpuemu daemon start` | Start the validation daemon |
| `gpuemu daemon stop` | Stop the daemon |
| `gpuemu daemon status` | Check daemon status |
| `gpuemu init` | Initialize `gpuemu.toml` in current directory |
| `gpuemu build` | Compile kernels and extract artifacts |
| `gpuemu test` | Run CPU mirror and correctness checks |
| `gpuemu test --quick` | Fast validation (subset of shapes/dtypes) |
| `gpuemu test --thorough` | Exhaustive validation |
| `gpuemu lint` | Apply artifact policies and report regressions |
| `gpuemu report` | Generate validation summary |
| `gpuemu ci` | Run full pipeline (build + test + lint + report) |
| `gpuemu baseline --tag <name>` | Store current results as baseline |
| `gpuemu diff --baseline <name>` | Compare against stored baseline |
| `gpuemu reproduce --seed <n>` | Reproduce a specific failure |
| `gpuemu minimize --seed <n>` | Find minimal failing case |

## Reproducibility

Every failing case is stored in sled with:

- Input shapes and layout metadata
- Random seeds used for fuzzing
- Toolchain version and policy snapshot
- Exact tensor values (serialized with rkyv)

```bash
# List stored failures
gpuemu seeds list

# Reproduce specific failure
gpuemu reproduce --seed 12345

# Export for sharing
gpuemu seeds export --seed 12345 > failure.json

# Import on another machine
gpuemu seeds import < failure.json
gpuemu reproduce --seed 12345
```

## Data Storage

```
~/.gpuemu/
├── db/              # sled database
│   ├── results/     # validation results
│   ├── baselines/   # stored baselines
│   └── artifacts/   # PTX/IR snapshots
├── gpuemu.sock      # nng IPC socket
└── logs/            # daemon logs
```

Project-local storage:

```
.gpuemu/
├── results/         # exported results
└── cache/           # build cache
```
