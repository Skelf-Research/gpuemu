# gpuemu

GPU-less development, assessment, and validation for deep learning kernels.

This project enables teams to develop and validate GPU-oriented deep learning code without requiring access to physical GPUs. It provides CPU-based execution mirrors, static analysis of GPU artifacts, and CI-friendly validation so you can catch correctness and structural performance regressions early.

## Quick Start

### Prerequisites

- Rust 1.70+ (for the daemon and CLI)
- Python 3.9+ with NumPy (for the client and reference scripts)
- [pynng](https://pypi.org/project/pynng/) (for the Python client IPC)
- VS Code 1.85+ (for the editor extension)

### Build & Install

```bash
# Build the Rust workspace (daemon + CLI)
cargo build --release

# Install the Python client
cd gpuemu-py && pip install -e .

# Install the VS Code extension
cd vscode-gpuemu && npm install && npx vsce package
code --install-extension gpuemu-0.1.0.vsix
```

### Initialize a Project

```bash
# Create a new gpuemu project with example ops
gpuemu init --name my-project --framework pytorch --with-examples

# This creates:
#   gpuemu.toml      - Configuration file
#   scripts/         - Reference implementation scripts
#   .gpuemu/         - Daemon data directory
```

### Start the Daemon

```bash
gpuemu daemon start     # Start in background
gpuemu daemon start --foreground  # Start in foreground
gpuemu status           # Check daemon status
```

### Run Validation

```bash
# Quick validation (10 iterations)
gpuemu test --quick

# Standard validation (50 iterations)
gpuemu test

# Fuzz a specific op
gpuemu fuzz --op matmul --iterations 100

# Run full CI validation with JSON output
gpuemu ci --quick --format json --output report.json
```

### Python Client

```python
from gpuemu_py import Client, SeededRng

# Connect and check daemon (auto-verifies protocol version)
client = Client()
info = client.ping()
print(f"Daemon v{info['version']}, uptime {info['uptime_secs']}s")

# Validate an op (single-shot)
result = client.validate_op("my_op", {"x": x_tensor}, output_tensor)
print(f"Passed: {result.passed}")

# Fuzz test (daemon generates inputs, validates against reference)
results = client.fuzz_op("my_op", iterations=100, seed=42)
print(f"{results.passed}/{results.total} passed")
```

## What this is

- A development and CI toolchain for GPU-targeted kernels that works without GPUs.
- A correctness and validation harness that runs kernels on CPU with deterministic checks.
- A policy-driven analyzer that inspects build artifacts (e.g., PTX/IR) to flag regressions.
- An editor-integrated workflow that surfaces failures as diagnostics in your IDE.

## What this is not

- A cycle-accurate GPU emulator.
- A replacement for running on real hardware for performance benchmarking.
- A framework for training or inference itself.

## Core capabilities

- **CPU mirror execution**: Run kernel logic on CPU to validate math, indexing, shapes, and layout handling.
- **Shape and layout fuzzing**: Automatically generate edge-case tensors to expose boundary and stride bugs.
- **Numerical stability checks**: Validate accumulation precision, reduction stability, and tolerance envelopes.
- **Artifact linting**: Inspect compiled artifacts for register pressure, spills, or missing patterns.
- **CI-first workflow**: Deterministic tests, reproducible seeds, and policy-driven pass/fail gates.
- **Cross-language RNG**: Bit-for-bit identical xorshift128+ PRNG in both Rust and Python for reproducibility.
- **Editor integration**: Failures appear as diagnostics in VS Code with code actions for reproduction and minimization.

## Architecture

- **gpuemu-daemon** (Rust): IPC server (NNG/REP0) handling validation, fuzzing, artifact analysis, and storage.
- **gpuemu** (Rust CLI): Command-line interface for daemon control, testing, fuzzing, and CI.
- **gpuemu-py** (Python client): Python API for programmatic validation and fuzzing.
- **vscode-gpuemu** (VS Code extension): Editor integration with diagnostics, code actions, and test explorer.

All IPC uses JSON serialization over NNG Unix-domain sockets for cross-language compatibility (with a protocol version check for forward/backward compatibility).

## Execution Modes

gpuemu supports three execution modes, depending on where the op under test runs:

### 1. Client-Side (Recommended for GPU developers)

The daemon generates random inputs; the **client runs the actual GPU op** and submits the output for validation. This is the primary drop-in path.

```python
from gpuemu_py import Client

client = Client()

# The lambda runs YOUR GPU kernel — gpuemu handles validation
results = client.fuzz_op_client_side(
    "flash_attention",
    run_op=lambda inputs: my_flash_attn(inputs["q"], inputs["k"], inputs["v"]),
    iterations=100,
)
print(f"Passed: {results.passed}/{results.total}")
```

### 2. Daemon-Orchestrated

Fine-grained control: fetch test cases from the daemon, run the op yourself, submit outputs one at a time.

```python
client = Client()

# Get test cases
cases = client.get_test_batch("my_op", count=50)

for case in cases:
    output = my_gpu_op(case["inputs"])       # Run your op
    result = client.submit_output("my_op", case["inputs"], output, case["seed"])
    if not result.passed:
        print(f"FAIL at seed {case['seed']}: {result.failures[0]['message']}")
```

### 3. Script-Based

For ops that can be executed from the daemon machine (e.g., the daemon has GPU access). Configure `op_script` in `gpuemu.toml` and the daemon runs both scripts automatically.

```toml
[[ops]]
name = "my_op"
reference = "scripts/ref_my_op.py"
op_script = "scripts/run_my_op.py"
execution_mode = "script_based"
```

Set `execution_mode` per op in `gpuemu.toml`: `client_side` (default), `daemon_orchestrated`, or `script_based`.

## VS Code Extension

The gpuemu extension provides a pseudo-LSP experience — validation failures appear as **red squiggles in your source code** with actionable code actions:

### Features

| Feature | Description |
|---------|-------------|
| **Problems panel** | Validation failures mapped to reference scripts with seed, dtype, and shape info |
| **Code actions** | Right-click a diagnostic → "Reproduce failure", "Minimize test case" |
| **Test Explorer** | Ops from `gpuemu.toml` appear in the Testing sidebar |
| **On-save validation** | Saving a `ref_*.py` or `.cu` file auto-triggers validation |
| **Config linting** | `gpuemu.toml` errors (invalid dtypes, missing references, wrong execution_mode) surface as diagnostics |
| **Status bar** | Shows daemon version and uptime; click to start/stop |
| **Failures tree** | Sidebar view of all stored failures; click to reproduce |

### How it fits your workflow

```
1. Edit ref_flash_attn.py → save
2. ValidationWatcher triggers → daemon runs quick validation
3. DiagnosticManager pushes failures to Problems panel
4. Red squiggle appears on the reference script
5. Right-click → "Reproduce failure (seed: 4242)"
6. Terminal opens with exact repro info
7. Fix the bug → save → squiggle clears
```

### Extension Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `gpuemu.binaryPath` | auto-detect | Path to the `gpuemu` CLI binary |
| `gpuemu.autoStartDaemon` | `true` | Auto-start daemon when `gpuemu.toml` is present |
| `gpuemu.showStatusBar` | `true` | Show daemon status in the status bar |

## Running Tests

```bash
# Rust tests (58 tests)
cargo test

# Python smoke tests (11 tests)
cd gpuemu-py && PYTHONPATH=. pytest tests/test_smoke.py -v

# TypeScript compilation check
cd vscode-gpuemu && npx -p typescript tsc --noEmit
```

## Platform goals

- **Linux**: Primary development and CI target.
- **macOS**: Core validation workflow runs without GPUs, where toolchains allow.
- **Windows**: Not a current target; may be considered after Linux/macOS parity.

## Repository map

- `crates/gpuemu-daemon/` — Rust IPC daemon (validation, fuzzing, storage)
- `crates/gpuemu-cli/` — Rust CLI binary
- `crates/gpuemu-common/` — Shared types, protocol, RNG, config
- `gpuemu-py/` — Python client package with framework adapters
- `vscode-gpuemu/` — VS Code extension (pseudo-LSP)
- `docs/` — Architecture, configuration, integration guides

**Key docs:**
- **[docs/INTEGRATIONS.md](docs/INTEGRATIONS.md)** — Complete PyTorch, JAX, and TensorFlow integration guide with API reference
- `docs/ARCHITECTURE.md` — Component design and data flow
- `docs/CONFIGURATION.md` — TOML configuration schema reference
- `docs/VALIDATION.md` — Validation taxonomy and policies

## Contributing

If you want to contribute, start with `docs/PROJECT_SCOPE.md` and `docs/ARCHITECTURE.md`, then propose changes as small, reviewable steps.