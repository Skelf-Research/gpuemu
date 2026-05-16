<p align="center">
  <img src="docs/assets/logo.svg" alt="gpuemu" width="120" height="120">
</p>

<h1 align="center">gpuemu</h1>

<p align="center">
  <strong>Ship GPU kernels with confidence — no GPU required.</strong>
</p>

<p align="center">
  <a href="https://github.com/skelfresearch/gpuemu/actions"><img src="https://img.shields.io/github/actions/workflow/status/skelfresearch/gpuemu/ci.yml?branch=main&style=flat-square&logo=github" alt="CI"></a>
  <a href="https://crates.io/crates/gpuemu"><img src="https://img.shields.io/crates/v/gpuemu?style=flat-square&logo=rust&color=orange" alt="Crates.io"></a>
  <a href="https://pypi.org/project/gpuemu-py/"><img src="https://img.shields.io/pypi/v/gpuemu-py?style=flat-square&logo=python&logoColor=white" alt="PyPI"></a>
  <a href="https://marketplace.visualstudio.com/items?itemName=gpuemu.gpuemu"><img src="https://img.shields.io/visual-studio-marketplace/v/gpuemu.gpuemu?style=flat-square&logo=visualstudiocode&logoColor=white&label=VS%20Code" alt="VS Code"></a>
  <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue?style=flat-square" alt="License"></a>
</p>

<p align="center">
  <a href="https://docs.skelfresearch.com/gpuemu">Documentation</a> •
  <a href="https://docs.skelfresearch.com/gpuemu/quickstart">Quick Start</a> •
  <a href="https://docs.skelfresearch.com/gpuemu/integrations">Integrations</a> •
  <a href="https://github.com/skelfresearch/gpuemu/discussions">Community</a>
</p>

---

## Why gpuemu?

Building GPU kernels is hard. Validating them shouldn't require a GPU farm.

**gpuemu** is a validation and testing toolkit that lets you catch correctness bugs, numerical instabilities, and edge-case failures in your CUDA/GPU kernels — all from your laptop, your CI runner, or anywhere without GPU access.

```
┌─────────────────────────────────────────────────────────────────┐
│  Your GPU Kernel (CUDA, Triton, custom)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  gpuemu                                                         │
│  ├── CPU mirror execution     → validate math & indexing       │
│  ├── Shape & layout fuzzing   → expose boundary bugs           │
│  ├── Numerical stability      → catch precision issues         │
│  └── Artifact analysis        → flag register spills & issues  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ✓ Deterministic, reproducible, CI-ready
```

### Built for kernel developers

- **Validate without GPUs** — Run correctness tests on any machine. Perfect for CI pipelines.
- **Fuzz your kernels** — Automatically generate edge-case inputs that expose boundary, stride, and accumulation bugs.
- **Reproducible failures** — Every failure includes a seed for exact reproduction.
- **IDE integration** — Failures appear as diagnostics in VS Code. Right-click to reproduce.
- **Framework agnostic** — Works with PyTorch, JAX, TensorFlow, or raw CUDA.

---

## Quick Start

### Install

```bash
# Rust daemon + CLI
cargo install gpuemu

# Python client
pip install gpuemu-py

# VS Code extension (optional)
code --install-extension gpuemu.gpuemu
```

### Initialize & Run

```bash
# Create a new project
gpuemu init --name my-kernels --framework pytorch

# Start the daemon
gpuemu daemon start

# Run validation
gpuemu test --quick
```

### Python API

```python
from gpuemu_py import Client

client = Client()

# Fuzz your kernel with 100 random inputs
results = client.fuzz_op_client_side(
    "flash_attention",
    run_op=lambda inputs: my_flash_attn(inputs["q"], inputs["k"], inputs["v"]),
    iterations=100,
)

print(f"Passed: {results.passed}/{results.total}")
```

---

## Core Capabilities

| Capability | What it does |
|------------|--------------|
| **CPU Mirror Execution** | Run kernel logic on CPU to validate correctness without GPU hardware |
| **Shape & Layout Fuzzing** | Auto-generate edge-case tensors (boundary sizes, non-contiguous strides) |
| **Numerical Stability Checks** | Validate accumulation precision, reduction stability, tolerance envelopes |
| **Artifact Linting** | Inspect PTX/IR for register pressure, spills, or missing optimizations |
| **Deterministic CI** | Reproducible seeds, policy-driven pass/fail gates, JSON output |
| **Cross-language RNG** | Bit-identical xorshift128+ in Rust and Python for reproducibility |

---

## Execution Modes

gpuemu supports three ways to run your kernels:

### Client-Side (Recommended)

Your client runs the GPU kernel; gpuemu validates the output.

```python
results = client.fuzz_op_client_side(
    "matmul",
    run_op=lambda inputs: torch.matmul(inputs["a"], inputs["b"]),
    iterations=100,
)
```

### Daemon-Orchestrated

Fine-grained control: fetch test cases, run ops yourself, submit outputs.

```python
cases = client.get_test_batch("my_op", count=50)
for case in cases:
    output = my_gpu_op(case["inputs"])
    result = client.submit_output("my_op", case["inputs"], output, case["seed"])
```

### Script-Based

Configure reference scripts in `gpuemu.toml` — the daemon runs everything.

```toml
[[ops]]
name = "my_op"
reference = "scripts/ref_my_op.py"
op_script = "scripts/run_my_op.py"
execution_mode = "script_based"
```

---

## VS Code Integration

Validation failures appear as **red squiggles** in your editor with actionable code actions:

- **Problems panel** — Failures mapped to source with seed, dtype, and shape info
- **Code actions** — Right-click → "Reproduce failure" or "Minimize test case"
- **Test Explorer** — Ops appear in the Testing sidebar
- **On-save validation** — Auto-triggers when you save reference scripts

---

## Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   gpuemu CLI     │     │   Python Client  │     │  VS Code Ext     │
│   (Rust)         │     │   (gpuemu-py)    │     │  (TypeScript)    │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │ IPC (NNG/Unix sockets)
                                  ▼
                    ┌─────────────────────────────┐
                    │      gpuemu-daemon          │
                    │  ├── Validation engine      │
                    │  ├── Fuzz test generator    │
                    │  ├── Artifact analyzer      │
                    │  └── Failure storage (sled) │
                    └─────────────────────────────┘
```

---

## What gpuemu is NOT

- **Not a cycle-accurate GPU emulator** — We validate correctness, not performance timing.
- **Not a replacement for real hardware** — Use gpuemu for development; benchmark on real GPUs.
- **Not a training framework** — We test kernels, not models.

---

## Framework Support

| Framework | Status | Install |
|-----------|--------|---------|
| PyTorch | Stable | `pip install gpuemu-py[torch]` |
| JAX | Stable | `pip install gpuemu-py[jax]` |
| TensorFlow | Stable | `pip install gpuemu-py[tensorflow]` |
| Raw CUDA/Triton | Stable | `pip install gpuemu-py` |

---

## Documentation

Full documentation is available at **[docs.skelfresearch.com/gpuemu](https://docs.skelfresearch.com/gpuemu)**

- [Getting Started](https://docs.skelfresearch.com/gpuemu/quickstart)
- [Configuration Reference](https://docs.skelfresearch.com/gpuemu/configuration)
- [Integration Guides](https://docs.skelfresearch.com/gpuemu/integrations)
- [Validation Policies](https://docs.skelfresearch.com/gpuemu/validation)
- [Architecture Deep Dive](https://docs.skelfresearch.com/gpuemu/architecture)

---

## Platform Support

| Platform | Status |
|----------|--------|
| Linux | Primary target |
| macOS | Fully supported |
| Windows | Planned |

---

## Contributing

We welcome contributions. See our [Contributing Guide](https://docs.skelfresearch.com/gpuemu/contributing) for details.

```bash
# Run tests
cargo test                    # Rust (58 tests)
cd gpuemu-py && pytest -v     # Python (11 tests)
```

---

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE) at your option.

---

<p align="center">
  <sub>Built with care by the <a href="https://skelfresearch.com">Skelf Research</a> team</sub>
</p>
