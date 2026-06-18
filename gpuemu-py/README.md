<h1 align="center">gpuemu</h1>

<p align="center">
  <strong>Catch silently-wrong GPU kernels from Python — before they reach production.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/gpuemu/"><img src="https://img.shields.io/pypi/v/gpuemu?style=flat-square&logo=python&logoColor=white" alt="PyPI"></a>
  <a href="https://pypi.org/project/gpuemu/"><img src="https://img.shields.io/pypi/pyversions/gpuemu?style=flat-square&logo=python&logoColor=white" alt="Python versions"></a>
  <a href="https://github.com/Skelf-Research/gpuemu/blob/main/LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue?style=flat-square" alt="License"></a>
  <a href="https://docs.skelfresearch.com/gpuemu"><img src="https://img.shields.io/badge/docs-skelfresearch.com-informational?style=flat-square" alt="Docs"></a>
</p>

`gpuemu` is the Python client for [**gpuemu**](https://github.com/Skelf-Research/gpuemu),
a GPU-less correctness oracle for deep-learning kernels. It plugs into PyTorch, JAX, and
TensorFlow and validates your CUDA/Triton kernels against a high-precision fp64 reference
with op-schema-aware, adversarial inputs — finding the silent numerical bugs that
`torch.allclose` misses.

---

## The problem

The industry-standard correctness check for a GPU kernel is one line:

```python
torch.allclose(my_kernel(x), reference(x), atol=1e-5, rtol=1e-2)
```

One shape, one dtype, one seed. In a measured 26-op corpus that oracle **accepts 9/9**
LLM-style buggy kernels — tail-mask leaks, accumulator-scale bugs, missing normalisation,
online-softmax rescale errors — as "correct". Those kernels then ship and run at scale:
GPU-hours wasted on broken work, quality regressions that survive months of green CI.

`gpuemu` replaces that one-line check with an operator-aware regime that caught
**100%** of those bugs across 5 GPU classes with **zero** false positives on controls (P1).

## Install

```bash
pip install gpuemu            # core client
pip install gpuemu[torch]     # + PyTorch adapter
pip install gpuemu[jax]       # + JAX adapter
pip install gpuemu[tensorflow]
pip install gpuemu[all]       # everything
```

The client talks to the `gpuemu` daemon over IPC and will start one on demand. To run the
daemon yourself, install the CLI: `cargo install gpuemu`.

## Quick start

```python
from gpuemu import Client

client = Client()

# Fuzz with op-schema-aware inputs and an fp64 reference oracle.
results = client.fuzz_op_client_side(
    "flash_attention",
    run_op=lambda inputs: my_flash_attn(inputs["q"], inputs["k"], inputs["v"]),
    iterations=100,
    value_distribution="adversarial",   # the P3 default — 99% bug recall
)

print(f"Passed: {results.passed}/{results.total}")
```

A failure reports the **seed, dtype, shape, and a base64 snapshot** of the failing input —
re-run it byte-for-byte from any machine, with or without a GPU. The client's `SeededRng` is
bit-identical to the Rust daemon, so reproduction is exact across languages.

## Execution modes

```python
# 1. Client-side (recommended): your code runs the GPU op; gpuemu validates.
results = client.fuzz_op_client_side(
    "matmul",
    run_op=lambda i: torch.matmul(i["a"], i["b"]),
    iterations=100,
)

# 2. Daemon-orchestrated: fetch cases, run them yourself, submit outputs.
for case in client.get_test_batch("my_op", count=50):
    out = my_gpu_op(case["inputs"])
    client.submit_output("my_op", case["inputs"], out, case["seed"])

# 3. Reproduce / minimise a known failure from its seed.
repro = client.reproduce(seed)
small = client.minimize(seed)
```

## What you get

| Feature | What it does |
|---|---|
| **fp64 reference oracle** | Validates kernel output against a high-precision CPU reference per dtype |
| **Op-schema-aware fuzzing** | Boundary + regular + adversarial input distributions, per op |
| **Calibrated tolerances** | `calibrate_tolerance()` / `get_recommended_tolerance()` — p95-of-controls × 1.5 envelope (P2: 65% → 82% recall) |
| **Deterministic RNG** | `SeededRng` reproduces failures byte-for-byte, identical to the Rust daemon |
| **Framework adapters** | PyTorch, JAX, TensorFlow — `from gpuemu.frameworks.pytorch import validate_pytorch` |
| **Static lint** | `client.lint_kernel(...)` surfaces PTX/SASS register pressure and spills |

## The research backing (P1–P4)

Each default is anchored to a measured study — fp64 oracle (P1: 9/9 bugs caught, 0 false
positives), calibrated tolerances (P2: +23 pp recall), adversarial fuzzing (P3: 99% recall),
and PTX lint (P4). See **[The Evidence](https://docs.skelfresearch.com/gpuemu/why-gpuemu/the-evidence)**.

## Documentation

- Quick start: **[5-minute first validation](https://docs.skelfresearch.com/gpuemu/getting-started/quickstart)**
- Project docs: **[docs.skelfresearch.com/gpuemu](https://docs.skelfresearch.com/gpuemu)**
- Source & issues: **[github.com/Skelf-Research/gpuemu](https://github.com/Skelf-Research/gpuemu)**

## Development

```bash
pip install -e .[dev]
pytest -v          # 11 tests, +7 daemon-live tests
```

## License

Dual-licensed under [MIT](https://github.com/Skelf-Research/gpuemu/blob/main/LICENSE-MIT) or
[Apache 2.0](https://github.com/Skelf-Research/gpuemu/blob/main/LICENSE-APACHE) at your option.
