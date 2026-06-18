<p align="center">
  <img src="docs/assets/logo.svg" alt="gpuemu" width="120" height="120">
</p>

<h1 align="center">gpuemu</h1>

<p align="center">
  <strong>Catch silently-wrong GPU kernels before they reach production.</strong>
</p>

<p align="center">
  <a href="https://github.com/skelfresearch/gpuemu/actions"><img src="https://img.shields.io/github/actions/workflow/status/skelfresearch/gpuemu/ci.yml?branch=main&style=flat-square&logo=github" alt="CI"></a>
  <a href="https://crates.io/crates/gpuemu"><img src="https://img.shields.io/crates/v/gpuemu?style=flat-square&logo=rust&color=orange" alt="Crates.io"></a>
  <a href="https://pypi.org/project/gpuemu/"><img src="https://img.shields.io/pypi/v/gpuemu?style=flat-square&logo=python&logoColor=white" alt="PyPI"></a>
  <a href="https://marketplace.visualstudio.com/items?itemName=gpuemu.gpuemu"><img src="https://img.shields.io/visual-studio-marketplace/v/gpuemu.gpuemu?style=flat-square&logo=visualstudiocode&logoColor=white&label=VS%20Code" alt="VS Code"></a>
  <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue?style=flat-square" alt="License"></a>
</p>

<p align="center">
  <a href="https://docs.skelfresearch.com/gpuemu">Documentation</a> •
  <a href="https://docs.skelfresearch.com/gpuemu/quickstart">Quick Start</a> •
  <a href="https://docs.skelfresearch.com/gpuemu/why-gpuemu/the-problem">The Problem</a> •
  <a href="https://github.com/skelfresearch/gpuemu/discussions">Community</a>
</p>

---

## The problem: every benchmark says your kernel is "correct"

The industry-standard correctness oracle for a GPU kernel is one line:

```python
torch.allclose(my_kernel(x), reference(x), atol=1e-5, rtol=1e-2)
```

One shape. One dtype. One seed. Every modern LLM-kernel benchmark — **KernelBench**,
**TritonBench**, **GEAK**, **KernelBand**, **STARK** — uses the same oracle. Kernels that
pass it ship to production.

That oracle is blind to entire bug classes that LLM-generated CUDA/Triton code routinely
contains:

| Bug class | Example | Why allclose misses it |
|---|---|---|
| **Tail-mask leak** | softmax forgets to mask the last partial tile | Only fires when `H` isn't a multiple of `BLOCK` — `H=256` looks fine |
| **Accumulator scale** | matmul writes `acc =` instead of `acc +=` | Result happens to match within `rtol` on the chosen shape |
| **Missing normalisation** | attention without `1/√D` | Saturates softmax differently; one shape looks correct |
| **Online-softmax rescale** | flash-attention forgets `acc *= α` after max update | Only wrong when `N > BLOCK_N` |

In our measured 26-op corpus, the standard one-shape oracle **accepts 9/9** of these
LLM-style buggy kernels as correct (P1, [run records on B2][b2]).

## Why it matters: silent correctness regressions ship at LLM scale

Every modern LLM training and inference stack now ships LLM-generated CUDA/Triton kernels
— fused attention, custom matmul variants, normalisation layers — and a silently-wrong
kernel runs at scale. A miscompiled matmul propagates through every forward pass; a
broken flash-attention degrades long-context quality without crashing; an unmasked
reduction taints metrics no one looks at. The cost is **GPU-hours wasted on
silently-broken work** and **slow, untraceable quality regressions** that survive months
of CI green builds.

This is not hypothetical. Every benchmark in the LLM-kernel literature shares the same
oracle gap; the kernels they bless are the kernels that ship.

## What gpuemu does

gpuemu replaces "allclose on one shape" with an operator-domain–aware correctness regime
that runs **without a GPU** for the validation step and with one for the artifact step.
The product of four measured studies (P1–P4) shapes every default:

| Capability | What it does | Measured finding |
|---|---|---|
| **fp64 reference oracle** | Validates GPU kernel output against a high-precision CPU reference per dtype | **P1**: 100% illusion catch on 9/9 LLM-style bugs across 5 GPU classes; 0 false positives on 15/15 controls |
| **Op-schema-aware fuzzing** | Per-op shape generator with boundary + regular + adversarial value distributions | **P3**: 99% bug recall under adversarial values; +28 pp over the field-standard default |
| **Per-op calibrated tolerances** | p95-of-controls × 1.5 envelope; fits each op/dtype individually | **P2**: +23 pp recall (65 → 82%) over a single hand-picked `atol=1e-5,rtol=1e-2` |
| **Static PTX/SASS lint** | Register pressure, spills, instruction count from compiled artifacts | **P4**: structural Δregs predicts Δperf% consistently across H100/A100/L40S/A10/3060; semantic bugs (identical PTX) are blind — pair with the fp64 oracle |
| **Reproducible RNG** | Bit-identical xorshift128+ in Rust and Python; exact input snapshots | Every flagged failure replays byte-for-byte from its seed |

The full research backing lives in the [gpuemu-paper][paper-repo] artefact (P1–P4, with
LaTeX manuscripts, run-id records on B2, and a kernel corpus you can replay).

---

## Quick start

### Install

```bash
# Rust daemon + CLI
cargo install gpuemu

# Python client
pip install gpuemu

# VS Code extension (optional)
code --install-extension gpuemu.gpuemu
```

### Validate a kernel

```python
from gpuemu import Client

client = Client()

# Fuzz with op-schema-aware inputs and an fp64 reference oracle.
results = client.fuzz_op_client_side(
    "flash_attention",
    run_op=lambda inputs: my_flash_attn(inputs["q"], inputs["k"], inputs["v"]),
    iterations=100,
    value_distribution="adversarial",  # the P3 default — 99% recall
)

print(f"Passed: {results.passed}/{results.total}")
```

A failure reports the seed, dtype, shape, and a base64 snapshot of the failing input. Re-run
it byte-for-byte from any machine.

---

## Compared to

The first question a champion gets asked is "isn't this just
`torch.testing.assert_close`?" or "isn't this what KernelBench does?". Short version:

| Tool | What it does well | The gap gpuemu fills |
|---|---|---|
| `torch.testing.assert_close` | Standard, simple, in-tree | One shape, one dtype, one seed — measured to catch 0/9 LLM-style bugs in our corpus (P1) |
| KernelBench / TritonBench / GEAK / KernelBand / STARK | Leaderboards for LLM-generated kernels | Use the same one-shape oracle inside; not user-facing |
| NVIDIA Compute Sanitizer | Memcheck / racecheck / synccheck | Memory bugs only — silent numerical wrong-output is invisible to it |
| Triton built-in testing | Same `assert_close` semantics | No op-schema fuzz, no fp64 reference |
| HF Kernel Hub | Distribution + ABI checks | Explicitly assumes a correctness tool upstream — that's gpuemu's slot |
| ncu / cuobjdump / ptxas | Static PTX/SASS introspection | No lint policy, no baseline diffing, no regression gate |
| FreeFuzz / DocTer / NablaFuzz / FuzzGPT | API-level DL framework fuzzers | API layer, not kernel; ACL TOSEM 2025 measured 6.5 % real-world bug catch |

The full walk-through, citations, and the five moat signals it surfaces live in
[the Compared to alternatives page](https://docs.skelfresearch.com/gpuemu/why-gpuemu/compared-to).

## Used by / built for

gpuemu serves three distinct customer profiles. Each page leads with a real cited issue
the workflow prevents:

- **[Frontier-lab kernel teams](https://docs.skelfresearch.com/gpuemu/who-uses-gpuemu/frontier-lab-kernel-team)** —
  Anthropic / OpenAI / DeepMind / Meta / xAI. Pre-merge correctness gate scaled to
  100s of ops; PR-blocking with replay-seed links.
- **[OSS-inference maintainers](https://docs.skelfresearch.com/gpuemu/who-uses-gpuemu/oss-inference-maintainer)** —
  vLLM / SGLang / TensorRT-LLM / llama.cpp / MLC-LLM. One-line GitHub Action;
  responds to issues like SGLang #15996, #21238 and vLLM #26378.
- **[Inference-as-a-service vendors](https://docs.skelfresearch.com/gpuemu/who-uses-gpuemu/inference-vendor)** —
  Fireworks / Together / Anyscale / Modal / Replicate / Baseten / Modular. Signed
  Kernel Correctness Report customers verify offline; SLA evidence artefact.

If your team fits one of these profiles and you want to pilot the enterprise tier
(private rule packs, on-prem daemon, signed reports), see
[Design Partners](https://docs.skelfresearch.com/gpuemu/why-gpuemu/design-partners).

## The research backing (P1–P4)

Each capability above is anchored to a measured study. All four ship as LaTeX manuscripts
plus replayable run records on B2.

- **[P1] The correctness illusion in LLM-generated GPU kernels** — Hardware-free fuzz oracle
  catches 9/9 LLM-style bugs across 5 GPU classes (RTX 3060, A10, L40S, A100 SXM4, H100 NVL)
  with 0 false positives on 15/15 controls.
- **[P2] Operator-aware mixed-precision tolerance calibration** — p95-of-controls × 1.5
  envelope raises kernel-bug recall from 65% to 82% over the field-standard fixed
  `atol/rtol`, at zero precision cost.
- **[P3] Test-input generation for tensor programs** — Seven-strategy ablation; adversarial
  value sampling wins at 99% recall; "regular shape only" misses 100% of tail-mask bugs.
- **[P4] Static PTX metrics track structural regressions but miss semantic ones** —
  Structural Δregs / Δinstrs predicts Δperf% consistently across 5 GPU classes; semantic
  bugs compile to identical PTX and need the correctness oracle.

---

## Execution modes

gpuemu supports three ways to run your kernels:

```python
# Client-side (recommended): your code runs the GPU op; gpuemu validates.
results = client.fuzz_op_client_side("matmul",
    run_op=lambda i: torch.matmul(i["a"], i["b"]),
    iterations=100)

# Daemon-orchestrated: fetch cases, run yourself, submit outputs.
for case in client.get_test_batch("my_op", count=50):
    out = my_gpu_op(case["inputs"])
    client.submit_output("my_op", case["inputs"], out, case["seed"])

# Script-based: register reference + op scripts in gpuemu.toml; daemon runs everything.
```

---

## VS Code integration

Validation failures appear as red squiggles with code actions:

- **Problems panel** — seed, dtype, shape per failure
- **Code actions** — "Reproduce failure" or "Minimize test case"
- **Test Explorer** — ops appear in the Testing sidebar
- **On-save validation** — auto-triggers on reference-script save

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
                    │  ├── Op-schema fuzzer       │
                    │  ├── Artifact analyzer      │
                    │  └── Failure storage (sled) │
                    └─────────────────────────────┘
```

---

## Framework support

| Framework | Status | Install |
|---|---|---|
| PyTorch | Stable | `pip install gpuemu[torch]` |
| JAX | Stable | `pip install gpuemu[jax]` |
| TensorFlow | Stable | `pip install gpuemu[tensorflow]` |
| Raw CUDA/Triton | Stable | `pip install gpuemu` |

---

## What gpuemu is NOT

- **Not a cycle-accurate GPU emulator** — correctness, not timing simulation.
- **Not a replacement for real hardware** — final benchmarks still belong on the target GPU.
- **Not a training framework** — kernel-level oracle, not a model-level one.

---

## Documentation

Full docs: **[docs.skelfresearch.com/gpuemu](https://docs.skelfresearch.com/gpuemu)**

- [The Problem](https://docs.skelfresearch.com/gpuemu/why-gpuemu/the-problem) — what allclose misses
- [Industry Impact](https://docs.skelfresearch.com/gpuemu/why-gpuemu/the-industry-impact) — what silent bugs cost
- [The Evidence](https://docs.skelfresearch.com/gpuemu/why-gpuemu/the-evidence) — P1–P4 in one page
- [Quick Start](https://docs.skelfresearch.com/gpuemu/getting-started/quickstart) — first validation in 5 minutes
- [Architecture Deep Dive](https://docs.skelfresearch.com/gpuemu/concepts/architecture)

---

## Platform support

| Platform | Status |
|---|---|
| Linux | Primary target |
| macOS | Fully supported (CPU validation) |
| Windows | Planned |

---

## Contributing

```bash
cargo test                    # Rust (58 tests)
cd gpuemu-py && pytest -v     # Python (11 tests, +7 daemon-live tests)
```

See the [Contributing Guide](https://docs.skelfresearch.com/gpuemu/development/contributing).

---

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE) at your option.

---

<p align="center">
  <sub>Built with care by the <a href="https://skelfresearch.com">Skelf Research</a> team</sub>
</p>

[b2]: https://github.com/sarkar-dipankar/gpuemu-paper
[paper-repo]: https://github.com/sarkar-dipankar/gpuemu-paper
