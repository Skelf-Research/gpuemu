# gpuemu (CLI)

[![Crates.io](https://img.shields.io/crates/v/gpuemu?style=flat-square&logo=rust&color=orange)](https://crates.io/crates/gpuemu)
[![docs.rs](https://img.shields.io/docsrs/gpuemu?style=flat-square&logo=docsdotrs)](https://docs.rs/gpuemu)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue?style=flat-square)](https://github.com/Skelf-Research/gpuemu)

**The `gpuemu` command line — catch silently-wrong GPU kernels before they ship.**

This crate installs the `gpuemu` binary: a single command to fuzz deep-learning kernels
against an fp64 reference oracle, lint compiled PTX/SASS, gate CI on correctness
regressions, and emit signed correctness reports. It's the front door to the
[gpuemu](https://github.com/Skelf-Research/gpuemu) project — it manages the
[`gpuemu-daemon`](https://crates.io/crates/gpuemu-daemon) for you.

---

## Why

The field-standard correctness check — `torch.allclose(kernel(x), ref(x), atol=1e-5, rtol=1e-2)`
— is one shape, one dtype, one seed. In a measured 26-op corpus it **accepts 9/9** LLM-style
buggy kernels (tail-mask leaks, accumulator-scale bugs, missing normalisation,
online-softmax rescale errors) as correct. `gpuemu` replaces it with an operator-aware
regime that caught **100%** of those bugs across 5 GPU classes with **zero** false positives
on controls (P1). See the [project README](https://github.com/Skelf-Research/gpuemu#readme)
for the P1–P4 evidence.

## Install

```bash
cargo install gpuemu      # installs the `gpuemu` binary
```

```bash
gpuemu --version
gpuemu init my-kernels --framework pytorch --ci github   # scaffold a project
gpuemu daemon start                                       # start the validation engine
```

> Requires a `python3` interpreter on `PATH` for reference-script execution.

## What you can do

```bash
# Fuzz an op with op-schema-aware, adversarial inputs against the fp64 oracle
gpuemu fuzz flash_attention --iterations 100

# Reproduce or minimise a stored failure, byte-for-byte, from its seed
gpuemu reproduce <seed>
gpuemu minimize  <seed> --strategy binary-search-dims

# Lint compiled artifacts and gate on register/spill regressions vs a baseline
gpuemu lint matmul --ptx target/matmul.ptx
gpuemu baseline v1.0
gpuemu diff v1.0 --fail-on-regression

# One command for CI: fuzz + lint + baseline-diff, emit machine-readable output
gpuemu ci --baseline v1.0 --format sarif --output gpuemu.sarif

# Signed Kernel Correctness Report (HTML + ed25519) for offline/SLA verification
gpuemu report --format html --signed --output report.html
```

### Command reference

| Command | Purpose |
|---|---|
| `init` | Scaffold a new gpuemu project (framework + CI templates) |
| `daemon start/stop` | Manage the validation daemon |
| `test` | Run the configured validation suite |
| `fuzz` | Op-schema-aware fuzzing against the fp64 reference oracle |
| `reproduce` / `minimize` | Replay or shrink a failing case from its seed |
| `failures` | List stored failures |
| `lint` | Lint kernel PTX/SASS against policy rules |
| `baseline` / `diff` | Store and compare artifact baselines; gate on regressions |
| `artifacts` | Show artifact metrics for kernels |
| `ci` | Combined fuzz + lint + diff suite for pipelines |
| `coverage` | Correctness-coverage report (Codecov / Sonar consumable) |
| `report` | Generate a report — `text`, `json`, `junit`, `sarif`, `pr-comment`, `html` (`--signed`) |
| `debug` | Interactive failure-investigation REPL |
| `status` / `version` | Daemon status and build info |

Output formats for `ci`/`report` (`junit`, `sarif`, `pr-comment`, `codecov`) drop straight
into GitHub Actions, GitLab CI, Codecov, and Sonar.

## Prefer Python?

The [`gpuemu`](https://pypi.org/project/gpuemu/) client gives you the same engine from
PyTorch / JAX / TensorFlow with `pip install gpuemu`.

## Documentation

- Quick start: **[5-minute first validation](https://docs.skelfresearch.com/gpuemu/getting-started/quickstart)**
- Project docs: **[docs.skelfresearch.com/gpuemu](https://docs.skelfresearch.com/gpuemu)**
- Compared to alternatives: **[the gap gpuemu fills](https://docs.skelfresearch.com/gpuemu/why-gpuemu/compared-to)**

## License

Dual-licensed under [MIT](../../LICENSE-MIT) or [Apache 2.0](../../LICENSE-APACHE) at your option.
