# gpuemu-daemon

[![Crates.io](https://img.shields.io/crates/v/gpuemu-daemon?style=flat-square&logo=rust&color=orange)](https://crates.io/crates/gpuemu-daemon)
[![docs.rs](https://img.shields.io/docsrs/gpuemu-daemon?style=flat-square&logo=docsdotrs)](https://docs.rs/gpuemu-daemon)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue?style=flat-square)](https://github.com/Skelf-Research/gpuemu)

**The validation engine behind [gpuemu](https://github.com/Skelf-Research/gpuemu) — catch silently-wrong GPU kernels before they reach production, without a GPU.**

`gpuemu-daemon` is the long-running process that does the actual correctness work:
op-schema-aware input generation, an fp64 reference oracle, per-op calibrated tolerances,
static PTX/SASS lint, and durable failure storage. Clients — the
[`gpuemu`](https://crates.io/crates/gpuemu-cli) CLI, the
[`gpuemu-py`](https://pypi.org/project/gpuemu-py/) Python package, and the VS Code
extension — talk to it over IPC (NNG / Unix sockets).

---

## The problem it solves

Every modern training and inference stack ships LLM-generated CUDA/Triton kernels, and the
field-standard correctness oracle — `torch.allclose` on one shape — is blind to the bug
classes those kernels routinely contain. In a measured 26-op corpus it **accepts 9/9**
LLM-style buggy kernels as correct. A silently-wrong matmul or flash-attention then runs at
scale: GPU-hours wasted on broken work, quality regressions that survive months of green CI.

The daemon replaces that one-line check with an operator-domain–aware regime.

## What the daemon does

| Component (`src/`) | Capability | Measured finding |
|---|---|---|
| `validator` | fp64 reference oracle — validates kernel output against a high-precision CPU reference per dtype | **P1**: 100% illusion catch on 9/9 LLM-style bugs across 5 GPU classes; 0 false positives on 15/15 controls |
| `fuzzer` | Op-schema-aware shape generator with boundary + regular + adversarial value distributions | **P3**: 99% bug recall under adversarial values; +28 pp over the field-standard default |
| `validator` (tolerances) | Per-op p95-of-controls × 1.5 envelope, fit per op/dtype | **P2**: +23 pp recall (65 → 82%) over a single hand-picked `atol/rtol` |
| `artifact` | Static PTX/SASS lint — register pressure, spills, instruction counts | **P4**: structural Δregs predicts Δperf% across H100/A100/L40S/A10/3060 |
| `executor` | Runs reference + kernel scripts and snapshots exact inputs | Every flagged failure replays byte-for-byte from its seed |
| `storage` | Durable failure + baseline store (sled) | Reproduce and minimise any past failure |
| `server` | NNG/Unix-socket IPC server, parallel job execution | Drives CI fan-out from the CLI |

## Install & run

```bash
cargo install gpuemu-daemon

# Start the daemon (reads ~/.gpuemu/gpuemu.toml; falls back to sensible defaults)
gpuemu-daemon
```

Most users never invoke the daemon directly — the [`gpuemu`](https://crates.io/crates/gpuemu-cli)
CLI starts and manages it for you (`gpuemu daemon start`), and the Python client spawns it on
demand. Install it standalone when you want to run the engine as a service, in a container, or
behind your own client.

> **Requires** a `python3` interpreter on `PATH` for reference-script execution. The daemon
> checks this at startup and reports what's missing.

## Architecture

```
 gpuemu CLI ─┐
 Python      ├─ IPC (NNG / Unix sockets) ─▶  gpuemu-daemon
 VS Code     ┘                                ├── validator  (fp64 oracle + tolerances)
                                              ├── fuzzer     (op-schema input gen)
                                              ├── artifact   (PTX/SASS lint)
                                              ├── executor   (runs reference + kernel)
                                              └── storage    (sled: failures + baselines)
```

## Documentation

- API docs: **[docs.rs/gpuemu-daemon](https://docs.rs/gpuemu-daemon)**
- Project docs: **[docs.skelfresearch.com/gpuemu](https://docs.skelfresearch.com/gpuemu)**
- The evidence (P1–P4): **[The Evidence](https://docs.skelfresearch.com/gpuemu/why-gpuemu/the-evidence)**

## License

Dual-licensed under [MIT](../../LICENSE-MIT) or [Apache 2.0](../../LICENSE-APACHE) at your option.
