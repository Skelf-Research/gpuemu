# gpuemu-common

[![Crates.io](https://img.shields.io/crates/v/gpuemu-common?style=flat-square&logo=rust&color=orange)](https://crates.io/crates/gpuemu-common)
[![docs.rs](https://img.shields.io/docsrs/gpuemu-common?style=flat-square&logo=docsdotrs)](https://docs.rs/gpuemu-common)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue?style=flat-square)](https://github.com/Skelf-Research/gpuemu)

**Shared foundation for [gpuemu](https://github.com/Skelf-Research/gpuemu) — the GPU-less correctness oracle that catches silently-wrong GPU kernels before they reach production.**

This crate holds the types, wire protocol, deterministic RNG, and configuration that the
[`gpuemu-daemon`](https://crates.io/crates/gpuemu-daemon), the
[`gpuemu`](https://crates.io/crates/gpuemu) CLI, and the
[`gpuemu`](https://pypi.org/project/gpuemu/) Python client all speak. If you are
*using* gpuemu, install the CLI or the Python client — you only depend on `gpuemu-common`
directly when you are building a new gpuemu client, transport, or integration in Rust.

---

## Why gpuemu exists

The industry-standard correctness check for a GPU kernel is one line:

```python
torch.allclose(my_kernel(x), reference(x), atol=1e-5, rtol=1e-2)
```

One shape, one dtype, one seed. In a measured 26-op corpus that oracle **accepts 9/9**
LLM-style buggy kernels — tail-mask leaks, accumulator-scale bugs, missing normalisation,
online-softmax rescale errors — as "correct". gpuemu replaces it with an
operator-domain–aware regime: a CPU reference oracle, op-schema-aware (shape) fuzzing,
per-op tolerances, and static PTX/SASS lint. See the
[full project README](https://github.com/Skelf-Research/gpuemu#readme) for the P1–P4
evidence and the "Shipped vs. research regime" table marking what ships today.

## What's in this crate

| Module | What it provides |
|---|---|
| `types` | Core data model — ops, dtypes, tensor descriptors, validation results, failure records |
| `protocol` | The request/response messages exchanged over IPC between client and daemon |
| `rng` | `SeededRng` — a deterministic **xorshift128+** generator that is bit-identical to the Python client, so every flagged failure replays byte-for-byte |
| `config` | `GpuemuConfig`, `OpConfig`, `ValidationConfig`, `ExecutionMode` — the `gpuemu.toml` schema |

### Deterministic, cross-language reproducibility

The reproducibility guarantee that makes gpuemu failures actionable lives here. The same
seed produces the same input tensors in Rust and in Python:

```rust
use gpuemu_common::SeededRng;

let mut rng = SeededRng::new(0xC0FFEE);
let a = rng.next_u64();
// The Python client's SeededRng(0xC0FFEE) yields the identical stream — a failure found
// on a GPU box reproduces exactly on a laptop with no GPU.
```

## Install

```toml
[dependencies]
gpuemu-common = "0.1"
```

## Documentation

- API docs: **[docs.rs/gpuemu-common](https://docs.rs/gpuemu-common)**
- Project docs: **[docs.skelfresearch.com/gpuemu](https://docs.skelfresearch.com/gpuemu)**
- Architecture: **[Architecture deep dive](https://docs.skelfresearch.com/gpuemu/concepts/architecture)**

## License

Dual-licensed under [MIT](../../LICENSE-MIT) or [Apache 2.0](../../LICENSE-APACHE) at your option.
