# Architecture

This project is organized around a CPU-first validation pipeline that can optionally inspect GPU toolchain artifacts. The goal is to provide early correctness and structural performance checks without hardware.

## System Architecture

gpuemu uses a daemon + client architecture built in Rust:

```
┌─────────────────────────────────────────────────────────────────┐
│                        gpuemu daemon                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Validation │  │  Artifact   │  │  Storage (sled)         │  │
│  │  Engine     │  │  Inspector  │  │  - results, baselines   │  │
│  └─────────────┘  └─────────────┘  │  - artifact history     │  │
│         │               │          └─────────────────────────┘  │
│         └───────┬───────┘                     │                 │
│                 │                             │                 │
│         ┌───────▼───────┐            ┌───────▼───────┐         │
│         │ Policy Engine │            │ Serialization │         │
│         │               │            │ (rkyv)        │         │
│         └───────────────┘            └───────────────┘         │
│                          │                                      │
│                  ┌───────▼───────┐                              │
│                  │  IPC Layer    │                              │
│                  │  (async-nng)  │                              │
│                  └───────┬───────┘                              │
└──────────────────────────┼──────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
    ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
    │ Python      │ │ CLI         │ │ CI Runner   │
    │ Client      │ │ gpuemu      │ │             │
    │ (gpuemu-py) │ │             │ │             │
    └─────────────┘ └─────────────┘ └─────────────┘
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Core daemon | Rust | Performance, safety, cross-platform |
| Storage | sled | Embedded DB for results, baselines, artifact history |
| Serialization | rkyv | Zero-copy deserialization for fast tensor metadata handling |
| IPC | async-nng | Async messaging between daemon and clients |
| Python client | gpuemu-py | Thin client for PyTorch/JAX/TF integration |

## Components

1. **Kernel Contract**
   - A structural convention that separates kernel math from launch mapping.
   - The same math function is used by both GPU and CPU paths.
   - Defined in TOML configuration, not code.

2. **CPU Mirror Runner**
   - Executes kernel logic on CPU with deterministic indexing.
   - Supports layout and shape fuzzing with reproducible seeds.
   - Runs in the daemon process for isolation.

3. **Reference Implementations**
   - Canonical CPU implementations for each kernel/operator.
   - Defined as external executables or Python scripts referenced in config.
   - Daemon invokes them and captures outputs.

4. **Validation Engine**
   - Compares CPU mirror results against reference outputs.
   - Enforces numeric tolerances, NaN/Inf policies, and invariants.
   - Results stored in sled for history and diffing.

5. **Artifact Inspector**
   - Parses compiled GPU artifacts (PTX/IR/metadata when available).
   - Applies policy checks for register pressure, spills, and expected patterns.
   - Artifact snapshots stored in sled for regression detection.

6. **Policy Layer**
   - Defines pass/fail thresholds and warning levels.
   - Configured via TOML files.
   - Enables org-specific gating rules for CI.

7. **CLI + Client Libraries**
   - `gpuemu` CLI for direct daemon interaction.
   - `gpuemu` Python package for framework integration.
   - Both communicate with daemon via async-nng.

## Data flow (high level)

```
Client Request (async-nng)
       │
       ▼
┌──────────────────┐
│ 1. Build         │ Compile kernels, extract artifacts
│    (optional)    │ Store artifact metadata in sled
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 2. Mirror        │ Execute CPU reference implementations
│                  │ Fuzz shapes/layouts with seeded RNG
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 3. Validate      │ Compare outputs against references
│                  │ Check tolerances, NaN/Inf, invariants
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 4. Inspect       │ Parse PTX/IR artifacts
│                  │ Apply policy rules, diff against baseline
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 5. Report        │ Serialize results (rkyv)
│                  │ Store in sled, return to client
└──────────────────┘
```

## Daemon Lifecycle

```bash
# Start daemon (background)
gpuemu daemon start

# Daemon stores data in ~/.gpuemu/
#   ~/.gpuemu/db/          # sled database
#   ~/.gpuemu/gpuemu.sock  # nng socket

# Stop daemon
gpuemu daemon stop
```

## Kernel Contract (concept)

The kernel contract is a design rule that makes GPU logic portable to CPU:

- Kernel math lives in a `host+device`-friendly function.
- The mapping from threads to elements is expressed as a pluggable indexer.
- CPU execution uses the same math function with a virtualized thread indexer.

Contracts are declared in TOML, not code:

```toml
# gpuemu.toml
[[kernels]]
name = "fused_add_relu"
reference = "python scripts/ref_fused_add_relu.py"
tolerances = { float32 = 1e-5, float16 = 1e-3 }
invariants = ["non_negative", "shape_preserved"]
```

## Platform considerations

- The daemon runs on macOS and Linux (Rust binary).
- Windows support planned for future releases.
- Artifact inspection requires vendor toolchains (nvcc, cuobjdump) but not GPUs.
- Python client (`gpuemu`) supports Python 3.9+.

