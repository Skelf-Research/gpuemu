# Roadmap

This roadmap is structured as staged deliverables that keep the workflow useful from day one.

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Core daemon | Rust | Performance, safety, cross-platform |
| Storage | sled | Embedded DB for results, baselines, artifacts |
| Serialization | rkyv | Zero-copy deserialization for tensor metadata |
| IPC | async-nng | Async messaging between daemon and clients |
| Python client | gpuemu-py | Framework integration (PyTorch, JAX, TensorFlow) |

## Phase 1: Core Validation Loop

**Goal**: Basic CPU mirror execution and validation

- [ ] Rust daemon skeleton with async-nng IPC
- [ ] TOML configuration parser (`gpuemu.toml`)
- [ ] Reference script executor (subprocess + pickle protocol)
- [ ] Basic validation engine (tolerance comparison)
- [ ] sled storage for results
- [ ] CLI: `gpuemu daemon start/stop`, `gpuemu test`
- [ ] Python client: `gpuemu-py` with `validate()` context manager

**Deliverable**: Run a single op validation against CPU reference

## Phase 2: Fuzzing and Reproducibility

**Goal**: Shape/layout fuzzing with deterministic seeds

- [ ] Seeded RNG for reproducible fuzzing
- [ ] Shape fuzzer (batch, sequence, edge cases)
- [ ] Layout fuzzer (contiguous, strided, transposed)
- [ ] rkyv serialization for tensor metadata
- [ ] Failure storage with full repro info
- [ ] CLI: `gpuemu reproduce --seed`, `gpuemu minimize`

**Deliverable**: Reproduce any failure with exact inputs

## Phase 3: Artifact Inspection

**Goal**: PTX/SASS analysis without GPU execution

- [ ] PTX parser for register/spill extraction
- [ ] SASS parser (via cuobjdump when available)
- [ ] Policy rules (max registers, required patterns)
- [ ] Baseline storage and diffing
- [ ] CLI: `gpuemu lint`, `gpuemu diff --baseline`

**Deliverable**: Detect artifact regressions in CI

## Phase 4: CI Integration

**Goal**: Production-ready CI workflows

- [ ] GitHub Actions template
- [ ] GitLab CI template
- [ ] JSON/JUnit output formats
- [ ] Parallel validation jobs
- [ ] CLI: `gpuemu ci`, `gpuemu report --format json`

**Deliverable**: Drop-in CI validation without GPU runners

## Phase 5: Framework Integrations

**Goal**: Deep integration with ML frameworks

- [ ] PyTorch: custom op validation, autograd checks
- [ ] JAX: primitive validation, vmap/pmap compatibility
- [ ] TensorFlow: custom op, gradient tape validation
- [ ] Cross-framework tolerance calibration

**Deliverable**: Validate any framework's custom ops

## Phase 6: Developer UX

**Goal**: Polished developer experience

- [ ] `gpuemu init` project scaffolding
- [ ] Interactive debugging mode
- [ ] VS Code extension
- [ ] Documentation site
- [ ] Binary releases (no cargo required)

**Deliverable**: Frictionless onboarding

## Phase 7: Expanded Platform Support

**Goal**: Broader platform and toolchain support

- [ ] macOS full workflow validation
- [ ] Windows daemon support
- [ ] AMD HIP artifact inspection
- [ ] Intel oneAPI support

**Deliverable**: Cross-platform, cross-vendor validation
