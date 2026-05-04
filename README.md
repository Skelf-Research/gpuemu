# gpuemu

GPU-less development, assessment, and validation for deep learning kernels.

This project is focused on enabling teams to develop and validate GPU-oriented deep learning code without requiring access to physical GPUs. It provides CPU-based execution mirrors, static analysis of GPU artifacts, and CI-friendly validation so you can catch correctness and structural performance regressions early.

## What this is

- A development and CI toolchain for GPU-targeted kernels that works without GPUs.
- A correctness and validation harness that runs kernels on CPU with deterministic checks.
- A policy-driven analyzer that inspects build artifacts (e.g., PTX/IR) to flag regressions.

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

## Platform goals

- **Linux**: Primary development and CI target.
- **macOS**: Core validation workflow should run without GPUs, where toolchains allow.
- **Windows**: Not a current target; may be considered after Linux/macOS parity.

macOS support focuses on CPU mirror execution and validation. Vendor GPU toolchains may be unavailable or limited on macOS, so those checks are optional and environment-gated.

## Repository map

- `docs/DEVELOPER_GUIDE.md`: **start here** — lifecycle guide for all developer personas.
- `docs/PROJECT_SCOPE.md`: goals, non-goals, target users, and constraints.
- `docs/ARCHITECTURE.md`: component design and data flow (Rust daemon + Python client).
- `docs/WORKFLOW.md`: local + CI workflows and expected lifecycle.
- `docs/CONFIGURATION.md`: TOML configuration schema reference.
- `docs/VALIDATION.md`: validation taxonomy and policies.
- `docs/INTEGRATIONS.md`: PyTorch, JAX, and TensorFlow integration patterns.
- `docs/PLATFORM_SUPPORT.md`: OS/toolchain assumptions and caveats.
- `docs/ROADMAP.md`: staged build plan and milestones.

## Status

This repository is in a planning and design phase. The documents above define the intended behavior and interfaces so implementation can proceed with clear boundaries.

## Contributing

If you want to contribute, start with `docs/PROJECT_SCOPE.md` and `docs/ARCHITECTURE.md`, then propose changes as small, reviewable steps.
