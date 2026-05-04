# Project Scope

## Mission

Enable GPU-less development, assessment, and validation of deep learning kernels by providing CPU-based execution mirrors, rigorous correctness checks, and policy-driven artifact inspection.

## Primary users

- ML infrastructure teams building custom kernels or operators for PyTorch, JAX, or TensorFlow.
- Compiler/kernel teams validating kernel logic before hardware access.
- Startups or research teams without dedicated GPU CI runners.
- Teams maintaining custom CUDA extensions across multiple ML frameworks.

## Goals

1. **Correctness-first validation** without requiring a GPU.
2. **Deterministic, CI-friendly testing** with reproducible seeds and clear failure modes.
3. **Deep learning-aware checks** for shapes, layouts, dtypes, and numerical stability.
4. **Artifact-based performance guardrails** to detect structural regressions early.
5. **Cross-platform developer workflow**, with macOS support where feasible.

## Non-goals

- Cycle-accurate GPU simulation or performance benchmarking.
- Replacing hardware validation for final performance and hardware-specific behavior.
- Building or hosting a training/inference runtime.

## Success criteria

- Teams can validate kernel logic and layout assumptions without a GPU.
- CI failures are actionable and point to specific correctness or structural issues.
- Artifact diffs reveal regressions (register pressure, spills, missing vectorization patterns).
- The core workflow runs on macOS and Linux without special hardware.

## Constraints

- Vendor toolchains may be required for artifact generation and are not always available on macOS.
- Performance validation remains a hardware-dependent phase outside the scope of this tool.

