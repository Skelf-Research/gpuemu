# Platform Support

This project is designed to run without GPUs, and to operate on common developer machines and CI runners.

## Linux

- Primary development and CI target.
- Full workflow intended to work when toolchains are available.
- Artifact inspection depends on installed vendor tooling.

## macOS

- Core CPU mirror execution and validation should work.
- Artifact inspection is optional and gated by tool availability.
- GPU toolchains may be limited or unavailable; the workflow must gracefully degrade.

## Windows

- Not currently targeted.
- Consideration after Linux and macOS parity is achieved.

## Toolchain assumptions

- CPU-based validation is the baseline and should have minimal dependencies.
- Optional toolchain integrations are modular and should not block core workflows.

