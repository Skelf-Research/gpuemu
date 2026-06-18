# Contributing

Thank you for your interest in contributing to gpuemu. This guide covers everything you need to set up a development environment, understand the codebase, and submit changes.

---

## Prerequisites

Before you begin, make sure you have the following tools installed:

| Tool | Minimum Version | Purpose |
|------|-----------------|---------|
| **Rust** | 1.70+ | CLI and daemon development |
| **Python** | 3.9+ | Python client and reference scripts |
| **Node.js** | 18+ | VS Code extension development (optional) |
| **npm** | 9+ | VS Code extension dependencies (optional) |
| **Git** | any recent | Version control |

!!! tip "Check your versions"

    ```bash
    rustc --version    # 1.70.0 or higher
    python3 --version  # 3.9 or higher
    node --version     # 18.0.0 or higher (only for extension work)
    ```

---

## Repository Structure

The gpuemu repository is organized as a Rust workspace with co-located Python and TypeScript packages:

```
gpuemu/
├── Cargo.toml              # Workspace root
├── Cargo.lock
├── crates/
│   ├── gpuemu-common/      # Shared types, protocol, config, RNG
│   ├── gpuemu-daemon/      # Daemon server, validator, executor, fuzzer, storage
│    └── gpuemu/             # CLI entry point, debug REPL, init scaffolding, reports
├── gpuemu-py/              # Python client library
│   ├── gpuemu/          # Package source
│   │   ├── client.py       # IPC client
│   │   ├── validate.py     # Test generation
│   │   ├── rng.py          # Cross-language RNG
│   │   ├── tolerances.py   # Tolerance calibration
│   │   └── frameworks/     # PyTorch, JAX, TensorFlow adapters
│   ├── tests/              # Python test suite
│   └── pyproject.toml
├── vscode-gpuemu/          # VS Code extension
│   ├── src/
│   │   ├── extension.ts    # Extension entry point
│   │   ├── runner.ts       # CLI runner
│   │   ├── providers/      # Diagnostics, test controller, watchers
│   │   └── commands/       # Command palette commands
│   ├── package.json
│   └── tsconfig.json
├── documentation/          # MkDocs documentation site
│   ├── mkdocs.yml
│   └── docs/
├── scripts/                # Example reference scripts
└── templates/              # Init scaffolding templates
```

---

## Building

### Rust (CLI and Daemon)

Build all crates from the workspace root:

```bash
cargo build
```

For a release build (optimized):

```bash
cargo build --release
```

The resulting binaries are:

| Binary | Location | Crate |
|--------|----------|-------|
| `gpuemu` | `target/debug/gpuemu` or `target/release/gpuemu` | `gpuemu` |
| `gpuemu-daemon` | `target/debug/gpuemu-daemon` or `target/release/gpuemu-daemon` | `gpuemu-daemon` |

### Python Client

Install the Python package in editable (development) mode:

```bash
pip install -e ./gpuemu-py
```

This installs the package so that changes to the source files take effect immediately without reinstalling.

To install with framework extras for testing:

```bash
pip install -e ./gpuemu-py[torch,jax,tensorflow]
```

### VS Code Extension

```bash
cd vscode-gpuemu
npm install
npm run compile
```

To launch the extension in development mode, open the `vscode-gpuemu/` directory in VS Code and press ++f5++ to start the Extension Development Host.

---

## Testing

### Rust Tests

Run the full Rust test suite from the workspace root:

```bash
cargo test
```

This runs tests across all three crates (58 tests total). To run tests for a specific crate:

```bash
cargo test -p gpuemu-common
cargo test -p gpuemu-daemon
cargo test -p gpuemu
```

### Python Tests

Run the Python test suite using pytest:

```bash
cd gpuemu-py && pytest
```

The Python tests (11 tests) are organized into categories:

| Category | What it tests |
|----------|---------------|
| `smoke` | Basic client connectivity and protocol handshake |
| `pytorch` | PyTorch adapter integration |
| `jax` | JAX adapter integration |
| `tensorflow` | TensorFlow adapter integration |
| `tolerances` | Tolerance calibration and recommendation logic |

To run a specific category:

```bash
cd gpuemu-py && pytest -k "smoke"
cd gpuemu-py && pytest -k "tolerances"
```

!!! note "Framework tests require dependencies"

    The `pytorch`, `jax`, and `tensorflow` test categories require the respective framework to be installed. Tests for unavailable frameworks are automatically skipped.

### TypeScript Type Check

The VS Code extension does not have a dedicated test runner, but you can verify type correctness:

```bash
cd vscode-gpuemu && npm run compile
```

A successful compile with no errors confirms the TypeScript types are consistent.

---

## Code Style

### Rust

Rust code follows standard formatting and linting conventions:

```bash
# Format all Rust code
cargo fmt

# Run the linter
cargo clippy -- -D warnings
```

All code must pass `cargo fmt --check` and `cargo clippy -- -D warnings` with zero warnings before merging.

### Python

Python code uses [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check for lint issues
ruff check gpuemu-py/

# Auto-fix what can be fixed
ruff check --fix gpuemu-py/

# Format code
ruff format gpuemu-py/
```

### TypeScript

The VS Code extension uses ESLint:

```bash
cd vscode-gpuemu && npx eslint src/
```

---

## Pull Request Workflow

### 1. Fork and Clone

Fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/YOUR_USERNAME/gpuemu.git
cd gpuemu
```

### 2. Create a Branch

Create a descriptive branch name:

```bash
git checkout -b fix/tolerance-calibration-overflow
```

Use prefixes that indicate the type of change:

| Prefix | Use for |
|--------|---------|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `docs/` | Documentation changes |
| `refactor/` | Code restructuring without behavior change |
| `test/` | Adding or improving tests |

### 3. Implement and Test

Make your changes, then verify everything passes:

```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test
cd gpuemu-py && pytest && cd ..
```

### 4. Commit and Push

Write clear commit messages that explain the **why**, not just the what:

```bash
git add -A
git commit -m "Fix overflow in tolerance calibration for float16 dtypes

The calibration loop could overflow when accumulating max_diff values
for float16 tensors with large element counts. Switch to float64
accumulation to avoid precision loss."
```

Push to your fork:

```bash
git push origin fix/tolerance-calibration-overflow
```

### 5. Open a Pull Request

Open a PR against the `main` branch. In your PR description:

- **Reference any related issues** (e.g., "Fixes #42")
- **Describe what changed and why**
- **Include test evidence** (test output, new tests added)
- **Keep PRs focused** -- one logical change per PR

!!! tip "Small PRs merge faster"

    Large PRs are harder to review and more likely to conflict. If your change is substantial, consider splitting it into a stack of smaller PRs that each make sense independently.

---

## Architecture Guide

For a deep understanding of how the codebase is organized internally -- crate responsibilities, IPC protocol details, the validation pipeline, fuzzer design, and extension architecture -- see the [Architecture Internals](architecture-internals.md) page.

---

## License

gpuemu is dual-licensed under:

- [MIT License](https://opensource.org/licenses/MIT)
- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

You may choose either license. Contributions are accepted under the same dual-license terms.
