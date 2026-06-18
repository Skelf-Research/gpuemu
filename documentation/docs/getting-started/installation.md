# Installation

This guide walks you through installing all four gpuemu components: the Rust CLI, the Python client library, and the VS Code extension.

---

## Prerequisites Checklist

Before you begin, make sure you have the following installed:

- [x] **Git** (any recent version)
- [ ] **Rust 1.70+** (for the CLI)
- [ ] **Python 3.9+** (for the Python client)
- [ ] **pip** or **uv** (Python package manager)
- [ ] **VS Code 1.85+** (optional, for the editor extension)
- [ ] **Node.js 18+** and **npm** (optional, only if building the VS Code extension from source)

!!! tip "Check your existing versions"

    ```bash
    rustc --version    # Should be 1.70.0 or higher
    python3 --version  # Should be 3.9 or higher
    code --version     # Should be 1.85 or higher
    ```

---

## 1. Rust CLI (`gpuemu`)

The CLI is the primary interface for running validations, controlling the daemon, fuzzing, and CI integration.

### Install Rust (if needed)

=== "Linux"

    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source "$HOME/.cargo/env"
    ```

=== "macOS"

    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source "$HOME/.cargo/env"
    ```

    !!! note "Xcode Command Line Tools"

        On macOS you may need to install Xcode Command Line Tools first:

        ```bash
        xcode-select --install
        ```

### Install from crates.io (recommended)

```bash
cargo install gpuemu
```

This builds and installs the `gpuemu` binary onto your `PATH` (under `~/.cargo/bin`). Verify it:

```bash
gpuemu --version
```

That's it — skip ahead to the [Python client](#2-python-client-gpuemu). The steps below are only needed if you'd rather build from a checkout.

### Build from source (alternative)

For contributors, or to track `main`, clone the repository and build the release binary:

```bash
git clone https://github.com/Skelf-Research/gpuemu.git
cd gpuemu
cargo build --release
```

The compiled binary will be at `target/release/gpuemu`.

#### Install the binary

=== "Linux"

    Copy the binary to a well-known location and add it to your `PATH`:

    ```bash
    mkdir -p ~/.gpuemu/bin
    cp target/release/gpuemu ~/.gpuemu/bin/gpuemu
    ```

    Add the following to your `~/.bashrc` or `~/.zshrc`:

    ```bash
    export PATH="$HOME/.gpuemu/bin:$PATH"
    ```

    Then reload your shell:

    ```bash
    source ~/.bashrc  # or source ~/.zshrc
    ```

=== "macOS"

    Copy the binary to a well-known location and add it to your `PATH`:

    ```bash
    mkdir -p ~/.gpuemu/bin
    cp target/release/gpuemu ~/.gpuemu/bin/gpuemu
    ```

    Add the following to your `~/.zshrc` (the default shell on macOS):

    ```bash
    export PATH="$HOME/.gpuemu/bin:$PATH"
    ```

    Then reload your shell:

    ```bash
    source ~/.zshrc
    ```

!!! info "Why `~/.gpuemu/bin`?"

    The `~/.gpuemu/` directory is also where the daemon stores its socket file (`gpuemu.sock`), database, and logs. Keeping the binary here keeps all gpuemu runtime artifacts in one place. You can install the binary anywhere on your `PATH` if you prefer.

---

## 2. Python Client (`gpuemu`)

The Python client provides programmatic access to the gpuemu daemon, including framework-specific adapters for PyTorch, JAX, and TensorFlow.

### Core installation

Install the package from PyPI:

```bash
pip install gpuemu
```

This installs the core library with the following dependencies:

| Dependency | Minimum Version |
|------------|-----------------|
| `pynng` | >= 0.8.0 |
| `numpy` | >= 1.20.0 |

!!! tip "Using a virtual environment"

    It is strongly recommended to install `gpuemu` inside a virtual environment:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install gpuemu
    ```

### Framework extras

Install optional framework-specific adapters using pip extras:

=== "PyTorch"

    ```bash
    pip install gpuemu[torch]
    ```

=== "JAX"

    ```bash
    pip install gpuemu[jax]
    ```

=== "TensorFlow"

    ```bash
    pip install gpuemu[tensorflow]
    ```

You can combine extras if you work with multiple frameworks:

```bash
pip install gpuemu[torch,jax]
```

---

## 3. VS Code Extension

The VS Code extension provides live diagnostics, code actions, test explorer integration, and on-save validation directly in your editor.

### Option A: Install from VSIX

If a pre-built `.vsix` file is available (e.g., from a release or CI artifact):

1. Open VS Code.
2. Open the Command Palette (++ctrl+shift+p++ on Linux, ++cmd+shift+p++ on macOS).
3. Type **"Extensions: Install from VSIX..."** and select it.
4. Browse to the `.vsix` file and install.

### Option B: Build from source

Navigate to the extension directory, install dependencies, and compile:

```bash
cd vscode-gpuemu/
npm install
npm run compile
```

Then either:

- **Package as VSIX**: Run `npx vsce package` to create a `.vsix` file, then install it via VS Code as described above.
- **Development mode**: Open the `vscode-gpuemu/` folder in VS Code and press ++f5++ to launch a development Extension Host.

!!! warning "Requirements"

    - **VS Code 1.85+** is required.
    - **Node.js 18+** and **npm** are required for building from source.
    - The extension expects the `gpuemu` CLI to be available on your `PATH`.

---

## 4. Verifying Installation

After installing each component, verify that everything is working correctly.

### CLI

```bash
gpuemu version
```

You should see output like:

```
gpuemu 0.1.0 (release)
```

### Python client

```bash
python -c "import gpuemu; print('ok')"
```

You should see:

```
ok
```

### VS Code extension

1. Open VS Code.
2. Open the Extensions panel (++ctrl+shift+x++ on Linux, ++cmd+shift+x++ on macOS).
3. Search for **"gpuemu"** in the installed extensions list.
4. Confirm the extension is listed and enabled.

!!! success "All set"

    If all three checks pass, your gpuemu installation is complete. Head to the [Quick Start](quickstart.md) guide to run your first validation.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `cargo build` fails with missing dependencies | Make sure you have a C linker installed. On Linux: `sudo apt install build-essential`. On macOS: `xcode-select --install`. |
| `pip install gpuemu` fails | Ensure you are using Python 3.9+ and pip is up to date: `pip install --upgrade pip`. |
| `gpuemu version` says "command not found" | Ensure `~/.gpuemu/bin` (or wherever you placed the binary) is in your `PATH`. Open a new terminal after editing your shell config. |
| VS Code extension does not activate | Check that the `gpuemu` CLI is on your `PATH` and that you are running VS Code 1.85 or higher. |
| `import gpuemu` raises `ModuleNotFoundError` | Make sure you installed the package in the same Python environment you are running. Check `which python` and `pip list | grep gpuemu`. |

---

## Next Steps

- [Quick Start](quickstart.md) -- Run your first validation in 5 minutes.
- [Configuration](configuration.md) -- Customize tolerances, dtypes, and policies.
