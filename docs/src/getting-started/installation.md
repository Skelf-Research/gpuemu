# Installation

## Quick Install (Recommended)

### Linux / macOS

```bash
curl -fsSL https://gpuemu.dev/install.sh | sh
```

This installs gpuemu to `~/.gpuemu/bin`. Add it to your PATH:

```bash
# bash/zsh
echo 'export PATH="$HOME/.gpuemu/bin:$PATH"' >> ~/.bashrc

# fish
fish_add_path ~/.gpuemu/bin
```

### Windows

Download the latest release from [GitHub Releases](https://github.com/example/gpuemu/releases) and extract to a directory in your PATH.

## From Source

### Prerequisites

- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- Python 3.8+ (for reference scripts)

### Build

```bash
git clone https://github.com/example/gpuemu.git
cd gpuemu
cargo build --release

# Binaries are in target/release/
./target/release/gpuemu --version
```

## Python Client

Install the Python client for programmatic access:

```bash
pip install gpuemu
```

Or with framework-specific extras:

```bash
pip install gpuemu[pytorch]   # PyTorch support
pip install gpuemu[jax]       # JAX support
pip install gpuemu[all]       # All frameworks
```

## VS Code Extension

Install from the VS Code marketplace:

1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "gpuemu"
4. Click Install

Or install from VSIX:

```bash
code --install-extension gpuemu-0.1.0.vsix
```

## Verify Installation

```bash
# Check CLI version
gpuemu --version

# Check daemon can start
gpuemu daemon start
gpuemu status
gpuemu daemon stop
```

## Next Steps

- [Quick Start Guide](./quickstart.md)
- [Configuration Reference](./configuration.md)
