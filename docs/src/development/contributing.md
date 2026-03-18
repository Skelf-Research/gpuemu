# Contributing

## Development Setup

```bash
git clone https://github.com/example/gpuemu.git
cd gpuemu

# Build all crates
cargo build

# Run tests
cargo test

# Run CLI
cargo run --bin gpuemu -- --help
```

## Code Style

- Run `cargo fmt` before committing
- Run `cargo clippy` and fix warnings
- Add tests for new functionality

## Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `cargo test`
5. Submit PR

## Reporting Issues

Please include:
- gpuemu version (`gpuemu --version`)
- OS and architecture
- Steps to reproduce
- Expected vs actual behavior
