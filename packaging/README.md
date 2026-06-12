# Packaging

Distribution recipes for gpuemu beyond `cargo install` / `pip install` /
`docker run`. The release workflow `.github/workflows/release.yml` already
publishes:

- pre-built binaries to GitHub Releases (linux/macos/windows × x86_64/arm64),
- the Rust crates to crates.io,
- the Python client to PyPI,
- the Docker image to ghcr.io.

This directory adds two more channels for users who don't reach for `cargo
install`:

## `conda/meta.yaml` — conda-forge recipe

Submit by forking [conda-forge/staged-recipes][cfsr] and PR'ing
`recipes/gpuemu/meta.yaml`. The `sha256` placeholder is filled by the
conda-forge bot using the release tarball; we only need to bump the `version`
on tag.

Once accepted, users install with:

```bash
conda install -c conda-forge gpuemu
```

[cfsr]: https://github.com/conda-forge/staged-recipes

## `homebrew/gpuemu.rb` — Homebrew formula

Two delivery paths. Until acceptance into homebrew-core, ship via a tap:

```bash
brew tap skelf-research/gpuemu https://github.com/Skelf-Research/homebrew-gpuemu
brew install gpuemu
```

The tap repo is a new minimal repo `Skelf-Research/homebrew-gpuemu` with a
single `Formula/gpuemu.rb` that we copy from here on every release. The
SHA-256 placeholders are filled by running `brew fetch --build-from-source
gpuemu` against the release tag and recording the resulting hashes.

Once stable, PR the formula into `homebrew-core`. Their acceptance criteria
(maintained ≥ 3 months, ≥ 50 GitHub stars, no vendored binaries beyond what
HB-core ships for) align with gpuemu's distribution shape.

## Future channels

- **Nix** — `pkgs/development/tools/gpuemu/default.nix` with the same
  release-tarball + SHA-256 pattern. Low-priority; the `cargo install` route
  works fine for Nix users today.
- **APT / DEB** — out of scope until the binary release shape stabilises.
