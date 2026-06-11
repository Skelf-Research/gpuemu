# Multi-stage build for the gpuemu daemon + CLI.
#
# Stage 1 (builder) — Rust toolchain on bookworm to compile the gpuemu workspace
# in release mode. Uses cargo-chef-style caching: copy Cargo.{toml,lock} +
# stub-out the src/ trees first so the dependency graph is cached separately from
# the source.
#
# Stage 2 (runtime) — debian:bookworm-slim with the gpuemu binaries and Python
# bindings installed via pip. PyTorch / Triton are NOT preinstalled by default
# (they bloat the image to >5 GB); set --build-arg WITH_TORCH=1 to opt in for
# the CI image used by the gpuemu validate-action.

# ----- Stage 1: builder -----
FROM rust:1.83-bookworm AS builder

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates/gpuemu-common/Cargo.toml crates/gpuemu-common/Cargo.toml
COPY crates/gpuemu-daemon/Cargo.toml crates/gpuemu-daemon/Cargo.toml
COPY crates/gpuemu-cli/Cargo.toml crates/gpuemu-cli/Cargo.toml

# Stub crates so the dependency build can be cached separately from the source.
RUN mkdir -p crates/gpuemu-common/src crates/gpuemu-daemon/src crates/gpuemu-cli/src && \
    echo "fn main() {}" > crates/gpuemu-daemon/src/main.rs && \
    echo "fn main() {}" > crates/gpuemu-cli/src/main.rs && \
    echo "" > crates/gpuemu-common/src/lib.rs && \
    cargo build --release --workspace --locked && \
    rm -rf crates/gpuemu-common/src crates/gpuemu-daemon/src crates/gpuemu-cli/src

# Now copy the real source and build for real.
COPY crates/ crates/
RUN cargo build --release --workspace --locked

# ----- Stage 2: runtime -----
FROM debian:bookworm-slim AS runtime

ARG WITH_TORCH=0

# Runtime deps + Python for the gpuemu-py client.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/gpuemu /usr/local/bin/gpuemu
COPY --from=builder /build/target/release/gpuemu-daemon /usr/local/bin/gpuemu-daemon
RUN chmod +x /usr/local/bin/gpuemu /usr/local/bin/gpuemu-daemon

# Install the Python client. Without --break-system-packages this fails on
# Debian 12 (PEP 668); the image is single-purpose so the warning is moot.
COPY gpuemu-py /tmp/gpuemu-py
RUN pip install --break-system-packages --no-cache-dir /tmp/gpuemu-py && \
    rm -rf /tmp/gpuemu-py

# Optional: PyTorch + Triton for the CI variant used by validate-action.
RUN if [ "$WITH_TORCH" = "1" ]; then \
        pip install --break-system-packages --no-cache-dir \
            torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu && \
        pip install --break-system-packages --no-cache-dir triton numpy ; \
    fi

# Default working dir for `docker run -v $PWD:/work`.
WORKDIR /work

# The image's default ENTRYPOINT is the gpuemu CLI; commands like
# `docker run ghcr.io/skelf-research/gpuemu:vX ci --format sarif` work as-is.
ENTRYPOINT ["gpuemu"]
CMD ["--help"]
