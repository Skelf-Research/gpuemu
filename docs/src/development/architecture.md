# Architecture

## Overview

gpuemu consists of three main components:

```
┌─────────────┐     IPC      ┌─────────────────┐
│  gpuemu CLI │◄────────────►│  gpuemu-daemon  │
└─────────────┘              └────────┬────────┘
                                      │
┌─────────────┐     IPC               │
│  gpuemu-py  │◄──────────────────────┘
└─────────────┘
```

### CLI (`gpuemu`)
Command-line interface for user interaction. Sends requests to daemon.

### Daemon (`gpuemu-daemon`)
Long-running service that:
- Manages validation workers
- Stores results in sled database
- Executes reference scripts
- Coordinates fuzz testing

### Python Client (`gpuemu-py`)
Python library for programmatic access. Framework-specific adapters for PyTorch, JAX, TensorFlow.

## Communication

Components communicate via NNG (nanomsg-next-gen) over Unix domain sockets.

Protocol: Request/Reply pattern with rkyv serialization.

## Storage

Results stored in sled embedded database at `~/.gpuemu/db/`.

## Crate Structure

```
crates/
├── gpuemu-common/   # Shared types, protocol, config
├── gpuemu-daemon/   # Daemon implementation
└── gpuemu-cli/      # CLI implementation
```
