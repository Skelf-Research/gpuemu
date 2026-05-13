# Environment Variables

All environment variables recognized by gpuemu, including the daemon, CLI, and Python
client.

---

## Summary

| Variable | Purpose | Default |
|----------|---------|---------|
| `GPUEMU_CONFIG` | Path to `gpuemu.toml` configuration file | Auto-discovered |
| `GPUEMU_SOCKET` | Path to daemon Unix socket | `~/.gpuemu/gpuemu.sock` |
| `GPUEMU_LOG_LEVEL` | Daemon log verbosity level | `info` |
| `GPUEMU_NO_COLOR` | Disable colored CLI output | *(unset)* |
| `RUST_LOG` | Rust tracing filter for daemon internals | *(unset)* |

---

## `GPUEMU_CONFIG`

Override the automatic configuration file discovery. When set, gpuemu loads the
configuration from the specified path instead of searching up the directory tree.

**Type:** File path (absolute or relative)

**Default:** gpuemu walks from the current directory upward, looking for `gpuemu.toml`.

```bash
# Use a specific config file
export GPUEMU_CONFIG=/path/to/my/gpuemu.toml
gpuemu test
```

```bash
# One-off override
GPUEMU_CONFIG=./configs/ci.toml gpuemu ci --quick
```

!!! tip
    This is useful for monorepos or CI environments where different configurations
    are needed for different contexts (e.g., quick CI vs. thorough nightly runs).

---

## `GPUEMU_SOCKET`

Override the Unix domain socket path used for daemon communication. Both the daemon
and all clients (CLI, Python library) respect this variable.

**Type:** File path (absolute or relative)

**Default:** `~/.gpuemu/gpuemu.sock`

```bash
# Use a custom socket location
export GPUEMU_SOCKET=/tmp/gpuemu-dev.sock

# Start the daemon on the custom socket
gpuemu daemon start --background

# All subsequent commands use the same socket
gpuemu status
gpuemu test
```

!!! note
    When using `GPUEMU_SOCKET`, ensure both the daemon and clients use the same
    value. The Python client also accepts `socket_path` as a constructor argument,
    which takes precedence over this variable.

---

## `GPUEMU_LOG_LEVEL`

Control the log verbosity of the gpuemu daemon.

**Type:** One of `trace`, `debug`, `info`, `warn`, `error`

**Default:** `info`

| Level | Description |
|-------|-------------|
| `trace` | Extremely detailed output, including per-element comparisons |
| `debug` | Detailed diagnostic information for troubleshooting |
| `info` | General operational messages (default) |
| `warn` | Warnings about potential issues |
| `error` | Only error messages |

```bash
# Start the daemon with debug logging
GPUEMU_LOG_LEVEL=debug gpuemu daemon start
```

```bash
# Trace-level logging for deep debugging
GPUEMU_LOG_LEVEL=trace gpuemu daemon start 2>&1 | tee daemon.log
```

!!! warning
    `trace` level logging produces a very large volume of output and can impact
    performance. Use it only for targeted debugging sessions.

---

## `GPUEMU_NO_COLOR`

Disable colored and styled output from the gpuemu CLI. Set to any non-empty value
to activate.

**Type:** Any value (presence is checked, not content)

**Default:** *(unset -- colors enabled)*

```bash
# Disable colors
export GPUEMU_NO_COLOR=1
gpuemu test
```

```bash
# Useful for piping output to a file
GPUEMU_NO_COLOR=1 gpuemu report > report.txt
```

!!! info
    Colors are also automatically disabled when stdout is not a TTY (e.g., when
    piping output). This variable forces colors off even in TTY contexts.

---

## `RUST_LOG`

The gpuemu daemon is built in Rust and uses the `tracing` framework. The `RUST_LOG`
environment variable provides fine-grained control over which internal modules emit
log output.

**Type:** Tracing filter directive string

**Default:** *(unset -- defers to `GPUEMU_LOG_LEVEL`)*

!!! info
    `RUST_LOG` takes precedence over `GPUEMU_LOG_LEVEL` when set. The two can be
    used together, but `RUST_LOG` provides more granular control.

```bash
# Enable debug logging for the daemon's validation engine only
RUST_LOG=gpuemu_daemon::validation=debug gpuemu daemon start
```

```bash
# Trace NNG socket activity and info for everything else
RUST_LOG=info,gpuemu_daemon::ipc=trace gpuemu daemon start
```

```bash
# Silence everything except errors, but show validation warnings
RUST_LOG=error,gpuemu_daemon::validation=warn gpuemu daemon start
```

**Common module paths:**

| Module | Description |
|--------|-------------|
| `gpuemu_daemon` | Top-level daemon module |
| `gpuemu_daemon::ipc` | IPC message handling |
| `gpuemu_daemon::validation` | Op and kernel validation engine |
| `gpuemu_daemon::fuzz` | Fuzz testing engine |
| `gpuemu_daemon::artifacts` | Kernel artifact analysis |
| `gpuemu_daemon::ci` | CI runner |

---

## Usage in CI

A typical CI environment might configure multiple variables together:

=== "GitHub Actions"

    ```yaml
    env:
      GPUEMU_CONFIG: ./ci/gpuemu-ci.toml
      GPUEMU_LOG_LEVEL: warn
      GPUEMU_NO_COLOR: "1"

    steps:
      - name: Start daemon
        run: gpuemu daemon start --background

      - name: Run CI suite
        run: gpuemu ci --quick --format junit --output results.xml
    ```

=== "GitLab CI"

    ```yaml
    variables:
      GPUEMU_CONFIG: ./ci/gpuemu-ci.toml
      GPUEMU_LOG_LEVEL: warn
      GPUEMU_NO_COLOR: "1"

    test:
      script:
        - gpuemu daemon start --background
        - gpuemu ci --quick --format json --output report.json
      artifacts:
        reports:
          junit: report.json
    ```

=== "Shell script"

    ```bash
    #!/usr/bin/env bash
    set -euo pipefail

    export GPUEMU_CONFIG="./ci/gpuemu-ci.toml"
    export GPUEMU_LOG_LEVEL="warn"
    export GPUEMU_NO_COLOR=1

    gpuemu daemon start --background
    gpuemu ci --baseline main --fail-on-regression --format junit --output results.xml
    gpuemu daemon stop
    ```
