# IPC Protocol Specification

This document describes the inter-process communication protocol used between the
gpuemu daemon and its clients (CLI, Python library, VS Code extension).

---

## Transport

| Property | Value |
|----------|-------|
| **Library** | NNG (nanomsg next generation) |
| **Pattern** | REQ/REP (request-reply) |
| **Socket type** | Unix domain socket |
| **Default path** | `~/.gpuemu/gpuemu.sock` |
| **Protocol version** | `1` |
| **Serialization** | JSON |

The daemon binds a REP socket. Clients connect with REQ sockets. Each message is a
single JSON object serialized as UTF-8. All communication is synchronous from the
client's perspective: one request produces exactly one response.

!!! info
    Override the socket path with the `GPUEMU_SOCKET` environment variable or the
    `socket_path` parameter in the Python client.

---

## Message Envelope

Every request and response follows the same envelope structure:

=== "Request"

    ```json
    {
      "type": "<RequestType>",
      "payload": { ... }
    }
    ```

=== "Response"

    ```json
    {
      "type": "<ResponseType>",
      "payload": { ... }
    }
    ```

The `type` field is a string tag that determines how `payload` is interpreted.

---

## Request Types

All request types recognized by the daemon.

| Type | Payload | Description |
|------|---------|-------------|
| `Ping` | `{}` | Health check |
| `Shutdown` | `{}` | Gracefully shut down the daemon |
| `ValidateOp` | `{ op_name, inputs, output, dtype?, seed? }` | Validate an op output against its reference |
| `GetResult` | `{ seed }` | Retrieve a stored validation result |
| `ListResults` | `{}` | List all stored validation results |
| `StoreBaseline` | `{ tag }` | Store current results as a named baseline |
| `CompareBaseline` | `{ tag, fail_on_regression? }` | Compare current results against a baseline |
| `FuzzOp` | `{ op_name, iterations?, seed? }` | Run fuzz testing on an op (daemon-side) |
| `Reproduce` | `{ seed }` | Reproduce a specific failure |
| `Minimize` | `{ seed, strategy?, max_iters? }` | Minimize a failing test case |
| `ListFailures` | `{ limit? }` | List stored fuzz failures |
| `LintKernel` | `{ kernel_name?, ptx_path? }` | Lint kernel artifacts |
| `StoreArtifact` | `{ kernel_name, metrics }` | Store artifact metrics for a kernel |
| `StoreArtifactBaseline` | `{ tag }` | Store current artifact metrics as a baseline |
| `DiffArtifactBaseline` | `{ tag, fail_on_regression? }` | Compare artifact metrics against a baseline |
| `GetArtifact` | `{ kernel_name }` | Retrieve artifact metrics for a kernel |
| `ListArtifacts` | `{}` | List all stored artifact metrics |
| `RunCi` | `{ quick?, baseline?, parallel?, format? }` | Run the CI validation suite |
| `GetCiSummary` | `{}` | Retrieve the most recent CI run summary |
| `GetTestCase` | `{ op_name, seed }` | Get a test case for client-side execution |
| `GetTestBatch` | `{ op_name, seeds }` | Get a batch of test cases for client-side execution |
| `SubmitOutput` | `{ op_name, seed, output }` | Submit client-side output for validation |

---

## Response Types

All response types returned by the daemon.

| Type | Payload | Description |
|------|---------|-------------|
| `Ok` | `{}` | Generic success acknowledgment |
| `Pong` | `{}` | Response to `Ping` |
| `ValidationResult` | `{ passed, seed, op_name, max_diff, ... }` | Result of a validation run |
| `Results` | `{ results: [...] }` | List of validation results |
| `BaselineComparison` | `{ regressions, improvements, unchanged }` | Baseline diff results |
| `Error` | `{ code, message }` | Error response |
| `FuzzResults` | `{ seed, total, passed, failed, failures }` | Fuzz testing results |
| `ReproduceResult` | `{ result, inputs }` | Reproduction of a failure |
| `MinimizeResult` | `{ original_seed, minimized_seed, ... }` | Minimized test case |
| `LintResults` | `{ warnings, errors, info }` | Kernel lint results |
| `ArtifactMetricsResult` | `{ kernel_name, registers, spills, ... }` | Artifact metrics |
| `ArtifactList` | `{ artifacts: [...] }` | List of artifact metrics |
| `ArtifactDiffs` | `{ diffs: [...] }` | Artifact baseline comparison |
| `CiRunComplete` | `{ summary, results, duration_ms }` | CI run results |
| `TestCase` | `{ seed, inputs, metadata }` | A single test case for client-side execution |
| `TestBatch` | `{ test_cases: [...] }` | A batch of test cases |
| `SubmitResult` | `{ result }` | Validation result after output submission |

---

## Error Codes

When the daemon returns an `Error` response, the `code` field contains one of the
following string values:

| Code | Description |
|------|-------------|
| `OpNotFound` | The requested op name is not defined in `gpuemu.toml` |
| `ReferenceScriptFailed` | The reference Python script exited with an error |
| `InvalidRequest` | The request payload is malformed or missing required fields |
| `InternalError` | An unexpected internal daemon error |
| `NotFound` | A generic "not found" error for results, seeds, etc. |
| `ConfigError` | The `gpuemu.toml` configuration is invalid |
| `KernelNotFound` | The requested kernel name is not defined in `gpuemu.toml` |
| `ArtifactNotFound` | No artifact metrics stored for the requested kernel |
| `BaselineNotFound` | The requested baseline tag does not exist |
| `PtxParseError` | Failed to parse the provided PTX assembly |
| `CuobjdumpNotAvailable` | The `cuobjdump` tool is not available on the system |
| `VersionMismatch` | Client protocol version does not match daemon protocol version |

**Error response example:**

```json
{
  "type": "Error",
  "payload": {
    "code": "OpNotFound",
    "message": "No op named 'softmax2' found in gpuemu.toml"
  }
}
```

!!! warning
    A `VersionMismatch` error indicates that the client and daemon are running
    incompatible protocol versions. Restart the daemon after upgrading gpuemu.

---

## TensorData Format

Tensors are serialized using the `TensorData` format across all messages that include
tensor values (inputs, outputs, gradients).

```json
{
  "shape": [4, 64],
  "strides": [64, 1],
  "dtype": "Float32",
  "data": "AAAA..."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `shape` | `Array<int>` | Tensor dimensions |
| `strides` | `Array<int>` | Element strides for each dimension |
| `dtype` | `String` | One of the `DType` enum values (see below) |
| `data` | `String` | Raw tensor bytes, base64-encoded |

!!! note
    The `data` field uses standard base64 encoding (RFC 4648). The byte order is
    little-endian, matching NumPy's default on x86/ARM systems.

---

## DType Values

The `dtype` field in `TensorData` uses the following enum values:

| Value | Size (bytes) | Description |
|-------|-------------|-------------|
| `Float16` | 2 | IEEE 754 half-precision float |
| `BFloat16` | 2 | Brain floating-point format |
| `Float32` | 4 | IEEE 754 single-precision float |
| `Float64` | 8 | IEEE 754 double-precision float |
| `Int8` | 1 | Signed 8-bit integer |
| `Int16` | 2 | Signed 16-bit integer |
| `Int32` | 4 | Signed 32-bit integer |
| `Int64` | 8 | Signed 64-bit integer |
| `UInt8` | 1 | Unsigned 8-bit integer |
| `UInt16` | 2 | Unsigned 16-bit integer |
| `UInt32` | 4 | Unsigned 32-bit integer |
| `UInt64` | 8 | Unsigned 64-bit integer |
| `Bool` | 1 | Boolean (0 or 1) |

---

## Example Conversations

### Ping / Pong

=== "Request"

    ```json
    {
      "type": "Ping",
      "payload": {}
    }
    ```

=== "Response"

    ```json
    {
      "type": "Pong",
      "payload": {}
    }
    ```

### Validate an Op

=== "Request"

    ```json
    {
      "type": "ValidateOp",
      "payload": {
        "op_name": "softmax",
        "inputs": {
          "logits": {
            "shape": [2, 8],
            "strides": [8, 1],
            "dtype": "Float32",
            "data": "AAAAPwAAAD8AAAA/..."
          }
        },
        "output": {
          "shape": [2, 8],
          "strides": [8, 1],
          "dtype": "Float32",
          "data": "zczMPc3MzD3NzMw9..."
        },
        "dtype": "float32"
      }
    }
    ```

=== "Response (pass)"

    ```json
    {
      "type": "ValidationResult",
      "payload": {
        "passed": true,
        "seed": 7293481056,
        "op_name": "softmax",
        "max_diff": 2.384185791015625e-07,
        "max_rel_diff": 1.9073486328125e-06,
        "failures": [],
        "timestamp": "2026-04-13T10:30:00Z",
        "duration_ms": 42,
        "repro_info": null
      }
    }
    ```

=== "Response (fail)"

    ```json
    {
      "type": "ValidationResult",
      "payload": {
        "passed": false,
        "seed": 7293481056,
        "op_name": "softmax",
        "max_diff": 0.015625,
        "max_rel_diff": 0.125,
        "failures": [
          "Absolute tolerance exceeded: max_diff=0.015625 > atol=1e-05"
        ],
        "timestamp": "2026-04-13T10:30:00Z",
        "duration_ms": 45,
        "repro_info": {
          "seed": 7293481056,
          "shape": [2, 8],
          "strides": [8, 1],
          "dtype": "float32",
          "layout": "contiguous",
          "fuzz_config": null,
          "input_snapshot": {}
        }
      }
    }
    ```

### Fuzz an Op

=== "Request"

    ```json
    {
      "type": "FuzzOp",
      "payload": {
        "op_name": "softmax",
        "iterations": 200,
        "seed": 42
      }
    }
    ```

=== "Response"

    ```json
    {
      "type": "FuzzResults",
      "payload": {
        "seed": 42,
        "total": 200,
        "passed": 198,
        "failed": 2,
        "failures": [
          {
            "passed": false,
            "seed": 9128374650,
            "op_name": "softmax",
            "max_diff": 0.03125,
            "max_rel_diff": 0.25,
            "failures": ["Absolute tolerance exceeded"],
            "timestamp": "2026-04-13T10:31:00Z",
            "duration_ms": 12,
            "repro_info": { "seed": 9128374650, "shape": [128, 512], "strides": [512, 1], "dtype": "float32", "layout": "contiguous", "fuzz_config": null, "input_snapshot": {} }
          }
        ]
      }
    }
    ```
