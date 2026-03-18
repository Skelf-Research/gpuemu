# Protocol Reference

gpuemu uses a request/response protocol over Unix domain sockets.

## Socket Location

Default: `~/.gpuemu/daemon.sock`

Override with `GPUEMU_SOCKET` environment variable.

## Serialization

Messages are serialized using rkyv for zero-copy deserialization.

## Request Types

- `Ping` - Health check
- `ValidateOp` - Validate single op
- `FuzzOp` - Run fuzz tests
- `Reproduce` - Reproduce failure
- `Minimize` - Minimize failure
- `ListFailures` - Get stored failures
- `LintKernel` - Analyze PTX
- `StoreBaseline` - Store artifact baseline
- `DiffBaseline` - Compare against baseline
- `RunCi` - Run CI suite
- `Shutdown` - Stop daemon

## Response Types

- `Pong` - Health check response
- `ValidationResult` - Single validation result
- `FuzzResults` - Fuzz test results
- `ReproduceResult` - Reproduction result
- `MinimizeResult` - Minimization result
- `Results` - List of results
- `LintResults` - Lint analysis
- `ArtifactDiffs` - Baseline comparison
- `CiRunComplete` - CI run summary
- `Error` - Error response
