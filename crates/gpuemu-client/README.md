# gpuemu-client

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue?style=flat-square)](https://github.com/Skelf-Research/gpuemu)

**Rust-native client for the [gpuemu](https://github.com/Skelf-Research/gpuemu) validation daemon.**

Drive the daemon's op-schema fuzzer and reference oracle directly from Rust —
generate fuzzed inputs, run your kernel in a closure, and validate the output —
without going through the Python client. This is the Rust equivalent of the
Python `Client.fuzz_op_client_side` path.

```rust
use gpuemu_client::Client;
use gpuemu_common::types::ValueDistribution;

let client = Client::connect()?;                 // dials the daemon, checks protocol version

let summary = client.fuzz_builtin(
    "matmul",
    100,                                          // iterations
    42,                                           // seed
    ValueDistribution::Adversarial,               // boundary / regular / adversarial
    |inputs| my_matmul(&inputs["a"], &inputs["b"]),  // your kernel under test
)?;

println!("{}/{} passed", summary.passed, summary.total);
for failure in &summary.failures {
    // each failure carries the seed — replay with `gpuemu reproduce <seed>`
    println!("seed {} failed: {:?}", failure.seed, failure.failures);
}
```

For ops without a built-in schema, build a `FuzzConfig` (optionally with your
own `OpSchema`) and call `fuzz_op_client_side(op, cfg, iters, run_op)`. Lower
level, `get_test_batch` and `submit_output` mirror the daemon primitives.

## What it does / doesn't

- **Does** remove Python from the *driver*: input generation, the fuzz loop,
  and result handling are all Rust, over the JSON-over-NNG protocol.
- **Doesn't** remove Python from the *oracle*: the reference is still a Python
  script the daemon executes (see `[validation] oracle_fp64` and the
  `gpuemu.references` library for ready-made fp64 references).

Requires a running `gpuemu-daemon`. Connection verifies the protocol version on
connect and returns a typed `ClientError` on transport/codec/daemon errors.

## License

Dual-licensed under [MIT](../../LICENSE-MIT) or [Apache 2.0](../../LICENSE-APACHE) at your option.
