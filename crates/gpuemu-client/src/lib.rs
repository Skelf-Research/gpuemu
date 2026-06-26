//! Rust-native client for the gpuemu validation daemon.
//!
//! This is the Rust equivalent of the Python `Client.fuzz_op_client_side` path:
//! it lets a Rust caller drive the daemon's fuzzer + fp64 oracle directly over
//! the JSON-over-NNG protocol, without going through Python/torch. The kernel
//! under test runs in a caller-supplied closure (`run_op`); gpuemu generates
//! fuzzed inputs and validates the closure's output against the reference.
//!
//! ```no_run
//! use gpuemu_client::Client;
//! use gpuemu_common::types::ValueDistribution;
//!
//! let client = Client::connect()?;
//! let summary = client.fuzz_builtin(
//!     "matmul",
//!     64,
//!     /* seed */ 42,
//!     ValueDistribution::Adversarial,
//!     |inputs| {
//!         // run your kernel on `inputs`, return its output tensor
//!         my_matmul(&inputs["a"], &inputs["b"])
//!     },
//! )?;
//! println!("{}/{} passed", summary.passed, summary.total);
//! # fn my_matmul(_a: &gpuemu_common::types::TensorData, _b: &gpuemu_common::types::TensorData)
//! #   -> gpuemu_common::types::TensorData { unimplemented!() }
//! # Ok::<(), gpuemu_client::ClientError>(())
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use gpuemu_common::default_socket_path;
use gpuemu_common::protocol::{
    deserialize_response, serialize_request, ErrorCode, Request, Response, PROTOCOL_VERSION,
};
use gpuemu_common::types::{FuzzConfig, OpSchema, TensorData, ValidationResult, ValueDistribution};
use nng::options::Options;
use nng::{Protocol, Socket};

/// Default receive timeout for daemon requests.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Errors returned by the client.
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    /// Failed to create/configure/connect the NNG socket.
    #[error("transport error: {0}")]
    Transport(String),
    /// Failed to (de)serialize a protocol message.
    #[error("protocol (de)serialization error: {0}")]
    Codec(String),
    /// The daemon returned an `Error` response.
    #[error("daemon error [{code:?}]: {message}")]
    Daemon {
        /// Daemon-supplied error code.
        code: ErrorCode,
        /// Human-readable message.
        message: String,
    },
    /// The daemon replied with an unexpected response variant.
    #[error("unexpected response to {request}: {got}")]
    Unexpected {
        /// The request kind that was sent.
        request: &'static str,
        /// A short description of what came back.
        got: String,
    },
    /// Client and daemon disagree on protocol version.
    #[error("protocol version mismatch: client {client}, daemon {daemon}")]
    VersionMismatch {
        /// This client's protocol version.
        client: u32,
        /// The daemon's protocol version.
        daemon: u32,
    },
}

type Result<T> = std::result::Result<T, ClientError>;

/// Outcome of a client-side fuzzing run.
#[derive(Debug, Clone, Default)]
pub struct FuzzSummary {
    /// Total cases attempted.
    pub total: usize,
    /// Cases that passed validation.
    pub passed: usize,
    /// Cases that failed validation.
    pub failed: usize,
    /// The failing validation results (each carries the replay seed).
    pub failures: Vec<ValidationResult>,
}

/// A connected gpuemu daemon client.
///
/// Holds a single `Req0` socket; requests are synchronous request/reply.
pub struct Client {
    socket: Socket,
}

impl Client {
    /// Connect to the daemon at the default socket path and verify protocol
    /// compatibility.
    pub fn connect() -> Result<Self> {
        Self::connect_to(&default_socket_path(), DEFAULT_TIMEOUT)
    }

    /// Connect to the daemon at a specific socket path with a custom timeout.
    pub fn connect_to(socket_path: &Path, timeout: Duration) -> Result<Self> {
        let socket_url = format!("ipc://{}", socket_path.display());
        let socket = Socket::new(Protocol::Req0)
            .map_err(|e| ClientError::Transport(format!("create socket: {e}")))?;
        socket
            .set_opt::<nng::options::RecvTimeout>(Some(timeout))
            .map_err(|e| ClientError::Transport(format!("set timeout: {e}")))?;
        socket
            .dial(&socket_url)
            .map_err(|e| ClientError::Transport(format!("dial {socket_url}: {e}")))?;

        let client = Self { socket };
        client.check_version()?;
        Ok(client)
    }

    /// Ping the daemon: returns `(version, protocol_version, uptime_secs)`.
    pub fn ping(&self) -> Result<(String, u32, u64)> {
        match self.request("Ping", &Request::Ping)? {
            Response::Pong {
                version,
                protocol_version,
                uptime_secs,
            } => Ok((version, protocol_version, uptime_secs)),
            other => Err(unexpected("Ping", &other)),
        }
    }

    fn check_version(&self) -> Result<()> {
        let (_, daemon, _) = self.ping()?;
        if daemon != PROTOCOL_VERSION {
            return Err(ClientError::VersionMismatch {
                client: PROTOCOL_VERSION,
                daemon,
            });
        }
        Ok(())
    }

    /// Fetch a batch of fuzzed test cases from the daemon.
    pub fn get_test_batch(
        &self,
        op_name: &str,
        fuzz_config: FuzzConfig,
        count: usize,
    ) -> Result<Vec<gpuemu_common::protocol::TestCaseData>> {
        let req = Request::GetTestBatch {
            op_name: op_name.to_string(),
            fuzz_config,
            count,
        };
        match self.request("GetTestBatch", &req)? {
            Response::TestBatch { cases } => Ok(cases),
            Response::Error { code, message } => Err(ClientError::Daemon { code, message }),
            other => Err(unexpected("GetTestBatch", &other)),
        }
    }

    /// Submit an op output for validation against the (fp64) reference.
    pub fn submit_output(
        &self,
        op_name: &str,
        inputs: HashMap<String, TensorData>,
        output: TensorData,
        seed: u64,
        kwargs: HashMap<String, String>,
    ) -> Result<ValidationResult> {
        let req = Request::SubmitOutput {
            op_name: op_name.to_string(),
            inputs,
            output,
            seed,
            kwargs,
        };
        match self.request("SubmitOutput", &req)? {
            Response::SubmitResult { result } => Ok(result),
            Response::Error { code, message } => Err(ClientError::Daemon { code, message }),
            other => Err(unexpected("SubmitOutput", &other)),
        }
    }

    /// Client-side fuzzing: generate `iterations` cases, run each through
    /// `run_op`, and validate the output against the reference oracle.
    ///
    /// `run_op` receives the fuzzed inputs and returns the op's output tensor —
    /// this is where the caller invokes the kernel under test.
    pub fn fuzz_op_client_side<F>(
        &self,
        op_name: &str,
        fuzz_config: FuzzConfig,
        iterations: usize,
        mut run_op: F,
    ) -> Result<FuzzSummary>
    where
        F: FnMut(&HashMap<String, TensorData>) -> TensorData,
    {
        let cases = self.get_test_batch(op_name, fuzz_config, iterations)?;
        let mut summary = FuzzSummary::default();
        for case in cases {
            summary.total += 1;
            let output = run_op(&case.inputs);
            let result =
                self.submit_output(op_name, case.inputs, output, case.seed, HashMap::new())?;
            if result.passed {
                summary.passed += 1;
            } else {
                summary.failed += 1;
                summary.failures.push(result);
            }
        }
        Ok(summary)
    }

    /// Convenience wrapper over [`fuzz_op_client_side`](Self::fuzz_op_client_side)
    /// for an op that has a built-in schema ([`OpSchema::builtin`]). Builds a
    /// `FuzzConfig` from the schema with the requested value distribution.
    pub fn fuzz_builtin<F>(
        &self,
        op_name: &str,
        iterations: usize,
        seed: u64,
        value_distribution: ValueDistribution,
        run_op: F,
    ) -> Result<FuzzSummary>
    where
        F: FnMut(&HashMap<String, TensorData>) -> TensorData,
    {
        let mut cfg = FuzzConfig::with_seed(seed);
        cfg.value_distribution = value_distribution;
        cfg.op_schema = OpSchema::builtin(op_name);
        self.fuzz_op_client_side(op_name, cfg, iterations, run_op)
    }

    /// Send a request and return the response, surfacing transport/codec errors.
    fn request(&self, kind: &'static str, request: &Request) -> Result<Response> {
        let bytes = serialize_request(request).map_err(|e| ClientError::Codec(format!("{e:?}")))?;
        self.socket
            .send(&bytes)
            .map_err(|(_, e)| ClientError::Transport(format!("send {kind}: {e}")))?;
        let response = self
            .socket
            .recv()
            .map_err(|e| ClientError::Transport(format!("recv {kind}: {e}")))?;
        deserialize_response(&response).map_err(|e| ClientError::Codec(format!("{e:?}")))
    }
}

fn unexpected(request: &'static str, got: &Response) -> ClientError {
    ClientError::Unexpected {
        request,
        got: format!("{got:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuzz_builtin_uses_schema() {
        // Building a config for a known builtin attaches its schema.
        let mut cfg = FuzzConfig::with_seed(1);
        cfg.value_distribution = ValueDistribution::Boundary;
        cfg.op_schema = OpSchema::builtin("rmsnorm");
        assert!(cfg.op_schema.is_some());
        assert_eq!(cfg.value_distribution, ValueDistribution::Boundary);
    }

    #[test]
    fn fuzz_summary_defaults_are_zero() {
        let s = FuzzSummary::default();
        assert_eq!(s.total, 0);
        assert_eq!(s.passed, 0);
        assert!(s.failures.is_empty());
    }
}
