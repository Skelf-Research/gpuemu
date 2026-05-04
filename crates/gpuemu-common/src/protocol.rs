//! IPC protocol messages for daemon-client communication.

use crate::types::{FuzzConfig, TensorData, ValidationResult};
use rkyv::{Archive, Deserialize, Serialize};
use std::collections::HashMap;

/// Strategy for minimizing a failing test case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub enum MinimizeStrategy {
    /// Binary search on tensor dimensions to find smallest failing shape.
    BinarySearchDims,
    /// Binary search on tensor values to find minimal failing input.
    BinarySearchValues,
}

/// Request messages sent from clients to the daemon.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub enum Request {
    /// Validate an op against its reference implementation.
    ValidateOp {
        /// Name of the op to validate.
        op_name: String,
        /// Input tensors by name.
        inputs: HashMap<String, TensorData>,
        /// Output tensor from the op being validated.
        output: TensorData,
        /// Optional kwargs for the reference script.
        kwargs: HashMap<String, String>,
    },

    /// Get a stored validation result by seed.
    GetResult {
        /// Seed of the validation run.
        seed: u64,
    },

    /// List recent validation results.
    ListResults {
        /// Maximum number of results to return.
        limit: usize,
    },

    /// Store a baseline for comparison.
    StoreBaseline {
        /// Tag name for the baseline.
        tag: String,
    },

    /// Compare current results against a baseline.
    CompareBaseline {
        /// Tag name of the baseline to compare against.
        tag: String,
    },

    /// Check daemon health.
    Ping,

    /// Request daemon shutdown.
    Shutdown,

    // =========================================================================
    // Phase 2: Fuzzing and Reproducibility
    // =========================================================================

    /// Fuzz an op with seeded random inputs.
    FuzzOp {
        /// Name of the op to fuzz.
        op_name: String,
        /// Fuzzing configuration.
        fuzz_config: FuzzConfig,
        /// Number of test iterations to run.
        iterations: usize,
        /// Stop on first failure.
        fail_fast: bool,
    },

    /// Reproduce a specific failure by seed.
    Reproduce {
        /// Seed of the failing test case.
        seed: u64,
    },

    /// Minimize a failing test case to find smallest failing input.
    Minimize {
        /// Seed of the failing test case.
        seed: u64,
        /// Minimization strategy to use.
        strategy: MinimizeStrategy,
        /// Maximum iterations for minimization.
        max_iters: usize,
    },

    /// List stored failures.
    ListFailures {
        /// Maximum number of failures to return.
        limit: usize,
    },
}

/// Response messages sent from the daemon to clients.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub enum Response {
    /// Validation completed.
    ValidationResult(ValidationResult),

    /// Multiple validation results.
    Results(Vec<ValidationResult>),

    /// Simple acknowledgment.
    Ok,

    /// Pong response to ping.
    Pong {
        /// Daemon version.
        version: String,
        /// Uptime in seconds.
        uptime_secs: u64,
    },

    /// Error response.
    Error {
        /// Error code.
        code: ErrorCode,
        /// Error message.
        message: String,
    },

    // =========================================================================
    // Phase 2: Fuzzing and Reproducibility
    // =========================================================================

    /// Fuzz run completed.
    FuzzResults {
        /// Master seed used for the fuzz run.
        seed: u64,
        /// Total number of test cases run.
        total: usize,
        /// Number of passing test cases.
        passed: usize,
        /// Number of failing test cases.
        failed: usize,
        /// Details of failed test cases.
        failures: Vec<ValidationResult>,
    },

    /// Reproduction result.
    ReproduceResult {
        /// The reproduced validation result.
        result: ValidationResult,
        /// The regenerated inputs (for debugging).
        inputs: HashMap<String, TensorData>,
    },

    /// Minimization result.
    MinimizeResult {
        /// Original seed that was minimized.
        original_seed: u64,
        /// Seed of the minimized test case.
        minimized_seed: u64,
        /// Minimized shape.
        minimized_shape: Vec<usize>,
        /// The minimized validation result.
        result: ValidationResult,
    },
}

/// Error codes for daemon responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub enum ErrorCode {
    /// Op not found in configuration.
    OpNotFound,
    /// Reference script failed to execute.
    ReferenceScriptFailed,
    /// Invalid request format.
    InvalidRequest,
    /// Internal daemon error.
    InternalError,
    /// Result not found.
    NotFound,
    /// Configuration error.
    ConfigError,
}

/// Serialize a request to bytes using rkyv.
pub fn serialize_request(request: &Request) -> Result<Vec<u8>, SerializeError> {
    rkyv::to_bytes::<_, 256>(request)
        .map(|v| v.to_vec())
        .map_err(|e| SerializeError(format!("{:?}", e)))
}

/// Deserialize a request from bytes.
pub fn deserialize_request(bytes: &[u8]) -> Result<Request, DeserializeError> {
    // Copy to aligned buffer to satisfy rkyv's alignment requirements
    let mut aligned = rkyv::AlignedVec::new();
    aligned.extend_from_slice(bytes);

    let archived = rkyv::check_archived_root::<Request>(&aligned)
        .map_err(|e| DeserializeError::Validation(e.to_string()))?;
    let request: Request = archived.deserialize(&mut rkyv::Infallible)
        .map_err(|e| DeserializeError::Deserialize(format!("{:?}", e)))?;
    Ok(request)
}

/// Serialize a response to bytes using rkyv.
pub fn serialize_response(response: &Response) -> Result<Vec<u8>, SerializeError> {
    rkyv::to_bytes::<_, 256>(response)
        .map(|v| v.to_vec())
        .map_err(|e| SerializeError(format!("{:?}", e)))
}

/// Deserialize a response from bytes.
pub fn deserialize_response(bytes: &[u8]) -> Result<Response, DeserializeError> {
    // Copy to aligned buffer to satisfy rkyv's alignment requirements
    let mut aligned = rkyv::AlignedVec::new();
    aligned.extend_from_slice(bytes);

    let archived = rkyv::check_archived_root::<Response>(&aligned)
        .map_err(|e| DeserializeError::Validation(e.to_string()))?;
    let response: Response = archived.deserialize(&mut rkyv::Infallible)
        .map_err(|e| DeserializeError::Deserialize(format!("{:?}", e)))?;
    Ok(response)
}

/// Errors during serialization.
#[derive(Debug, thiserror::Error)]
#[error("Serialization failed: {0}")]
pub struct SerializeError(String);

/// Errors during deserialization.
#[derive(Debug, thiserror::Error)]
pub enum DeserializeError {
    #[error("Validation failed: {0}")]
    Validation(String),
    #[error("Deserialization failed: {0}")]
    Deserialize(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ping_roundtrip() {
        let request = Request::Ping;
        let bytes = serialize_request(&request).unwrap();
        let decoded = deserialize_request(&bytes).unwrap();
        assert!(matches!(decoded, Request::Ping));
    }

    #[test]
    fn test_response_roundtrip() {
        let response = Response::Pong {
            version: "0.1.0".to_string(),
            uptime_secs: 100,
        };
        let bytes = serialize_response(&response).unwrap();
        let decoded = deserialize_response(&bytes).unwrap();
        match decoded {
            Response::Pong { version, uptime_secs } => {
                assert_eq!(version, "0.1.0");
                assert_eq!(uptime_secs, 100);
            }
            _ => panic!("Expected Pong response"),
        }
    }
}
