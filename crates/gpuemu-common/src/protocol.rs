//! IPC protocol messages for daemon-client communication.
//!
//! Uses JSON serialization for cross-language compatibility between
//! the Rust daemon and Python client.

use crate::types::{
    ArtifactDiff, ArtifactMetrics, BaselineComparison, CiRunSummary, FuzzConfig, LintResult,
    TensorData, ValidationResult,
};
use serde::{Deserialize as SerdeDeserialize, Serialize as SerdeSerialize};
use std::collections::HashMap;

/// Protocol version for compatibility checking between daemon and clients.
/// Increment when making breaking changes to the IPC protocol.
pub const PROTOCOL_VERSION: u32 = 1;

/// Strategy for minimizing a failing test case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, SerdeSerialize, SerdeDeserialize)]
pub enum MinimizeStrategy {
    /// Binary search on tensor dimensions to find smallest failing shape.
    BinarySearchDims,
    /// Binary search on tensor values to find minimal failing input.
    BinarySearchValues,
}

/// Request messages sent from clients to the daemon.
#[derive(Debug, Clone, SerdeSerialize, SerdeDeserialize)]
#[serde(tag = "type")]
pub enum Request {
    /// Validate an op against its reference implementation.
    #[serde(rename = "ValidateOp")]
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
    #[serde(rename = "GetResult")]
    GetResult {
        /// Seed of the validation run.
        seed: u64,
    },

    /// List recent validation results.
    #[serde(rename = "ListResults")]
    ListResults {
        /// Maximum number of results to return.
        limit: usize,
    },

    /// Store a baseline for comparison.
    #[serde(rename = "StoreBaseline")]
    StoreBaseline {
        /// Tag name for the baseline.
        tag: String,
    },

    /// Compare current results against a baseline.
    #[serde(rename = "CompareBaseline")]
    CompareBaseline {
        /// Tag name of the baseline to compare against.
        tag: String,
    },

    /// Check daemon health.
    #[serde(rename = "Ping")]
    Ping,

    /// Request daemon shutdown.
    #[serde(rename = "Shutdown")]
    Shutdown,

    // =========================================================================
    // Phase 2: Fuzzing and Reproducibility
    // =========================================================================
    /// Fuzz an op with seeded random inputs.
    #[serde(rename = "FuzzOp")]
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
    #[serde(rename = "Reproduce")]
    Reproduce {
        /// Seed of the failing test case.
        seed: u64,
    },

    /// Minimize a failing test case to find smallest failing input.
    #[serde(rename = "Minimize")]
    Minimize {
        /// Seed of the failing test case.
        seed: u64,
        /// Minimization strategy to use.
        strategy: MinimizeStrategy,
        /// Maximum iterations for minimization.
        max_iters: usize,
    },

    /// List stored failures.
    #[serde(rename = "ListFailures")]
    ListFailures {
        /// Maximum number of failures to return.
        limit: usize,
    },

    // =========================================================================
    // Phase 3: Artifact Inspection
    // =========================================================================
    /// Lint kernel artifacts against configured policies.
    #[serde(rename = "LintKernel")]
    LintKernel {
        /// Name of the kernel to lint (or None for all).
        kernel_name: Option<String>,
        /// PTX content to analyze.
        ptx_content: String,
    },

    /// Store artifact metrics for a kernel.
    #[serde(rename = "StoreArtifact")]
    StoreArtifact {
        /// Kernel name.
        kernel_name: String,
        /// Artifact metrics to store.
        metrics: ArtifactMetrics,
    },

    /// Store current artifacts as a baseline.
    #[serde(rename = "StoreArtifactBaseline")]
    StoreArtifactBaseline {
        /// Tag for the baseline.
        tag: String,
    },

    /// Compare current artifacts against a baseline.
    #[serde(rename = "DiffArtifactBaseline")]
    DiffArtifactBaseline {
        /// Baseline tag to compare against.
        tag: String,
    },

    /// Get artifact metrics for a kernel.
    #[serde(rename = "GetArtifact")]
    GetArtifact {
        /// Kernel name.
        kernel_name: String,
    },

    /// List all stored artifact metrics.
    #[serde(rename = "ListArtifacts")]
    ListArtifacts,

    // =========================================================================
    // Phase 4: CI Integration
    // =========================================================================
    /// Run full CI validation suite.
    #[serde(rename = "RunCi")]
    RunCi {
        /// Run quick validation only (fewer dtypes, smaller shapes).
        quick: bool,
        /// Baseline tag to compare artifacts against (optional).
        baseline: Option<String>,
        /// Number of parallel jobs (0 = auto from config).
        parallel_jobs: u32,
    },

    /// Get the last CI run summary.
    #[serde(rename = "GetCiSummary")]
    GetCiSummary,

    // =========================================================================
    // Execution Modes: Daemon-Orchestrated
    // =========================================================================
    /// Generate a single test case for client-side execution.
    /// Returns the generated inputs and metadata (seed, shape, dtype, etc).
    #[serde(rename = "GetTestCase")]
    GetTestCase {
        /// Name of the op.
        op_name: String,
        /// Fuzzing configuration.
        fuzz_config: FuzzConfig,
    },

    /// Generate a batch of test cases for client-side execution.
    #[serde(rename = "GetTestBatch")]
    GetTestBatch {
        /// Name of the op.
        op_name: String,
        /// Fuzzing configuration.
        fuzz_config: FuzzConfig,
        /// Number of test cases in the batch.
        count: usize,
    },

    /// Submit an op output for validation against the reference.
    /// Used in daemon-orchestrated and client-side modes.
    #[serde(rename = "SubmitOutput")]
    SubmitOutput {
        /// Name of the op.
        op_name: String,
        /// Input tensors (for reference computation).
        inputs: HashMap<String, TensorData>,
        /// Output from the op under test.
        output: TensorData,
        /// Seed for this test case (for result tracking).
        seed: u64,
        /// Optional kwargs for the reference script.
        kwargs: HashMap<String, String>,
    },
}

/// Response messages sent from the daemon to clients.
#[derive(Debug, Clone, SerdeSerialize, SerdeDeserialize)]
#[serde(tag = "type")]
pub enum Response {
    /// Validation completed.
    #[serde(rename = "ValidationResult")]
    ValidationResult { result: ValidationResult },

    /// Multiple validation results.
    #[serde(rename = "Results")]
    Results { results: Vec<ValidationResult> },

    /// Baseline comparison results.
    #[serde(rename = "BaselineComparison")]
    BaselineComparison {
        /// Baseline tag compared against.
        baseline_tag: String,
        /// Per-result comparisons.
        comparisons: Vec<BaselineComparison>,
        /// Whether any regressions were found.
        has_regressions: bool,
    },

    /// Simple acknowledgment.
    #[serde(rename = "Ok")]
    Ok,

    /// Pong response to ping.
    #[serde(rename = "Pong")]
    Pong {
        /// Daemon version.
        version: String,
        /// Protocol version for compatibility checking.
        protocol_version: u32,
        /// Uptime in seconds.
        uptime_secs: u64,
    },

    /// Error response.
    #[serde(rename = "Error")]
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
    #[serde(rename = "FuzzResults")]
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
    #[serde(rename = "ReproduceResult")]
    ReproduceResult {
        /// The reproduced validation result.
        result: ValidationResult,
        /// The regenerated inputs (for debugging).
        inputs: HashMap<String, TensorData>,
    },

    /// Minimization result.
    #[serde(rename = "MinimizeResult")]
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

    // =========================================================================
    // Phase 3: Artifact Inspection
    // =========================================================================
    /// Lint results for one or more kernels.
    #[serde(rename = "LintResults")]
    LintResults(Vec<LintResult>),

    /// Artifact metrics for a single kernel.
    #[serde(rename = "ArtifactMetricsResult")]
    ArtifactMetricsResult(ArtifactMetrics),

    /// List of artifact metrics.
    #[serde(rename = "ArtifactList")]
    ArtifactList(Vec<ArtifactMetrics>),

    /// Diff results against baseline.
    #[serde(rename = "ArtifactDiffs")]
    ArtifactDiffs {
        /// Baseline tag used.
        baseline_tag: String,
        /// List of diffs for each kernel.
        diffs: Vec<ArtifactDiff>,
        /// Overall regression detected.
        has_regressions: bool,
    },

    // =========================================================================
    // Phase 4: CI Integration
    // =========================================================================
    /// CI run completed.
    #[serde(rename = "CiRunComplete")]
    CiRunComplete(CiRunSummary),

    // =========================================================================
    // Execution Modes: Daemon-Orchestrated Responses
    // =========================================================================
    /// A single test case with generated inputs.
    #[serde(rename = "TestCase")]
    TestCase {
        /// Seed for this test case.
        seed: u64,
        /// Generated input tensors.
        inputs: HashMap<String, TensorData>,
        /// Shape of the test case.
        shape: Vec<usize>,
        /// Dtype of the test case.
        dtype: String,
        /// Layout of the test case.
        layout: String,
    },

    /// A batch of test cases.
    #[serde(rename = "TestBatch")]
    TestBatch {
        /// The test cases in this batch.
        cases: Vec<TestCaseData>,
    },

    /// Result of submitting an op output for validation.
    #[serde(rename = "SubmitResult")]
    SubmitResult {
        /// The validation result.
        result: ValidationResult,
    },
}

/// A single test case for transmission to clients.
#[derive(Debug, Clone, SerdeSerialize, SerdeDeserialize)]
pub struct TestCaseData {
    /// Seed for this test case.
    pub seed: u64,
    /// Generated input tensors.
    pub inputs: HashMap<String, TensorData>,
    /// Shape of the test case.
    pub shape: Vec<usize>,
    /// Dtype of the test case.
    pub dtype: String,
    /// Layout of the test case.
    pub layout: String,
}

/// Error codes for daemon responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, SerdeSerialize, SerdeDeserialize)]
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

    // Phase 3: Artifact Inspection
    /// Kernel not found in configuration.
    KernelNotFound,
    /// Artifact not found in storage.
    ArtifactNotFound,
    /// Baseline not found in storage.
    BaselineNotFound,
    /// PTX parse error.
    PtxParseError,
    /// cuobjdump not available.
    CuobjdumpNotAvailable,
    /// Client/daemon protocol version mismatch.
    VersionMismatch,
}

/// Serialize a request to JSON bytes.
pub fn serialize_request(request: &Request) -> Result<Vec<u8>, SerializeError> {
    serde_json::to_vec(request).map_err(|e| SerializeError(e.to_string()))
}

/// Deserialize a request from JSON bytes.
pub fn deserialize_request(bytes: &[u8]) -> Result<Request, DeserializeError> {
    serde_json::from_slice(bytes).map_err(|e| DeserializeError::Json(e.to_string()))
}

/// Serialize a response to JSON bytes.
pub fn serialize_response(response: &Response) -> Result<Vec<u8>, SerializeError> {
    serde_json::to_vec(response).map_err(|e| SerializeError(e.to_string()))
}

/// Deserialize a response from JSON bytes.
pub fn deserialize_response(bytes: &[u8]) -> Result<Response, DeserializeError> {
    serde_json::from_slice(bytes).map_err(|e| DeserializeError::Json(e.to_string()))
}

/// Errors during serialization.
#[derive(Debug, thiserror::Error)]
#[error("Serialization failed: {0}")]
pub struct SerializeError(String);

/// Errors during deserialization.
#[derive(Debug, thiserror::Error)]
pub enum DeserializeError {
    #[error("JSON error: {0}")]
    Json(String),
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
            protocol_version: PROTOCOL_VERSION,
            uptime_secs: 100,
        };
        let bytes = serialize_response(&response).unwrap();
        let decoded = deserialize_response(&bytes).unwrap();
        match decoded {
            Response::Pong {
                version,
                protocol_version,
                uptime_secs,
            } => {
                assert_eq!(version, "0.1.0");
                assert_eq!(protocol_version, PROTOCOL_VERSION);
                assert_eq!(uptime_secs, 100);
            }
            _ => panic!("Expected Pong response"),
        }
    }

    #[test]
    fn test_request_json_tag() {
        let request = Request::Ping;
        let json_str = serde_json::to_string(&request).unwrap();
        assert!(
            json_str.contains("\"type\":\"Ping\""),
            "Expected JSON tag: {}",
            json_str
        );
    }

    #[test]
    fn test_response_json_tag() {
        let response = Response::Ok;
        let json_str = serde_json::to_string(&response).unwrap();
        assert!(
            json_str.contains("\"type\":\"Ok\""),
            "Expected JSON tag: {}",
            json_str
        );
    }
}
