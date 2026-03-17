//! Core types for gpuemu validation.

use rkyv::{Archive, Deserialize, Serialize};
use serde::{Deserialize as SerdeDeserialize, Serialize as SerdeSerialize};

/// Supported data types for tensor validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
#[derive(SerdeSerialize, SerdeDeserialize)]
#[archive(check_bytes)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    Float16,
    BFloat16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Bool,
}

impl DType {
    /// Get the size in bytes for this dtype.
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::Bool | DType::Int8 | DType::UInt8 => 1,
            DType::Float16 | DType::BFloat16 | DType::Int16 | DType::UInt16 => 2,
            DType::Float32 | DType::Int32 | DType::UInt32 => 4,
            DType::Float64 | DType::Int64 | DType::UInt64 => 8,
        }
    }
}

/// Tensor metadata and data.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct TensorData {
    /// Shape of the tensor.
    pub shape: Vec<usize>,
    /// Strides in elements (not bytes).
    pub strides: Vec<usize>,
    /// Data type.
    pub dtype: DType,
    /// Raw bytes of tensor data.
    pub data: Vec<u8>,
}

impl TensorData {
    /// Create a new tensor with the given shape, dtype, and data.
    pub fn new(shape: Vec<usize>, dtype: DType, data: Vec<u8>) -> Self {
        let strides = Self::compute_contiguous_strides(&shape);
        Self {
            shape,
            strides,
            dtype,
            data,
        }
    }

    /// Compute contiguous strides for a given shape (row-major).
    pub fn compute_contiguous_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Total number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if tensor is contiguous in memory.
    pub fn is_contiguous(&self) -> bool {
        self.strides == Self::compute_contiguous_strides(&self.shape)
    }
}

/// A single validation failure.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[derive(SerdeSerialize, SerdeDeserialize)]
#[archive(check_bytes)]
pub struct ValidationFailure {
    /// Type of failure.
    pub kind: FailureKind,
    /// Human-readable message.
    pub message: String,
    /// Index in flattened tensor where failure occurred (if applicable).
    pub index: Option<usize>,
    /// Expected value (if applicable).
    pub expected: Option<f64>,
    /// Actual value (if applicable).
    pub actual: Option<f64>,
}

/// Types of validation failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[derive(SerdeSerialize, SerdeDeserialize)]
#[archive(check_bytes)]
pub enum FailureKind {
    /// Tolerance exceeded.
    ToleranceExceeded,
    /// NaN detected in output.
    NaNDetected,
    /// Inf detected in output.
    InfDetected,
    /// Shape mismatch.
    ShapeMismatch,
    /// DType mismatch.
    DTypeMismatch,
    /// Invariant violation.
    InvariantViolation,
    /// Reference script failed.
    ReferenceError,
}

/// Result of a validation run.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[derive(SerdeSerialize, SerdeDeserialize)]
#[archive(check_bytes)]
pub struct ValidationResult {
    /// Whether validation passed.
    pub passed: bool,
    /// Seed used for this validation run.
    pub seed: u64,
    /// Name of the op/kernel being validated.
    pub op_name: String,
    /// Maximum absolute difference found.
    pub max_diff: f64,
    /// Maximum relative difference found.
    pub max_rel_diff: f64,
    /// List of failures (empty if passed).
    pub failures: Vec<ValidationFailure>,
    /// Timestamp of validation (Unix epoch seconds).
    pub timestamp: u64,
    /// Duration of validation in milliseconds.
    pub duration_ms: u64,
    /// Full reproduction info (populated on failures during fuzzing).
    pub repro_info: Option<ReproductionInfo>,
}

// =============================================================================
// Fuzzing Types (Phase 2)
// =============================================================================

/// Memory layout type for tensor fuzzing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[derive(SerdeSerialize, SerdeDeserialize)]
#[archive(check_bytes)]
pub enum LayoutType {
    /// Standard row-major contiguous layout.
    Contiguous,
    /// Strided layout with gaps in memory.
    Strided,
    /// Transposed layout (dimensions swapped).
    Transposed,
}

/// Shape options for fuzzing.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[derive(SerdeSerialize, SerdeDeserialize)]
#[archive(check_bytes)]
pub struct ShapeOptions {
    /// Batch size options to fuzz.
    pub batch_sizes: Vec<usize>,
    /// Sequence length options to fuzz.
    pub seq_lengths: Vec<usize>,
    /// Hidden dimension options to fuzz.
    pub hidden_dims: Vec<usize>,
    /// Edge cases to always include (e.g., [[1], [0], [1,1,1]]).
    pub edge_cases: Vec<Vec<usize>>,
}

impl Default for ShapeOptions {
    fn default() -> Self {
        Self {
            batch_sizes: vec![1, 2, 4, 8],
            seq_lengths: vec![64, 128, 256, 512],
            hidden_dims: vec![512, 768, 1024],
            edge_cases: vec![vec![1], vec![1, 1], vec![1, 1, 1]],
        }
    }
}

/// Configuration for fuzz testing.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[derive(SerdeSerialize, SerdeDeserialize)]
#[archive(check_bytes)]
pub struct FuzzConfig {
    /// Master seed for all random decisions.
    pub seed: u64,
    /// Shape options for fuzzing.
    pub shape_options: ShapeOptions,
    /// Data types to fuzz.
    pub dtypes: Vec<DType>,
    /// Layout types to fuzz.
    pub layouts: Vec<LayoutType>,
}

impl Default for FuzzConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            shape_options: ShapeOptions::default(),
            dtypes: vec![DType::Float32, DType::Float16],
            layouts: vec![LayoutType::Contiguous, LayoutType::Strided, LayoutType::Transposed],
        }
    }
}

impl FuzzConfig {
    /// Create a new fuzz config with the given seed.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            seed,
            ..Default::default()
        }
    }
}

/// Full reproduction info stored with failures.
///
/// Contains all information needed to reproduce a failing test case.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[derive(SerdeSerialize, SerdeDeserialize)]
#[archive(check_bytes)]
pub struct ReproductionInfo {
    /// Master seed that generated this test case.
    pub seed: u64,
    /// Fuzz config used (if fuzzing was enabled).
    pub fuzz_config: Option<FuzzConfig>,
    /// Exact shape used.
    pub shape: Vec<usize>,
    /// Exact strides used.
    pub strides: Vec<usize>,
    /// Data type used.
    pub dtype: DType,
    /// Layout type used.
    pub layout: LayoutType,
    /// Compressed input data snapshot (optional, for exact reproduction).
    pub input_snapshot: Option<Vec<u8>>,
}

impl ValidationResult {
    /// Create a passing result.
    pub fn pass(op_name: String, seed: u64, max_diff: f64, max_rel_diff: f64, duration_ms: u64) -> Self {
        Self {
            passed: true,
            seed,
            op_name,
            max_diff,
            max_rel_diff,
            failures: Vec::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            duration_ms,
            repro_info: None,
        }
    }

    /// Create a failing result.
    pub fn fail(op_name: String, seed: u64, failures: Vec<ValidationFailure>, duration_ms: u64) -> Self {
        Self {
            passed: false,
            seed,
            op_name,
            max_diff: 0.0,
            max_rel_diff: 0.0,
            failures,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            duration_ms,
            repro_info: None,
        }
    }

    /// Create a failing result with reproduction info.
    pub fn fail_with_repro(
        op_name: String,
        seed: u64,
        failures: Vec<ValidationFailure>,
        duration_ms: u64,
        repro_info: ReproductionInfo,
    ) -> Self {
        Self {
            passed: false,
            seed,
            op_name,
            max_diff: 0.0,
            max_rel_diff: 0.0,
            failures,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            duration_ms,
            repro_info: Some(repro_info),
        }
    }

    /// Add reproduction info to an existing result.
    pub fn with_repro_info(mut self, repro_info: ReproductionInfo) -> Self {
        self.repro_info = Some(repro_info);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contiguous_strides() {
        let shape = vec![2, 3, 4];
        let strides = TensorData::compute_contiguous_strides(&shape);
        assert_eq!(strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::Float32.size_bytes(), 4);
        assert_eq!(DType::Float16.size_bytes(), 2);
        assert_eq!(DType::Float64.size_bytes(), 8);
    }
}
