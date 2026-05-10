//! Core types for gpuemu validation.

use rkyv::{Archive, Deserialize, Serialize};
use serde::{Deserialize as SerdeDeserialize, Serialize as SerdeSerialize};

/// Supported data types for tensor validation.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Archive,
    Serialize,
    Deserialize,
    SerdeSerialize,
    SerdeDeserialize,
)]
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

    /// Get the numpy dtype string for this dtype.
    pub fn to_numpy_dtype(&self) -> &'static str {
        match self {
            DType::Float16 => "float16",
            DType::BFloat16 => "bfloat16",
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Int8 => "int8",
            DType::Int16 => "int16",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::UInt8 => "uint8",
            DType::UInt16 => "uint16",
            DType::UInt32 => "uint32",
            DType::UInt64 => "uint64",
            DType::Bool => "bool",
        }
    }

    /// Parse a dtype from a config string (case-insensitive).
    /// Accepts: "float16", "bfloat16", "float32", "float64",
    ///          "int8", "int16", "int32", "int64",
    ///          "uint8", "uint16", "uint32", "uint64", "bool".
    /// Returns None for unrecognized strings.
    pub fn from_config_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "float16" | "f16" => Some(DType::Float16),
            "bfloat16" | "bf16" => Some(DType::BFloat16),
            "float32" | "f32" => Some(DType::Float32),
            "float64" | "f64" => Some(DType::Float64),
            "int8" | "i8" => Some(DType::Int8),
            "int16" | "i16" => Some(DType::Int16),
            "int32" | "i32" => Some(DType::Int32),
            "int64" | "i64" => Some(DType::Int64),
            "uint8" | "u8" => Some(DType::UInt8),
            "uint16" | "u16" => Some(DType::UInt16),
            "uint32" | "u32" => Some(DType::UInt32),
            "uint64" | "u64" => Some(DType::UInt64),
            "bool" => Some(DType::Bool),
            _ => None,
        }
    }
}

/// Parse a list of dtype config strings into DType enums.
/// Unrecognized strings are silently skipped; if none match, defaults to [Float32].
pub fn parse_dtypes(strings: &[String]) -> Vec<DType> {
    let dtypes: Vec<DType> = strings
        .iter()
        .filter_map(|s| DType::from_config_str(s))
        .collect();
    if dtypes.is_empty() {
        vec![DType::Float32]
    } else {
        dtypes
    }
}

/// Parse a list of layout config strings into LayoutType enums.
/// Unrecognized strings are silently skipped; if none match, defaults to [Contiguous].
pub fn parse_layouts(strings: &[String]) -> Vec<LayoutType> {
    let layouts: Vec<LayoutType> = strings
        .iter()
        .filter_map(|s| LayoutType::from_config_str(s))
        .collect();
    if layouts.is_empty() {
        vec![LayoutType::Contiguous]
    } else {
        layouts
    }
}

/// Tensor metadata and data.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, SerdeSerialize, SerdeDeserialize)]
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
#[derive(Debug, Clone, Archive, Serialize, Deserialize, SerdeSerialize, SerdeDeserialize)]
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
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Archive,
    Serialize,
    Deserialize,
    SerdeSerialize,
    SerdeDeserialize,
)]
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
#[derive(Debug, Clone, Archive, Serialize, Deserialize, SerdeSerialize, SerdeDeserialize)]
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
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Archive,
    Serialize,
    Deserialize,
    SerdeSerialize,
    SerdeDeserialize,
)]
#[archive(check_bytes)]
pub enum LayoutType {
    /// Standard row-major contiguous layout.
    Contiguous,
    /// Strided layout with gaps in memory.
    Strided,
    /// Transposed layout (dimensions swapped).
    Transposed,
}

impl LayoutType {
    /// Parse from a config string (case-insensitive).
    pub fn from_config_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "contiguous" => Some(LayoutType::Contiguous),
            "strided" => Some(LayoutType::Strided),
            "transposed" => Some(LayoutType::Transposed),
            _ => None,
        }
    }

    /// Convert to a config/presentation string.
    pub fn to_config_str(&self) -> &'static str {
        match self {
            LayoutType::Contiguous => "contiguous",
            LayoutType::Strided => "strided",
            LayoutType::Transposed => "transposed",
        }
    }
}

/// Shape options for fuzzing.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, SerdeSerialize, SerdeDeserialize)]
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
#[derive(Debug, Clone, Archive, Serialize, Deserialize, SerdeSerialize, SerdeDeserialize)]
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
            layouts: vec![
                LayoutType::Contiguous,
                LayoutType::Strided,
                LayoutType::Transposed,
            ],
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
#[derive(Debug, Clone, Archive, Serialize, Deserialize, SerdeSerialize, SerdeDeserialize)]
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
    pub fn pass(
        op_name: String,
        seed: u64,
        max_diff: f64,
        max_rel_diff: f64,
        duration_ms: u64,
    ) -> Self {
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
    pub fn fail(
        op_name: String,
        seed: u64,
        failures: Vec<ValidationFailure>,
        duration_ms: u64,
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

// =============================================================================
// Artifact Types (Phase 3)
// =============================================================================

/// Source of artifact data.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Archive,
    Serialize,
    Deserialize,
    SerdeSerialize,
    SerdeDeserialize,
)]
#[archive(check_bytes)]
pub enum ArtifactSource {
    /// Parsed from PTX text.
    Ptx,
    /// Extracted from SASS via cuobjdump.
    Sass,
    /// Both PTX and SASS available.
    Both,
}

/// Extracted metrics from PTX/SASS artifacts.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, SerdeSerialize, SerdeDeserialize)]
#[archive(check_bytes)]
pub struct ArtifactMetrics {
    /// Kernel name this artifact belongs to.
    pub kernel_name: String,
    /// Number of registers used.
    pub register_count: u32,
    /// Number of spills detected (ld.local/st.local operations).
    pub spill_count: u32,
    /// Local memory usage in bytes.
    pub local_memory_bytes: u32,
    /// Shared memory usage in bytes.
    pub shared_memory_bytes: u32,
    /// Instruction count (approximate).
    pub instruction_count: u32,
    /// Instruction patterns/mnemonics found in the artifact.
    pub patterns_found: Vec<String>,
    /// Source type of the artifact.
    pub source: ArtifactSource,
    /// Timestamp of extraction (Unix epoch seconds).
    pub timestamp: u64,
    /// Raw PTX content (optional, for debugging).
    pub ptx_content: Option<String>,
}

impl Default for ArtifactMetrics {
    fn default() -> Self {
        Self {
            kernel_name: String::new(),
            register_count: 0,
            spill_count: 0,
            local_memory_bytes: 0,
            shared_memory_bytes: 0,
            instruction_count: 0,
            patterns_found: Vec::new(),
            source: ArtifactSource::Ptx,
            timestamp: 0,
            ptx_content: None,
        }
    }
}

/// Types of lint violations.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Archive,
    Serialize,
    Deserialize,
    SerdeSerialize,
    SerdeDeserialize,
)]
#[archive(check_bytes)]
pub enum LintViolationKind {
    /// Register count exceeds maximum.
    ExcessiveRegisters,
    /// Spill count exceeds maximum.
    ExcessiveSpills,
    /// Local memory exceeds maximum.
    ExcessiveLocalMemory,
    /// Required pattern not found.
    MissingRequiredPattern,
    /// Forbidden pattern detected.
    ForbiddenPatternFound,
}

/// A single lint violation.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, SerdeSerialize, SerdeDeserialize)]
#[archive(check_bytes)]
pub struct LintViolation {
    /// Type of violation.
    pub kind: LintViolationKind,
    /// Human-readable message.
    pub message: String,
    /// Actual value that violated the rule (if applicable).
    pub actual: Option<u32>,
    /// Threshold that was exceeded (if applicable).
    pub threshold: Option<u32>,
}

/// Result of linting a kernel's artifacts.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, SerdeSerialize, SerdeDeserialize)]
#[archive(check_bytes)]
pub struct LintResult {
    /// Kernel name.
    pub kernel_name: String,
    /// Whether lint passed (no violations).
    pub passed: bool,
    /// Extracted metrics.
    pub metrics: ArtifactMetrics,
    /// List of violations.
    pub violations: Vec<LintViolation>,
    /// Timestamp of lint run (Unix epoch seconds).
    pub timestamp: u64,
}

/// Difference between two artifact metrics.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, SerdeSerialize, SerdeDeserialize)]
#[archive(check_bytes)]
pub struct ArtifactDiff {
    /// Kernel name.
    pub kernel_name: String,
    /// Baseline metrics (if available).
    pub baseline: Option<ArtifactMetrics>,
    /// Current metrics.
    pub current: ArtifactMetrics,
    /// Register count change (current - baseline).
    pub register_delta: i32,
    /// Spill count change.
    pub spill_delta: i32,
    /// Local memory change.
    pub local_memory_delta: i32,
    /// Instruction count change.
    pub instruction_delta: i32,
    /// Whether this is a regression (any metric increased).
    pub is_regression: bool,
}

// =============================================================================
// CI Types (Phase 4)
// =============================================================================

/// Summary of artifact diff results for CI reporting.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, SerdeSerialize, SerdeDeserialize)]
#[archive(check_bytes)]
pub struct ArtifactDiffSummary {
    /// Baseline tag that was compared against.
    pub baseline_tag: String,
    /// Whether any regressions were detected.
    pub has_regressions: bool,
    /// List of diffs for each kernel.
    pub diffs: Vec<ArtifactDiff>,
}

/// Comparison of a single result against its baseline.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, SerdeSerialize, SerdeDeserialize)]
#[archive(check_bytes)]
pub struct BaselineComparison {
    /// Op name for this comparison.
    pub op_name: String,
    /// Seed of the current result.
    pub seed: u64,
    /// Maximum absolute diff in the current result.
    pub current_max_diff: f64,
    /// Maximum absolute diff in the baseline result (None if no baseline match).
    pub baseline_max_diff: Option<f64>,
    /// Whether the current result passed.
    pub current_passed: bool,
    /// Whether the baseline result passed (None if no baseline match).
    pub baseline_passed: Option<bool>,
    /// Relative change in max_diff vs baseline (None if no baseline).
    pub relative_change: Option<f64>,
    /// Whether this result represents a regression.
    pub is_regression: bool,
}

/// Summary of a CI run for reporting.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, SerdeSerialize, SerdeDeserialize)]
#[archive(check_bytes)]
pub struct CiRunSummary {
    /// Total number of tests run.
    pub total_tests: usize,
    /// Number of tests that passed.
    pub passed: usize,
    /// Number of tests that failed.
    pub failed: usize,
    /// Number of tests that were skipped.
    pub skipped: usize,
    /// Total duration of the CI run in milliseconds.
    pub duration_ms: u64,
    /// Timestamp of the CI run (Unix epoch seconds).
    pub timestamp: u64,
    /// Validation results from fuzz/test runs.
    pub validation_results: Vec<ValidationResult>,
    /// Lint results from artifact checks.
    pub lint_results: Vec<LintResult>,
    /// Artifact diff results (if baseline comparison was requested).
    pub artifact_diffs: Option<ArtifactDiffSummary>,
}

impl CiRunSummary {
    /// Check if the CI run has any failures.
    pub fn has_failures(&self) -> bool {
        self.failed > 0
    }

    /// Check if the CI run has any regressions.
    pub fn has_regressions(&self) -> bool {
        self.artifact_diffs
            .as_ref()
            .map(|d| d.has_regressions)
            .unwrap_or(false)
    }

    /// Get the exit code for CI (0 = success, 1 = failure).
    pub fn exit_code(&self) -> i32 {
        if self.has_failures() || self.has_regressions() {
            1
        } else {
            0
        }
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
