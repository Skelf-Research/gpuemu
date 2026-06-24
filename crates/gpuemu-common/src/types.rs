//! Core types for gpuemu validation.

use rkyv::{Archive, Deserialize, Serialize};
use serde::{Deserialize as SerdeDeserialize, Serialize as SerdeSerialize};

/// Serde helpers that represent byte buffers as base64 strings in JSON.
///
/// The protocol exchanges tensor bytes as base64 (compact, and what the Python
/// client encodes/decodes). serde's default for `Vec<u8>` is a JSON number
/// array — so these `#[serde(with = ...)]` adapters keep both sides in agreement.
/// Only affects serde (the JSON protocol); rkyv archiving is unaffected.
pub(crate) mod serde_b64 {
    use base64::Engine;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&base64::engine::general_purpose::STANDARD.encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(d)?;
        base64::engine::general_purpose::STANDARD
            .decode(s.as_bytes())
            .map_err(serde::de::Error::custom)
    }

    /// Variant for `Option<Vec<u8>>` fields.
    pub mod opt {
        use base64::Engine;
        use serde::{Deserialize, Deserializer, Serializer};

        pub fn serialize<S: Serializer>(bytes: &Option<Vec<u8>>, s: S) -> Result<S::Ok, S::Error> {
            match bytes {
                Some(b) => {
                    s.serialize_some(&base64::engine::general_purpose::STANDARD.encode(b))
                }
                None => s.serialize_none(),
            }
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(
            d: D,
        ) -> Result<Option<Vec<u8>>, D::Error> {
            let opt = Option::<String>::deserialize(d)?;
            match opt {
                Some(s) => base64::engine::general_purpose::STANDARD
                    .decode(s.as_bytes())
                    .map(Some)
                    .map_err(serde::de::Error::custom),
                None => Ok(None),
            }
        }
    }
}

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
    /// Raw bytes of tensor data (base64 in the JSON protocol).
    #[serde(with = "serde_b64")]
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

/// Distribution of element-wise error between an output and its reference.
///
/// Captures the full error picture (not just the first failure) so numerical
/// studies — e.g. mixed-precision tolerance calibration — have raw material.
/// Populated for float dtypes; `None` for integer/bool outputs.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, SerdeSerialize, SerdeDeserialize)]
#[archive(check_bytes)]
pub struct ErrorStats {
    /// Number of elements compared.
    pub count: usize,
    /// Number of elements whose absolute error exceeded the tolerance.
    pub num_exceeding: usize,
    /// Maximum absolute error.
    pub max_abs: f64,
    /// Mean absolute error.
    pub mean_abs: f64,
    /// 50th / 90th / 99th percentile absolute error.
    pub p50_abs: f64,
    pub p90_abs: f64,
    pub p99_abs: f64,
    /// Maximum relative error (|o-r| / |r|, skipping r == 0).
    pub max_rel: f64,
    /// Mean relative error over elements with r != 0.
    pub mean_rel: f64,
    /// Maximum ULP (units-in-the-last-place) distance, in the output dtype.
    pub max_ulp: u64,
    /// Mean ULP distance.
    pub mean_ulp: f64,
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
    /// Element-wise error distribution (float outputs only).
    #[serde(default)]
    pub error_stats: Option<ErrorStats>,
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

/// A symbolic tensor dimension with the concrete sizes the fuzzer may sample.
///
/// Authors include boundary/prime/edge values (e.g. `1`, `2`, `7`, `127`, `256`)
/// so a single dim covers the cases that expose tail and stride bugs.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, SerdeSerialize, SerdeDeserialize)]
#[archive(check_bytes)]
pub struct DimSpec {
    /// Symbolic name of the dimension (e.g. "M", "K", "N", "S").
    pub name: String,
    /// Concrete candidate sizes to sample from.
    pub candidates: Vec<usize>,
}

/// One tensor's shape expressed as an ordered list of dimension names.
///
/// The names reference [`DimSpec`] entries in the owning [`OpSchema`], so two
/// tensors that share a dim name (e.g. matmul's `K`) always get the same size.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, SerdeSerialize, SerdeDeserialize)]
#[archive(check_bytes)]
pub struct TensorSchema {
    /// Tensor name; for inputs this must match the op's `input_names`.
    pub name: String,
    /// Ordered dimension names that form this tensor's shape.
    pub dims: Vec<String>,
}

/// An operator-aware shape schema: shared symbolic dims plus per-tensor shapes.
///
/// This lets the fuzzer cover real operator domains where inputs have *different*
/// but *linked* shapes (matmul `A[M,K] · B[K,N]`, attention `Q,K,V[B,H,S,D]`),
/// instead of forcing one shape onto every input.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, SerdeSerialize, SerdeDeserialize)]
#[archive(check_bytes)]
pub struct OpSchema {
    /// Schema name (usually the op name).
    pub name: String,
    /// Symbolic dimensions shared across the tensors below.
    pub dims: Vec<DimSpec>,
    /// Input tensor shapes, keyed by name.
    pub inputs: Vec<TensorSchema>,
    /// Optional output tensor shape (used as the representative shape for repro).
    #[serde(default)]
    pub output: Option<TensorSchema>,
}

impl OpSchema {
    /// Built-in schemas for common ops. Candidate sizes deliberately mix small,
    /// prime, power-of-two, and off-by-one values to stress boundaries.
    pub fn builtin(name: &str) -> Option<Self> {
        let dim = |n: &str, c: Vec<usize>| DimSpec {
            name: n.to_string(),
            candidates: c,
        };
        let t = |n: &str, dims: &[&str]| TensorSchema {
            name: n.to_string(),
            dims: dims.iter().map(|s| s.to_string()).collect(),
        };
        match name {
            // C[M,N] = A[M,K] · B[K,N]
            "matmul" => Some(OpSchema {
                name: "matmul".to_string(),
                dims: vec![
                    dim("M", vec![1, 2, 7, 31, 128, 257]),
                    dim("K", vec![1, 3, 16, 127, 256]),
                    dim("N", vec![1, 2, 15, 64, 255]),
                ],
                inputs: vec![t("a", &["M", "K"]), t("b", &["K", "N"])],
                output: Some(t("out", &["M", "N"])),
            }),
            // Scaled dot-product attention over Q,K,V of [B,H,S,D].
            "attention" => Some(OpSchema {
                name: "attention".to_string(),
                dims: vec![
                    dim("B", vec![1, 2, 5]),
                    dim("H", vec![1, 3, 8]),
                    dim("S", vec![1, 2, 17, 128, 257]),
                    dim("D", vec![8, 16, 64, 65]),
                ],
                inputs: vec![
                    t("q", &["B", "H", "S", "D"]),
                    t("k", &["B", "H", "S", "D"]),
                    t("v", &["B", "H", "S", "D"]),
                ],
                output: Some(t("out", &["B", "H", "S", "D"])),
            }),
            // Elementwise / reduction over a single tensor [B,S,H].
            "elementwise" => Some(OpSchema {
                name: "elementwise".to_string(),
                dims: vec![
                    dim("B", vec![1, 2, 8, 33]),
                    dim("S", vec![1, 7, 128, 513]),
                    dim("H", vec![1, 3, 256, 1025]),
                ],
                inputs: vec![t("input", &["B", "S", "H"])],
                output: Some(t("out", &["B", "S", "H"])),
            }),
            _ => None,
        }
    }
}

/// Element-value sampling strategy, orthogonal to shape/layout fuzzing.
///
/// Controls the *values* placed in fuzzed tensors. `Regular` is the historical
/// uniform distribution and is preserved bit-for-bit (so old seeds reproduce);
/// `Boundary` and `Adversarial` are the P3 coverage modes that surface the
/// partial-tile / sign-cancellation / special-value bugs a uniform sample misses.
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
#[serde(rename_all = "snake_case")]
pub enum ValueDistribution {
    /// Uniform in [-10, 10] for floats; small-range ints. The field-standard
    /// default and the historical behaviour (byte-identical for old seeds).
    Regular,
    /// Emphasise edge magnitudes: exact `0`, `±1`, near-denormal tiny values,
    /// and large finite values — the inputs that expose tail-mask / partial-tile
    /// and normalisation bugs.
    Boundary,
    /// Inject pathological values: `NaN`, `±Inf`, very large/small magnitudes,
    /// and sign-cancellation pairs, mixed with regular draws.
    Adversarial,
}

impl Default for ValueDistribution {
    fn default() -> Self {
        ValueDistribution::Regular
    }
}

/// Configuration for fuzz testing.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, SerdeSerialize, SerdeDeserialize)]
#[archive(check_bytes)]
pub struct FuzzConfig {
    /// Master seed for all random decisions.
    pub seed: u64,
    /// Shape options for fuzzing (legacy rank-3 generation when no `op_schema`).
    pub shape_options: ShapeOptions,
    /// Data types to fuzz.
    pub dtypes: Vec<DType>,
    /// Layout types to fuzz.
    pub layouts: Vec<LayoutType>,
    /// Optional operator-aware shape schema. When present, the fuzzer generates
    /// per-input shapes from shared dims instead of one shape for all inputs.
    #[serde(default)]
    pub op_schema: Option<OpSchema>,
    /// Element-value sampling strategy. Defaults to [`ValueDistribution::Regular`]
    /// (preserving historical byte-for-byte generation when absent).
    #[serde(default)]
    pub value_distribution: ValueDistribution,
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
            op_schema: None,
            value_distribution: ValueDistribution::Regular,
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
    #[serde(with = "serde_b64::opt", default)]
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
            error_stats: None,
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
            error_stats: None,
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
            error_stats: None,
        }
    }

    /// Add reproduction info to an existing result.
    pub fn with_repro_info(mut self, repro_info: ReproductionInfo) -> Self {
        self.repro_info = Some(repro_info);
        self
    }

    /// Attach element-wise error statistics to an existing result.
    pub fn with_error_stats(mut self, error_stats: Option<ErrorStats>) -> Self {
        self.error_stats = error_stats;
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

    #[test]
    fn test_fuzzconfig_json_backcompat_without_op_schema() {
        // Clients on the old protocol omit op_schema; it must default to None.
        let json = r#"{
            "seed": 7,
            "shape_options": {"batch_sizes": [1], "seq_lengths": [8],
                              "hidden_dims": [16], "edge_cases": []},
            "dtypes": ["float32"],
            "layouts": ["Contiguous"]
        }"#;
        let cfg: FuzzConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.seed, 7);
        assert!(cfg.op_schema.is_none());
    }

    #[test]
    fn test_fuzzconfig_json_with_op_schema() {
        // The exact JSON shape the Python client emits for op_schema.
        let json = r#"{
            "seed": 1,
            "shape_options": {"batch_sizes": [1], "seq_lengths": [8],
                              "hidden_dims": [16], "edge_cases": []},
            "dtypes": ["float16"],
            "layouts": ["Contiguous"],
            "op_schema": {
                "name": "matmul",
                "dims": [{"name": "M", "candidates": [2, 4]},
                         {"name": "K", "candidates": [3]},
                         {"name": "N", "candidates": [5, 7]}],
                "inputs": [{"name": "a", "dims": ["M", "K"]},
                           {"name": "b", "dims": ["K", "N"]}],
                "output": {"name": "out", "dims": ["M", "N"]}
            }
        }"#;
        let cfg: FuzzConfig = serde_json::from_str(json).unwrap();
        let schema = cfg.op_schema.expect("schema present");
        assert_eq!(schema.name, "matmul");
        assert_eq!(schema.dims.len(), 3);
        assert_eq!(schema.inputs[1].dims, vec!["K".to_string(), "N".to_string()]);
        assert_eq!(schema.output.unwrap().dims, vec!["M".to_string(), "N".to_string()]);
    }

    #[test]
    fn test_builtin_schemas_present() {
        for name in ["matmul", "attention", "elementwise"] {
            assert!(OpSchema::builtin(name).is_some(), "missing builtin {name}");
        }
        assert!(OpSchema::builtin("nope").is_none());
    }
}
