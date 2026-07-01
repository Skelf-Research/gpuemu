//! External reproducer bridge.
//!
//! Maps a [`ValidationResult`] (with its [`ReproductionInfo`]) into a stable,
//! self-contained JSON document that an external tool can consume as its own
//! reproducer format — e.g. smid's `reproducer.json` — or that smid can adopt
//! directly. The seed + shape/strides/dtype/layout + base64 input snapshot is
//! everything needed to replay a failure byte-for-byte.
//!
//! Every gpuemu type already serialises via serde; this module just reshapes
//! the relevant fields into a documented, language-neutral schema (the SARIF
//! emitter in the CLI is the precedent for a second external schema).

use serde::{Deserialize, Serialize};

use crate::types::{DType, LayoutType, ReproductionInfo, ValidationResult, ValueDistribution};

/// Schema identifier embedded in every emitted reproducer.
pub const REPRODUCER_SCHEMA: &str = "gpuemu.reproducer/v1";

/// A self-contained, replayable description of a validation outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reproducer {
    /// Schema tag for forward compatibility.
    pub schema: String,
    /// Op under test.
    pub op_name: String,
    /// Master seed — replay with `gpuemu reproduce <seed>`.
    pub seed: u64,
    /// Whether validation passed (reproducers are usually emitted for failures).
    pub passed: bool,
    /// Max absolute error observed.
    pub max_diff: f64,
    /// Max relative error observed.
    pub max_rel_diff: f64,
    /// Human-readable failure messages (empty when `passed`).
    pub failures: Vec<String>,
    /// Everything needed to regenerate the exact inputs, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repro: Option<ReproPayload>,
}

/// The replay payload: shape/strides/dtype/layout plus the exact input bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproPayload {
    /// Representative tensor shape.
    pub shape: Vec<usize>,
    /// Representative strides.
    pub strides: Vec<usize>,
    /// Element dtype (lowercase, e.g. `float32`, `bfloat16`).
    pub dtype: String,
    /// Memory layout (`contiguous` | `strided` | `transposed`).
    pub layout: String,
    /// Value-distribution mode the failing run used, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value_distribution: Option<String>,
    /// Base64 of the compressed exact-input snapshot (absent if not captured —
    /// then replay falls back to seed regeneration).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_snapshot_b64: Option<String>,
}

/// Lowercase dtype name used in the external schema.
pub fn dtype_name(d: DType) -> &'static str {
    match d {
        DType::Float16 => "float16",
        DType::BFloat16 => "bfloat16",
        DType::Float32 => "float32",
        DType::Float64 => "float64",
        DType::Int8 => "int8",
        DType::UInt8 => "uint8",
        DType::Int16 => "int16",
        DType::UInt16 => "uint16",
        DType::Int32 => "int32",
        DType::UInt32 => "uint32",
        DType::Int64 => "int64",
        DType::UInt64 => "uint64",
        DType::Bool => "bool",
    }
}

/// Lowercase layout name used in the external schema.
pub fn layout_name(l: LayoutType) -> &'static str {
    match l {
        LayoutType::Contiguous => "contiguous",
        LayoutType::Strided => "strided",
        LayoutType::Transposed => "transposed",
    }
}

fn value_distribution_name(v: ValueDistribution) -> &'static str {
    match v {
        ValueDistribution::Regular => "regular",
        ValueDistribution::Boundary => "boundary",
        ValueDistribution::Adversarial => "adversarial",
    }
}

impl ReproPayload {
    fn from_info(info: &ReproductionInfo) -> Self {
        use base64::Engine;
        ReproPayload {
            shape: info.shape.clone(),
            strides: info.strides.clone(),
            dtype: dtype_name(info.dtype).to_string(),
            layout: layout_name(info.layout).to_string(),
            value_distribution: info
                .fuzz_config
                .as_ref()
                .map(|c| value_distribution_name(c.value_distribution).to_string()),
            input_snapshot_b64: info
                .input_snapshot
                .as_ref()
                .map(|bytes| base64::engine::general_purpose::STANDARD.encode(bytes)),
        }
    }
}

impl Reproducer {
    /// Build a reproducer from a validation result.
    pub fn from_result(result: &ValidationResult) -> Self {
        Reproducer {
            schema: REPRODUCER_SCHEMA.to_string(),
            op_name: result.op_name.clone(),
            seed: result.seed,
            passed: result.passed,
            max_diff: result.max_diff,
            max_rel_diff: result.max_rel_diff,
            failures: result.failures.iter().map(|f| f.message.clone()).collect(),
            repro: result.repro_info.as_ref().map(ReproPayload::from_info),
        }
    }

    /// Serialise to a pretty JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// The CLI command that replays this failure.
    pub fn replay_command(&self) -> String {
        format!("gpuemu reproduce {}", self.seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FailureKind, FuzzConfig, ValidationFailure};

    fn failing_result_with_repro() -> ValidationResult {
        let mut cfg = FuzzConfig::with_seed(99);
        cfg.value_distribution = ValueDistribution::Adversarial;
        let info = ReproductionInfo {
            seed: 99,
            fuzz_config: Some(cfg),
            shape: vec![2, 3],
            strides: vec![3, 1],
            dtype: DType::BFloat16,
            layout: LayoutType::Contiguous,
            input_snapshot: Some(vec![1, 2, 3, 4]),
        };
        let mut r = ValidationResult::fail(
            "matmul_silu".to_string(),
            99,
            vec![ValidationFailure {
                kind: FailureKind::ToleranceExceeded,
                message: "value mismatch at index 5".to_string(),
                index: Some(5),
                expected: Some(1.0),
                actual: Some(2.0),
            }],
            7,
        );
        r.max_diff = 1.0;
        r.max_rel_diff = 0.5;
        r.repro_info = Some(info);
        r
    }

    #[test]
    fn maps_result_fields_and_snapshot() {
        let r = failing_result_with_repro();
        let repro = Reproducer::from_result(&r);
        assert_eq!(repro.schema, REPRODUCER_SCHEMA);
        assert_eq!(repro.op_name, "matmul_silu");
        assert_eq!(repro.seed, 99);
        assert!(!repro.passed);
        assert_eq!(repro.failures.len(), 1);

        let payload = repro.repro.expect("repro payload");
        assert_eq!(payload.dtype, "bfloat16");
        assert_eq!(payload.layout, "contiguous");
        assert_eq!(payload.value_distribution.as_deref(), Some("adversarial"));
        // base64 of [1,2,3,4]
        use base64::Engine;
        assert_eq!(
            payload.input_snapshot_b64.as_deref(),
            Some(
                base64::engine::general_purpose::STANDARD
                    .encode([1u8, 2, 3, 4])
                    .as_str()
            )
        );
    }

    #[test]
    fn json_round_trips() {
        let r = failing_result_with_repro();
        let repro = Reproducer::from_result(&r);
        let json = repro.to_json().unwrap();
        let back: Reproducer = serde_json::from_str(&json).unwrap();
        assert_eq!(back.seed, 99);
        assert_eq!(back.op_name, "matmul_silu");
        assert_eq!(back.repro.unwrap().dtype, "bfloat16");
        assert!(json.contains("gpuemu.reproducer/v1"));
    }

    #[test]
    fn replay_command_uses_seed() {
        let r = failing_result_with_repro();
        assert_eq!(
            Reproducer::from_result(&r).replay_command(),
            "gpuemu reproduce 99"
        );
    }

    #[test]
    fn passing_result_has_no_failures() {
        let r = ValidationResult::pass("silu".to_string(), 1, 0.0, 0.0, 1);
        let repro = Reproducer::from_result(&r);
        assert!(repro.passed);
        assert!(repro.failures.is_empty());
    }
}
