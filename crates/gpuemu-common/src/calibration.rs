//! Tolerance calibration and the precision/fusion tolerance class.
//!
//! Two related pieces:
//!
//! * [`calibrate_tolerance`] — fit an absolute-error envelope from a set of
//!   *control* (known-correct) error magnitudes: the p-th percentile times a
//!   safety margin. The research default is p95 × 1.5 (vs. a single hand-picked
//!   `atol/rtol`), which is what [`calibrate_p95`] uses.
//!
//! * [`tolerance_multiplier`] — a *tolerance class* that widens the base
//!   per-dtype tolerance to account for error compounding across fused ops and
//!   for low-precision (bf16/fp16) tensor-core accumulation. This is the
//!   automatic safety net that keeps a legitimately-lower-precision fused
//!   kernel (e.g. a bf16 fused matmul) from being flagged as buggy.

use crate::types::DType;

/// Default percentile (0.95) for the research calibration envelope.
pub const DEFAULT_PERCENTILE: f64 = 0.95;
/// Default safety margin (×1.5) for the research calibration envelope.
pub const DEFAULT_SAFETY_MARGIN: f64 = 1.5;

/// Fit an absolute-tolerance envelope from control error magnitudes.
///
/// `samples` are the per-run maximum absolute errors observed on
/// known-correct controls. Returns the `percentile`-th value scaled by
/// `safety_margin`. `percentile` is a fraction in `[0, 1]`. Returns `0.0` for
/// an empty sample set. Non-finite samples sort last (treated as the worst
/// case), matching the validator's error-stats convention.
pub fn calibrate_tolerance(samples: &[f64], percentile: f64, safety_margin: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
    let p = percentile.clamp(0.0, 1.0);
    let idx = ((p * (sorted.len() as f64 - 1.0)).round() as usize).min(sorted.len() - 1);
    sorted[idx] * safety_margin
}

/// [`calibrate_tolerance`] with the research defaults (p95 × 1.5).
pub fn calibrate_p95(samples: &[f64]) -> f64 {
    calibrate_tolerance(samples, DEFAULT_PERCENTILE, DEFAULT_SAFETY_MARGIN)
}

/// Per-operation multiplier reflecting how much numerical error an op
/// legitimately accumulates (matmul FMAs, softmax exp amplification, fused
/// epilogues, attention's matmul+softmax+matmul chain).
pub fn op_multiplier(op_name: &str) -> f64 {
    match op_name.to_lowercase().as_str() {
        // pure elementwise — no accumulation
        "silu" | "gelu" | "relu" | "elementwise" => 1.0,
        // single reduction / trig
        "matmul" | "softmax" | "rope" => 1.5,
        // normalisation (variance-sensitive)
        "rmsnorm" | "layernorm" | "batchnorm" => 2.0,
        // fused matmul + activation epilogue (matmul error then nonlinearity)
        "matmul_silu" | "matmul_gelu" | "conv2d" => 2.0,
        // attention: matmul -> softmax -> matmul
        "attention" | "kv_cache_attention" => 2.5,
        _ => 1.0,
    }
}

/// Per-dtype multiplier reflecting low-precision accumulation. bf16's ~8-bit
/// mantissa rounds inputs hard even with fp32 tensor-core accumulation, so it
/// needs the widest envelope; fp16 is intermediate; fp32/fp64 are 1.0.
pub fn dtype_multiplier(dtype: DType) -> f64 {
    match dtype {
        DType::BFloat16 => 4.0,
        DType::Float16 => 2.0,
        _ => 1.0,
    }
}

/// The combined tolerance-class multiplier for an `(op, dtype)` pair: the
/// product of the op and dtype multipliers. Multiply the base per-dtype
/// tolerance by this to get the effective envelope.
pub fn tolerance_multiplier(op_name: &str, dtype: DType) -> f64 {
    op_multiplier(op_name) * dtype_multiplier(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibrate_empty_is_zero() {
        assert_eq!(calibrate_p95(&[]), 0.0);
    }

    #[test]
    fn calibrate_p95_picks_percentile_times_margin() {
        // 0..=100 -> p95 index = round(0.95*100) = 95 -> value 95.0, ×1.5.
        let samples: Vec<f64> = (0..=100).map(|i| i as f64).collect();
        let tol = calibrate_p95(&samples);
        assert!((tol - 95.0 * 1.5).abs() < 1e-9, "got {tol}");
    }

    #[test]
    fn calibrate_custom_percentile_and_margin() {
        let samples: Vec<f64> = (0..=100).map(|i| i as f64).collect();
        // p50 = 50.0, margin 2.0 -> 100.0
        assert!((calibrate_tolerance(&samples, 0.50, 2.0) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn fused_and_bf16_widen_the_envelope() {
        // A pure f32 elementwise op gets the base envelope...
        assert_eq!(tolerance_multiplier("silu", DType::Float32), 1.0);
        // ...a bf16 fused matmul gets a much wider one (2.0 * 4.0).
        assert_eq!(tolerance_multiplier("matmul_silu", DType::BFloat16), 8.0);
        // attention in fp16: 2.5 * 2.0
        assert_eq!(tolerance_multiplier("attention", DType::Float16), 5.0);
        // unknown op, f32: neutral
        assert_eq!(tolerance_multiplier("mystery", DType::Float32), 1.0);
    }
}
