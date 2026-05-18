//! Validation engine for comparing outputs against references.

use gpuemu_common::config::{InvariantConfig, ValidationConfig};
use gpuemu_common::types::{
    DType, ErrorStats, FailureKind, TensorData, ValidationFailure, ValidationResult,
};
use std::time::Instant;
use tracing::{debug, warn};

/// Validator for comparing tensors.
pub struct Validator {
    config: ValidationConfig,
}

impl Validator {
    /// Create a new validator with the given configuration.
    pub fn new(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Validate output against reference.
    pub fn validate(
        &self,
        op_name: &str,
        output: &TensorData,
        reference: &TensorData,
        seed: u64,
        invariants: Option<&InvariantConfig>,
    ) -> ValidationResult {
        let start = Instant::now();
        let mut failures = Vec::new();

        // Check shape match
        if output.shape != reference.shape {
            failures.push(ValidationFailure {
                kind: FailureKind::ShapeMismatch,
                message: format!(
                    "Shape mismatch: output {:?} vs reference {:?}",
                    output.shape, reference.shape
                ),
                index: None,
                expected: None,
                actual: None,
            });
        }

        // Check dtype match
        if output.dtype != reference.dtype {
            failures.push(ValidationFailure {
                kind: FailureKind::DTypeMismatch,
                message: format!(
                    "DType mismatch: output {:?} vs reference {:?}",
                    output.dtype, reference.dtype
                ),
                index: None,
                expected: None,
                actual: None,
            });
        }

        // If shapes don't match, can't compare values
        if !failures.is_empty() {
            return ValidationResult::fail(
                op_name.to_string(),
                seed,
                failures,
                start.elapsed().as_millis() as u64,
            );
        }

        // Compare values
        let (max_diff, max_rel_diff, value_failures) =
            self.compare_values(output, reference, invariants);
        failures.extend(value_failures);

        // Check NaN/Inf if configured
        if self.config.check_nan {
            if let Some(nan_failures) = self.check_nan(output) {
                failures.extend(nan_failures);
            }
        }

        if self.config.check_inf {
            if let Some(inf_failures) = self.check_inf(output) {
                failures.extend(inf_failures);
            }
        }

        // Check invariants
        if let Some(inv) = invariants {
            if let Some(inv_failures) = self.check_invariants(output, inv) {
                failures.extend(inv_failures);
            }
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        // Capture the full element-wise error distribution (float dtypes only).
        let error_stats = self.compute_error_stats(output, reference);

        if failures.is_empty() {
            debug!(
                "Validation passed for {} (max_diff={:.2e}, max_rel_diff={:.2e})",
                op_name, max_diff, max_rel_diff
            );
            ValidationResult::pass(
                op_name.to_string(),
                seed,
                max_diff,
                max_rel_diff,
                duration_ms,
            )
            .with_error_stats(error_stats)
        } else {
            warn!(
                "Validation failed for {} with {} failures",
                op_name,
                failures.len()
            );
            let mut result =
                ValidationResult::fail(op_name.to_string(), seed, failures, duration_ms);
            result.max_diff = max_diff;
            result.max_rel_diff = max_rel_diff;
            result.error_stats = error_stats;
            result
        }
    }

    /// Compare tensor values and return (max_diff, max_rel_diff, failures).
    fn compare_values(
        &self,
        output: &TensorData,
        reference: &TensorData,
        _invariants: Option<&InvariantConfig>,
    ) -> (f64, f64, Vec<ValidationFailure>) {
        let dtype_str = match output.dtype {
            DType::Float32 => "float32",
            DType::Float16 => "float16",
            DType::BFloat16 => "bfloat16",
            DType::Float64 => "float64",
            _ => "float32",
        };
        let tolerance = self.config.get_tolerance(dtype_str);

        let mut max_diff: f64 = 0.0;
        let mut max_rel_diff: f64 = 0.0;
        let mut failures = Vec::new();
        let mut first_failure_idx: Option<usize> = None;

        match output.dtype {
            DType::Float32 => {
                let output_vals = bytemuck_cast_slice::<f32>(&output.data);
                let ref_vals = bytemuck_cast_slice::<f32>(&reference.data);

                for (i, (o, r)) in output_vals.iter().zip(ref_vals.iter()).enumerate() {
                    let diff = (*o as f64 - *r as f64).abs();
                    let rel_diff = if *r != 0.0 {
                        diff / (*r as f64).abs()
                    } else {
                        diff
                    };

                    max_diff = max_diff.max(diff);
                    max_rel_diff = max_rel_diff.max(rel_diff);

                    if diff > tolerance && first_failure_idx.is_none() {
                        first_failure_idx = Some(i);
                        failures.push(ValidationFailure {
                            kind: FailureKind::ToleranceExceeded,
                            message: format!(
                                "Tolerance exceeded at index {}: diff={:.2e} > tol={:.2e}",
                                i, diff, tolerance
                            ),
                            index: Some(i),
                            expected: Some(*r as f64),
                            actual: Some(*o as f64),
                        });
                    }
                }
            }
            DType::Float64 => {
                let output_vals = bytemuck_cast_slice::<f64>(&output.data);
                let ref_vals = bytemuck_cast_slice::<f64>(&reference.data);

                for (i, (o, r)) in output_vals.iter().zip(ref_vals.iter()).enumerate() {
                    let diff = (*o - *r).abs();
                    let rel_diff = if *r != 0.0 { diff / r.abs() } else { diff };

                    max_diff = max_diff.max(diff);
                    max_rel_diff = max_rel_diff.max(rel_diff);

                    if diff > tolerance && first_failure_idx.is_none() {
                        first_failure_idx = Some(i);
                        failures.push(ValidationFailure {
                            kind: FailureKind::ToleranceExceeded,
                            message: format!(
                                "Tolerance exceeded at index {}: diff={:.2e} > tol={:.2e}",
                                i, diff, tolerance
                            ),
                            index: Some(i),
                            expected: Some(*r),
                            actual: Some(*o),
                        });
                    }
                }
            }
            DType::Float16 => {
                let output_vals = bytemuck_cast_slice::<u16>(&output.data);
                let ref_vals = bytemuck_cast_slice::<u16>(&reference.data);

                for (i, (o, r)) in output_vals.iter().zip(ref_vals.iter()).enumerate() {
                    let o_f = f16_to_f32(*o);
                    let r_f = f16_to_f32(*r);
                    let diff = (o_f - r_f).abs() as f64;
                    let rel_diff = if r_f != 0.0 {
                        diff / (r_f as f64).abs()
                    } else {
                        diff
                    };

                    max_diff = max_diff.max(diff);
                    max_rel_diff = max_rel_diff.max(rel_diff);

                    if diff > tolerance && first_failure_idx.is_none() {
                        first_failure_idx = Some(i);
                        failures.push(ValidationFailure {
                            kind: FailureKind::ToleranceExceeded,
                            message: format!(
                                "Tolerance exceeded at index {}: diff={:.2e} > tol={:.2e}",
                                i, diff, tolerance
                            ),
                            index: Some(i),
                            expected: Some(r_f as f64),
                            actual: Some(o_f as f64),
                        });
                    }
                }
            }
            DType::BFloat16 => {
                let output_vals = bytemuck_cast_slice::<u16>(&output.data);
                let ref_vals = bytemuck_cast_slice::<u16>(&reference.data);

                for (i, (o, r)) in output_vals.iter().zip(ref_vals.iter()).enumerate() {
                    let o_f = bf16_to_f32(*o);
                    let r_f = bf16_to_f32(*r);
                    let diff = (o_f - r_f).abs() as f64;
                    let rel_diff = if r_f != 0.0 {
                        diff / (r_f as f64).abs()
                    } else {
                        diff
                    };

                    max_diff = max_diff.max(diff);
                    max_rel_diff = max_rel_diff.max(rel_diff);

                    if diff > tolerance && first_failure_idx.is_none() {
                        first_failure_idx = Some(i);
                        failures.push(ValidationFailure {
                            kind: FailureKind::ToleranceExceeded,
                            message: format!(
                                "Tolerance exceeded at index {}: diff={:.2e} > tol={:.2e}",
                                i, diff, tolerance
                            ),
                            index: Some(i),
                            expected: Some(r_f as f64),
                            actual: Some(o_f as f64),
                        });
                    }
                }
            }
            DType::Int8 | DType::UInt8 => {
                let output_vals: &[u8] = &output.data;
                let ref_vals: &[u8] = &reference.data;
                max_diff = self.compare_bytes(output_vals, ref_vals, &mut failures, tolerance);
            }
            DType::Int16 => {
                let output_vals = bytemuck_cast_slice::<i16>(&output.data);
                let ref_vals = bytemuck_cast_slice::<i16>(&reference.data);
                max_diff = self.compare_integers(output_vals, ref_vals, &mut failures, tolerance);
            }
            DType::UInt16 => {
                let output_vals = bytemuck_cast_slice::<u16>(&output.data);
                let ref_vals = bytemuck_cast_slice::<u16>(&reference.data);
                max_diff =
                    self.compare_unsigned_integers(output_vals, ref_vals, &mut failures, tolerance);
            }
            DType::Int32 => {
                let output_vals = bytemuck_cast_slice::<i32>(&output.data);
                let ref_vals = bytemuck_cast_slice::<i32>(&reference.data);
                max_diff = self.compare_integers(output_vals, ref_vals, &mut failures, tolerance);
            }
            DType::UInt32 => {
                let output_vals = bytemuck_cast_slice::<u32>(&output.data);
                let ref_vals = bytemuck_cast_slice::<u32>(&reference.data);
                max_diff =
                    self.compare_unsigned_integers(output_vals, ref_vals, &mut failures, tolerance);
            }
            DType::Int64 => {
                let output_vals = bytemuck_cast_slice::<i64>(&output.data);
                let ref_vals = bytemuck_cast_slice::<i64>(&reference.data);
                max_diff = self.compare_integers(output_vals, ref_vals, &mut failures, tolerance);
            }
            DType::UInt64 => {
                let output_vals = bytemuck_cast_slice::<u64>(&output.data);
                let ref_vals = bytemuck_cast_slice::<u64>(&reference.data);
                max_diff =
                    self.compare_unsigned_integers(output_vals, ref_vals, &mut failures, tolerance);
            }
            DType::Bool => {
                let output_vals = &output.data;
                let ref_vals = &reference.data;
                max_diff = self.compare_bytes(output_vals, ref_vals, &mut failures, tolerance);
            }
        }

        (max_diff, max_rel_diff, failures)
    }

    fn compare_integers<T: Copy + Into<i64> + std::fmt::Debug>(
        &self,
        output: &[T],
        reference: &[T],
        failures: &mut Vec<ValidationFailure>,
        tolerance: f64,
    ) -> f64 {
        let mut max_diff = 0.0f64;
        for (i, (o, r)) in output.iter().zip(reference.iter()).enumerate() {
            let diff = ((*o).into() - (*r).into()).abs() as f64;
            max_diff = max_diff.max(diff);
            if diff > tolerance && failures.is_empty() {
                failures.push(ValidationFailure {
                    kind: FailureKind::ToleranceExceeded,
                    message: format!(
                        "Integer mismatch at index {}: expected {:?}, got {:?}, diff={:.0}",
                        i, r, o, diff
                    ),
                    index: Some(i),
                    expected: Some((*r).into() as f64),
                    actual: Some((*o).into() as f64),
                });
            }
        }
        max_diff
    }

    fn compare_unsigned_integers<T: Copy + Into<u64> + std::fmt::Debug>(
        &self,
        output: &[T],
        reference: &[T],
        failures: &mut Vec<ValidationFailure>,
        tolerance: f64,
    ) -> f64 {
        let mut max_diff = 0.0f64;
        for (i, (o, r)) in output.iter().zip(reference.iter()).enumerate() {
            let diff = if (*o).into() >= (*r).into() {
                ((*o).into() - (*r).into()) as f64
            } else {
                ((*r).into() - (*o).into()) as f64
            };
            max_diff = max_diff.max(diff);
            if diff > tolerance && failures.is_empty() {
                failures.push(ValidationFailure {
                    kind: FailureKind::ToleranceExceeded,
                    message: format!(
                        "Unsigned integer mismatch at index {}: expected {:?}, got {:?}, diff={:.0}",
                        i, r, o, diff
                    ),
                    index: Some(i),
                    expected: Some((*r).into() as f64),
                    actual: Some((*o).into() as f64),
                });
            }
        }
        max_diff
    }

    fn compare_bytes(
        &self,
        output: &[u8],
        reference: &[u8],
        failures: &mut Vec<ValidationFailure>,
        tolerance: f64,
    ) -> f64 {
        let mut max_diff = 0.0f64;
        for (i, (o, r)) in output.iter().zip(reference.iter()).enumerate() {
            let diff = (*o as i64 - *r as i64).abs() as f64;
            max_diff = max_diff.max(diff);
            if diff > tolerance && failures.is_empty() {
                failures.push(ValidationFailure {
                    kind: FailureKind::ToleranceExceeded,
                    message: format!("Byte mismatch at index {}: expected {}, got {}", i, r, o),
                    index: Some(i),
                    expected: Some(*r as f64),
                    actual: Some(*o as f64),
                });
            }
        }
        max_diff
    }

    /// Check for NaN values in the output.
    fn check_nan(&self, output: &TensorData) -> Option<Vec<ValidationFailure>> {
        let mut failures = Vec::new();

        match output.dtype {
            DType::Float32 => {
                let vals = bytemuck_cast_slice::<f32>(&output.data);
                for (i, v) in vals.iter().enumerate() {
                    if v.is_nan() {
                        failures.push(ValidationFailure {
                            kind: FailureKind::NaNDetected,
                            message: format!("NaN detected at index {}", i),
                            index: Some(i),
                            expected: None,
                            actual: None,
                        });
                        break; // Report first occurrence only
                    }
                }
            }
            DType::Float64 => {
                let vals = bytemuck_cast_slice::<f64>(&output.data);
                for (i, v) in vals.iter().enumerate() {
                    if v.is_nan() {
                        failures.push(ValidationFailure {
                            kind: FailureKind::NaNDetected,
                            message: format!("NaN detected at index {}", i),
                            index: Some(i),
                            expected: None,
                            actual: None,
                        });
                        break;
                    }
                }
            }
            DType::Float16 | DType::BFloat16 => {
                let convert = if output.dtype == DType::Float16 {
                    f16_to_f32
                } else {
                    bf16_to_f32
                };
                let vals = bytemuck_cast_slice::<u16>(&output.data);
                for (i, bits) in vals.iter().enumerate() {
                    if convert(*bits).is_nan() {
                        failures.push(ValidationFailure {
                            kind: FailureKind::NaNDetected,
                            message: format!("NaN detected at index {}", i),
                            index: Some(i),
                            expected: None,
                            actual: None,
                        });
                        break;
                    }
                }
            }
            _ => {}
        }

        if failures.is_empty() {
            None
        } else {
            Some(failures)
        }
    }

    /// Check for Inf values in the output.
    fn check_inf(&self, output: &TensorData) -> Option<Vec<ValidationFailure>> {
        let mut failures = Vec::new();

        match output.dtype {
            DType::Float32 => {
                let vals = bytemuck_cast_slice::<f32>(&output.data);
                for (i, v) in vals.iter().enumerate() {
                    if v.is_infinite() {
                        failures.push(ValidationFailure {
                            kind: FailureKind::InfDetected,
                            message: format!("Inf detected at index {}", i),
                            index: Some(i),
                            expected: None,
                            actual: None,
                        });
                        break;
                    }
                }
            }
            DType::Float64 => {
                let vals = bytemuck_cast_slice::<f64>(&output.data);
                for (i, v) in vals.iter().enumerate() {
                    if v.is_infinite() {
                        failures.push(ValidationFailure {
                            kind: FailureKind::InfDetected,
                            message: format!("Inf detected at index {}", i),
                            index: Some(i),
                            expected: None,
                            actual: None,
                        });
                        break;
                    }
                }
            }
            DType::Float16 | DType::BFloat16 => {
                let convert = if output.dtype == DType::Float16 {
                    f16_to_f32
                } else {
                    bf16_to_f32
                };
                let vals = bytemuck_cast_slice::<u16>(&output.data);
                for (i, bits) in vals.iter().enumerate() {
                    if convert(*bits).is_infinite() {
                        failures.push(ValidationFailure {
                            kind: FailureKind::InfDetected,
                            message: format!("Inf detected at index {}", i),
                            index: Some(i),
                            expected: None,
                            actual: None,
                        });
                        break;
                    }
                }
            }
            _ => {}
        }

        if failures.is_empty() {
            None
        } else {
            Some(failures)
        }
    }

    /// Check invariants on the output.
    fn check_invariants(
        &self,
        output: &TensorData,
        invariants: &InvariantConfig,
    ) -> Option<Vec<ValidationFailure>> {
        let mut failures = Vec::new();

        if invariants.non_negative {
            match output.dtype {
                DType::Float32 => {
                    let vals = bytemuck_cast_slice::<f32>(&output.data);
                    for (i, v) in vals.iter().enumerate() {
                        if *v < 0.0 {
                            failures.push(ValidationFailure {
                                kind: FailureKind::InvariantViolation,
                                message: format!(
                                    "Non-negative invariant violated at index {}: value={}",
                                    i, v
                                ),
                                index: Some(i),
                                expected: None,
                                actual: Some(*v as f64),
                            });
                            break;
                        }
                    }
                }
                DType::Float64 => {
                    let vals = bytemuck_cast_slice::<f64>(&output.data);
                    for (i, v) in vals.iter().enumerate() {
                        if *v < 0.0 {
                            failures.push(ValidationFailure {
                                kind: FailureKind::InvariantViolation,
                                message: format!(
                                    "Non-negative invariant violated at index {}: value={}",
                                    i, v
                                ),
                                index: Some(i),
                                expected: None,
                                actual: Some(*v),
                            });
                            break;
                        }
                    }
                }
                _ => {}
            }
        }

        if invariants.shape_preserved {
            let expected_strides = TensorData::compute_contiguous_strides(&output.shape);
            if output.strides != expected_strides || !output.is_contiguous() {
                // Check that the number of elements is preserved regardless of layout
                let total_elements: usize = output.shape.iter().product();
                let expected_elements: usize = output.shape.iter().product();
                if total_elements != expected_elements {
                    failures.push(ValidationFailure {
                        kind: FailureKind::InvariantViolation,
                        message: format!(
                            "Shape preserved invariant violated: expected {} elements, got {}",
                            expected_elements, total_elements
                        ),
                        index: None,
                        expected: None,
                        actual: None,
                    });
                }
            }
        }

        if failures.is_empty() {
            None
        } else {
            Some(failures)
        }
    }
}

/// Helper to cast byte slices to typed slices.
pub(crate) fn bytemuck_cast_slice<T: bytemuck::Pod>(bytes: &[u8]) -> &[T] {
    bytemuck::cast_slice(bytes)
}

/// Helper to cast mutable byte slices to typed slices.
pub(crate) fn bytemuck_cast_slice_mut<T: bytemuck::Pod>(bytes: &mut [u8]) -> &mut [T] {
    bytemuck::cast_slice_mut(bytes)
}

// Add bytemuck dependency
use bytemuck;

// =============================================================================
// Float conversions (shared by value comparison, NaN/Inf checks, error stats)
// =============================================================================

/// Convert IEEE-754 half-precision (fp16) bits to f32.
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            let sign32 = sign << 31;
            let mant32 = mant << 13;
            let exp32 = 127 - 15 + 1;
            f32::from_bits(sign32 | ((exp32 as u32) << 23) | mant32) - f32::from_bits(1u32 << 23)
        }
    } else if exp == 31 {
        // fp16 Inf/NaN -> f32 Inf/NaN: exponent field must be all ones (0xFF).
        let signed = (sign << 31) | (0xFFu32 << 23) | (mant << 13);
        f32::from_bits(signed)
    } else {
        let sign32 = sign << 31;
        let exp32 = (exp + 127 - 15) as u32;
        let mant32 = mant << 13;
        f32::from_bits(sign32 | (exp32 << 23) | mant32)
    }
}

/// Convert bfloat16 bits (1 sign, 8 exp, 7 mantissa) to f32.
pub(crate) fn bf16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 7) & 0xFF) as u32;
    let mant = (bits & 0x7F) as u32;
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            let sign32 = sign << 31;
            let mant32 = mant << 16;
            let exp32 = 127 - 127 + 1;
            f32::from_bits(sign32 | (exp32 << 23) | mant32) - f32::from_bits(1u32 << 23)
        }
    } else if exp == 255 {
        f32::from_bits((sign << 31) | (0xFFu32 << 23) | (mant << 16))
    } else {
        let sign32 = sign << 31;
        let exp32 = exp;
        let mant32 = mant << 16;
        f32::from_bits(sign32 | (exp32 << 23) | mant32)
    }
}

// =============================================================================
// ULP (units-in-the-last-place) distance
// =============================================================================
//
// Sign-magnitude floats become monotonically-ordered integers via the standard
// transform (flip sign bit for positives, flip all bits for negatives); the
// absolute difference of the ordered keys is the ULP distance. Non-finite
// inputs yield u64::MAX (treated as "infinitely far").

fn f32_ulp(a: f32, b: f32) -> u64 {
    if !a.is_finite() || !b.is_finite() {
        return u64::MAX;
    }
    let key = |x: f32| -> u32 {
        let bits = x.to_bits();
        if bits >> 31 == 0 {
            bits | 0x8000_0000
        } else {
            !bits
        }
    };
    let (ka, kb) = (key(a), key(b));
    (ka.max(kb) - ka.min(kb)) as u64
}

fn f64_ulp(a: f64, b: f64) -> u64 {
    if !a.is_finite() || !b.is_finite() {
        return u64::MAX;
    }
    let key = |x: f64| -> u64 {
        let bits = x.to_bits();
        if bits >> 63 == 0 {
            bits | 0x8000_0000_0000_0000
        } else {
            !bits
        }
    };
    let (ka, kb) = (key(a), key(b));
    ka.max(kb) - ka.min(kb)
}

/// ULP distance for 16-bit sign-magnitude floats (fp16 or bf16) from raw bits.
/// `finite` reports whether both values are finite in f32 space.
fn u16_ulp(a: u16, b: u16, finite: bool) -> u64 {
    if !finite {
        return u64::MAX;
    }
    let key = |x: u16| -> u16 {
        if x >> 15 == 0 {
            x | 0x8000
        } else {
            !x
        }
    };
    let (ka, kb) = (key(a), key(b));
    (ka.max(kb) - ka.min(kb)) as u64
}

/// Streaming accumulator for element-wise error statistics.
struct StatsAccum {
    abs_errs: Vec<f64>,
    sum_abs: f64,
    max_abs: f64,
    sum_rel: f64,
    max_rel: f64,
    rel_count: usize,
    sum_ulp: f64,
    max_ulp: u64,
    num_exceeding: usize,
    tol: f64,
}

impl StatsAccum {
    fn with_capacity(n: usize, tol: f64) -> Self {
        Self {
            abs_errs: Vec::with_capacity(n),
            sum_abs: 0.0,
            max_abs: 0.0,
            sum_rel: 0.0,
            max_rel: 0.0,
            rel_count: 0,
            sum_ulp: 0.0,
            max_ulp: 0,
            num_exceeding: 0,
            tol,
        }
    }

    fn push(&mut self, o: f64, r: f64, ulp: u64) {
        let mut abs = (o - r).abs();
        if !abs.is_finite() {
            abs = f64::INFINITY; // sort last; signals a blow-up
        }
        self.abs_errs.push(abs);
        self.sum_abs += abs;
        self.max_abs = self.max_abs.max(abs);
        if r != 0.0 {
            let rel = abs / r.abs();
            self.sum_rel += rel;
            self.max_rel = self.max_rel.max(rel);
            self.rel_count += 1;
        }
        self.sum_ulp += ulp as f64;
        self.max_ulp = self.max_ulp.max(ulp);
        if abs > self.tol {
            self.num_exceeding += 1;
        }
    }

    fn finish(mut self) -> ErrorStats {
        let count = self.abs_errs.len();
        self.abs_errs
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pct = |p: f64| -> f64 {
            if count == 0 {
                0.0
            } else {
                let idx = ((p * (count as f64 - 1.0)).round() as usize).min(count - 1);
                self.abs_errs[idx]
            }
        };
        ErrorStats {
            count,
            num_exceeding: self.num_exceeding,
            max_abs: self.max_abs,
            mean_abs: if count > 0 { self.sum_abs / count as f64 } else { 0.0 },
            p50_abs: pct(0.50),
            p90_abs: pct(0.90),
            p99_abs: pct(0.99),
            max_rel: self.max_rel,
            mean_rel: if self.rel_count > 0 {
                self.sum_rel / self.rel_count as f64
            } else {
                0.0
            },
            max_ulp: self.max_ulp,
            mean_ulp: if count > 0 { self.sum_ulp / count as f64 } else { 0.0 },
        }
    }
}

impl Validator {
    /// Tolerance for a dtype (mirrors the mapping used in `compare_values`).
    fn tolerance_for(&self, dtype: DType) -> f64 {
        let s = match dtype {
            DType::Float32 => "float32",
            DType::Float16 => "float16",
            DType::BFloat16 => "bfloat16",
            DType::Float64 => "float64",
            _ => "float32",
        };
        self.config.get_tolerance(s)
    }

    /// Compute the full element-wise error distribution for float outputs.
    /// Returns `None` for non-float dtypes or on shape/length mismatch.
    pub fn compute_error_stats(
        &self,
        output: &TensorData,
        reference: &TensorData,
    ) -> Option<ErrorStats> {
        if output.shape != reference.shape || output.dtype != reference.dtype {
            return None;
        }
        let tol = self.tolerance_for(output.dtype);
        // bytemuck's cast_slice requires the buffer's *pointer* to be aligned
        // to the target type, which empty Vec<u8>::new() does not guarantee.
        // For empty tensors, return zero-stats directly.
        if output.data.is_empty() {
            return Some(StatsAccum::with_capacity(0, tol).finish());
        }
        match output.dtype {
            DType::Float32 => {
                let o = bytemuck_cast_slice::<f32>(&output.data);
                let r = bytemuck_cast_slice::<f32>(&reference.data);
                let mut acc = StatsAccum::with_capacity(o.len(), tol);
                for (a, b) in o.iter().zip(r.iter()) {
                    acc.push(*a as f64, *b as f64, f32_ulp(*a, *b));
                }
                Some(acc.finish())
            }
            DType::Float64 => {
                let o = bytemuck_cast_slice::<f64>(&output.data);
                let r = bytemuck_cast_slice::<f64>(&reference.data);
                let mut acc = StatsAccum::with_capacity(o.len(), tol);
                for (a, b) in o.iter().zip(r.iter()) {
                    acc.push(*a, *b, f64_ulp(*a, *b));
                }
                Some(acc.finish())
            }
            DType::Float16 => {
                let o = bytemuck_cast_slice::<u16>(&output.data);
                let r = bytemuck_cast_slice::<u16>(&reference.data);
                let mut acc = StatsAccum::with_capacity(o.len(), tol);
                for (a, b) in o.iter().zip(r.iter()) {
                    let (af, bf) = (f16_to_f32(*a), f16_to_f32(*b));
                    let ulp = u16_ulp(*a, *b, af.is_finite() && bf.is_finite());
                    acc.push(af as f64, bf as f64, ulp);
                }
                Some(acc.finish())
            }
            DType::BFloat16 => {
                let o = bytemuck_cast_slice::<u16>(&output.data);
                let r = bytemuck_cast_slice::<u16>(&reference.data);
                let mut acc = StatsAccum::with_capacity(o.len(), tol);
                for (a, b) in o.iter().zip(r.iter()) {
                    let (af, bf) = (bf16_to_f32(*a), bf16_to_f32(*b));
                    let ulp = u16_ulp(*a, *b, af.is_finite() && bf.is_finite());
                    acc.push(af as f64, bf as f64, ulp);
                }
                Some(acc.finish())
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_f32_tensor(shape: Vec<usize>, values: Vec<f32>) -> TensorData {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        TensorData::new(shape, DType::Float32, data)
    }

    #[test]
    fn test_validate_pass() {
        let config = ValidationConfig::default();
        let validator = Validator::new(config);

        let output = make_f32_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let reference = make_f32_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);

        let result = validator.validate("test", &output, &reference, 123, None);
        assert!(result.passed);
    }

    #[test]
    fn test_validate_tolerance_fail() {
        let config = ValidationConfig::default();
        let validator = Validator::new(config);

        let output = make_f32_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let reference = make_f32_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 5.0]); // Last value differs

        let result = validator.validate("test", &output, &reference, 123, None);
        assert!(!result.passed);
        assert!(!result.failures.is_empty());
    }

    #[test]
    fn test_validate_shape_mismatch() {
        let config = ValidationConfig::default();
        let validator = Validator::new(config);

        let output = make_f32_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let reference = make_f32_tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0]);

        let result = validator.validate("test", &output, &reference, 123, None);
        assert!(!result.passed);
        assert!(result
            .failures
            .iter()
            .any(|f| f.kind == FailureKind::ShapeMismatch));
    }

    #[test]
    fn test_validate_nan_detection() {
        let config = ValidationConfig::default();
        let validator = Validator::new(config);

        let output = make_f32_tensor(vec![4], vec![1.0, f32::NAN, 3.0, 4.0]);
        let reference = make_f32_tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0]);

        let result = validator.validate("test", &output, &reference, 123, None);
        assert!(!result.passed);
        assert!(result
            .failures
            .iter()
            .any(|f| f.kind == FailureKind::NaNDetected));
    }

    #[test]
    fn test_f32_ulp_distance() {
        assert_eq!(f32_ulp(1.0, 1.0), 0);
        // One ULP above 1.0.
        let next = f32::from_bits(1.0f32.to_bits() + 1);
        assert_eq!(f32_ulp(1.0, next), 1);
        // Under the monotonic-key transform, -0.0 and +0.0 are adjacent (1 ULP).
        assert_eq!(f32_ulp(-0.0, 0.0), 1);
        // Non-finite -> saturates.
        assert_eq!(f32_ulp(1.0, f32::NAN), u64::MAX);
        assert_eq!(f32_ulp(1.0, f32::INFINITY), u64::MAX);
    }

    #[test]
    fn test_f16_inf_nan_conversion_fixed() {
        // fp16 +Inf = 0x7C00, NaN = 0x7E00 must decode to f32 Inf/NaN.
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7E00).is_nan());
        // A normal fp16 value (1.0 = 0x3C00) still decodes correctly.
        assert_eq!(f16_to_f32(0x3C00), 1.0);
    }

    #[test]
    fn test_error_stats_basic() {
        let config = ValidationConfig::default(); // float32 tol = 1e-5
        let validator = Validator::new(config);

        // diffs: 0, 0, 0.5, 2.0  -> two exceed tol; max_abs = 2.0
        let output = make_f32_tensor(vec![4], vec![1.0, 2.0, 3.5, 6.0]);
        let reference = make_f32_tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0]);

        let stats = validator
            .compute_error_stats(&output, &reference)
            .expect("float stats");
        assert_eq!(stats.count, 4);
        assert_eq!(stats.num_exceeding, 2);
        assert!((stats.max_abs - 2.0).abs() < 1e-9);
        assert!((stats.mean_abs - (0.0 + 0.0 + 0.5 + 2.0) / 4.0).abs() < 1e-9);
        // max relative error is at the last element: 2.0 / 4.0 = 0.5
        assert!((stats.max_rel - 0.5).abs() < 1e-9);
        assert!(stats.max_ulp > 0);
    }

    #[test]
    fn test_error_stats_attached_to_result() {
        let validator = Validator::new(ValidationConfig::default());
        let output = make_f32_tensor(vec![3], vec![1.0, 2.0, 3.0]);
        let reference = make_f32_tensor(vec![3], vec![1.0, 2.0, 3.0]);
        let result = validator.validate("test", &output, &reference, 1, None);
        assert!(result.passed);
        let stats = result.error_stats.expect("stats present on pass");
        assert_eq!(stats.count, 3);
        assert_eq!(stats.max_abs, 0.0);
    }

    #[test]
    fn test_error_stats_empty() {
        let v = Validator::new(ValidationConfig::default());
        let out = make_f32_tensor(vec![0], vec![]);
        let r = make_f32_tensor(vec![0], vec![]);
        let s = v.compute_error_stats(&out, &r).expect("empty stats");
        assert_eq!(s.count, 0);
        assert_eq!(s.max_abs, 0.0);
        assert_eq!(s.num_exceeding, 0);
    }

    #[test]
    fn test_error_stats_all_nan() {
        // When the kernel returns all-NaN, |o - r| is NaN per element; the
        // stats path coerces NaN to +inf so max_abs is finite-infinite and
        // num_exceeding still counts. mean_rel is over r != 0 only.
        let v = Validator::new(ValidationConfig::default());
        let out = make_f32_tensor(vec![3], vec![f32::NAN, f32::NAN, f32::NAN]);
        let r = make_f32_tensor(vec![3], vec![1.0, 2.0, 3.0]);
        let s = v.compute_error_stats(&out, &r).expect("nan stats");
        assert_eq!(s.count, 3);
        assert!(s.max_abs.is_infinite());
        assert_eq!(s.num_exceeding, 3);
    }

    #[test]
    fn test_value_distribution_default_is_uniform() {
        // FuzzConfig default keeps Uniform — preserves existing run determinism.
        let c = gpuemu_common::types::FuzzConfig::default();
        assert!(matches!(c.value_distribution,
                         gpuemu_common::types::ValueDistribution::Uniform));
    }

    #[test]
    fn test_f16_nan_detected() {
        let validator = Validator::new(ValidationConfig::default());
        // fp16 tensor [1.0, NaN] = [0x3C00, 0x7E00]
        let bits: [u16; 2] = [0x3C00, 0x7E00];
        let data: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let output = TensorData::new(vec![2], DType::Float16, data.clone());
        let reference = TensorData::new(vec![2], DType::Float16, data);

        let result = validator.validate("test", &output, &reference, 1, None);
        assert!(!result.passed);
        assert!(result
            .failures
            .iter()
            .any(|f| f.kind == FailureKind::NaNDetected));
    }
}
