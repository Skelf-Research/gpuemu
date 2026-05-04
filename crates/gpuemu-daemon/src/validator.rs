//! Validation engine for comparing outputs against references.

use gpuemu_common::config::{InvariantConfig, ValidationConfig};
use gpuemu_common::types::{DType, FailureKind, TensorData, ValidationFailure, ValidationResult};
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

        if failures.is_empty() {
            debug!(
                "Validation passed for {} (max_diff={:.2e}, max_rel_diff={:.2e})",
                op_name, max_diff, max_rel_diff
            );
            ValidationResult::pass(op_name.to_string(), seed, max_diff, max_rel_diff, duration_ms)
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
            // For other dtypes, skip value comparison for now
            _ => {}
        }

        (max_diff, max_rel_diff, failures)
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

        if failures.is_empty() {
            None
        } else {
            Some(failures)
        }
    }
}

/// Helper to cast byte slices to typed slices.
fn bytemuck_cast_slice<T: bytemuck::Pod>(bytes: &[u8]) -> &[T] {
    bytemuck::cast_slice(bytes)
}

// Add bytemuck dependency
use bytemuck;

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
        assert!(result.failures.iter().any(|f| f.kind == FailureKind::ShapeMismatch));
    }

    #[test]
    fn test_validate_nan_detection() {
        let config = ValidationConfig::default();
        let validator = Validator::new(config);

        let output = make_f32_tensor(vec![4], vec![1.0, f32::NAN, 3.0, 4.0]);
        let reference = make_f32_tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0]);

        let result = validator.validate("test", &output, &reference, 123, None);
        assert!(!result.passed);
        assert!(result.failures.iter().any(|f| f.kind == FailureKind::NaNDetected));
    }
}
