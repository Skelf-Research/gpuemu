//! Fuzzer for generating test cases from FuzzConfig.

use gpuemu_common::rng::SeededRng;
use gpuemu_common::types::{
    DType, FuzzConfig, LayoutType, ReproductionInfo, ShapeOptions, TensorData,
};
use std::collections::HashMap;

/// A generated test case for fuzzing.
#[derive(Debug, Clone)]
pub struct TestCase {
    /// Seed for this specific test case.
    pub seed: u64,
    /// Generated shape.
    pub shape: Vec<usize>,
    /// Generated strides.
    pub strides: Vec<usize>,
    /// Generated dtype.
    pub dtype: DType,
    /// Generated layout type.
    pub layout: LayoutType,
    /// Generated input tensors.
    pub inputs: HashMap<String, TensorData>,
}

impl TestCase {
    /// Create reproduction info from this test case.
    pub fn to_repro_info(&self, fuzz_config: Option<FuzzConfig>) -> ReproductionInfo {
        ReproductionInfo {
            seed: self.seed,
            fuzz_config,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
            layout: self.layout,
            input_snapshot: None, // Could compress inputs here if needed
        }
    }
}

/// Fuzzer that generates test cases from a FuzzConfig.
pub struct Fuzzer {
    config: FuzzConfig,
    rng: SeededRng,
    iteration: usize,
}

impl Fuzzer {
    /// Create a new fuzzer from a FuzzConfig.
    pub fn new(config: FuzzConfig) -> Self {
        let rng = SeededRng::new(config.seed);
        Self {
            config,
            rng,
            iteration: 0,
        }
    }

    /// Generate the next test case.
    pub fn next_test_case(&mut self, input_names: &[&str]) -> TestCase {
        // Derive a seed for this iteration
        let iter_seed = self.rng.derive(&format!("iter_{}", self.iteration)).seed();
        self.iteration += 1;

        let mut iter_rng = SeededRng::new(iter_seed);

        // Generate shape
        let shape = self.generate_shape(&mut iter_rng.derive("shape"));

        // Generate dtype
        let dtype = iter_rng.derive("dtype").choice(&self.config.dtypes);

        // Generate layout
        let layout = iter_rng.derive("layout").choice(&self.config.layouts);

        // Compute strides based on layout
        let strides = self.compute_strides(&shape, layout);

        // Generate input tensors
        let mut inputs = HashMap::new();
        let mut data_rng = iter_rng.derive("data");
        for (i, name) in input_names.iter().enumerate() {
            let input = self.generate_tensor(&mut data_rng.derive(&format!("input_{}", i)), &shape, dtype, &strides);
            inputs.insert(name.to_string(), input);
        }

        TestCase {
            seed: iter_seed,
            shape,
            strides,
            dtype,
            layout,
            inputs,
        }
    }

    /// Generate a shape from the shape options.
    fn generate_shape(&self, rng: &mut SeededRng) -> Vec<usize> {
        let opts = &self.config.shape_options;

        // Randomly decide whether to use an edge case
        if !opts.edge_cases.is_empty() && rng.gen_bool(0.1) {
            return rng.choice(&opts.edge_cases);
        }

        // Generate a shape from the options
        let batch = rng.choice(&opts.batch_sizes);
        let seq = rng.choice(&opts.seq_lengths);
        let hidden = rng.choice(&opts.hidden_dims);

        vec![batch, seq, hidden]
    }

    /// Compute strides for a shape based on layout type.
    fn compute_strides(&self, shape: &[usize], layout: LayoutType) -> Vec<usize> {
        match layout {
            LayoutType::Contiguous => TensorData::compute_contiguous_strides(shape),
            LayoutType::Strided => {
                // Add gaps by multiplying all strides by 2
                let contiguous = TensorData::compute_contiguous_strides(shape);
                contiguous.into_iter().map(|s| s * 2).collect()
            }
            LayoutType::Transposed => {
                // Transpose last two dimensions if possible
                if shape.len() >= 2 {
                    let mut transposed_shape = shape.to_vec();
                    let n = transposed_shape.len();
                    transposed_shape.swap(n - 1, n - 2);
                    TensorData::compute_contiguous_strides(&transposed_shape)
                } else {
                    TensorData::compute_contiguous_strides(shape)
                }
            }
        }
    }

    /// Generate a random tensor with the given shape and dtype.
    fn generate_tensor(
        &self,
        rng: &mut SeededRng,
        shape: &[usize],
        dtype: DType,
        _strides: &[usize],
    ) -> TensorData {
        let numel: usize = shape.iter().product();
        let data = self.generate_random_data(rng, numel, dtype);

        TensorData {
            shape: shape.to_vec(),
            strides: TensorData::compute_contiguous_strides(shape),
            dtype,
            data,
        }
    }

    /// Generate random data for a tensor.
    fn generate_random_data(&self, rng: &mut SeededRng, numel: usize, dtype: DType) -> Vec<u8> {
        match dtype {
            DType::Float32 => {
                let mut data = vec![0u8; numel * 4];
                for i in 0..numel {
                    // Generate random float in [-1, 1] using Box-Muller or simple uniform
                    let val: f32 = (rng.gen_f32() * 2.0 - 1.0) * 10.0; // Scale to [-10, 10]
                    let bytes = val.to_le_bytes();
                    data[i * 4..i * 4 + 4].copy_from_slice(&bytes);
                }
                data
            }
            DType::Float64 => {
                let mut data = vec![0u8; numel * 8];
                for i in 0..numel {
                    let val: f64 = (rng.gen_f64() * 2.0 - 1.0) * 10.0;
                    let bytes = val.to_le_bytes();
                    data[i * 8..i * 8 + 8].copy_from_slice(&bytes);
                }
                data
            }
            DType::Float16 | DType::BFloat16 => {
                // For f16/bf16, generate f32 and truncate (simplified)
                let mut data = vec![0u8; numel * 2];
                for i in 0..numel {
                    let val: f32 = (rng.gen_f32() * 2.0 - 1.0) * 10.0;
                    // Simple f32 to f16 conversion (truncate mantissa)
                    let bits = val.to_bits();
                    let sign = (bits >> 16) & 0x8000;
                    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
                    let mant = (bits >> 13) & 0x3FF;
                    let f16_bits = if exp <= 0 {
                        sign as u16
                    } else if exp >= 31 {
                        (sign | 0x7C00) as u16
                    } else {
                        (sign | ((exp as u32) << 10) | mant) as u16
                    };
                    let bytes = f16_bits.to_le_bytes();
                    data[i * 2..i * 2 + 2].copy_from_slice(&bytes);
                }
                data
            }
            DType::Int32 => {
                let mut data = vec![0u8; numel * 4];
                for i in 0..numel {
                    let val: i32 = rng.gen_range(0..200) as i32 - 100;
                    let bytes = val.to_le_bytes();
                    data[i * 4..i * 4 + 4].copy_from_slice(&bytes);
                }
                data
            }
            DType::Int64 => {
                let mut data = vec![0u8; numel * 8];
                for i in 0..numel {
                    let val: i64 = rng.gen_range(0..200) as i64 - 100;
                    let bytes = val.to_le_bytes();
                    data[i * 8..i * 8 + 8].copy_from_slice(&bytes);
                }
                data
            }
            _ => {
                // For other types, generate zeros
                vec![0u8; numel * dtype.size_bytes()]
            }
        }
    }

    /// Get the FuzzConfig.
    pub fn config(&self) -> &FuzzConfig {
        &self.config
    }
}

/// Regenerate a test case from reproduction info.
pub fn regenerate_test_case(repro: &ReproductionInfo, input_names: &[&str]) -> TestCase {
    let mut rng = SeededRng::new(repro.seed);

    // Use the exact shape/dtype/layout from repro info
    let shape = repro.shape.clone();
    let dtype = repro.dtype;
    let layout = repro.layout;
    let strides = repro.strides.clone();

    // Regenerate input data using the same seed derivation
    let mut inputs = HashMap::new();
    let mut data_rng = rng.derive("data");
    for (i, name) in input_names.iter().enumerate() {
        let numel: usize = shape.iter().product();
        let data = generate_data_from_seed(&mut data_rng.derive(&format!("input_{}", i)), numel, dtype);
        let input = TensorData {
            shape: shape.clone(),
            strides: strides.clone(),
            dtype,
            data,
        };
        inputs.insert(name.to_string(), input);
    }

    TestCase {
        seed: repro.seed,
        shape,
        strides,
        dtype,
        layout,
        inputs,
    }
}

/// Generate tensor data from a seed (helper for regeneration).
fn generate_data_from_seed(rng: &mut SeededRng, numel: usize, dtype: DType) -> Vec<u8> {
    match dtype {
        DType::Float32 => {
            let mut data = vec![0u8; numel * 4];
            for i in 0..numel {
                let val: f32 = (rng.gen_f32() * 2.0 - 1.0) * 10.0;
                let bytes = val.to_le_bytes();
                data[i * 4..i * 4 + 4].copy_from_slice(&bytes);
            }
            data
        }
        DType::Float64 => {
            let mut data = vec![0u8; numel * 8];
            for i in 0..numel {
                let val: f64 = (rng.gen_f64() * 2.0 - 1.0) * 10.0;
                let bytes = val.to_le_bytes();
                data[i * 8..i * 8 + 8].copy_from_slice(&bytes);
            }
            data
        }
        _ => vec![0u8; numel * dtype.size_bytes()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuzzer_deterministic() {
        let config = FuzzConfig::with_seed(12345);

        let mut fuzzer1 = Fuzzer::new(config.clone());
        let mut fuzzer2 = Fuzzer::new(config);

        let tc1 = fuzzer1.next_test_case(&["input"]);
        let tc2 = fuzzer2.next_test_case(&["input"]);

        assert_eq!(tc1.seed, tc2.seed);
        assert_eq!(tc1.shape, tc2.shape);
        assert_eq!(tc1.dtype, tc2.dtype);
        assert_eq!(tc1.layout, tc2.layout);
    }

    #[test]
    fn test_fuzzer_different_iterations() {
        let config = FuzzConfig::with_seed(12345);
        let mut fuzzer = Fuzzer::new(config);

        let tc1 = fuzzer.next_test_case(&["input"]);
        let tc2 = fuzzer.next_test_case(&["input"]);

        // Different iterations should produce different seeds
        assert_ne!(tc1.seed, tc2.seed);
    }

    #[test]
    fn test_regenerate_test_case() {
        let config = FuzzConfig::with_seed(12345);
        let mut fuzzer = Fuzzer::new(config.clone());

        let original = fuzzer.next_test_case(&["input"]);
        let repro = original.to_repro_info(Some(config));

        let regenerated = regenerate_test_case(&repro, &["input"]);

        assert_eq!(original.seed, regenerated.seed);
        assert_eq!(original.shape, regenerated.shape);
        assert_eq!(original.dtype, regenerated.dtype);
        assert_eq!(original.inputs["input"].data, regenerated.inputs["input"].data);
    }
}
