//! Fuzzer for generating test cases from FuzzConfig.

use gpuemu_common::rng::SeededRng;
use gpuemu_common::types::{
    DType, FuzzConfig, LayoutType, OpSchema, ReproductionInfo, TensorData, TensorSchema,
    ValueDistribution,
};
use std::collections::HashMap;
use tracing::warn;

/// Serialize input tensors into a compact binary snapshot.
///
/// Format: for each input, write [name_len:u16, name:bytes, shape_len:u16,
/// shape items as u32 each, strides_len:u16, strides as u32 each,
/// dtype:u8, data_len:u32, data:bytes].
pub fn serialize_input_snapshot(inputs: &HashMap<String, TensorData>) -> Option<Vec<u8>> {
    let mut buf = Vec::new();
    let count = inputs.len() as u16;
    buf.extend_from_slice(&count.to_le_bytes());
    for (name, tensor) in inputs {
        let name_bytes = name.as_bytes();
        if name_bytes.len() > u16::MAX as usize {
            continue;
        }
        buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(name_bytes);

        if tensor.shape.len() > u16::MAX as usize {
            continue;
        }
        buf.extend_from_slice(&(tensor.shape.len() as u16).to_le_bytes());
        for &dim in &tensor.shape {
            buf.extend_from_slice(&(dim as u32).to_le_bytes());
        }

        buf.extend_from_slice(&(tensor.strides.len() as u16).to_le_bytes());
        for &s in &tensor.strides {
            buf.extend_from_slice(&(s as u32).to_le_bytes());
        }

        buf.push(tensor.dtype as u8);

        if tensor.data.len() > u32::MAX as usize {
            continue;
        }
        buf.extend_from_slice(&(tensor.data.len() as u32).to_le_bytes());
        buf.extend_from_slice(&tensor.data);
    }
    Some(buf)
}

/// Deserialize input tensors from a compact binary snapshot.
pub fn deserialize_input_snapshot(snapshot: &[u8]) -> Option<HashMap<String, TensorData>> {
    let mut inputs = HashMap::new();
    let mut offset;

    if snapshot.len() < 2 {
        return None;
    }
    let count = u16::from_le_bytes([snapshot[0], snapshot[1]]) as usize;
    offset = 2;

    let dtypes: &[DType] = &[
        DType::Float16,
        DType::BFloat16,
        DType::Float32,
        DType::Float64,
        DType::Int8,
        DType::Int16,
        DType::Int32,
        DType::Int64,
        DType::UInt8,
        DType::UInt16,
        DType::UInt32,
        DType::UInt64,
        DType::Bool,
    ];

    for _ in 0..count {
        if offset + 2 > snapshot.len() {
            return None;
        }
        let name_len = u16::from_le_bytes([snapshot[offset], snapshot[offset + 1]]) as usize;
        offset += 2;
        if offset + name_len > snapshot.len() {
            return None;
        }
        let name = String::from_utf8_lossy(&snapshot[offset..offset + name_len]).to_string();
        offset += name_len;

        if offset + 2 > snapshot.len() {
            return None;
        }
        let shape_len = u16::from_le_bytes([snapshot[offset], snapshot[offset + 1]]) as usize;
        offset += 2;
        let mut shape = Vec::with_capacity(shape_len);
        for _ in 0..shape_len {
            if offset + 4 > snapshot.len() {
                return None;
            }
            let dim = u32::from_le_bytes([
                snapshot[offset],
                snapshot[offset + 1],
                snapshot[offset + 2],
                snapshot[offset + 3],
            ]) as usize;
            shape.push(dim);
            offset += 4;
        }

        if offset + 2 > snapshot.len() {
            return None;
        }
        let strides_len = u16::from_le_bytes([snapshot[offset], snapshot[offset + 1]]) as usize;
        offset += 2;
        let mut strides = Vec::with_capacity(strides_len);
        for _ in 0..strides_len {
            if offset + 4 > snapshot.len() {
                return None;
            }
            let s = u32::from_le_bytes([
                snapshot[offset],
                snapshot[offset + 1],
                snapshot[offset + 2],
                snapshot[offset + 3],
            ]) as usize;
            strides.push(s);
            offset += 4;
        }

        if offset + 1 > snapshot.len() {
            return None;
        }
        let dtype_idx = snapshot[offset] as usize;
        let dtype = dtypes.get(dtype_idx).copied().unwrap_or(DType::Float32);
        offset += 1;

        if offset + 4 > snapshot.len() {
            return None;
        }
        let data_len = u32::from_le_bytes([
            snapshot[offset],
            snapshot[offset + 1],
            snapshot[offset + 2],
            snapshot[offset + 3],
        ]) as usize;
        offset += 4;
        if offset + data_len > snapshot.len() {
            return None;
        }
        let data = snapshot[offset..offset + data_len].to_vec();
        offset += data_len;

        inputs.insert(
            name,
            TensorData {
                shape,
                strides,
                dtype,
                data,
            },
        );
    }
    Some(inputs)
}

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
        let input_snapshot = serialize_input_snapshot(&self.inputs);
        ReproductionInfo {
            seed: self.seed,
            fuzz_config,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
            layout: self.layout,
            input_snapshot,
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
    ///
    /// When the config carries an [`OpSchema`], inputs get *per-input* shapes
    /// derived from shared symbolic dims (matmul `A[M,K]·B[K,N]`, attention,
    /// ...). Otherwise the legacy path generates one rank-3 shape applied to
    /// every input (preserved bit-for-bit for reproducibility).
    pub fn next_test_case(&mut self, input_names: &[&str]) -> TestCase {
        // Derive a seed for this iteration
        let iter_seed = self.rng.derive(&format!("iter_{}", self.iteration)).seed();
        self.iteration += 1;

        let iter_rng = SeededRng::new(iter_seed);

        // Generate dtype and layout (single per test case, applied to all inputs).
        let dtype = iter_rng.derive("dtype").choice(&self.config.dtypes);
        let layout = iter_rng.derive("layout").choice(&self.config.layouts);

        // Resolve per-input shapes and a representative shape (for repro info).
        let (input_shapes, repr_shape) = match &self.config.op_schema {
            Some(schema) => {
                let dim_values = self.sample_schema_dims(&iter_rng.derive("shape"), schema);
                let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
                for ts in &schema.inputs {
                    shapes.insert(ts.name.clone(), Self::shape_from_schema(&dim_values, ts));
                }
                let repr = schema
                    .output
                    .as_ref()
                    .map(|o| Self::shape_from_schema(&dim_values, o))
                    .or_else(|| {
                        schema
                            .inputs
                            .first()
                            .map(|i| Self::shape_from_schema(&dim_values, i))
                    })
                    .unwrap_or_else(|| vec![1]);
                (shapes, repr)
            }
            None => {
                let shape = self.generate_shape(&mut iter_rng.derive("shape"));
                let shapes = input_names
                    .iter()
                    .map(|n| (n.to_string(), shape.clone()))
                    .collect();
                (shapes, shape)
            }
        };

        // Representative strides (from the representative shape) for repro info.
        let strides = self.compute_strides(&repr_shape, layout);

        // Generate input tensors, each with its own shape/strides.
        let mut inputs = HashMap::new();
        let data_rng = iter_rng.derive("data");
        for (i, name) in input_names.iter().enumerate() {
            let shape = input_shapes
                .get(*name)
                .cloned()
                .unwrap_or_else(|| repr_shape.clone());
            let input_strides = self.compute_strides(&shape, layout);
            let input = self.generate_tensor(
                &mut data_rng.derive(&format!("input_{}", i)),
                &shape,
                dtype,
                &input_strides,
            );
            inputs.insert(name.to_string(), input);
        }

        TestCase {
            seed: iter_seed,
            shape: repr_shape,
            strides,
            dtype,
            layout,
            inputs,
        }
    }

    /// Sample one concrete size for each symbolic dim in the schema.
    ///
    /// Each dim derives an independent sub-RNG from `rng`, so sampling is
    /// deterministic and independent of dim ordering.
    fn sample_schema_dims(
        &self,
        rng: &SeededRng,
        schema: &OpSchema,
    ) -> HashMap<String, usize> {
        let mut dims = HashMap::new();
        for d in &schema.dims {
            let value = if d.candidates.is_empty() {
                1
            } else {
                rng.derive(&d.name).choice(&d.candidates)
            };
            dims.insert(d.name.clone(), value);
        }
        dims
    }

    /// Build a concrete shape for a tensor schema from sampled dim values.
    /// Unknown dim names fall back to size 1.
    fn shape_from_schema(dim_values: &HashMap<String, usize>, t: &TensorSchema) -> Vec<usize> {
        t.dims
            .iter()
            .map(|n| *dim_values.get(n).unwrap_or(&1))
            .collect()
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
        strides: &[usize],
    ) -> TensorData {
        let numel: usize = shape.iter().product();
        let data = self.generate_random_data(rng, numel, dtype);

        TensorData {
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            dtype,
            data,
        }
    }

    /// Generate random data for a tensor.
    fn generate_random_data(&self, rng: &mut SeededRng, numel: usize, dtype: DType) -> Vec<u8> {
        generate_data_from_seed(rng, numel, dtype, self.config.value_distribution)
    }

    /// Get the FuzzConfig.
    #[allow(dead_code)]
    pub fn config(&self) -> &FuzzConfig {
        &self.config
    }
}

/// Regenerate a test case from reproduction info.
///
/// If `input_snapshot` is present in the repro info, returns exact inputs.
/// Otherwise, regenerates from seed (which may diverge for some dtypes).
pub fn regenerate_test_case(repro: &ReproductionInfo, input_names: &[&str]) -> TestCase {
    let inputs = if let Some(ref snapshot) = repro.input_snapshot {
        if let Some(deserialized) = deserialize_input_snapshot(snapshot) {
            deserialized
        } else {
            warn!("Failed to deserialize input snapshot, falling back to seed regeneration");
            regenerate_inputs_from_seed(repro, input_names)
        }
    } else {
        regenerate_inputs_from_seed(repro, input_names)
    };

    TestCase {
        seed: repro.seed,
        shape: repro.shape.clone(),
        strides: repro.strides.clone(),
        dtype: repro.dtype,
        layout: repro.layout,
        inputs,
    }
}

fn regenerate_inputs_from_seed(
    repro: &ReproductionInfo,
    input_names: &[&str],
) -> HashMap<String, TensorData> {
    let rng = SeededRng::new(repro.seed);
    let shape = repro.shape.clone();
    let dtype = repro.dtype;
    let strides = repro.strides.clone();
    // Replay with the same value distribution the failing run used (Regular if
    // the repro predates the field or carries no fuzz_config).
    let dist = repro
        .fuzz_config
        .as_ref()
        .map(|c| c.value_distribution)
        .unwrap_or(ValueDistribution::Regular);

    let mut inputs = HashMap::new();
    let data_rng = rng.derive("data");
    for (i, name) in input_names.iter().enumerate() {
        let numel: usize = shape.iter().product();
        let data = generate_data_from_seed(
            &mut data_rng.derive(&format!("input_{}", i)),
            numel,
            dtype,
            dist,
        );
        let input = TensorData {
            shape: shape.clone(),
            strides: strides.clone(),
            dtype,
            data,
        };
        inputs.insert(name.to_string(), input);
    }
    inputs
}

/// Sample one float value under the chosen [`ValueDistribution`].
///
/// `Regular` consumes exactly one `gen_f64()` and reproduces the historical
/// uniform[-10, 10] draw byte-for-byte. `Boundary`/`Adversarial` first draw a
/// category selector, then either emit a special value or fall back to a
/// regular draw — so the special values are interleaved with normal ones.
fn sample_float(rng: &mut SeededRng, dist: ValueDistribution) -> f64 {
    let regular = |rng: &mut SeededRng| (rng.gen_f64() * 2.0 - 1.0) * 10.0;
    match dist {
        ValueDistribution::Regular => regular(rng),
        ValueDistribution::Boundary => match rng.gen_range(0..8) {
            0 => 0.0,
            1 => 1.0,
            2 => -1.0,
            3 => f64::MIN_POSITIVE, // smallest normal — exercises near-denormal paths
            4 => -f64::MIN_POSITIVE,
            5 => 1.0e30,
            6 => -1.0e30,
            _ => regular(rng),
        },
        ValueDistribution::Adversarial => match rng.gen_range(0..10) {
            0 => f64::NAN,
            1 => f64::INFINITY,
            2 => f64::NEG_INFINITY,
            3 => 0.0,
            4 => f64::MIN_POSITIVE,
            5 => 1.0e300,
            6 => -1.0e300,
            7 => 1.0e-300, // subnormal-adjacent — sign-cancellation / underflow
            _ => regular(rng),
        },
    }
}

/// Convert an `f32` to IEEE-754 half-precision bits.
///
/// Finite values follow the original inline conversion exactly (so `Regular`
/// generation stays byte-identical); `NaN`/`Inf` are now preserved rather than
/// collapsing to infinity, which matters for adversarial inputs.
fn f32_to_half_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    if val.is_nan() {
        return (sign | 0x7E00) as u16;
    }
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3FF;
    if exp <= 0 {
        sign as u16
    } else if exp >= 31 {
        (sign | 0x7C00) as u16
    } else {
        (sign | ((exp as u32) << 10) | mant) as u16
    }
}

/// Generate tensor data from a seed (helper for regeneration).
///
/// `dist` selects the float value distribution; integer/bool generation is
/// distribution-independent (shape fuzzing already covers their boundaries).
fn generate_data_from_seed(
    rng: &mut SeededRng,
    numel: usize,
    dtype: DType,
    dist: ValueDistribution,
) -> Vec<u8> {
    match dtype {
        DType::Float32 => {
            let mut data = vec![0u8; numel * 4];
            for i in 0..numel {
                let val = sample_float(rng, dist) as f32;
                let bytes = val.to_le_bytes();
                data[i * 4..i * 4 + 4].copy_from_slice(&bytes);
            }
            data
        }
        DType::Float64 => {
            let mut data = vec![0u8; numel * 8];
            for i in 0..numel {
                let val: f64 = sample_float(rng, dist);
                let bytes = val.to_le_bytes();
                data[i * 8..i * 8 + 8].copy_from_slice(&bytes);
            }
            data
        }
        DType::Float16 | DType::BFloat16 => {
            let mut data = vec![0u8; numel * 2];
            for i in 0..numel {
                let val = sample_float(rng, dist) as f32;
                let f16_bits = f32_to_half_bits(val);
                let bytes = f16_bits.to_le_bytes();
                data[i * 2..i * 2 + 2].copy_from_slice(&bytes);
            }
            data
        }
        DType::Int8 => {
            let mut data = vec![0u8; numel];
            for i in 0..numel {
                let val: i8 = (rng.gen_range(0..200) as i32 - 100) as i8;
                data[i] = val as u8;
            }
            data
        }
        DType::UInt8 => {
            let mut data = vec![0u8; numel];
            for i in 0..numel {
                data[i] = (rng.gen_u64() & 0xFF) as u8;
            }
            data
        }
        DType::Int16 => {
            let mut data = vec![0u8; numel * 2];
            for i in 0..numel {
                let val: i16 = (rng.gen_range(0..200) as i32 - 100) as i16;
                let bytes = val.to_le_bytes();
                data[i * 2..i * 2 + 2].copy_from_slice(&bytes);
            }
            data
        }
        DType::UInt16 => {
            let mut data = vec![0u8; numel * 2];
            for i in 0..numel {
                let val: u16 = (rng.gen_u64() & 0xFFFF) as u16;
                let bytes = val.to_le_bytes();
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
        DType::UInt32 => {
            let mut data = vec![0u8; numel * 4];
            for i in 0..numel {
                let val: u32 = (rng.gen_u64() & 0xFFFFFFFF) as u32;
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
        DType::UInt64 => {
            let mut data = vec![0u8; numel * 8];
            for i in 0..numel {
                let bytes = rng.gen_u64().to_le_bytes();
                data[i * 8..i * 8 + 8].copy_from_slice(&bytes);
            }
            data
        }
        DType::Bool => {
            let mut data = vec![0u8; numel];
            for i in 0..numel {
                data[i] = if rng.gen_bool(0.5) { 1 } else { 0 };
            }
            data
        }
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
        assert_eq!(
            original.inputs["input"].data,
            regenerated.inputs["input"].data
        );
    }

    fn decode_f32(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn test_regular_distribution_byte_identical() {
        // Regular mode must reproduce the historical uniform[-10,10] draw
        // exactly (one gen_f64 per element), so old seeds replay byte-for-byte.
        let mut rng = SeededRng::new(98765);
        let got = generate_data_from_seed(&mut rng, 64, DType::Float32, ValueDistribution::Regular);

        let mut ref_rng = SeededRng::new(98765);
        let mut expected = vec![0u8; 64 * 4];
        for i in 0..64 {
            let val = ((ref_rng.gen_f64() * 2.0 - 1.0) * 10.0) as f32;
            expected[i * 4..i * 4 + 4].copy_from_slice(&val.to_le_bytes());
        }
        assert_eq!(got, expected, "Regular mode diverged from historical formula");
    }

    #[test]
    fn test_adversarial_injects_special_values() {
        let mut rng = SeededRng::new(13);
        let data =
            generate_data_from_seed(&mut rng, 512, DType::Float32, ValueDistribution::Adversarial);
        let vals = decode_f32(&data);
        assert!(vals.iter().any(|v| v.is_nan()), "expected at least one NaN");
        assert!(
            vals.iter().any(|v| v.is_infinite()),
            "expected at least one Inf"
        );
        assert!(
            vals.iter().any(|v| v.is_finite()),
            "adversarial should still mix in finite values"
        );
    }

    #[test]
    fn test_boundary_injects_edge_values() {
        let mut rng = SeededRng::new(7);
        let data =
            generate_data_from_seed(&mut rng, 512, DType::Float32, ValueDistribution::Boundary);
        let vals = decode_f32(&data);
        assert!(vals.iter().all(|v| v.is_finite()), "boundary stays finite");
        assert!(vals.iter().any(|v| *v == 0.0), "expected exact zeros");
        assert!(vals.iter().any(|v| *v == 1.0), "expected exact ones");
    }

    #[test]
    fn test_adversarial_half_preserves_nan() {
        // The improved f16 conversion must keep NaN as NaN (0x7E00 pattern),
        // not collapse it to infinity.
        assert_eq!(f32_to_half_bits(f32::NAN) & 0x7C00, 0x7C00);
        assert_ne!(f32_to_half_bits(f32::NAN) & 0x03FF, 0, "NaN must keep mantissa");
        assert_eq!(f32_to_half_bits(f32::INFINITY), 0x7C00);
        assert_eq!(f32_to_half_bits(f32::NEG_INFINITY), 0xFC00);
    }

    #[test]
    fn test_schema_matmul_shares_k() {
        let mut config = FuzzConfig::with_seed(2024);
        config.op_schema = OpSchema::builtin("matmul");
        let mut fuzzer = Fuzzer::new(config);

        for _ in 0..50 {
            let tc = fuzzer.next_test_case(&["a", "b"]);
            let a = &tc.inputs["a"].shape;
            let b = &tc.inputs["b"].shape;
            assert_eq!(a.len(), 2, "A is M x K");
            assert_eq!(b.len(), 2, "B is K x N");
            // Inner dim K is shared between A and B.
            assert_eq!(a[1], b[0], "shared K dim must match: A={a:?} B={b:?}");
            // Representative (output) shape is M x N.
            assert_eq!(tc.shape, vec![a[0], b[1]]);
            // Data length matches each input's own element count.
            assert_eq!(tc.inputs["a"].data.len(), a.iter().product::<usize>() * tc.dtype.size_bytes());
            assert_eq!(tc.inputs["b"].data.len(), b.iter().product::<usize>() * tc.dtype.size_bytes());
        }
    }

    #[test]
    fn test_schema_deterministic() {
        let mut config = FuzzConfig::with_seed(7);
        config.op_schema = OpSchema::builtin("matmul");

        let mut f1 = Fuzzer::new(config.clone());
        let mut f2 = Fuzzer::new(config);

        for _ in 0..20 {
            let t1 = f1.next_test_case(&["a", "b"]);
            let t2 = f2.next_test_case(&["a", "b"]);
            assert_eq!(t1.inputs["a"].shape, t2.inputs["a"].shape);
            assert_eq!(t1.inputs["b"].shape, t2.inputs["b"].shape);
            assert_eq!(t1.inputs["a"].data, t2.inputs["a"].data);
        }
    }

    #[test]
    fn test_schema_attention_qkv_equal_rank4() {
        let mut config = FuzzConfig::with_seed(11);
        config.op_schema = OpSchema::builtin("attention");
        let mut fuzzer = Fuzzer::new(config);

        let tc = fuzzer.next_test_case(&["q", "k", "v"]);
        for name in ["q", "k", "v"] {
            assert_eq!(tc.inputs[name].shape.len(), 4, "{name} is B,H,S,D");
        }
        // Q, K, V share the same [B,H,S,D].
        assert_eq!(tc.inputs["q"].shape, tc.inputs["k"].shape);
        assert_eq!(tc.inputs["k"].shape, tc.inputs["v"].shape);
    }

    #[test]
    fn test_no_schema_preserves_legacy_single_shape() {
        // Without a schema, every input gets the same (rank-3) shape.
        let config = FuzzConfig::with_seed(123);
        let mut fuzzer = Fuzzer::new(config);
        let tc = fuzzer.next_test_case(&["x", "y"]);
        assert_eq!(tc.inputs["x"].shape, tc.inputs["y"].shape);
        assert_eq!(tc.inputs["x"].shape, tc.shape);
    }
}
