//! Seeded RNG for reproducible fuzzing across Rust and Python.
//!
//! This module provides a deterministic RNG that can derive sub-RNGs
//! for different domains (shapes, dtypes, layouts, data) from a master seed.
//! The derivation uses Blake2b to ensure cross-language compatibility.
//!
//! The underlying PRNG uses xorshift128+ which is simple to implement
//! in any language, ensuring bit-for-bit reproducibility between Rust and Python.

use blake2::{Blake2b, Digest};

/// xorshift128+ state: two 64-bit words.
#[derive(Clone)]
struct Xorshift128State {
    s0: u64,
    s1: u64,
}

impl Xorshift128State {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.s0;
        let y = self.s1;
        self.s0 = y;
        x ^= x << 23;
        x ^= x >> 17;
        x ^= y ^ (y >> 26);
        self.s1 = x ^ y;
        x.wrapping_add(y)
    }
}

/// Deterministic RNG wrapper for reproducible fuzzing.
///
/// Uses xorshift128+ as the underlying RNG for cross-language compatibility.
/// Seed derivation uses Blake2b-64 for deterministic sub-seed generation.
/// The initialization follows SplitMix64 to expand a single u64 seed into
/// the 128-bit state, matching the Python implementation exactly.
#[derive(Clone)]
pub struct SeededRng {
    seed: u64,
    state: Xorshift128State,
}

impl SeededRng {
    /// Create a new RNG from a seed using SplitMix64 initialization.
    ///
    /// The initialization matches the Python implementation exactly:
    /// state[0] = splitmix64(seed)
    /// state[1] = splitmix64(seed + 1)
    pub fn new(seed: u64) -> Self {
        let s0 = splitmix64(seed);
        let s1 = splitmix64(seed.wrapping_add(1));
        // Ensure non-zero state
        let state = Xorshift128State {
            s0: if s0 == 0 && s1 == 0 { 1 } else { s0 },
            s1,
        };
        Self { seed, state }
    }

    /// Get the seed used to create this RNG.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Derive a sub-RNG for a specific domain.
    ///
    /// This uses Blake2b to hash the seed with the domain string,
    /// producing a deterministic sub-seed. The algorithm matches
    /// the Python implementation for cross-language reproducibility.
    pub fn derive(&self, domain: &str) -> Self {
        let sub_seed = derive_seed(self.seed, domain);
        Self::new(sub_seed)
    }

    /// Choose a random element from a slice.
    pub fn choice<T: Clone>(&mut self, options: &[T]) -> T {
        assert!(!options.is_empty(), "Cannot choose from empty slice");
        let idx = self.gen_range(0..options.len());
        options[idx].clone()
    }

    /// Choose a random element and return its index.
    pub fn choice_index(&mut self, len: usize) -> usize {
        assert!(len > 0, "Cannot choose from empty range");
        self.gen_range(0..len)
    }

    /// Generate a random integer in the given range.
    pub fn gen_range(&mut self, range: std::ops::Range<usize>) -> usize {
        assert!(range.start < range.end, "Invalid range");
        let width = range.end - range.start;
        range.start + (self.gen_u64() as usize % width)
    }

    /// Generate a random u64.
    pub fn gen_u64(&mut self) -> u64 {
        self.state.next_u64()
    }

    /// Generate a random f32 in [0, 1).
    pub fn gen_f32(&mut self) -> f32 {
        let x = self.gen_u64();
        // Use upper 24 bits for mantissa, result in [0, 1)
        (x >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Generate a random f64 in [0, 1).
    pub fn gen_f64(&mut self) -> f64 {
        let x = self.gen_u64();
        // Use upper 53 bits for mantissa, result in [0, 1)
        (x >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Fill a slice with random f32 values from standard normal distribution.
    pub fn fill_normal_f32(&mut self, dest: &mut [f32]) {
        for i in (0..dest.len()).step_by(2) {
            let (n1, n2) = box_muller_f32(self.gen_f64(), self.gen_f64());
            dest[i] = n1;
            if i + 1 < dest.len() {
                dest[i + 1] = n2;
            }
        }
    }

    /// Fill a slice with random f64 values from standard normal distribution.
    pub fn fill_normal_f64(&mut self, dest: &mut [f64]) {
        for i in (0..dest.len()).step_by(2) {
            let (n1, n2) = box_muller_f64(self.gen_f64(), self.gen_f64());
            dest[i] = n1;
            if i + 1 < dest.len() {
                dest[i + 1] = n2;
            }
        }
    }

    /// Fill a slice with random bytes.
    pub fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(8) {
            let val = self.gen_u64();
            let bytes = val.to_le_bytes();
            for (i, b) in chunk.iter_mut().enumerate() {
                if i < 8 {
                    *b = bytes[i];
                }
            }
        }
    }

    /// Shuffle a slice in place.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.gen_range(0..(i + 1));
            slice.swap(i, j);
        }
    }

    /// Generate a random boolean with given probability of true.
    pub fn gen_bool(&mut self, probability: f64) -> bool {
        self.gen_f64() < probability
    }
}

/// Box-Muller transform for generating normal-distributed f64 values.
fn box_muller_f64(u1: f64, u2: f64) -> (f64, f64) {
    let u1 = u1.max(1e-300); // Avoid log(0)
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

/// Box-Muller transform for generating normal-distributed f32 values.
fn box_muller_f32(u1: f64, u2: f64) -> (f32, f32) {
    let (a, b) = box_muller_f64(u1, u2);
    (a as f32, b as f32)
}

/// SplitMix64: deterministic state expansion from a single u64.
///
/// This matches the Python implementation exactly:
/// z = (seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
/// z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
/// z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
/// z = (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF
fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Derive a sub-seed from a master seed and domain string.
///
/// Uses Blake2b-64 to hash the seed (little-endian bytes) concatenated
/// with the domain string (UTF-8 bytes). This algorithm is implemented
/// identically in Python for cross-language reproducibility.
pub fn derive_seed(seed: u64, domain: &str) -> u64 {
    type Blake2b64 = Blake2b<blake2::digest::consts::U8>;
    let mut hasher = Blake2b64::new();
    hasher.update(&seed.to_le_bytes());
    hasher.update(domain.as_bytes());
    let result = hasher.finalize();
    u64::from_le_bytes(result.as_slice().try_into().unwrap())
}

/// Generate a timestamp-based seed (for when no seed is specified).
pub fn generate_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_derivation_deterministic() {
        let seed1 = derive_seed(12345, "shapes");
        let seed2 = derive_seed(12345, "shapes");
        assert_eq!(seed1, seed2);
    }

    #[test]
    fn test_seed_derivation_different_domains() {
        let seed1 = derive_seed(12345, "shapes");
        let seed2 = derive_seed(12345, "dtypes");
        assert_ne!(seed1, seed2);
    }

    #[test]
    fn test_seed_derivation_different_seeds() {
        let seed1 = derive_seed(12345, "shapes");
        let seed2 = derive_seed(54321, "shapes");
        assert_ne!(seed1, seed2);
    }

    #[test]
    fn test_rng_deterministic() {
        let mut rng1 = SeededRng::new(12345);
        let mut rng2 = SeededRng::new(12345);

        for _ in 0..100 {
            assert_eq!(rng1.gen_u64(), rng2.gen_u64());
        }
    }

    #[test]
    fn test_derive_deterministic() {
        let master1 = SeededRng::new(12345);
        let master2 = SeededRng::new(12345);

        let mut shapes1 = master1.derive("shapes");
        let mut shapes2 = master2.derive("shapes");

        for _ in 0..100 {
            assert_eq!(shapes1.gen_u64(), shapes2.gen_u64());
        }
    }

    #[test]
    fn test_choice() {
        let mut rng = SeededRng::new(12345);
        let options = vec![1, 2, 3, 4, 5];

        for _ in 0..100 {
            let choice = rng.choice(&options);
            assert!(options.contains(&choice));
        }
    }

    #[test]
    fn test_shuffle_deterministic() {
        let mut rng1 = SeededRng::new(12345);
        let mut rng2 = SeededRng::new(12345);

        let mut vec1 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut vec2 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        rng1.shuffle(&mut vec1);
        rng2.shuffle(&mut vec2);

        assert_eq!(vec1, vec2);
    }

    #[test]
    fn test_known_seed_output() {
        // These values must match the Python implementation exactly
        let seed = derive_seed(42, "test");
        println!("derive_seed(42, 'test') = {}", seed);

        let mut rng = SeededRng::new(42);
        let first_u64 = rng.gen_u64();
        println!("SeededRng(42).gen_u64() = {}", first_u64);

        // Verify SplitMix64 initialization
        let s0 = splitmix64(42);
        let s1 = splitmix64(43);
        println!("splitmix64(42) = {}", s0);
        println!("splitmix64(43) = {}", s1);
    }

    #[test]
    fn test_gen_range() {
        let mut rng = SeededRng::new(12345);
        for _ in 0..100 {
            let val = rng.gen_range(0..10);
            assert!(val < 10);
        }
    }

    #[test]
    fn test_gen_f64_range() {
        let mut rng = SeededRng::new(12345);
        for _ in 0..100 {
            let val = rng.gen_f64();
            assert!(val >= 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_gen_f32_range() {
        let mut rng = SeededRng::new(12345);
        for _ in 0..100 {
            let val = rng.gen_f32();
            assert!(val >= 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_python_rng_compatibility() {
        use std::process::Command;

        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root");
        let python_package_dir = workspace_root.join("gpuemu-py");

        let mut rust_rng = SeededRng::new(42);
        let expected = serde_json::json!({
            "derive_seed": derive_seed(42, "test"),
            "gen_u64": rust_rng.gen_u64(),
            "gen_f64": rust_rng.gen_f64(),
            "derived_gen_u64": SeededRng::new(42).derive("shape").gen_u64(),
        });

        let python = Command::new("python3")
            .env("PYTHONPATH", &python_package_dir)
            .arg("-c")
            .arg(
                "import json; from gpuemu_py.rng import derive_seed, SeededRng; rng=SeededRng(42); out={'derive_seed': derive_seed(42, 'test'), 'gen_u64': rng.gen_u64(), 'gen_f64': rng.gen_f64(), 'derived_gen_u64': SeededRng(42).derive('shape').gen_u64()}; print(json.dumps(out))",
            )
            .output()
            .expect("python3 should be available");

        assert!(
            python.status.success(),
            "python failed: {}",
            String::from_utf8_lossy(&python.stderr)
        );

        let actual: serde_json::Value =
            serde_json::from_slice(&python.stdout).expect("valid python json");
        assert_eq!(actual["derive_seed"], expected["derive_seed"]);
        assert_eq!(actual["gen_u64"], expected["gen_u64"]);
        assert_eq!(actual["derived_gen_u64"], expected["derived_gen_u64"]);

        let py_f64 = actual["gen_f64"].as_f64().expect("python gen_f64");
        let rs_f64 = expected["gen_f64"].as_f64().expect("rust gen_f64");
        assert!(
            (py_f64 - rs_f64).abs() < f64::EPSILON,
            "python gen_f64 {} != rust {}",
            py_f64,
            rs_f64
        );
    }
}
