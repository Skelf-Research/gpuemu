//! Seeded RNG for reproducible fuzzing across Rust and Python.
//!
//! This module provides a deterministic RNG that can derive sub-RNGs
//! for different domains (shapes, dtypes, layouts, data) from a master seed.
//! The derivation uses Blake2b to ensure cross-language compatibility.

use blake2::{Blake2b, Digest};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::ops::Range;

/// Deterministic RNG wrapper for reproducible fuzzing.
///
/// Uses ChaCha8 as the underlying RNG for speed and quality.
/// Seed derivation uses Blake2b-64 for cross-language compatibility.
#[derive(Clone)]
pub struct SeededRng {
    seed: u64,
    rng: ChaCha8Rng,
}

impl SeededRng {
    /// Create a new RNG from a seed.
    pub fn new(seed: u64) -> Self {
        let rng = ChaCha8Rng::seed_from_u64(seed);
        Self { seed, rng }
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
    ///
    /// # Example
    /// ```
    /// use gpuemu_common::rng::SeededRng;
    ///
    /// let master = SeededRng::new(12345);
    /// let shapes_rng = master.derive("shapes");
    /// let data_rng = master.derive("data");
    /// // shapes_rng and data_rng are independent but deterministic
    /// ```
    pub fn derive(&self, domain: &str) -> Self {
        let sub_seed = derive_seed(self.seed, domain);
        Self::new(sub_seed)
    }

    /// Choose a random element from a slice.
    pub fn choice<T: Clone>(&mut self, options: &[T]) -> T {
        assert!(!options.is_empty(), "Cannot choose from empty slice");
        let idx = self.rng.gen_range(0..options.len());
        options[idx].clone()
    }

    /// Choose a random element and return its index.
    pub fn choice_index(&mut self, len: usize) -> usize {
        assert!(len > 0, "Cannot choose from empty range");
        self.rng.gen_range(0..len)
    }

    /// Generate a random integer in the given range.
    pub fn gen_range(&mut self, range: Range<usize>) -> usize {
        self.rng.gen_range(range)
    }

    /// Generate a random u64.
    pub fn gen_u64(&mut self) -> u64 {
        self.rng.gen()
    }

    /// Generate a random f32 in [0, 1).
    pub fn gen_f32(&mut self) -> f32 {
        self.rng.gen()
    }

    /// Generate a random f64 in [0, 1).
    pub fn gen_f64(&mut self) -> f64 {
        self.rng.gen()
    }

    /// Fill a slice with random f32 values from standard normal distribution.
    pub fn fill_normal_f32(&mut self, dest: &mut [f32]) {
        use rand_distr::{Distribution, StandardNormal};
        for v in dest.iter_mut() {
            *v = StandardNormal.sample(&mut self.rng);
        }
    }

    /// Fill a slice with random f64 values from standard normal distribution.
    pub fn fill_normal_f64(&mut self, dest: &mut [f64]) {
        use rand_distr::{Distribution, StandardNormal};
        for v in dest.iter_mut() {
            *v = StandardNormal.sample(&mut self.rng);
        }
    }

    /// Fill a slice with random bytes.
    pub fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.rng.fill(dest);
    }

    /// Shuffle a slice in place.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        use rand::seq::SliceRandom;
        slice.shuffle(&mut self.rng);
    }

    /// Generate a random boolean with given probability of true.
    pub fn gen_bool(&mut self, probability: f64) -> bool {
        self.rng.gen_bool(probability)
    }
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

        // Should not panic and return valid choices
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

    /// Test that known seed produces known output.
    /// This can be used to verify cross-language compatibility.
    #[test]
    fn test_known_seed_output() {
        let seed = derive_seed(42, "test");
        // Store this value and verify Python produces the same
        println!("derive_seed(42, 'test') = {}", seed);

        let mut rng = SeededRng::new(42);
        let first_u64 = rng.gen_u64();
        println!("SeededRng(42).gen_u64() = {}", first_u64);
        // These values can be compared with Python implementation
    }
}
