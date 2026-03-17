//! Storage layer using sled embedded database.

use anyhow::{Context, Result};
use gpuemu_common::types::{FuzzConfig, ValidationResult};
use rkyv::Deserialize;
use sled::Db;
use std::path::Path;

/// Storage wrapper for sled database.
pub struct Storage {
    db: Db,
}

impl Storage {
    /// Open or create a database at the given path.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let db = sled::open(path.as_ref())
            .with_context(|| format!("Failed to open database at {:?}", path.as_ref()))?;
        Ok(Self { db })
    }

    /// Store a validation result.
    pub fn store_result(&self, result: &ValidationResult) -> Result<()> {
        let tree = self.db.open_tree("results")?;
        let key = result.seed.to_be_bytes();
        let value = rkyv::to_bytes::<_, 1024>(result)
            .map_err(|e| anyhow::anyhow!("Serialization error: {:?}", e))?;
        tree.insert(key, value.as_slice())?;
        tree.flush()?;
        Ok(())
    }

    /// Get a validation result by seed.
    pub fn get_result(&self, seed: u64) -> Result<Option<ValidationResult>> {
        let tree = self.db.open_tree("results")?;
        let key = seed.to_be_bytes();
        match tree.get(key)? {
            Some(bytes) => {
                let archived = rkyv::check_archived_root::<ValidationResult>(&bytes)
                    .map_err(|e| anyhow::anyhow!("Validation error: {:?}", e))?;
                let result: ValidationResult = archived
                    .deserialize(&mut rkyv::Infallible)
                    .map_err(|e| anyhow::anyhow!("Deserialization error: {:?}", e))?;
                Ok(Some(result))
            }
            None => Ok(None),
        }
    }

    /// List recent validation results.
    pub fn list_results(&self, limit: usize) -> Result<Vec<ValidationResult>> {
        let tree = self.db.open_tree("results")?;
        let mut results = Vec::new();

        // Iterate in reverse order (most recent first, assuming seeds increase)
        for item in tree.iter().rev().take(limit) {
            let (_, bytes) = item?;
            let archived = rkyv::check_archived_root::<ValidationResult>(&bytes)
                .map_err(|e| anyhow::anyhow!("Validation error: {:?}", e))?;
            let result: ValidationResult = archived
                .deserialize(&mut rkyv::Infallible)
                .map_err(|e| anyhow::anyhow!("Deserialization error: {:?}", e))?;
            results.push(result);
        }

        Ok(results)
    }

    /// Store a baseline tag pointing to current results.
    pub fn store_baseline(&self, tag: &str) -> Result<()> {
        let baseline_tree = self.db.open_tree("baselines")?;
        let results_tree = self.db.open_tree("results")?;

        // Copy current results to baseline
        let baseline_results_tree = self.db.open_tree(format!("baseline:{}", tag))?;
        for item in results_tree.iter() {
            let (key, value) = item?;
            baseline_results_tree.insert(key, value)?;
        }

        // Store baseline metadata
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        baseline_tree.insert(tag, &timestamp.to_be_bytes())?;

        self.db.flush()?;
        Ok(())
    }

    /// Get baseline results for comparison.
    pub fn get_baseline(&self, tag: &str) -> Result<Vec<ValidationResult>> {
        let tree = self.db.open_tree(format!("baseline:{}", tag))?;
        let mut results = Vec::new();

        for item in tree.iter() {
            let (_, bytes) = item?;
            let archived = rkyv::check_archived_root::<ValidationResult>(&bytes)
                .map_err(|e| anyhow::anyhow!("Validation error: {:?}", e))?;
            let result: ValidationResult = archived
                .deserialize(&mut rkyv::Infallible)
                .map_err(|e| anyhow::anyhow!("Deserialization error: {:?}", e))?;
            results.push(result);
        }

        Ok(results)
    }

    /// Clear all results (for testing).
    pub fn clear_results(&self) -> Result<()> {
        let tree = self.db.open_tree("results")?;
        tree.clear()?;
        tree.flush()?;
        Ok(())
    }

    /// Flush all pending writes to disk.
    pub fn flush(&self) -> Result<()> {
        self.db.flush()?;
        Ok(())
    }

    // =========================================================================
    // Phase 2: Failure Storage
    // =========================================================================

    /// Store a failed validation result.
    ///
    /// Failures are stored separately from regular results to allow
    /// quick access for reproduction and debugging.
    pub fn store_failure(&self, result: &ValidationResult) -> Result<()> {
        let tree = self.db.open_tree("failures")?;
        let key = result.seed.to_be_bytes();
        // Use larger buffer for failures which may contain reproduction info
        let value = rkyv::to_bytes::<_, 8192>(result)
            .map_err(|e| anyhow::anyhow!("Serialization error: {:?}", e))?;
        tree.insert(key, value.as_slice())?;
        tree.flush()?;
        Ok(())
    }

    /// Get a stored failure by seed.
    pub fn get_failure(&self, seed: u64) -> Result<Option<ValidationResult>> {
        let tree = self.db.open_tree("failures")?;
        let key = seed.to_be_bytes();
        match tree.get(key)? {
            Some(bytes) => {
                let archived = rkyv::check_archived_root::<ValidationResult>(&bytes)
                    .map_err(|e| anyhow::anyhow!("Validation error: {:?}", e))?;
                let result: ValidationResult = archived
                    .deserialize(&mut rkyv::Infallible)
                    .map_err(|e| anyhow::anyhow!("Deserialization error: {:?}", e))?;
                Ok(Some(result))
            }
            None => Ok(None),
        }
    }

    /// List stored failures.
    pub fn list_failures(&self, limit: usize) -> Result<Vec<ValidationResult>> {
        let tree = self.db.open_tree("failures")?;
        let mut failures = Vec::new();

        // Iterate in reverse order (most recent first)
        for item in tree.iter().rev().take(limit) {
            let (_, bytes) = item?;
            let archived = rkyv::check_archived_root::<ValidationResult>(&bytes)
                .map_err(|e| anyhow::anyhow!("Validation error: {:?}", e))?;
            let result: ValidationResult = archived
                .deserialize(&mut rkyv::Infallible)
                .map_err(|e| anyhow::anyhow!("Deserialization error: {:?}", e))?;
            failures.push(result);
        }

        Ok(failures)
    }

    /// Count total stored failures.
    pub fn count_failures(&self) -> Result<usize> {
        let tree = self.db.open_tree("failures")?;
        Ok(tree.len())
    }

    /// Clear all failures (for testing).
    pub fn clear_failures(&self) -> Result<()> {
        let tree = self.db.open_tree("failures")?;
        tree.clear()?;
        tree.flush()?;
        Ok(())
    }

    /// Store a fuzz configuration for reproduction.
    ///
    /// The config is stored by its seed, allowing later retrieval
    /// when reproducing failures.
    pub fn store_fuzz_config(&self, seed: u64, config: &FuzzConfig) -> Result<()> {
        let tree = self.db.open_tree("fuzz_configs")?;
        let key = seed.to_be_bytes();
        let value = rkyv::to_bytes::<_, 2048>(config)
            .map_err(|e| anyhow::anyhow!("Serialization error: {:?}", e))?;
        tree.insert(key, value.as_slice())?;
        tree.flush()?;
        Ok(())
    }

    /// Get a stored fuzz configuration by seed.
    pub fn get_fuzz_config(&self, seed: u64) -> Result<Option<FuzzConfig>> {
        let tree = self.db.open_tree("fuzz_configs")?;
        let key = seed.to_be_bytes();
        match tree.get(key)? {
            Some(bytes) => {
                let archived = rkyv::check_archived_root::<FuzzConfig>(&bytes)
                    .map_err(|e| anyhow::anyhow!("Validation error: {:?}", e))?;
                let config: FuzzConfig = archived
                    .deserialize(&mut rkyv::Infallible)
                    .map_err(|e| anyhow::anyhow!("Deserialization error: {:?}", e))?;
                Ok(Some(config))
            }
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_store_and_retrieve_result() {
        let tmp = TempDir::new().unwrap();
        let storage = Storage::open(tmp.path().join("test.db")).unwrap();

        let result = ValidationResult::pass(
            "test_op".to_string(),
            12345,
            1e-6,
            1e-7,
            100,
        );

        storage.store_result(&result).unwrap();

        let retrieved = storage.get_result(12345).unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.op_name, "test_op");
        assert!(retrieved.passed);
    }

    #[test]
    fn test_list_results() {
        let tmp = TempDir::new().unwrap();
        let storage = Storage::open(tmp.path().join("test.db")).unwrap();

        for i in 0..5 {
            let result = ValidationResult::pass(
                format!("op_{}", i),
                i as u64,
                1e-6,
                1e-7,
                100,
            );
            storage.store_result(&result).unwrap();
        }

        let results = storage.list_results(3).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_store_and_retrieve_failure() {
        use gpuemu_common::types::{FailureKind, ValidationFailure};

        let tmp = TempDir::new().unwrap();
        let storage = Storage::open(tmp.path().join("test.db")).unwrap();

        let failure = ValidationResult::fail(
            "failing_op".to_string(),
            99999,
            vec![ValidationFailure {
                kind: FailureKind::ToleranceExceeded,
                message: "Value mismatch".to_string(),
                index: Some(42),
                expected: Some(1.0),
                actual: Some(1.1),
            }],
            50,
        );

        storage.store_failure(&failure).unwrap();

        let retrieved = storage.get_failure(99999).unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.op_name, "failing_op");
        assert!(!retrieved.passed);
        assert_eq!(retrieved.failures.len(), 1);
    }

    #[test]
    fn test_list_failures() {
        use gpuemu_common::types::{FailureKind, ValidationFailure};

        let tmp = TempDir::new().unwrap();
        let storage = Storage::open(tmp.path().join("test.db")).unwrap();

        for i in 0..5 {
            let failure = ValidationResult::fail(
                format!("op_{}", i),
                (100 + i) as u64,
                vec![ValidationFailure {
                    kind: FailureKind::NaNDetected,
                    message: "NaN found".to_string(),
                    index: Some(i),
                    expected: None,
                    actual: None,
                }],
                10,
            );
            storage.store_failure(&failure).unwrap();
        }

        let failures = storage.list_failures(3).unwrap();
        assert_eq!(failures.len(), 3);
        assert_eq!(storage.count_failures().unwrap(), 5);
    }

    #[test]
    fn test_store_and_retrieve_fuzz_config() {
        use gpuemu_common::types::DType;

        let tmp = TempDir::new().unwrap();
        let storage = Storage::open(tmp.path().join("test.db")).unwrap();

        let config = FuzzConfig {
            seed: 12345,
            shape_options: gpuemu_common::types::ShapeOptions {
                batch_sizes: vec![1, 2, 4],
                seq_lengths: vec![128, 256],
                hidden_dims: vec![512],
                edge_cases: vec![vec![1]],
            },
            dtypes: vec![DType::Float32],
            layouts: vec![gpuemu_common::types::LayoutType::Contiguous],
        };

        storage.store_fuzz_config(12345, &config).unwrap();

        let retrieved = storage.get_fuzz_config(12345).unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.seed, 12345);
        assert_eq!(retrieved.shape_options.batch_sizes, vec![1, 2, 4]);
    }
}
