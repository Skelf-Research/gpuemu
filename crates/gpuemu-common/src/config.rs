//! Configuration parsing for gpuemu.toml files.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

/// Errors that can occur when loading configuration.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}")]
    Io(#[from] std::io::Error),
    #[error("Failed to parse TOML: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("Config file not found: {0}")]
    NotFound(String),
}

/// Root configuration structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuemuConfig {
    /// Project metadata.
    #[serde(default)]
    pub project: ProjectConfig,
    /// Validation settings.
    #[serde(default)]
    pub validation: ValidationConfig,
    /// Registered ops.
    #[serde(default)]
    pub ops: Vec<OpConfig>,
    /// Registered kernels.
    #[serde(default)]
    pub kernels: Vec<KernelConfig>,
    /// Policy settings.
    #[serde(default)]
    pub policies: PolicyConfig,
    /// CI settings.
    #[serde(default)]
    pub ci: CiConfig,
}

impl GpuemuConfig {
    /// Load configuration from a file path.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(ConfigError::NotFound(path.display().to_string()));
        }
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }

    /// Load configuration from the current directory or parent directories.
    pub fn find_and_load() -> Result<Self, ConfigError> {
        let mut dir = std::env::current_dir().map_err(ConfigError::Io)?;
        loop {
            let config_path = dir.join("gpuemu.toml");
            if config_path.exists() {
                return Self::load(&config_path);
            }
            if !dir.pop() {
                break;
            }
        }
        Err(ConfigError::NotFound("gpuemu.toml".to_string()))
    }

    /// Generate a default configuration file content.
    pub fn default_template() -> String {
        r#"[project]
name = "my-project"
framework = "pytorch"

[validation]
dtypes = ["float32", "float16"]
check_nan = true
check_inf = true

[validation.tolerances]
float32 = 1e-5
float16 = 1e-3
bfloat16 = 1e-3

# Example op registration:
# [[ops]]
# name = "my_custom_op"
# module = "my_module.custom_op"
# reference = "scripts/ref_my_op.py"
#
# [ops.tolerances]
# float32 = 1e-5
"#
        .to_string()
    }
}

impl Default for GpuemuConfig {
    fn default() -> Self {
        Self {
            project: ProjectConfig::default(),
            validation: ValidationConfig::default(),
            ops: Vec::new(),
            kernels: Vec::new(),
            policies: PolicyConfig::default(),
            ci: CiConfig::default(),
        }
    }
}

/// Project metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    /// Project name.
    #[serde(default = "default_project_name")]
    pub name: String,
    /// Project version.
    #[serde(default)]
    pub version: Option<String>,
    /// Target framework (pytorch, jax, tensorflow).
    #[serde(default)]
    pub framework: Option<String>,
}

fn default_project_name() -> String {
    "unnamed".to_string()
}

impl Default for ProjectConfig {
    fn default() -> Self {
        Self {
            name: default_project_name(),
            version: None,
            framework: None,
        }
    }
}

/// Validation settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Data types to validate.
    #[serde(default = "default_dtypes")]
    pub dtypes: Vec<String>,
    /// Check for NaN in outputs.
    #[serde(default = "default_true")]
    pub check_nan: bool,
    /// Check for Inf in outputs.
    #[serde(default = "default_true")]
    pub check_inf: bool,
    /// Fixed seed for reproducibility (optional).
    #[serde(default)]
    pub seed: Option<u64>,
    /// Tolerance settings per dtype.
    #[serde(default)]
    pub tolerances: HashMap<String, f64>,
    /// Promote float inputs to fp64 before running the reference oracle, so the
    /// reference is a high-precision ground truth. Defaults to `true`. Disable
    /// (`oracle_fp64 = false`) to compute the reference in the kernel's input
    /// dtype (the historical behaviour).
    #[serde(default = "default_true")]
    pub oracle_fp64: bool,
}

fn default_dtypes() -> Vec<String> {
    vec!["float32".to_string()]
}

fn default_true() -> bool {
    true
}

impl Default for ValidationConfig {
    fn default() -> Self {
        let mut tolerances = HashMap::new();
        tolerances.insert("float32".to_string(), 1e-5);
        tolerances.insert("float16".to_string(), 1e-3);
        tolerances.insert("bfloat16".to_string(), 1e-3);

        Self {
            dtypes: default_dtypes(),
            check_nan: true,
            check_inf: true,
            seed: None,
            tolerances,
            oracle_fp64: true,
        }
    }
}

impl ValidationConfig {
    /// Get tolerance for a specific dtype.
    pub fn get_tolerance(&self, dtype: &str) -> f64 {
        self.tolerances.get(dtype).copied().unwrap_or(1e-5)
    }
}

/// How the op under test is executed during fuzzing/CI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    /// Client generates inputs, runs the op itself, and submits the output
    /// to the daemon for comparison against the reference. The daemon never
    /// invokes the op — it only computes the reference and validates.
    #[default]
    ClientSide,
    /// Daemon generates test cases and returns them to the client (via
    /// GetTestCase/GetTestBatch requests). The client runs the op and
    /// submits each output. The daemon orchestrates iteration and fail-fast.
    DaemonOrchestrated,
    /// Daemon spawns both the reference script and an op script, feeding
    /// the same inputs to both. Requires the op script to be runnable
    /// from the daemon machine (e.g. has GPU access).
    ScriptBased,
}

/// Op registration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpConfig {
    /// Op name (identifier).
    pub name: String,
    /// Python module path (optional).
    #[serde(default)]
    pub module: Option<String>,
    /// Path to reference implementation script.
    pub reference: String,
    /// Path to op implementation script (used only in ScriptBased mode).
    /// Receives the same JSON+base64 input as the reference script,
    /// and must produce the same JSON+base64 output format.
    #[serde(default)]
    pub op_script: Option<String>,
    /// Input names used by the reference implementation during fuzzing/repro.
    #[serde(default)]
    pub input_names: Vec<String>,
    /// How this op is executed during fuzzing/CI runs.
    #[serde(default)]
    pub execution_mode: ExecutionMode,
    /// Frameworks this op supports.
    #[serde(default)]
    pub frameworks: Vec<String>,
    /// Op-specific tolerances.
    #[serde(default)]
    pub tolerances: HashMap<String, f64>,
    /// Invariants to check.
    #[serde(default)]
    pub invariants: InvariantConfig,
}

/// Kernel registration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelConfig {
    /// Kernel name (identifier).
    pub name: String,
    /// Path to CUDA source (optional).
    #[serde(default)]
    pub source: Option<String>,
    /// Path to reference implementation script.
    pub reference: String,
    /// Kernel-specific tolerances.
    #[serde(default)]
    pub tolerances: HashMap<String, f64>,
    /// Invariants to check.
    #[serde(default)]
    pub invariants: InvariantConfig,
    /// Artifact checks.
    #[serde(default)]
    pub artifact_checks: ArtifactCheckConfig,
}

/// Invariant checks configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InvariantConfig {
    /// Output must be non-negative.
    #[serde(default)]
    pub non_negative: bool,
    /// Output shape must match input shape.
    #[serde(default)]
    pub shape_preserved: bool,
    /// No NaN in output.
    #[serde(default)]
    pub no_nan: bool,
    /// No Inf in output.
    #[serde(default)]
    pub no_inf: bool,
}

/// Artifact check configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactCheckConfig {
    /// Maximum register count.
    #[serde(default = "default_max_registers")]
    pub max_registers: u32,
    /// Maximum spill count.
    #[serde(default)]
    pub max_spills: u32,
    /// Maximum local memory bytes.
    #[serde(default)]
    pub max_local_memory: u32,
    /// Required instruction patterns.
    #[serde(default)]
    pub required_patterns: Vec<String>,
    /// Forbidden instruction patterns.
    #[serde(default)]
    pub forbidden_patterns: Vec<String>,
}

fn default_max_registers() -> u32 {
    64
}

impl Default for ArtifactCheckConfig {
    fn default() -> Self {
        Self {
            max_registers: default_max_registers(),
            max_spills: 0,
            max_local_memory: 0,
            required_patterns: Vec::new(),
            forbidden_patterns: Vec::new(),
        }
    }
}

/// Policy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyConfig {
    /// Fail on any regression.
    #[serde(default = "default_true")]
    pub fail_on_regression: bool,
    /// Warning threshold (percentage).
    #[serde(default = "default_warn_threshold")]
    pub warn_threshold: f64,
}

fn default_warn_threshold() -> f64 {
    0.1
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            fail_on_regression: true,
            warn_threshold: default_warn_threshold(),
        }
    }
}

/// CI configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiConfig {
    /// DTypes for quick validation.
    #[serde(default = "default_quick_dtypes")]
    pub quick_dtypes: Vec<String>,
    /// Timeout for thorough validation (seconds).
    #[serde(default = "default_thorough_timeout")]
    pub thorough_timeout: u64,
    /// Number of parallel jobs.
    #[serde(default = "default_parallel_jobs")]
    pub parallel_jobs: u32,
}

fn default_quick_dtypes() -> Vec<String> {
    vec!["float32".to_string()]
}

fn default_thorough_timeout() -> u64 {
    3600
}

fn default_parallel_jobs() -> u32 {
    4
}

impl Default for CiConfig {
    fn default() -> Self {
        Self {
            quick_dtypes: default_quick_dtypes(),
            thorough_timeout: default_thorough_timeout(),
            parallel_jobs: default_parallel_jobs(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GpuemuConfig::default();
        assert_eq!(config.project.name, "unnamed");
        assert!(config.validation.check_nan);
    }

    #[test]
    fn test_parse_minimal_config() {
        let toml = r#"
[project]
name = "test-project"
"#;
        let config: GpuemuConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.project.name, "test-project");
    }

    #[test]
    fn test_parse_full_config() {
        let toml = r#"
[project]
name = "full-test"
framework = "pytorch"

[validation]
dtypes = ["float32", "float16"]
check_nan = true

[validation.tolerances]
float32 = 1e-6

[[ops]]
name = "my_op"
reference = "scripts/ref.py"
"#;
        let config: GpuemuConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.project.name, "full-test");
        assert_eq!(config.ops.len(), 1);
        assert_eq!(config.ops[0].name, "my_op");
    }
}
