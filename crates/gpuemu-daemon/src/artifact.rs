//! Artifact analysis for PTX and SASS.
//!
//! This module provides parsing and linting capabilities for GPU compilation artifacts,
//! enabling detection of regressions without requiring GPU hardware.

use gpuemu_common::config::ArtifactCheckConfig;
use gpuemu_common::types::{
    ArtifactDiff, ArtifactMetrics, ArtifactSource, LintResult, LintViolation, LintViolationKind,
};
use regex::Regex;
use std::collections::HashSet;
use thiserror::Error;

/// Errors that can occur during artifact analysis.
#[derive(Debug, Error)]
pub enum ArtifactError {
    #[error("PTX parse error: {0}")]
    PtxParseError(String),
    #[error("cuobjdump failed: {0}")]
    CuobjdumpFailed(String),
    #[error("cuobjdump not available")]
    CuobjdumpNotAvailable,
}

/// PTX parser for extracting metrics from PTX assembly.
pub struct PtxParser {
    // Precompiled regex patterns
    reg_decl_re: Regex,
    local_decl_re: Regex,
    shared_decl_re: Regex,
    spill_re: Regex,
    instruction_re: Regex,
    kernel_re: Regex,
}

impl PtxParser {
    /// Create a new PTX parser with precompiled regex patterns.
    pub fn new() -> Self {
        Self {
            // Match: .reg .f32 %f<123>; or .reg .b32 %r<45>;
            reg_decl_re: Regex::new(r"\.reg\s+\.\w+\s+%\w+<(\d+)>").unwrap(),
            // Match: .local .align 4 .b8 __local_depot[256];
            local_decl_re: Regex::new(r"\.local\s+[^;]*\[(\d+)\]").unwrap(),
            // Match: .shared .align 16 .b8 smem[2048];
            shared_decl_re: Regex::new(r"\.shared\s+[^;]*\[(\d+)\]").unwrap(),
            // Match: st.local or ld.local (indicates spills)
            spill_re: Regex::new(r"\b(st|ld)\.local\b").unwrap(),
            // Match instruction lines (start with tab/spaces, have a mnemonic)
            // Handles both "add.f32 %f2, ..." and "ret;" patterns
            instruction_re: Regex::new(r"^\s+([a-z]\w*(?:\.\w+)*)(?:\s|;)").unwrap(),
            // Match kernel entry: .visible .entry kernel_name(
            kernel_re: Regex::new(r"\.visible\s+\.entry\s+(\w+)\s*\(").unwrap(),
        }
    }

    /// Parse PTX content and extract metrics.
    ///
    /// # Arguments
    /// * `kernel_name` - Name of the kernel (used for identification)
    /// * `ptx` - Raw PTX assembly text
    ///
    /// # Returns
    /// `ArtifactMetrics` with extracted information or an error
    pub fn parse(&self, kernel_name: &str, ptx: &str) -> Result<ArtifactMetrics, ArtifactError> {
        let mut register_count: u32 = 0;
        let mut local_memory_bytes: u32 = 0;
        let mut shared_memory_bytes: u32 = 0;
        let mut spill_count: u32 = 0;
        let mut instruction_count: u32 = 0;
        let mut patterns_found: HashSet<String> = HashSet::new();

        for line in ptx.lines() {
            // Count register declarations
            for caps in self.reg_decl_re.captures_iter(line) {
                if let Ok(count) = caps[1].parse::<u32>() {
                    register_count += count;
                }
            }

            // Count local memory
            for caps in self.local_decl_re.captures_iter(line) {
                if let Ok(bytes) = caps[1].parse::<u32>() {
                    local_memory_bytes += bytes;
                }
            }

            // Count shared memory
            for caps in self.shared_decl_re.captures_iter(line) {
                if let Ok(bytes) = caps[1].parse::<u32>() {
                    shared_memory_bytes += bytes;
                }
            }

            // Count spills (ld.local/st.local)
            spill_count += self.spill_re.find_iter(line).count() as u32;

            // Count instructions and extract mnemonics
            if let Some(caps) = self.instruction_re.captures(line) {
                instruction_count += 1;
                // Extract instruction mnemonic (e.g., "add.f32" -> "add.f32")
                let mnemonic = caps[1].to_string();
                patterns_found.insert(mnemonic);
            }
        }

        Ok(ArtifactMetrics {
            kernel_name: kernel_name.to_string(),
            register_count,
            spill_count,
            local_memory_bytes,
            shared_memory_bytes,
            instruction_count,
            patterns_found: patterns_found.into_iter().collect(),
            source: ArtifactSource::Ptx,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            ptx_content: Some(ptx.to_string()),
        })
    }

    /// Extract kernel name from PTX content if not provided.
    pub fn extract_kernel_name(&self, ptx: &str) -> Option<String> {
        self.kernel_re
            .captures(ptx)
            .map(|caps| caps[1].to_string())
    }
}

impl Default for PtxParser {
    fn default() -> Self {
        Self::new()
    }
}

/// SASS parser using cuobjdump (optional, requires NVIDIA tools).
pub struct SassParser;

impl SassParser {
    /// Check if cuobjdump is available on the system.
    pub fn is_available() -> bool {
        std::process::Command::new("cuobjdump")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Parse SASS from a cubin file.
    ///
    /// # Arguments
    /// * `kernel_name` - Name of the kernel to extract
    /// * `cubin_path` - Path to the cubin file
    ///
    /// # Returns
    /// `ArtifactMetrics` with extracted SASS information
    pub fn parse_cubin(
        kernel_name: &str,
        cubin_path: &std::path::Path,
    ) -> Result<ArtifactMetrics, ArtifactError> {
        if !Self::is_available() {
            return Err(ArtifactError::CuobjdumpNotAvailable);
        }

        let output = std::process::Command::new("cuobjdump")
            .args(["--dump-sass", "--function", kernel_name])
            .arg(cubin_path)
            .output()
            .map_err(|e| ArtifactError::CuobjdumpFailed(e.to_string()))?;

        if !output.status.success() {
            return Err(ArtifactError::CuobjdumpFailed(
                String::from_utf8_lossy(&output.stderr).to_string(),
            ));
        }

        let sass_output = String::from_utf8_lossy(&output.stdout);
        Self::parse_sass_output(kernel_name, &sass_output)
    }

    /// Parse SASS output text and extract metrics.
    fn parse_sass_output(kernel_name: &str, sass: &str) -> Result<ArtifactMetrics, ArtifactError> {
        let mut register_count: u32 = 0;
        let mut instruction_count: u32 = 0;
        let mut patterns_found: HashSet<String> = HashSet::new();

        // Look for "REG:xx" in function info
        let reg_re = Regex::new(r"REG:(\d+)").unwrap();
        if let Some(caps) = reg_re.captures(sass) {
            register_count = caps[1].parse().unwrap_or(0);
        }

        // SASS instruction pattern: /*address*/ OPCODE operands
        let sass_instr_re = Regex::new(r"/\*[0-9a-fA-F]+\*/\s+([A-Z][A-Z0-9\.]+)").unwrap();
        for caps in sass_instr_re.captures_iter(sass) {
            instruction_count += 1;
            patterns_found.insert(caps[1].to_string());
        }

        Ok(ArtifactMetrics {
            kernel_name: kernel_name.to_string(),
            register_count,
            spill_count: 0, // Would need more sophisticated SASS analysis
            local_memory_bytes: 0,
            shared_memory_bytes: 0,
            instruction_count,
            patterns_found: patterns_found.into_iter().collect(),
            source: ArtifactSource::Sass,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            ptx_content: None,
        })
    }
}

/// Linter that checks artifacts against policy rules.
pub struct ArtifactLinter;

impl ArtifactLinter {
    /// Lint artifact metrics against configuration.
    ///
    /// # Arguments
    /// * `metrics` - Extracted artifact metrics
    /// * `config` - Policy configuration to lint against
    ///
    /// # Returns
    /// `LintResult` with pass/fail status and any violations
    pub fn lint(metrics: &ArtifactMetrics, config: &ArtifactCheckConfig) -> LintResult {
        let mut violations = Vec::new();

        // Check register count
        if metrics.register_count > config.max_registers {
            violations.push(LintViolation {
                kind: LintViolationKind::ExcessiveRegisters,
                message: format!(
                    "Register count {} exceeds maximum {}",
                    metrics.register_count, config.max_registers
                ),
                actual: Some(metrics.register_count),
                threshold: Some(config.max_registers),
            });
        }

        // Check spill count
        if metrics.spill_count > config.max_spills {
            violations.push(LintViolation {
                kind: LintViolationKind::ExcessiveSpills,
                message: format!(
                    "Spill count {} exceeds maximum {}",
                    metrics.spill_count, config.max_spills
                ),
                actual: Some(metrics.spill_count),
                threshold: Some(config.max_spills),
            });
        }

        // Check local memory
        if metrics.local_memory_bytes > config.max_local_memory {
            violations.push(LintViolation {
                kind: LintViolationKind::ExcessiveLocalMemory,
                message: format!(
                    "Local memory {} bytes exceeds maximum {} bytes",
                    metrics.local_memory_bytes, config.max_local_memory
                ),
                actual: Some(metrics.local_memory_bytes),
                threshold: Some(config.max_local_memory),
            });
        }

        // Check required patterns
        for pattern in &config.required_patterns {
            let pattern_found = metrics
                .patterns_found
                .iter()
                .any(|p| p.contains(pattern.as_str()));
            if !pattern_found {
                violations.push(LintViolation {
                    kind: LintViolationKind::MissingRequiredPattern,
                    message: format!("Required pattern '{}' not found", pattern),
                    actual: None,
                    threshold: None,
                });
            }
        }

        // Check forbidden patterns
        for pattern in &config.forbidden_patterns {
            let pattern_found = metrics
                .patterns_found
                .iter()
                .any(|p| p.contains(pattern.as_str()));
            if pattern_found {
                violations.push(LintViolation {
                    kind: LintViolationKind::ForbiddenPatternFound,
                    message: format!("Forbidden pattern '{}' found", pattern),
                    actual: None,
                    threshold: None,
                });
            }
        }

        LintResult {
            kernel_name: metrics.kernel_name.clone(),
            passed: violations.is_empty(),
            metrics: metrics.clone(),
            violations,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

/// Computes diffs between artifact metrics.
pub struct ArtifactDiffer;

impl ArtifactDiffer {
    /// Compare current metrics against baseline.
    ///
    /// # Arguments
    /// * `baseline` - Previous artifact metrics (None if no baseline)
    /// * `current` - Current artifact metrics
    ///
    /// # Returns
    /// `ArtifactDiff` with deltas and regression flag
    pub fn diff(baseline: Option<&ArtifactMetrics>, current: &ArtifactMetrics) -> ArtifactDiff {
        let (register_delta, spill_delta, local_memory_delta, instruction_delta, is_regression) =
            match baseline {
                Some(base) => {
                    let reg_d = current.register_count as i32 - base.register_count as i32;
                    let spill_d = current.spill_count as i32 - base.spill_count as i32;
                    let local_d =
                        current.local_memory_bytes as i32 - base.local_memory_bytes as i32;
                    let inst_d = current.instruction_count as i32 - base.instruction_count as i32;

                    // Regression if any critical metric increased
                    let is_reg = reg_d > 0 || spill_d > 0 || local_d > 0;

                    (reg_d, spill_d, local_d, inst_d, is_reg)
                }
                None => (0, 0, 0, 0, false),
            };

        ArtifactDiff {
            kernel_name: current.kernel_name.clone(),
            baseline: baseline.cloned(),
            current: current.clone(),
            register_delta,
            spill_delta,
            local_memory_delta,
            instruction_delta,
            is_regression,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_PTX: &str = r#"
.version 7.0
.target sm_80
.address_size 64

.visible .entry my_kernel(
    .param .u64 param_0,
    .param .u64 param_1
)
{
    .reg .f32 %f<32>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<8>;
    .local .align 4 .b8 __local_depot[128];

    ld.param.u64 %rd1, [param_0];
    ld.local.f32 %f1, [__local_depot];
    add.f32 %f2, %f1, %f1;
    mul.f32 %f3, %f2, %f1;
    st.local.f32 [__local_depot], %f2;
    st.global.f32 [%rd1], %f3;
    ret;
}
"#;

    #[test]
    fn test_parse_registers() {
        let parser = PtxParser::new();
        let metrics = parser.parse("my_kernel", SAMPLE_PTX).unwrap();
        // 32 float + 16 b32 + 8 b64 = 56 registers
        assert_eq!(metrics.register_count, 56);
    }

    #[test]
    fn test_parse_local_memory() {
        let parser = PtxParser::new();
        let metrics = parser.parse("my_kernel", SAMPLE_PTX).unwrap();
        assert_eq!(metrics.local_memory_bytes, 128);
    }

    #[test]
    fn test_parse_spills() {
        let parser = PtxParser::new();
        let metrics = parser.parse("my_kernel", SAMPLE_PTX).unwrap();
        // ld.local + st.local = 2 spills
        assert_eq!(metrics.spill_count, 2);
    }

    #[test]
    fn test_parse_instructions() {
        let parser = PtxParser::new();
        let metrics = parser.parse("my_kernel", SAMPLE_PTX).unwrap();
        // ld.param, ld.local, add.f32, mul.f32, st.local, st.global, ret = 7
        assert_eq!(metrics.instruction_count, 7);
    }

    #[test]
    fn test_extract_kernel_name() {
        let parser = PtxParser::new();
        let name = parser.extract_kernel_name(SAMPLE_PTX);
        assert_eq!(name, Some("my_kernel".to_string()));
    }

    #[test]
    fn test_lint_pass() {
        let parser = PtxParser::new();
        let metrics = parser.parse("my_kernel", SAMPLE_PTX).unwrap();

        let config = ArtifactCheckConfig {
            max_registers: 64,
            max_spills: 10,
            max_local_memory: 256,
            required_patterns: vec![],
            forbidden_patterns: vec![],
        };

        let result = ArtifactLinter::lint(&metrics, &config);
        assert!(result.passed);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_lint_fail_registers() {
        let parser = PtxParser::new();
        let metrics = parser.parse("my_kernel", SAMPLE_PTX).unwrap();

        let config = ArtifactCheckConfig {
            max_registers: 32, // Too low - we have 56
            max_spills: 10,
            max_local_memory: 256,
            required_patterns: vec![],
            forbidden_patterns: vec![],
        };

        let result = ArtifactLinter::lint(&metrics, &config);
        assert!(!result.passed);
        assert!(result
            .violations
            .iter()
            .any(|v| v.kind == LintViolationKind::ExcessiveRegisters));
    }

    #[test]
    fn test_lint_required_pattern() {
        let parser = PtxParser::new();
        let metrics = parser.parse("my_kernel", SAMPLE_PTX).unwrap();

        let config = ArtifactCheckConfig {
            max_registers: 100,
            max_spills: 100,
            max_local_memory: 1000,
            required_patterns: vec!["add.f32".to_string()],
            forbidden_patterns: vec![],
        };

        let result = ArtifactLinter::lint(&metrics, &config);
        assert!(result.passed);
    }

    #[test]
    fn test_lint_missing_required_pattern() {
        let parser = PtxParser::new();
        let metrics = parser.parse("my_kernel", SAMPLE_PTX).unwrap();

        let config = ArtifactCheckConfig {
            max_registers: 100,
            max_spills: 100,
            max_local_memory: 1000,
            required_patterns: vec!["fma.f32".to_string()], // Not in sample
            forbidden_patterns: vec![],
        };

        let result = ArtifactLinter::lint(&metrics, &config);
        assert!(!result.passed);
        assert!(result
            .violations
            .iter()
            .any(|v| v.kind == LintViolationKind::MissingRequiredPattern));
    }

    #[test]
    fn test_lint_forbidden_pattern() {
        let parser = PtxParser::new();
        let metrics = parser.parse("my_kernel", SAMPLE_PTX).unwrap();

        let config = ArtifactCheckConfig {
            max_registers: 100,
            max_spills: 100,
            max_local_memory: 1000,
            required_patterns: vec![],
            forbidden_patterns: vec!["mul.f32".to_string()], // This IS in sample
        };

        let result = ArtifactLinter::lint(&metrics, &config);
        assert!(!result.passed);
        assert!(result
            .violations
            .iter()
            .any(|v| v.kind == LintViolationKind::ForbiddenPatternFound));
    }

    #[test]
    fn test_diff_no_baseline() {
        let current = ArtifactMetrics {
            kernel_name: "test".to_string(),
            register_count: 32,
            spill_count: 0,
            local_memory_bytes: 0,
            shared_memory_bytes: 0,
            instruction_count: 100,
            patterns_found: vec![],
            source: ArtifactSource::Ptx,
            timestamp: 0,
            ptx_content: None,
        };

        let diff = ArtifactDiffer::diff(None, &current);
        assert!(!diff.is_regression);
        assert_eq!(diff.register_delta, 0);
    }

    #[test]
    fn test_diff_regression() {
        let baseline = ArtifactMetrics {
            kernel_name: "test".to_string(),
            register_count: 32,
            spill_count: 0,
            local_memory_bytes: 0,
            shared_memory_bytes: 0,
            instruction_count: 100,
            patterns_found: vec![],
            source: ArtifactSource::Ptx,
            timestamp: 0,
            ptx_content: None,
        };

        let current = ArtifactMetrics {
            kernel_name: "test".to_string(),
            register_count: 48, // Increased!
            spill_count: 0,
            local_memory_bytes: 0,
            shared_memory_bytes: 0,
            instruction_count: 110,
            patterns_found: vec![],
            source: ArtifactSource::Ptx,
            timestamp: 1,
            ptx_content: None,
        };

        let diff = ArtifactDiffer::diff(Some(&baseline), &current);
        assert!(diff.is_regression);
        assert_eq!(diff.register_delta, 16);
        assert_eq!(diff.instruction_delta, 10);
    }

    #[test]
    fn test_diff_improvement() {
        let baseline = ArtifactMetrics {
            kernel_name: "test".to_string(),
            register_count: 48,
            spill_count: 2,
            local_memory_bytes: 128,
            shared_memory_bytes: 0,
            instruction_count: 100,
            patterns_found: vec![],
            source: ArtifactSource::Ptx,
            timestamp: 0,
            ptx_content: None,
        };

        let current = ArtifactMetrics {
            kernel_name: "test".to_string(),
            register_count: 32, // Decreased!
            spill_count: 0,     // Decreased!
            local_memory_bytes: 0, // Decreased!
            shared_memory_bytes: 0,
            instruction_count: 90,
            patterns_found: vec![],
            source: ArtifactSource::Ptx,
            timestamp: 1,
            ptx_content: None,
        };

        let diff = ArtifactDiffer::diff(Some(&baseline), &current);
        assert!(!diff.is_regression); // Improvement, not regression
        assert_eq!(diff.register_delta, -16);
        assert_eq!(diff.spill_delta, -2);
    }
}
