//! Report generators for CI integration.
//!
//! Supports JUnit XML, JSON, and plain text output formats.

use gpuemu_common::types::{CiRunSummary, LintResult, ValidationResult};

/// Output format for reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Text,
    Json,
    Junit,
}

impl OutputFormat {
    /// Parse output format from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "text" => Some(OutputFormat::Text),
            "json" => Some(OutputFormat::Json),
            "junit" | "xml" => Some(OutputFormat::Junit),
            _ => None,
        }
    }
}

/// Generate a report from a CI run summary in the specified format.
pub fn generate_report(summary: &CiRunSummary, format: OutputFormat) -> String {
    match format {
        OutputFormat::Text => TextReport::from_summary(summary),
        OutputFormat::Json => JsonReport::from_summary(summary),
        OutputFormat::Junit => JunitReport::from_summary(summary),
    }
}

// =============================================================================
// JUnit XML Report
// =============================================================================

/// JUnit XML report generator.
pub struct JunitReport;

impl JunitReport {
    /// Convert CiRunSummary to JUnit XML format.
    pub fn from_summary(summary: &CiRunSummary) -> String {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");

        let total_time = summary.duration_ms as f64 / 1000.0;
        let validation_failures = summary
            .validation_results
            .iter()
            .filter(|r| !r.passed)
            .count();
        let lint_failures = summary.lint_results.iter().filter(|r| !r.passed).count();
        let total_failures = validation_failures + lint_failures;

        xml.push_str(&format!(
            "<testsuites name=\"gpuemu\" tests=\"{}\" failures=\"{}\" errors=\"0\" time=\"{:.3}\">\n",
            summary.total_tests, total_failures, total_time
        ));

        // Validation testsuite
        if !summary.validation_results.is_empty() {
            let validation_time: f64 = summary
                .validation_results
                .iter()
                .map(|r| r.duration_ms as f64 / 1000.0)
                .sum();

            xml.push_str(&format!(
                "  <testsuite name=\"validation\" tests=\"{}\" failures=\"{}\" time=\"{:.3}\">\n",
                summary.validation_results.len(),
                validation_failures,
                validation_time
            ));

            for result in &summary.validation_results {
                xml.push_str(&Self::validation_to_testcase(result));
            }

            xml.push_str("  </testsuite>\n");
        }

        // Lint testsuite
        if !summary.lint_results.is_empty() {
            let lint_time = 0.1 * summary.lint_results.len() as f64; // Approximate

            xml.push_str(&format!(
                "  <testsuite name=\"lint\" tests=\"{}\" failures=\"{}\" time=\"{:.3}\">\n",
                summary.lint_results.len(),
                lint_failures,
                lint_time
            ));

            for result in &summary.lint_results {
                xml.push_str(&Self::lint_to_testcase(result));
            }

            xml.push_str("  </testsuite>\n");
        }

        // Artifact diff testsuite (if present)
        if let Some(ref diffs) = summary.artifact_diffs {
            let regression_count = diffs.diffs.iter().filter(|d| d.is_regression).count();

            xml.push_str(&format!(
                "  <testsuite name=\"artifacts\" tests=\"{}\" failures=\"{}\" time=\"0.000\">\n",
                diffs.diffs.len(),
                regression_count
            ));

            for diff in &diffs.diffs {
                let _status = if diff.is_regression {
                    "REGRESSION"
                } else {
                    "OK"
                };
                xml.push_str(&format!(
                    "    <testcase name=\"{}\" classname=\"artifacts\" time=\"0.000\">\n",
                    xml_escape(&diff.kernel_name)
                ));

                if diff.is_regression {
                    xml.push_str(&format!(
                        "      <failure type=\"Regression\" message=\"Artifact regression detected\">\n"
                    ));
                    xml.push_str(&format!("Kernel: {}\n", diff.kernel_name));
                    xml.push_str(&format!("Register delta: {:+}\n", diff.register_delta));
                    xml.push_str(&format!("Spill delta: {:+}\n", diff.spill_delta));
                    xml.push_str(&format!(
                        "Local memory delta: {:+}\n",
                        diff.local_memory_delta
                    ));
                    xml.push_str(&format!(
                        "Instruction delta: {:+}\n",
                        diff.instruction_delta
                    ));
                    xml.push_str("      </failure>\n");
                }

                xml.push_str("    </testcase>\n");
            }

            xml.push_str("  </testsuite>\n");
        }

        xml.push_str("</testsuites>\n");
        xml
    }

    /// Convert ValidationResult to JUnit testcase element.
    fn validation_to_testcase(result: &ValidationResult) -> String {
        let mut xml = String::new();
        let time = result.duration_ms as f64 / 1000.0;

        xml.push_str(&format!(
            "    <testcase name=\"{}\" classname=\"ops\" time=\"{:.3}\">\n",
            xml_escape(&result.op_name),
            time
        ));

        if !result.passed {
            // Get the first failure for the type and message
            if let Some(failure) = result.failures.first() {
                xml.push_str(&format!(
                    "      <failure type=\"{:?}\" message=\"{}\">\n",
                    failure.kind,
                    xml_escape(&failure.message)
                ));

                // Add details
                xml.push_str(&format!("Seed: {}\n", result.seed));
                if let Some(ref repro) = result.repro_info {
                    xml.push_str(&format!("Shape: {:?}\n", repro.shape));
                    xml.push_str(&format!("DType: {:?}\n", repro.dtype));
                    xml.push_str(&format!("Layout: {:?}\n", repro.layout));
                }
                xml.push_str(&format!("Max diff: {}\n", result.max_diff));
                xml.push_str(&format!("Max rel diff: {}\n", result.max_rel_diff));

                // List all failures
                if result.failures.len() > 1 {
                    xml.push_str("\nAll failures:\n");
                    for f in &result.failures {
                        xml.push_str(&format!("  - {:?}: {}\n", f.kind, f.message));
                    }
                }

                xml.push_str("      </failure>\n");
            }
        }

        xml.push_str("    </testcase>\n");
        xml
    }

    /// Convert LintResult to JUnit testcase element.
    fn lint_to_testcase(result: &LintResult) -> String {
        let mut xml = String::new();

        xml.push_str(&format!(
            "    <testcase name=\"{}\" classname=\"artifacts\" time=\"0.100\">\n",
            xml_escape(&result.kernel_name)
        ));

        if !result.passed {
            if let Some(violation) = result.violations.first() {
                xml.push_str(&format!(
                    "      <failure type=\"{:?}\" message=\"{}\">\n",
                    violation.kind,
                    xml_escape(&violation.message)
                ));

                // Add metrics
                xml.push_str(&format!("Registers: {}\n", result.metrics.register_count));
                xml.push_str(&format!("Spills: {}\n", result.metrics.spill_count));
                xml.push_str(&format!(
                    "Local memory: {} bytes\n",
                    result.metrics.local_memory_bytes
                ));
                xml.push_str(&format!(
                    "Shared memory: {} bytes\n",
                    result.metrics.shared_memory_bytes
                ));
                xml.push_str(&format!(
                    "Instructions: {}\n",
                    result.metrics.instruction_count
                ));

                // List all violations
                if result.violations.len() > 1 {
                    xml.push_str("\nAll violations:\n");
                    for v in &result.violations {
                        xml.push_str(&format!("  - {:?}: {}\n", v.kind, v.message));
                    }
                }

                xml.push_str("      </failure>\n");
            }
        }

        xml.push_str("    </testcase>\n");
        xml
    }
}

/// Escape special XML characters.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

// =============================================================================
// JSON Report
// =============================================================================

/// JSON report generator.
pub struct JsonReport;

impl JsonReport {
    /// Convert CiRunSummary to JSON format.
    pub fn from_summary(summary: &CiRunSummary) -> String {
        serde_json::to_string_pretty(summary).unwrap_or_else(|_| "{}".to_string())
    }
}

// =============================================================================
// Text Report
// =============================================================================

/// Plain text report generator.
pub struct TextReport;

impl TextReport {
    /// Convert CiRunSummary to human-readable text.
    pub fn from_summary(summary: &CiRunSummary) -> String {
        let mut text = String::new();

        // Header
        text.push_str(
            "═══════════════════════════════════════════════════════════════════════════════\n",
        );
        text.push_str("                              gpuemu CI Report\n");
        text.push_str(
            "═══════════════════════════════════════════════════════════════════════════════\n\n",
        );

        // Summary
        text.push_str(&format!("Total Tests:  {}\n", summary.total_tests));
        text.push_str(&format!("Passed:       {}\n", summary.passed));
        text.push_str(&format!("Failed:       {}\n", summary.failed));
        text.push_str(&format!("Skipped:      {}\n", summary.skipped));
        text.push_str(&format!(
            "Duration:     {:.2}s\n",
            summary.duration_ms as f64 / 1000.0
        ));
        text.push_str("\n");

        // Validation results
        if !summary.validation_results.is_empty() {
            text.push_str(
                "───────────────────────────────────────────────────────────────────────────────\n",
            );
            text.push_str("Validation Results\n");
            text.push_str(
                "───────────────────────────────────────────────────────────────────────────────\n",
            );

            for result in &summary.validation_results {
                let status = if result.passed { "PASS" } else { "FAIL" };
                text.push_str(&format!(
                    "[{}] {} (seed: {}, {:.2}ms)\n",
                    status, result.op_name, result.seed, result.duration_ms as f64
                ));

                if !result.passed {
                    for failure in &result.failures {
                        text.push_str(&format!("      {:?}: {}\n", failure.kind, failure.message));
                    }
                }
            }
            text.push_str("\n");
        }

        // Lint results
        if !summary.lint_results.is_empty() {
            text.push_str(
                "───────────────────────────────────────────────────────────────────────────────\n",
            );
            text.push_str("Lint Results\n");
            text.push_str(
                "───────────────────────────────────────────────────────────────────────────────\n",
            );

            for result in &summary.lint_results {
                let status = if result.passed { "PASS" } else { "FAIL" };
                text.push_str(&format!(
                    "[{}] {} (regs: {}, spills: {}, local: {}B)\n",
                    status,
                    result.kernel_name,
                    result.metrics.register_count,
                    result.metrics.spill_count,
                    result.metrics.local_memory_bytes
                ));

                if !result.passed {
                    for violation in &result.violations {
                        text.push_str(&format!(
                            "      {:?}: {}\n",
                            violation.kind, violation.message
                        ));
                    }
                }
            }
            text.push_str("\n");
        }

        // Artifact diffs
        if let Some(ref diffs) = summary.artifact_diffs {
            text.push_str(
                "───────────────────────────────────────────────────────────────────────────────\n",
            );
            text.push_str(&format!(
                "Artifact Diff (baseline: {})\n",
                diffs.baseline_tag
            ));
            text.push_str(
                "───────────────────────────────────────────────────────────────────────────────\n",
            );

            text.push_str(&format!(
                "{:<30} {:>10} {:>10} {:>12} {:>10} {}\n",
                "KERNEL", "REGS", "SPILLS", "LOCAL_MEM", "INSTRS", "STATUS"
            ));
            text.push_str(&format!("{}\n", "-".repeat(85)));

            for diff in &diffs.diffs {
                let status = if diff.is_regression {
                    "REGRESSION"
                } else if diff.baseline.is_none() {
                    "NEW"
                } else {
                    "OK"
                };

                let format_delta = |delta: i32| -> String {
                    if delta > 0 {
                        format!("+{}", delta)
                    } else if delta < 0 {
                        format!("{}", delta)
                    } else {
                        "0".to_string()
                    }
                };

                text.push_str(&format!(
                    "{:<30} {:>10} {:>10} {:>12} {:>10} {}\n",
                    &diff.kernel_name.chars().take(30).collect::<String>(),
                    format_delta(diff.register_delta),
                    format_delta(diff.spill_delta),
                    format_delta(diff.local_memory_delta),
                    format_delta(diff.instruction_delta),
                    status
                ));
            }

            if diffs.has_regressions {
                text.push_str("\n⚠ Regressions detected!\n");
            } else {
                text.push_str("\n✓ No regressions detected.\n");
            }
            text.push_str("\n");
        }

        // Footer
        text.push_str(
            "═══════════════════════════════════════════════════════════════════════════════\n",
        );
        if summary.has_failures() || summary.has_regressions() {
            text.push_str("Result: FAILED\n");
        } else {
            text.push_str("Result: PASSED\n");
        }
        text.push_str(
            "═══════════════════════════════════════════════════════════════════════════════\n",
        );

        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpuemu_common::types::{FailureKind, ValidationFailure};

    fn make_pass_result() -> ValidationResult {
        ValidationResult {
            passed: true,
            seed: 12345,
            op_name: "test_op".to_string(),
            max_diff: 0.0001,
            max_rel_diff: 0.00001,
            failures: vec![],
            timestamp: 1000000,
            duration_ms: 150,
            repro_info: None,
            error_stats: None,
        }
    }

    fn make_fail_result() -> ValidationResult {
        ValidationResult {
            passed: false,
            seed: 67890,
            op_name: "failing_op".to_string(),
            max_diff: 0.05,
            max_rel_diff: 0.1,
            failures: vec![ValidationFailure {
                kind: FailureKind::ToleranceExceeded,
                message: "Max diff 0.05 exceeds tolerance 0.001".to_string(),
                index: Some(42),
                expected: Some(1.0),
                actual: Some(1.05),
            }],
            timestamp: 1000001,
            duration_ms: 200,
            repro_info: None,
            error_stats: None,
        }
    }

    #[test]
    fn test_junit_xml_generation() {
        let summary = CiRunSummary {
            total_tests: 2,
            passed: 1,
            failed: 1,
            skipped: 0,
            duration_ms: 350,
            timestamp: 1000000,
            validation_results: vec![make_pass_result(), make_fail_result()],
            lint_results: vec![],
            artifact_diffs: None,
        };

        let xml = JunitReport::from_summary(&summary);
        assert!(xml.contains("<?xml version=\"1.0\""));
        assert!(xml.contains("<testsuites"));
        assert!(xml.contains("tests=\"2\""));
        assert!(xml.contains("failures=\"1\""));
        assert!(xml.contains("<testcase name=\"test_op\""));
        assert!(xml.contains("<testcase name=\"failing_op\""));
        assert!(xml.contains("<failure type=\"ToleranceExceeded\""));
    }

    #[test]
    fn test_json_generation() {
        let summary = CiRunSummary {
            total_tests: 1,
            passed: 1,
            failed: 0,
            skipped: 0,
            duration_ms: 150,
            timestamp: 1000000,
            validation_results: vec![make_pass_result()],
            lint_results: vec![],
            artifact_diffs: None,
        };

        let json = JsonReport::from_summary(&summary);
        assert!(json.contains("\"total_tests\": 1"));
        assert!(json.contains("\"passed\": 1"));
        assert!(json.contains("\"test_op\""));
    }

    #[test]
    fn test_text_generation() {
        let summary = CiRunSummary {
            total_tests: 2,
            passed: 1,
            failed: 1,
            skipped: 0,
            duration_ms: 350,
            timestamp: 1000000,
            validation_results: vec![make_pass_result(), make_fail_result()],
            lint_results: vec![],
            artifact_diffs: None,
        };

        let text = TextReport::from_summary(&summary);
        assert!(text.contains("gpuemu CI Report"));
        assert!(text.contains("Total Tests:  2"));
        assert!(text.contains("Passed:       1"));
        assert!(text.contains("Failed:       1"));
        assert!(text.contains("[PASS] test_op"));
        assert!(text.contains("[FAIL] failing_op"));
        assert!(text.contains("Result: FAILED"));
    }

    #[test]
    fn test_xml_escape() {
        assert_eq!(xml_escape("<test>"), "&lt;test&gt;");
        assert_eq!(xml_escape("a & b"), "a &amp; b");
        assert_eq!(xml_escape("\"quoted\""), "&quot;quoted&quot;");
    }
}
