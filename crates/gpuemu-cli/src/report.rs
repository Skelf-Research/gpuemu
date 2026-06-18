//! Report generators for CI integration.
//!
//! Supports plain text, JSON, JUnit XML, SARIF 2.1.0, and PR-comment Markdown
//! output formats. SARIF and PR-comment are the surfaces GitHub Code Scanning and
//! the gpuemu validate-action consume; see `documentation/docs/who-uses-gpuemu/`
//! for the workflow.

use gpuemu_common::types::{CiRunSummary, LintResult, ValidationResult};

/// Output format for reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Text,
    Json,
    Junit,
    Sarif,
    PrComment,
}

impl OutputFormat {
    /// Parse output format from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "text" => Some(OutputFormat::Text),
            "json" => Some(OutputFormat::Json),
            "junit" | "xml" => Some(OutputFormat::Junit),
            "sarif" => Some(OutputFormat::Sarif),
            "pr-comment" | "pr_comment" | "prcomment" => Some(OutputFormat::PrComment),
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
        OutputFormat::Sarif => SarifReport::from_summary(summary),
        OutputFormat::PrComment => PrCommentReport::from_summary(summary),
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

// =============================================================================
// SARIF 2.1.0 Report
// =============================================================================
//
// SARIF (Static Analysis Results Interchange Format) is the format GitHub Code
// Scanning, Sentry, and most code-quality dashboards consume. Spec:
// https://docs.oasis-open.org/sarif/sarif/v2.1.0/os/sarif-v2.1.0-os.html
//
// Each gpuemu ValidationFailure becomes one SARIF result with:
//   - ruleId = "validation/<failure_kind>"  (e.g. "validation/ToleranceExceeded")
//   - level  = "error"
//   - message.text = human-readable summary including seed + max_diff
//   - locations[0].logicalLocations[0].name = op_name (no source range — gpuemu
//     doesn't know the kernel's source line yet; consumers display the op as a
//     "function" location which is the SARIF logical-location idiom)
//   - partialFingerprints.kindAndOp = "<kind>:<op_name>"  (lets GitHub dedupe the
//     same finding across PR re-runs)
//
// Each gpuemu LintResult violation becomes one SARIF result with:
//   - ruleId = "lint/<violation_kind>"
//   - level  = "error"  (lint violations are gating; warn vs error is a future
//     refinement once we add ArtifactCheckConfig severity levels)
//   - message includes the offending metric value

/// SARIF 2.1.0 report generator.
pub struct SarifReport;

impl SarifReport {
    /// Convert CiRunSummary to a SARIF 2.1.0 JSON document.
    pub fn from_summary(summary: &CiRunSummary) -> String {
        let version = env!("CARGO_PKG_VERSION");

        // Collect rules used in this run (deduplicated by ruleId) so SARIF
        // consumers can group results.
        let mut rule_ids: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut results = Vec::<serde_json::Value>::new();

        for r in &summary.validation_results {
            if r.passed {
                continue;
            }
            for f in &r.failures {
                let rule_id = format!("validation/{:?}", f.kind);
                rule_ids.insert(rule_id.clone());
                let text = format!(
                    "{} (op: {}, seed: {}, max_diff: {}). Reproduce with `gpuemu reproduce {}`.",
                    f.message, r.op_name, r.seed, r.max_diff, r.seed
                );
                results.push(serde_json::json!({
                    "ruleId": rule_id,
                    "level": "error",
                    "message": { "text": text },
                    "locations": [{
                        "logicalLocations": [{
                            "name": r.op_name,
                            "kind": "function"
                        }]
                    }],
                    "partialFingerprints": {
                        "kindAndOp": format!("{:?}:{}", f.kind, r.op_name)
                    }
                }));
            }
        }

        for r in &summary.lint_results {
            if r.passed {
                continue;
            }
            for v in &r.violations {
                let rule_id = format!("lint/{:?}", v.kind);
                rule_ids.insert(rule_id.clone());
                let text = format!(
                    "{} (kernel: {}, regs: {}, spills: {}).",
                    v.message,
                    r.kernel_name,
                    r.metrics.register_count,
                    r.metrics.spill_count,
                );
                results.push(serde_json::json!({
                    "ruleId": rule_id,
                    "level": "error",
                    "message": { "text": text },
                    "locations": [{
                        "logicalLocations": [{
                            "name": r.kernel_name,
                            "kind": "function"
                        }]
                    }],
                    "partialFingerprints": {
                        "kindAndKernel": format!("{:?}:{}", v.kind, r.kernel_name)
                    }
                }));
            }
        }

        // Artifact-baseline regressions become "lint/Regression" results so the
        // PR is gated on them through the same SARIF pipeline.
        if let Some(diffs) = &summary.artifact_diffs {
            for d in &diffs.diffs {
                if !d.is_regression {
                    continue;
                }
                let rule_id = "lint/Regression".to_string();
                rule_ids.insert(rule_id.clone());
                let text = format!(
                    "Artifact regression on kernel {} vs baseline `{}` (Δregs={}, \
                     Δspills={}, Δinstrs={}).",
                    d.kernel_name,
                    diffs.baseline_tag,
                    d.register_delta,
                    d.spill_delta,
                    d.instruction_delta,
                );
                results.push(serde_json::json!({
                    "ruleId": rule_id,
                    "level": "error",
                    "message": { "text": text },
                    "locations": [{
                        "logicalLocations": [{
                            "name": d.kernel_name,
                            "kind": "function"
                        }]
                    }],
                    "partialFingerprints": {
                        "kernelAndBaseline": format!("{}:{}", d.kernel_name, diffs.baseline_tag)
                    }
                }));
            }
        }

        let rules: Vec<serde_json::Value> = rule_ids
            .iter()
            .map(|id| {
                serde_json::json!({
                    "id": id,
                    "name": id,
                    "shortDescription": { "text": id },
                    "helpUri": "https://docs.skelfresearch.com/gpuemu/why-gpuemu/the-problem"
                })
            })
            .collect();

        let doc = serde_json::json!({
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "version": "2.1.0",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "gpuemu",
                        "version": version,
                        "informationUri": "https://docs.skelfresearch.com/gpuemu",
                        "rules": rules
                    }
                },
                "results": results
            }]
        });

        serde_json::to_string_pretty(&doc).unwrap_or_else(|_| "{}".to_string())
    }
}

// =============================================================================
// PR-comment (Markdown) Report
// =============================================================================
//
// The output the `gpuemu/validate-action` GitHub Action posts as a single PR
// comment via `gh pr comment`. Reads cleanly on github.com; each failure has a
// one-line `gpuemu reproduce <seed>` link the reviewer can paste locally.

/// PR-comment Markdown report generator.
pub struct PrCommentReport;

impl PrCommentReport {
    /// Convert CiRunSummary to a Markdown payload suitable for `gh pr comment -F -`.
    pub fn from_summary(summary: &CiRunSummary) -> String {
        let mut md = String::new();
        let icon = if summary.has_failures() || summary.has_regressions() {
            "❌"
        } else {
            "✅"
        };
        md.push_str(&format!(
            "## {} gpuemu correctness report\n\n",
            icon
        ));
        md.push_str(&format!(
            "**{} passed**, **{} failed**, **{} skipped** ({:.2}s total).\n\n",
            summary.passed,
            summary.failed,
            summary.skipped,
            summary.duration_ms as f64 / 1000.0
        ));

        // Validation failures.
        let val_fails: Vec<&ValidationResult> = summary
            .validation_results
            .iter()
            .filter(|r| !r.passed)
            .collect();
        if !val_fails.is_empty() {
            md.push_str("### Validation failures\n\n");
            md.push_str("| op | failure | max_diff | seed | replay |\n");
            md.push_str("|---|---|---:|---:|---|\n");
            for r in &val_fails {
                let kind = r
                    .failures
                    .first()
                    .map(|f| format!("{:?}", f.kind))
                    .unwrap_or_else(|| "Unknown".to_string());
                md.push_str(&format!(
                    "| `{}` | {} | {:.4e} | `{}` | `gpuemu reproduce {}` |\n",
                    r.op_name, kind, r.max_diff, r.seed, r.seed
                ));
            }
            md.push('\n');
        }

        // Lint violations.
        let lint_fails: Vec<&LintResult> = summary
            .lint_results
            .iter()
            .filter(|r| !r.passed)
            .collect();
        if !lint_fails.is_empty() {
            md.push_str("### Static-PTX lint violations\n\n");
            md.push_str("| kernel | violation | regs | spills | local | instrs |\n");
            md.push_str("|---|---|---:|---:|---:|---:|\n");
            for r in &lint_fails {
                let kind = r
                    .violations
                    .first()
                    .map(|v| format!("{:?}", v.kind))
                    .unwrap_or_else(|| "Unknown".to_string());
                md.push_str(&format!(
                    "| `{}` | {} | {} | {} | {} | {} |\n",
                    r.kernel_name,
                    kind,
                    r.metrics.register_count,
                    r.metrics.spill_count,
                    r.metrics.local_memory_bytes,
                    r.metrics.instruction_count,
                ));
            }
            md.push('\n');
        }

        // Artifact regressions.
        if let Some(diffs) = &summary.artifact_diffs {
            let regs: Vec<_> = diffs.diffs.iter().filter(|d| d.is_regression).collect();
            if !regs.is_empty() {
                md.push_str(&format!(
                    "### Artifact regressions vs baseline `{}`\n\n",
                    diffs.baseline_tag
                ));
                md.push_str("| kernel | Δregs | Δspills | Δlocal | Δinstrs |\n");
                md.push_str("|---|---:|---:|---:|---:|\n");
                for d in &regs {
                    md.push_str(&format!(
                        "| `{}` | {:+} | {:+} | {:+} | {:+} |\n",
                        d.kernel_name,
                        d.register_delta,
                        d.spill_delta,
                        d.local_memory_delta,
                        d.instruction_delta,
                    ));
                }
                md.push('\n');
            }
        }

        if summary.has_failures() || summary.has_regressions() {
            md.push_str(
                "\n*Every flagged failure ships a seed for byte-for-byte replay. \
                 Run `gpuemu reproduce <seed>` locally to debug.*\n",
            );
        } else {
            md.push_str(
                "\n*All ops pass the schema-aware fuzz + fp64-reference oracle. \
                 See [the evidence](https://docs.skelfresearch.com/gpuemu/why-gpuemu/the-evidence) \
                 for what this validates.*\n",
            );
        }

        md
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

    #[test]
    fn test_output_format_parses_sarif_and_pr_comment() {
        assert_eq!(OutputFormat::from_str("sarif"), Some(OutputFormat::Sarif));
        assert_eq!(OutputFormat::from_str("SARIF"), Some(OutputFormat::Sarif));
        assert_eq!(
            OutputFormat::from_str("pr-comment"),
            Some(OutputFormat::PrComment)
        );
        assert_eq!(
            OutputFormat::from_str("pr_comment"),
            Some(OutputFormat::PrComment)
        );
        assert_eq!(
            OutputFormat::from_str("prcomment"),
            Some(OutputFormat::PrComment)
        );
        assert_eq!(OutputFormat::from_str("unknown"), None);
    }

    #[test]
    fn test_sarif_generation_shape() {
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
        let s = SarifReport::from_summary(&summary);
        let v: serde_json::Value = serde_json::from_str(&s).expect("sarif must be valid JSON");
        assert_eq!(v["version"], "2.1.0");
        assert!(v["$schema"].as_str().unwrap().contains("sarif-2.1.0"));
        let run = &v["runs"][0];
        assert_eq!(run["tool"]["driver"]["name"], "gpuemu");
        let results = run["results"].as_array().unwrap();
        // One failing validation result with one ValidationFailure → one SARIF result.
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["ruleId"], "validation/ToleranceExceeded");
        assert_eq!(results[0]["level"], "error");
        assert!(results[0]["message"]["text"]
            .as_str()
            .unwrap()
            .contains("gpuemu reproduce 67890"));
        assert_eq!(
            results[0]["locations"][0]["logicalLocations"][0]["name"],
            "failing_op"
        );
        // Rule is registered on the driver.
        let rules = run["tool"]["driver"]["rules"].as_array().unwrap();
        assert!(rules.iter().any(|r| r["id"] == "validation/ToleranceExceeded"));
    }

    #[test]
    fn test_sarif_clean_run_has_no_results() {
        let summary = CiRunSummary {
            total_tests: 1,
            passed: 1,
            failed: 0,
            skipped: 0,
            duration_ms: 100,
            timestamp: 1000000,
            validation_results: vec![make_pass_result()],
            lint_results: vec![],
            artifact_diffs: None,
        };
        let s = SarifReport::from_summary(&summary);
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["runs"][0]["results"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_pr_comment_failed_summary() {
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
        let md = PrCommentReport::from_summary(&summary);
        assert!(md.starts_with("## ❌ gpuemu correctness report"));
        assert!(md.contains("**1 passed**, **1 failed**"));
        assert!(md.contains("`failing_op`"));
        assert!(md.contains("`gpuemu reproduce 67890`"));
        // PR-comment is a single markdown payload — `gh pr comment -F -` reads it
        // as one body; size sanity-check.
        assert!(md.len() < 8000);
    }

    #[test]
    fn test_pr_comment_clean_run() {
        let summary = CiRunSummary {
            total_tests: 1,
            passed: 1,
            failed: 0,
            skipped: 0,
            duration_ms: 100,
            timestamp: 1000000,
            validation_results: vec![make_pass_result()],
            lint_results: vec![],
            artifact_diffs: None,
        };
        let md = PrCommentReport::from_summary(&summary);
        assert!(md.starts_with("## ✅ gpuemu correctness report"));
        assert!(md.contains("**1 passed**, **0 failed**"));
        assert!(!md.contains("### Validation failures"));
        assert!(md.contains("docs.skelfresearch.com/gpuemu/why-gpuemu/the-evidence"));
    }
}
