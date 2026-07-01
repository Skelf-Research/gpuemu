//! Signed Kernel Correctness Report (HTML + ed25519 signature).
//!
//! Customer-facing artefact for the inference-as-a-service ICP. The vendor runs
//! `gpuemu report --signed --format html --output report.html`; the file
//! contains a per-op pass matrix, the four P1-P4 finding summaries, the gpuemu
//! version and corpus version (when known), and a footer with:
//!
//!   - SHA-256 of the unsigned report body (the HTML up to and including the
//!     opening `<!-- gpuemu-signature -->` marker).
//!   - The ed25519 signature of that SHA-256 (base64-encoded).
//!   - The signing public key's SHA-256 fingerprint (so the customer can
//!     pin it).
//!
//! The signing keypair lives in ``~/.gpuemu/sign-ed25519.{sec,pub}``. The
//! private key is PKCS#8 PEM (PEM-encoded for openssl interop) and is created
//! with 0600 permissions if absent. The public key is PEM-encoded; the vendor
//! shares it once and the customer verifies offline with:
//!
//!   openssl pkeyutl -verify -pubin -inkey gpuemu.pub \
//!       -in report.sha256 -sigfile report.sig
//!
//! ed25519 is the same signature primitive Sigstore, SSH, and GitHub-Actions
//! OIDC use; openssl 3.x supports it out of the box.

use anyhow::{Context, Result};
use base64::Engine;
use ed25519_dalek::pkcs8::spki::EncodePublicKey;
use ed25519_dalek::pkcs8::DecodePublicKey;
use ed25519_dalek::pkcs8::{DecodePrivateKey, EncodePrivateKey};
use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey};
use gpuemu_common::types::{CiRunSummary, ValidationResult};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// The marker the signature block is written immediately after. SHA-256 is
/// computed over everything up to and including this marker, exclusive of the
/// `<div class="signature">…</div>` that follows. Customers can re-compute the
/// SHA-256 with `sed '/<!-- gpuemu-signature -->/q' report.html | sha256sum`.
const SIGNATURE_MARKER: &str = "<!-- gpuemu-signature -->";

/// Default keypair location: ~/.gpuemu/sign-ed25519.{sec,pub}.
fn keypair_paths() -> Result<(PathBuf, PathBuf)> {
    let base = dirs::home_dir()
        .context("could not determine home directory for ~/.gpuemu/")?
        .join(".gpuemu");
    std::fs::create_dir_all(&base).with_context(|| format!("creating {}", base.display()))?;
    Ok((base.join("sign-ed25519.sec"), base.join("sign-ed25519.pub")))
}

/// Load the ed25519 keypair from ~/.gpuemu/, generating + persisting it if
/// absent. The private key is PKCS#8 PEM with 0600 perms; the public key is PEM.
pub fn load_or_generate_keypair() -> Result<SigningKey> {
    let (sec_path, pub_path) = keypair_paths()?;

    if sec_path.exists() {
        let pem = std::fs::read_to_string(&sec_path)
            .with_context(|| format!("reading {}", sec_path.display()))?;
        return SigningKey::from_pkcs8_pem(&pem)
            .map_err(|e| anyhow::anyhow!("failed to decode {}: {}", sec_path.display(), e));
    }

    // Generate fresh keypair. ed25519-dalek's `rand_core` feature lets us
    // generate from the system RNG without pulling in `rand`.
    use rand_core::RngCore;
    let mut seed = [0u8; 32];
    rand_core::OsRng.fill_bytes(&mut seed);
    let signing = SigningKey::from_bytes(&seed);
    let pem = signing
        .to_pkcs8_pem(Default::default())
        .map_err(|e| anyhow::anyhow!("pkcs8 encode failed: {}", e))?;
    std::fs::write(&sec_path, pem.as_bytes())
        .with_context(|| format!("writing {}", sec_path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&sec_path, std::fs::Permissions::from_mode(0o600))?;
    }

    let verifying = signing.verifying_key();
    let pub_pem = verifying
        .to_public_key_pem(Default::default())
        .map_err(|e| anyhow::anyhow!("spki encode failed: {}", e))?;
    std::fs::write(&pub_path, pub_pem.as_bytes())
        .with_context(|| format!("writing {}", pub_path.display()))?;
    Ok(signing)
}

/// SHA-256 fingerprint of the public key bytes (raw, not PEM).
fn pubkey_fingerprint(key: &VerifyingKey) -> String {
    let bytes = key.to_bytes();
    let hash = Sha256::digest(bytes);
    hex(&hash)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Generate the unsigned HTML report body (everything up to the signature
/// marker). Tests verify against this so the signature flow stays
/// reproducible across builds.
pub fn render_html(summary: &CiRunSummary) -> String {
    let mut html = String::with_capacity(8 * 1024);
    let now = chrono::Utc::now().to_rfc3339();
    let version = env!("CARGO_PKG_VERSION");
    let status = if summary.has_failures() || summary.has_regressions() {
        ("FAILED", "#b91c1c")
    } else {
        ("PASSED", "#15803d")
    };

    html.push_str(&format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>gpuemu Kernel Correctness Report — {}</title>
<style>
:root {{ font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; line-height: 1.5; color: #111827; max-width: 960px; margin: 2em auto; padding: 0 1em; }}
h1 {{ font-size: 1.5em; border-bottom: 2px solid #111827; padding-bottom: 0.3em; }}
h2 {{ font-size: 1.15em; margin-top: 2em; }}
.status {{ display: inline-block; padding: 0.15em 0.7em; border-radius: 999px; color: white; font-weight: 600; background: {}; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.9em; }}
th, td {{ border-bottom: 1px solid #d1d5db; padding: 0.4em 0.6em; text-align: left; }}
th {{ background: #f3f4f6; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.findings {{ background: #f9fafb; border-left: 4px solid #6366f1; padding: 0.8em 1em; margin: 1em 0; border-radius: 3px; }}
.findings li {{ margin: 0.3em 0; }}
.signature {{ background: #fff7ed; border: 1px solid #fed7aa; padding: 0.8em 1em; margin: 2em 0 0; border-radius: 3px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.85em; word-break: break-all; }}
.signature .label {{ font-weight: 600; color: #9a3412; }}
.muted {{ color: #6b7280; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>gpuemu Kernel Correctness Report</h1>
<p>
  <span class="status">{}</span>
  &nbsp;Generated: <code>{}</code>
  &nbsp;Tool: <code>gpuemu v{}</code>
</p>

<h2>Summary</h2>
<table>
<tr><th>Total ops</th><td class="num">{}</td></tr>
<tr><th>Passed</th><td class="num">{}</td></tr>
<tr><th>Failed</th><td class="num">{}</td></tr>
<tr><th>Skipped</th><td class="num">{}</td></tr>
<tr><th>Duration</th><td class="num">{:.2}s</td></tr>
</table>

<h2>Per-op verdicts</h2>
<table>
<thead><tr>
  <th>Op</th><th>Verdict</th>
  <th class="num">Iters</th>
  <th class="num">Max abs diff</th>
  <th class="num">Max rel diff</th>
  <th>Seed (first failure)</th>
</tr></thead>
<tbody>
"#,
        now,
        status.1,
        status.0,
        now,
        version,
        summary.total_tests,
        summary.passed,
        summary.failed,
        summary.skipped,
        summary.duration_ms as f64 / 1000.0,
    ));

    // Aggregate by op_name.
    use std::collections::BTreeMap;
    let mut by_op: BTreeMap<&str, Vec<&ValidationResult>> = BTreeMap::new();
    for r in &summary.validation_results {
        by_op.entry(r.op_name.as_str()).or_default().push(r);
    }
    for (op, rs) in &by_op {
        let total = rs.len();
        let failed = rs.iter().filter(|r| !r.passed).count();
        let verdict = if failed == 0 {
            r#"<span style="color: #15803d; font-weight: 600;">PASS</span>"#
        } else {
            r#"<span style="color: #b91c1c; font-weight: 600;">FAIL</span>"#
        };
        let max_abs = rs.iter().map(|r| r.max_diff).fold(0f64, f64::max);
        let max_rel = rs.iter().map(|r| r.max_rel_diff).fold(0f64, f64::max);
        let first_fail_seed = rs
            .iter()
            .find(|r| !r.passed)
            .map(|r| r.seed.to_string())
            .unwrap_or_else(|| "—".to_string());
        html.push_str(&format!(
            "<tr><td><code>{}</code></td><td>{}</td><td class=\"num\">{}/{}</td>\
             <td class=\"num\">{:.4e}</td><td class=\"num\">{:.4e}</td>\
             <td><code>{}</code></td></tr>\n",
            op,
            verdict,
            total - failed,
            total,
            max_abs,
            max_rel,
            first_fail_seed,
        ));
    }

    html.push_str("</tbody>\n</table>\n");

    html.push_str(
        r#"
<h2>Validated against the gpuemu research backing</h2>
<div class="findings">
<p>Each default in this validation regime is anchored to a measured study
(see <a href="https://docs.skelfresearch.com/gpuemu/why-gpuemu/the-evidence">the evidence</a>).</p>
<ul>
  <li><strong>P1 — Correctness illusion:</strong> the gpuemu corpus oracle catches 9 / 9 LLM-style buggy kernels with 0 false positives on 15 controls, measured across RTX 3060, A10, L40S, A100 SXM4, H100 NVL.</li>
  <li><strong>P2 — Tolerance calibration:</strong> p95-of-controls × 1.5 envelopes raise kernel-bug recall from 65 % (fixed atol/rtol) to 82 %.</li>
  <li><strong>P3 — Test-input generation:</strong> adversarial value distribution scores 99 % bug recall vs 64 % for the field-standard regular-shape default.</li>
  <li><strong>P4 — Static PTX gating:</strong> Δregs / Δinstrs are architecture-independent across H100 / A100 / L40S / A10 / RTX 3060 for the same compiled PTX, so the structural-bug gate is portable; semantic bugs (identical PTX) need the fp64 oracle above.</li>
</ul>
</div>
"#,
    );

    html.push_str(SIGNATURE_MARKER);
    html.push('\n');
    html
}

/// Append the signature block to a rendered HTML body. Computes SHA-256 of
/// the body (which already ends with `SIGNATURE_MARKER`), signs that hash
/// with the user's ed25519 key, and emits the signature block. The customer
/// can re-derive the SHA-256 and verify offline.
pub fn sign_html(unsigned: &str, key: &SigningKey) -> Result<String> {
    let digest = Sha256::digest(unsigned.as_bytes());
    let digest_hex = hex(&digest);
    let signature = key.sign(&digest);
    let sig_b64 = base64::engine::general_purpose::STANDARD.encode(signature.to_bytes());
    let verifying = key.verifying_key();
    let fp = pubkey_fingerprint(&verifying);

    let mut signed = String::with_capacity(unsigned.len() + 1024);
    signed.push_str(unsigned);
    signed.push_str(&format!(
        r#"<div class="signature">
<p><span class="label">SHA-256 of report body:</span><br><code>{digest_hex}</code></p>
<p><span class="label">ed25519 signature (base64):</span><br><code>{sig_b64}</code></p>
<p><span class="label">Signing key SHA-256 fingerprint:</span><br><code>{fp}</code></p>
<p class="muted">Verify offline with:
<code>sed '/{marker}/q' report.html | sha256sum</code> — must equal the SHA-256 above.
Then <code>openssl pkeyutl -verify -pubin -inkey gpuemu.pub -in report.sha256 -sigfile report.sig</code>
where <code>report.sha256</code> is the raw 32-byte digest and <code>report.sig</code> is the base64-decoded signature.</p>
</div>
</body>
</html>
"#,
        marker = SIGNATURE_MARKER,
    ));
    Ok(signed)
}

/// Verify a signed HTML report against a public-key PEM file. Returns the
/// digest hex if the signature is valid, otherwise an error.
#[allow(dead_code)]
pub fn verify_html(signed: &str, pubkey_pem_path: &Path) -> Result<String> {
    let body_end = signed
        .find(SIGNATURE_MARKER)
        .context("no signature marker")?;
    let body = &signed[..body_end + SIGNATURE_MARKER.len()];
    let body_with_newline = format!("{}\n", body);
    let digest = Sha256::digest(body_with_newline.as_bytes());
    let digest_hex = hex(&digest);

    // Extract the signature line from the signed footer.
    let sig_b64 = signed
        .lines()
        .find(|l| l.contains("base64") && !l.contains("label"))
        .and_then(|_| {
            let start = signed.find("ed25519 signature (base64):")?;
            let after = &signed[start..];
            let code_open = after.find("<code>")? + "<code>".len();
            let code_close = after[code_open..].find("</code>")?;
            Some(&after[code_open..code_open + code_close])
        })
        .context("no signature in report")?;

    let sig_bytes = base64::engine::general_purpose::STANDARD
        .decode(sig_b64)
        .context("base64 decode failed")?;
    if sig_bytes.len() != 64 {
        anyhow::bail!(
            "expected 64-byte ed25519 signature, got {}",
            sig_bytes.len()
        );
    }
    let mut sig_arr = [0u8; 64];
    sig_arr.copy_from_slice(&sig_bytes);
    let signature = ed25519_dalek::Signature::from_bytes(&sig_arr);

    let pubkey_pem = std::fs::read_to_string(pubkey_pem_path)?;
    let verifying = VerifyingKey::from_public_key_pem(&pubkey_pem)
        .map_err(|e| anyhow::anyhow!("public key decode failed: {}", e))?;
    verifying
        .verify(&digest, &signature)
        .map_err(|e| anyhow::anyhow!("signature verification failed: {}", e))?;
    Ok(digest_hex)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpuemu_common::types::{FailureKind, ValidationFailure};

    fn sample_summary() -> CiRunSummary {
        CiRunSummary {
            total_tests: 2,
            passed: 1,
            failed: 1,
            skipped: 0,
            duration_ms: 350,
            timestamp: 1000000,
            validation_results: vec![
                ValidationResult {
                    passed: true,
                    seed: 1,
                    op_name: "softmax".into(),
                    max_diff: 0.0001,
                    max_rel_diff: 0.0,
                    failures: vec![],
                    timestamp: 1000000,
                    duration_ms: 150,
                    repro_info: None,
                    error_stats: None,
                },
                ValidationResult {
                    passed: false,
                    seed: 67890,
                    op_name: "matmul".into(),
                    max_diff: 0.05,
                    max_rel_diff: 0.1,
                    failures: vec![ValidationFailure {
                        kind: FailureKind::ToleranceExceeded,
                        message: "Max diff 0.05 exceeds tolerance 0.001".into(),
                        index: Some(42),
                        expected: Some(1.0),
                        actual: Some(1.05),
                    }],
                    timestamp: 1000001,
                    duration_ms: 200,
                    repro_info: None,
                    error_stats: None,
                },
            ],
            lint_results: vec![],
            artifact_diffs: None,
        }
    }

    #[test]
    fn renders_html_with_signature_marker() {
        let s = sample_summary();
        let html = render_html(&s);
        assert!(html.contains("Kernel Correctness Report"));
        assert!(html.contains("FAILED"));
        assert!(html.contains("<code>softmax</code>"));
        assert!(html.contains("<code>matmul</code>"));
        assert!(html.contains("<code>67890</code>"));
        assert!(html.contains(SIGNATURE_MARKER));
    }

    #[test]
    fn sign_and_verify_round_trip() {
        // Use a deterministic seed for the test; in production keys come from
        // ~/.gpuemu/.
        let seed = [42u8; 32];
        let signing = SigningKey::from_bytes(&seed);
        let s = sample_summary();
        let unsigned = render_html(&s);
        let signed = sign_html(&unsigned, &signing).expect("sign should succeed");

        // The signature block is present.
        assert!(signed.contains("ed25519 signature (base64)"));
        assert!(signed.contains("Signing key SHA-256 fingerprint"));

        // Persist the public key so verify_html can read it.
        let tmp = tempfile::tempdir().unwrap();
        let pub_path = tmp.path().join("gpuemu.pub");
        let pub_pem = signing
            .verifying_key()
            .to_public_key_pem(Default::default())
            .unwrap();
        std::fs::write(&pub_path, pub_pem).unwrap();

        let digest = verify_html(&signed, &pub_path).expect("verify should succeed");
        assert_eq!(digest.len(), 64); // 32 bytes hex
    }

    #[test]
    fn tampered_body_fails_verification() {
        let seed = [42u8; 32];
        let signing = SigningKey::from_bytes(&seed);
        let s = sample_summary();
        let unsigned = render_html(&s);
        let signed = sign_html(&unsigned, &signing).unwrap();

        // Flip a byte in the body before the signature marker.
        let i = signed.find("matmul").unwrap();
        let mut tampered = signed.clone();
        tampered.replace_range(i..i + 6, "MATMUL");

        let tmp = tempfile::tempdir().unwrap();
        let pub_path = tmp.path().join("gpuemu.pub");
        let pub_pem = signing
            .verifying_key()
            .to_public_key_pem(Default::default())
            .unwrap();
        std::fs::write(&pub_path, pub_pem).unwrap();

        let err = verify_html(&tampered, &pub_path).unwrap_err();
        assert!(format!("{}", err).contains("signature verification failed"));
    }
}
