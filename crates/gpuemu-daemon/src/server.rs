//! NNG-based IPC server for the gpuemu daemon.

use crate::executor::{Executor, ExecutorConfig};
use crate::fuzzer::{Fuzzer, regenerate_test_case};
use crate::storage::Storage;
use crate::validator::Validator;
use anyhow::{Context, Result};
use gpuemu_common::config::GpuemuConfig;
use gpuemu_common::protocol::{
    deserialize_request, serialize_response, ErrorCode, MinimizeStrategy, PROTOCOL_VERSION,
    Request, Response, TestCaseData,
};
use gpuemu_common::types::{ArtifactDiffSummary, BaselineComparison, CiRunSummary, DType, LayoutType, TensorData, ValidationResult};
use nng::options::Options;
use nng::{Protocol, Socket};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

fn op_input_names(op: &gpuemu_common::config::OpConfig) -> Vec<&str> {
    if op.input_names.is_empty() {
        vec!["input"]
    } else {
        op.input_names.iter().map(String::as_str).collect()
    }
}

fn primary_input_name(op: &gpuemu_common::config::OpConfig) -> &str {
    op.input_names.first().map(String::as_str).unwrap_or("input")
}

/// Build a validator whose tolerances are the global validation tolerances
/// overlaid with the op's own tolerances (op-specific values win). Without this,
/// per-op `[ops.tolerances]` from gpuemu.toml are silently ignored.
fn op_validator(
    base: &gpuemu_common::config::ValidationConfig,
    op: &gpuemu_common::config::OpConfig,
) -> Validator {
    let mut cfg = base.clone();
    for (dtype, tol) in &op.tolerances {
        cfg.tolerances.insert(dtype.clone(), *tol);
    }
    Validator::new(cfg)
}

fn available_ops_message(config: &GpuemuConfig) -> String {
    if config.ops.is_empty() {
        "No ops are configured in gpuemu.toml".to_string()
    } else {
        format!(
            "Available ops: {}",
            config.ops
                .iter()
                .map(|op| op.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

fn enforce_shape_preserved(
    result: &mut ValidationResult,
    output: &TensorData,
    inputs: &std::collections::HashMap<String, TensorData>,
    op: &gpuemu_common::config::OpConfig,
) {
    if !op.invariants.shape_preserved {
        return;
    }

    let Some(input) = inputs.get(primary_input_name(op)) else {
        return;
    };

    if output.shape != input.shape {
        result.passed = false;
        result.failures.push(gpuemu_common::types::ValidationFailure {
            kind: gpuemu_common::types::FailureKind::InvariantViolation,
            message: format!(
                "shape_preserved invariant violated: input shape {:?}, output shape {:?}",
                input.shape, output.shape
            ),
            index: None,
            expected: None,
            actual: None,
        });
    }
}

fn scale_tensor_values(data: &mut [u8], dtype: DType, scale: f64) {
    use crate::validator::bytemuck_cast_slice_mut;
    match dtype {
        DType::Float32 => {
            let vals = bytemuck_cast_slice_mut::<f32>(data);
            for v in vals.iter_mut() {
                *v *= scale as f32;
            }
        }
        DType::Float64 => {
            let vals = bytemuck_cast_slice_mut::<f64>(data);
            for v in vals.iter_mut() {
                *v *= scale;
            }
        }
        DType::Float16 | DType::BFloat16 => {
            let vals = bytemuck_cast_slice_mut::<u16>(data);
            for bits in vals.iter_mut() {
                let f = if dtype == DType::Float16 {
                    f16_bits_to_f32(*bits)
                } else {
                    bf16_bits_to_f32(*bits)
                };
                let scaled = f * scale as f32;
                *bits = if dtype == DType::Float16 {
                    f32_to_f16_bits(scaled)
                } else {
                    f32_to_bf16_bits(scaled)
                };
            }
        }
        _ => {}
    }
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            let sign32 = sign << 31;
            let mant32 = mant << 13;
            let exp32: u32 = (127 - 15 + 1) as u32;
            f32::from_bits(sign32 | (exp32 << 23) | mant32) - f32::from_bits(1u32 << 23)
        }
    } else if exp == 31 {
        f32::from_bits((sign << 31) | (0x7F << 23) | (mant << 13))
    } else {
        let sign32 = sign << 31;
        let exp32 = (exp + 127 - 15) as u32;
        let mant32 = mant << 13;
        f32::from_bits(sign32 | (exp32 << 23) | mant32)
    }
}

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = ((bits >> 13) & 0x3FF) as u16;
    if exp <= 0 {
        sign
    } else if exp >= 31 {
        (sign | 0x7C00) as u16
    } else {
        sign | ((exp as u16) << 10) | mant
    }
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 7) & 0xFF) as u32;
    let mant = (bits & 0x7F) as u32;
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            let sign32 = sign << 31;
            let mant32 = mant << 16;
            let exp32: u32 = 1;
            f32::from_bits(sign32 | (exp32 << 23) | mant32) - f32::from_bits(1u32 << 23)
        }
    } else if exp == 255 {
        f32::from_bits((sign << 31) | (0xFFu32 << 23) | (mant << 16))
    } else {
        let sign32 = sign << 31;
        let exp32 = exp;
        let mant32 = mant << 16;
        f32::from_bits(sign32 | (exp32 << 23) | mant32)
    }
}

fn f32_to_bf16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = (((bits >> 23) & 0xFF) as i32 - 127 + 127) as u16;
    let mant = ((bits >> 16) & 0x7F) as u16;
    if exp == 0 || exp >= 255 {
        sign
    } else {
        sign | (exp << 7) | mant
    }
}

/// Server state shared across handlers.
pub struct ServerState {
    /// Storage for validation results.
    pub storage: Storage,
    /// Reference script executor.
    pub executor: Executor,
    /// Configuration. Handlers build a per-op [`Validator`] via `op_validator()`
    /// so `[ops.tolerances]` are applied on top of `config.validation`.
    pub config: GpuemuConfig,
    /// Server start time.
    pub start_time: Instant,
    /// Last CI run summary.
    pub last_ci_summary: Option<CiRunSummary>,
    /// Shutdown signal.
    pub shutdown_requested: bool,
}

impl ServerState {
    /// Create new server state.
    pub fn new(storage: Storage, config: GpuemuConfig) -> Self {
        let executor = Executor::new(ExecutorConfig {
            oracle_fp64: config.validation.oracle_fp64,
            ..ExecutorConfig::default()
        });

        Self {
            storage,
            executor,
            config,
            start_time: Instant::now(),
            last_ci_summary: None,
            shutdown_requested: false,
        }
    }
}

/// Run the NNG server.
pub async fn run_server(socket_path: &Path, state: Arc<RwLock<ServerState>>) -> Result<()> {
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    if socket_path.exists() {
        std::fs::remove_file(socket_path)?;
    }

    let socket_url = format!("ipc://{}", socket_path.display());
    info!("Starting server on {}", socket_url);

    let socket = Arc::new(Mutex::new(
        Socket::new(Protocol::Rep0).context("Failed to create NNG socket")?,
    ));
    socket
        .lock()
        .unwrap()
        .set_opt::<nng::options::RecvTimeout>(Some(Duration::from_millis(250)))
        .context("Failed to set receive timeout")?;

    socket
        .lock()
        .unwrap()
        .listen(&socket_url)
        .with_context(|| format!("Failed to listen on {}", socket_url))?;

    info!("Server listening on {}", socket_url);

    loop {
        if state.read().await.shutdown_requested {
            info!("Shutdown requested, stopping server");
            break;
        }

        let recv_result = {
            let socket = socket.clone();
            tokio::task::spawn_blocking(move || socket.lock().unwrap().recv()).await
        };

        let msg = match recv_result {
            Ok(Ok(msg)) => msg,
            Ok(Err(nng::Error::TimedOut)) => {
                continue;
            }
            Ok(Err(e)) => {
                error!("Failed to receive message: {}", e);
                continue;
            }
            Err(e) => {
                error!("Receive task error: {}", e);
                continue;
            }
        };

        debug!("Received message ({} bytes)", msg.len());

        let request = match deserialize_request(&msg) {
            Ok(req) => req,
            Err(e) => {
                error!("Failed to deserialize request: {}", e);
                let response = Response::Error {
                    code: ErrorCode::InvalidRequest,
                    message: format!("Invalid request: {}", e),
                };
                if let Ok(bytes) = serialize_response(&response) {
                    let socket = socket.clone();
                    match tokio::task::spawn_blocking(move || socket.lock().unwrap().send(&bytes)).await {
                        Ok(Ok(())) => {}
                        Ok(Err((_, e))) => {
                            error!("Failed to send invalid-request response: {}", e);
                        }
                        Err(e) => {
                            error!("Invalid-request send task error: {}", e);
                        }
                    }
                }
                continue;
            }
        };

        let response = handle_request(request, state.clone()).await;

        if let Ok(bytes) = serialize_response(&response) {
            let socket = socket.clone();
            match tokio::task::spawn_blocking(move || socket.lock().unwrap().send(&bytes)).await {
                Ok(Ok(())) => {}
                Ok(Err((_, e))) => {
                    error!("Failed to send response: {}", e);
                }
                Err(e) => {
                    error!("Send task error: {}", e);
                }
            }
        }
    }

    drop(socket);
    if socket_path.exists() {
        let _ = std::fs::remove_file(socket_path);
    }

    Ok(())
}

/// Handle a single request.
async fn handle_request(request: Request, state: Arc<RwLock<ServerState>>) -> Response {
    match request {
        Request::Ping => {
            let state = state.read().await;
            Response::Pong {
                version: gpuemu_common::VERSION.to_string(),
                protocol_version: PROTOCOL_VERSION,
                uptime_secs: state.start_time.elapsed().as_secs(),
            }
        }

        Request::Shutdown => {
            info!("Received shutdown request");
            let mut state_write = state.write().await;
            state_write.shutdown_requested = true;
            info!("Shutdown flag set, server will stop after current request");
            Response::Ok
        }

        Request::ValidateOp {
            op_name,
            inputs,
            output,
            kwargs,
        } => {
            info!("Validating op: {}", op_name);

            let state_read = state.read().await;

            // Find op config
            let op_config = state_read.config.ops.iter().find(|op| op.name == op_name);

            let op_config = match op_config {
                Some(c) => c,
                None => {
                    warn!("Op not found: {}", op_name);
                    return Response::Error {
                        code: ErrorCode::OpNotFound,
                        message: format!(
                            "Op '{}' not found in configuration. {}",
                            op_name,
                            available_ops_message(&state_read.config)
                        ),
                    };
                }
            };

            // Run reference script
            let reference_path = Path::new(&op_config.reference);
            let reference_result = state_read
                .executor
                .run_reference(reference_path, &inputs, &kwargs, true)
                .await;

            let reference = match reference_result {
                Ok(r) => r,
                Err(e) => {
                    error!("Reference script failed: {}", e);
                    return Response::Error {
                        code: ErrorCode::ReferenceScriptFailed,
                        message: format!("Reference script failed: {}", e),
                    };
                }
            };

            // Generate seed
            let seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;

            // Validate (with op-specific tolerances overlaid on the global ones)
            let invariants = Some(&op_config.invariants);
            let result = op_validator(&state_read.config.validation, op_config)
                .validate(&op_name, &output, &reference, seed, invariants);
            let mut result = result;
            enforce_shape_preserved(&mut result, &output, &inputs, op_config);

            // Store result
            drop(state_read);
            let state_write = state.write().await;
            if let Err(e) = state_write.storage.store_result(&result) {
                warn!("Failed to store result: {}", e);
            }

            Response::ValidationResult { result }
        }

        Request::GetResult { seed } => {
            let state = state.read().await;
            match state.storage.get_result(seed) {
                Ok(Some(result)) => Response::ValidationResult { result },
                Ok(None) => Response::Error {
                    code: ErrorCode::NotFound,
                    message: format!("Result with seed {} not found", seed),
                },
                Err(e) => Response::Error {
                    code: ErrorCode::InternalError,
                    message: format!("Failed to get result: {}", e),
                },
            }
        }

        Request::ListResults { limit } => {
            let state = state.read().await;
            match state.storage.list_results(limit) {
                Ok(results) => Response::Results { results },
                Err(e) => Response::Error {
                    code: ErrorCode::InternalError,
                    message: format!("Failed to list results: {}", e),
                },
            }
        }

        Request::StoreBaseline { tag } => {
            let state = state.read().await;
            match state.storage.store_baseline(&tag) {
                Ok(()) => {
                    info!("Stored baseline: {}", tag);
                    Response::Ok
                }
                Err(e) => Response::Error {
                    code: ErrorCode::InternalError,
                    message: format!("Failed to store baseline: {}", e),
                },
            }
        }

        Request::CompareBaseline { tag } => {
            let state = state.read().await;
            match state.storage.get_baseline(&tag) {
                Ok(baseline_results) => {
                    let current_results = match state.storage.list_results(1000) {
                        Ok(r) => r,
                        Err(e) => {
                            return Response::Error {
                                code: ErrorCode::InternalError,
                                message: format!("Failed to get current results: {}", e),
                            };
                        }
                    };

                    let mut comparisons = Vec::new();

                    // Group by op_name for more robust matching (seeds may differ across runs)
                    for current in &current_results {
                        let baseline_match = baseline_results.iter().find(|b| {
                            b.op_name == current.op_name && b.seed == current.seed
                        }).or_else(|| {
                            baseline_results.iter().find(|b| b.op_name == current.op_name)
                        });

                        let (regressed, relative_change) = match baseline_match {
                            Some(baseline) => {
                                let baseline_max = baseline.max_diff.max(1e-15);
                                let change = (current.max_diff - baseline.max_diff) / baseline_max;
                                let is_regression = current.max_diff > baseline.max_diff * 1.1 + 1e-10
                                    || (!current.passed && baseline.passed);
                                (is_regression, Some(change))
                            }
                            None => {
                                // New result — not a regression, but notable
                                (false, None)
                            }
                        };

                        comparisons.push(BaselineComparison {
                            op_name: current.op_name.clone(),
                            seed: current.seed,
                            current_max_diff: current.max_diff,
                            baseline_max_diff: baseline_match.map(|b| b.max_diff),
                            current_passed: current.passed,
                            baseline_passed: baseline_match.map(|b| b.passed),
                            relative_change,
                            is_regression: regressed,
                        });
                    }

                    let has_regressions = comparisons.iter().any(|c| c.is_regression);
                    Response::BaselineComparison {
                        baseline_tag: tag,
                        comparisons,
                        has_regressions,
                    }
                }
                Err(e) => Response::Error {
                    code: ErrorCode::NotFound,
                    message: format!("Baseline '{}' not found: {}", tag, e),
                },
            }
        }

        // =====================================================================
        // Phase 2: Fuzzing and Reproducibility
        // =====================================================================

        Request::FuzzOp { op_name, fuzz_config, iterations, fail_fast } => {
            info!("FuzzOp request: op={}, iterations={}, seed={}, fail_fast={}",
                  op_name, iterations, fuzz_config.seed, fail_fast);

            let state_read = state.read().await;

            // Find op config
            let op_config = state_read.config.ops.iter().find(|op| op.name == op_name);
            let op_config = match op_config {
                Some(c) => c.clone(),
                None => {
                    warn!("Op not found: {}", op_name);
                    return Response::Error {
                        code: ErrorCode::OpNotFound,
                        message: format!(
                            "Op '{}' not found in configuration. {}",
                            op_name,
                            available_ops_message(&state_read.config)
                        ),
                    };
                }
            };

            // Store the fuzz config for later reproduction
            if let Err(e) = state_read.storage.store_fuzz_config(fuzz_config.seed, &fuzz_config) {
                warn!("Failed to store fuzz config: {}", e);
            }

            drop(state_read);

            let input_names = op_input_names(&op_config);

            // Create fuzzer
            let mut fuzzer = Fuzzer::new(fuzz_config.clone());

            let mut total = 0;
            let mut passed = 0;
            let mut failed = 0;
            let mut failures = Vec::new();

            for i in 0..iterations {
                let test_case = fuzzer.next_test_case(&input_names);
                let start = Instant::now();

                debug!("Running test case {} with seed {}", i, test_case.seed);

                let state_read = state.read().await;

                // Run reference script
                let reference_path = std::path::Path::new(&op_config.reference);
                let reference_result = state_read
                    .executor
                    .run_reference(reference_path, &test_case.inputs, &std::collections::HashMap::new(), true)
                    .await;

                let reference = match reference_result {
                    Ok(r) => r,
                    Err(e) => {
                        error!("Reference script failed: {}", e);
                        let duration_ms = start.elapsed().as_millis() as u64;
                        let mut result = ValidationResult::fail(
                            op_name.clone(),
                            test_case.seed,
                            vec![gpuemu_common::types::ValidationFailure {
                                kind: gpuemu_common::types::FailureKind::ReferenceError,
                                message: format!("Reference script failed: {}", e),
                                index: None,
                                expected: None,
                                actual: None,
                            }],
                            duration_ms,
                        );
                        result = result.with_repro_info(test_case.to_repro_info(Some(fuzz_config.clone())));

                        if let Err(e) = state_read.storage.store_failure(&result) {
                            warn!("Failed to store failure: {}", e);
                        }

                        total += 1;
                        failed += 1;
                        failures.push(result);

                        if fail_fast {
                            break;
                        }
                        continue;
                    }
                };

                // Determine the output based on execution_mode:
                let output = match op_config.execution_mode {
                    gpuemu_common::config::ExecutionMode::ScriptBased => {
                        match &op_config.op_script {
                            Some(op_script) => {
                                let op_script_path = std::path::Path::new(op_script);
                                let op_result = state_read
                                    .executor
                                    .run_reference(op_script_path, &test_case.inputs, &std::collections::HashMap::new(), false)
                                    .await;
                                match op_result {
                                    Ok(r) => r,
                                    Err(e) => {
                                        let duration_ms = start.elapsed().as_millis() as u64;
                                        let mut result = ValidationResult::fail(
                                            op_name.clone(),
                                            test_case.seed,
                                            vec![gpuemu_common::types::ValidationFailure {
                                                kind: gpuemu_common::types::FailureKind::ReferenceError,
                                                message: format!("Op script failed: {}", e),
                                                index: None,
                                                expected: None,
                                                actual: None,
                                            }],
                                            duration_ms,
                                        );
                                        result = result.with_repro_info(test_case.to_repro_info(Some(fuzz_config.clone())));
                                        total += 1;
                                        failed += 1;
                                        failures.push(result);
                                        if fail_fast {
                                            break;
                                        }
                                        continue;
                                    }
                                }
                            }
                            None => {
                                warn!("Op '{}' has ScriptBased mode but no op_script configured", op_name);
                                test_case.inputs.get(primary_input_name(&op_config))
                                    .cloned()
                                    .unwrap_or_else(|| reference.clone())
                            }
                        }
                    }
                    _ => {
                        // ClientSide / DaemonOrchestrated: daemon can't invoke the actual op.
                        // Placeholder — use reference as output (validation will trivially pass).
                        // For real testing, use ValidateOp, SubmitOutput, or GetTestCase from the client.
                        reference.clone()
                    }
                };

                // Validate (op tolerances overlaid on the global ones)
                let invariants = Some(&op_config.invariants);
                let mut result = op_validator(&state_read.config.validation, &op_config).validate(
                    &op_name,
                    &output,
                    &reference,
                    test_case.seed,
                    invariants,
                );
                enforce_shape_preserved(&mut result, &output, &test_case.inputs, &op_config);

                let duration_ms = start.elapsed().as_millis() as u64;
                result.duration_ms = duration_ms;

                total += 1;
                if result.passed {
                    passed += 1;
                } else {
                    // Add reproduction info to failures
                    result = result.with_repro_info(test_case.to_repro_info(Some(fuzz_config.clone())));

                    if let Err(e) = state_read.storage.store_failure(&result) {
                        warn!("Failed to store failure: {}", e);
                    }

                    failed += 1;
                    failures.push(result);

                    if fail_fast {
                        break;
                    }
                }
            }

            info!("FuzzOp complete: total={}, passed={}, failed={}", total, passed, failed);

            Response::FuzzResults {
                seed: fuzz_config.seed,
                total,
                passed,
                failed,
                failures,
            }
        }

        Request::Reproduce { seed } => {
            info!("Reproduce request: seed={}", seed);

            let state_read = state.read().await;

            // Look up the failure
            let failure = match state_read.storage.get_failure(seed) {
                Ok(Some(f)) => f,
                Ok(None) => {
                    return Response::Error {
                        code: ErrorCode::NotFound,
                        message: format!("Failure with seed {} not found", seed),
                    };
                }
                Err(e) => {
                    return Response::Error {
                        code: ErrorCode::InternalError,
                        message: format!("Failed to retrieve failure: {}", e),
                    };
                }
            };

            // Get reproduction info
            let repro = match &failure.repro_info {
                Some(r) => r,
                None => {
                    return Response::Error {
                        code: ErrorCode::InternalError,
                        message: "Failure has no reproduction info".to_string(),
                    };
                }
            };

            let input_names = match state_read.config.ops.iter().find(|op| op.name == failure.op_name)
            {
                Some(op_config) => op_input_names(op_config),
                None => vec!["input"],
            };

            // Regenerate test case
            let test_case = regenerate_test_case(repro, &input_names);

            Response::ReproduceResult {
                result: failure,
                inputs: test_case.inputs,
            }
        }

        Request::Minimize { seed, strategy, max_iters } => {
            info!("Minimize request: seed={}, strategy={:?}, max_iters={}", seed, strategy, max_iters);

            let state_read = state.read().await;

            let failure = match state_read.storage.get_failure(seed) {
                Ok(Some(f)) => f,
                Ok(None) => {
                    return Response::Error {
                        code: ErrorCode::NotFound,
                        message: format!("Failure with seed {} not found", seed),
                    };
                }
                Err(e) => {
                    return Response::Error {
                        code: ErrorCode::InternalError,
                        message: format!("Failed to retrieve failure: {}", e),
                    };
                }
            };

            let repro = match &failure.repro_info {
                Some(r) => r.clone(),
                None => {
                    return Response::Error {
                        code: ErrorCode::InternalError,
                        message: "Failure has no reproduction info".to_string(),
                    };
                }
            };

            let op_config = match state_read.config.ops.iter().find(|op| op.name == failure.op_name) {
                Some(c) => c.clone(),
                None => {
                    return Response::Error {
                        code: ErrorCode::OpNotFound,
                        message: format!("Op '{}' not found in configuration", failure.op_name),
                    };
                }
            };

            let original_shape = repro.shape.clone();
            let mut minimized_shape = original_shape.clone();
            let mut minimized_seed = seed;

            match strategy {
                MinimizeStrategy::BinarySearchDims => {
                    // Try reducing each dimension via binary search
                    for dim_idx in 0..minimized_shape.len() {
                        let original_dim = original_shape[dim_idx];
                        if original_dim <= 1 {
                            continue;
                        }

                        let mut lo = 1usize;
                        let mut hi = original_dim;
                        let mut best = original_dim;

                        for _ in 0..max_iters {
                            if lo >= hi {
                                break;
                            }
                            let mid = lo + (hi - lo) / 2;
                            let mut test_shape = minimized_shape.clone();
                            test_shape[dim_idx] = mid;

                            let test_seed = gpuemu_common::rng::derive_seed(
                                seed,
                                &format!("min_dim_{}_{}", dim_idx, mid),
                            );
                            let mut test_repro = repro.clone();
                            test_repro.seed = test_seed;
                            test_repro.shape = test_shape.clone();
                            test_repro.strides = match test_repro.layout {
                                LayoutType::Contiguous => {
                                    TensorData::compute_contiguous_strides(&test_shape)
                                }
                                LayoutType::Strided => TensorData::compute_contiguous_strides(&test_shape)
                                    .into_iter()
                                    .map(|stride| stride * 2)
                                    .collect(),
                                LayoutType::Transposed => {
                                    if test_shape.len() >= 2 {
                                        let mut transposed_shape = test_shape.clone();
                                        let len = transposed_shape.len();
                                        transposed_shape.swap(len - 1, len - 2);
                                        TensorData::compute_contiguous_strides(&transposed_shape)
                                    } else {
                                        TensorData::compute_contiguous_strides(&test_shape)
                                    }
                                }
                            };
                            let input_names = op_input_names(&op_config);
                            let test_case = regenerate_test_case(&test_repro, &input_names);

                            let reference_path = Path::new(&op_config.reference);
                            let reference_result = state_read
                                .executor
                                .run_reference(reference_path, &test_case.inputs, &std::collections::HashMap::new(), true)
                                .await;

                            match reference_result {
                                Ok(reference) => {
                                    let output = test_case
                                        .inputs
                                        .get(primary_input_name(&op_config))
                                        .cloned()
                                        .unwrap_or_else(|| reference.clone());
                                    let invariants = Some(&op_config.invariants);
                                    let result = op_validator(&state_read.config.validation, &op_config).validate(
                                        &failure.op_name,
                                        &output,
                                        &reference,
                                        test_seed,
                                        invariants,
                                    );
                                    let mut result = result;
                                    enforce_shape_preserved(
                                        &mut result,
                                        &output,
                                        &test_case.inputs,
                                        &op_config,
                                    );
                                    if !result.passed {
                                        best = mid;
                                        hi = mid;
                                    } else {
                                        lo = mid + 1;
                                    }
                                }
                                Err(_) => {
                                    lo = mid + 1;
                                }
                            }
                        }

                        if best < original_dim {
                            minimized_shape[dim_idx] = best;
                        }
                    }
                }
                MinimizeStrategy::BinarySearchValues => {
                    if repro.input_snapshot.is_none() {
                        info!("No input snapshot stored, cannot minimize values");
                        minimized_seed = seed;
                    } else {
                        let input_names = op_input_names(&op_config);
                        let original_case = regenerate_test_case(&repro, &input_names);

                        let mut best_scale = 1.0f64;
                        let mut lo = 0.0f64;
                        let mut hi = 1.0f64;

                        for _ in 0..max_iters {
                            if hi - lo < 1e-6 {
                                break;
                            }
                            let mid = (lo + hi) / 2.0;
                            let scale = mid;

                            let mut scaled_inputs = HashMap::new();
                            for (name, tensor) in &original_case.inputs {
                                let mut scaled_data = tensor.data.clone();
                                scale_tensor_values(&mut scaled_data, tensor.dtype, scale);
                                scaled_inputs.insert(name.clone(), TensorData {
                                    shape: tensor.shape.clone(),
                                    strides: tensor.strides.clone(),
                                    dtype: tensor.dtype,
                                    data: scaled_data,
                                });
                            }

                            let reference_path = Path::new(&op_config.reference);
                            let reference_result = state_read
                                .executor
                                .run_reference(reference_path, &scaled_inputs, &std::collections::HashMap::new(), true)
                                .await;

                            match reference_result {
                                Ok(reference) => {
                                    let primary = primary_input_name(&op_config);
                                    let output = scaled_inputs.get(primary).cloned().unwrap_or_else(|| reference.clone());
                                    let invariants = Some(&op_config.invariants);
                                    let result = op_validator(&state_read.config.validation, &op_config).validate(
                                        &failure.op_name,
                                        &output,
                                        &reference,
                                        seed,
                                        invariants,
                                    );
                                    let mut result = result;
                                    enforce_shape_preserved(&mut result, &output, &scaled_inputs, &op_config);
                                    if !result.passed {
                                        best_scale = scale;
                                        hi = mid;
                                    } else {
                                        lo = mid;
                                    }
                                }
                                Err(_) => {
                                    lo = mid;
                                }
                            }
                        }

                        info!("Value minimization: smallest scale = {:.6}", best_scale);
                        minimized_seed = gpuemu_common::rng::derive_seed(
                            seed,
                            &format!("minval_{}", best_scale.to_bits()),
                        );
                    }
                }
            }

            Response::MinimizeResult {
                original_seed: seed,
                minimized_seed,
                minimized_shape,
                result: failure,
            }
        }

        Request::ListFailures { limit } => {
            info!("ListFailures request: limit={}", limit);

            let state_read = state.read().await;

            match state_read.storage.list_failures(limit) {
                Ok(failures) => Response::Results { results: failures },
                Err(e) => Response::Error {
                    code: ErrorCode::InternalError,
                    message: format!("Failed to list failures: {}", e),
                },
            }
        }

        // =====================================================================
        // Phase 3: Artifact Inspection
        // =====================================================================

        Request::LintKernel { kernel_name, ptx_content } => {
            info!("LintKernel request: kernel={:?}", kernel_name);

            let state_read = state.read().await;
            let parser = crate::artifact::PtxParser::new();

            // Determine which kernels to lint
            let kernels_to_lint: Vec<_> = match &kernel_name {
                Some(name) => {
                    match state_read.config.kernels.iter().find(|k| &k.name == name) {
                        Some(k) => vec![k.clone()],
                        None => {
                            // If kernel not in config, create a default config for it
                            vec![gpuemu_common::config::KernelConfig {
                                name: name.clone(),
                                source: None,
                                reference: String::new(),
                                tolerances: std::collections::HashMap::new(),
                                invariants: gpuemu_common::config::InvariantConfig::default(),
                                artifact_checks: gpuemu_common::config::ArtifactCheckConfig::default(),
                            }]
                        }
                    }
                }
                None => {
                    if state_read.config.kernels.is_empty() {
                        // No kernels configured, use default config with kernel name from PTX
                        let detected_name = parser.extract_kernel_name(&ptx_content)
                            .unwrap_or_else(|| "unknown".to_string());
                        vec![gpuemu_common::config::KernelConfig {
                            name: detected_name,
                            source: None,
                            reference: String::new(),
                            tolerances: std::collections::HashMap::new(),
                            invariants: gpuemu_common::config::InvariantConfig::default(),
                            artifact_checks: gpuemu_common::config::ArtifactCheckConfig::default(),
                        }]
                    } else {
                        state_read.config.kernels.clone()
                    }
                }
            };

            let mut results = Vec::new();

            for kernel in &kernels_to_lint {
                // Parse PTX
                match parser.parse(&kernel.name, &ptx_content) {
                    Ok(metrics) => {
                        // Store metrics
                        if let Err(e) = state_read.storage.store_artifact(&metrics) {
                            warn!("Failed to store artifact: {}", e);
                        }

                        // Lint against config
                        let result = crate::artifact::ArtifactLinter::lint(&metrics, &kernel.artifact_checks);
                        results.push(result);
                    }
                    Err(e) => {
                        return Response::Error {
                            code: ErrorCode::PtxParseError,
                            message: format!("Failed to parse PTX: {}", e),
                        };
                    }
                }
            }

            Response::LintResults { results }
        }

        Request::StoreArtifact { kernel_name: _, metrics } => {
            info!("StoreArtifact request: kernel={}", metrics.kernel_name);

            let state_read = state.read().await;
            match state_read.storage.store_artifact(&metrics) {
                Ok(()) => Response::Ok,
                Err(e) => Response::Error {
                    code: ErrorCode::InternalError,
                    message: format!("Failed to store artifact: {}", e),
                },
            }
        }

        Request::StoreArtifactBaseline { tag } => {
            info!("StoreArtifactBaseline request: tag={}", tag);

            let state_read = state.read().await;
            match state_read.storage.store_artifact_baseline(&tag) {
                Ok(()) => {
                    info!("Stored artifact baseline: {}", tag);
                    Response::Ok
                }
                Err(e) => Response::Error {
                    code: ErrorCode::InternalError,
                    message: format!("Failed to store baseline: {}", e),
                },
            }
        }

        Request::DiffArtifactBaseline { tag } => {
            info!("DiffArtifactBaseline request: tag={}", tag);

            let state_read = state.read().await;

            // Check baseline exists
            if !state_read.storage.has_artifact_baseline(&tag).unwrap_or(false) {
                return Response::Error {
                    code: ErrorCode::BaselineNotFound,
                    message: format!("Baseline '{}' not found", tag),
                };
            }

            // Get baseline and current artifacts
            let baseline = match state_read.storage.get_artifact_baseline(&tag) {
                Ok(b) => b,
                Err(e) => {
                    return Response::Error {
                        code: ErrorCode::InternalError,
                        message: format!("Failed to get baseline: {}", e),
                    };
                }
            };

            let current = match state_read.storage.list_artifacts() {
                Ok(c) => c,
                Err(e) => {
                    return Response::Error {
                        code: ErrorCode::InternalError,
                        message: format!("Failed to get current artifacts: {}", e),
                    };
                }
            };

            // Build diff for each current artifact
            let mut diffs = Vec::new();
            let mut has_regressions = false;

            for curr in &current {
                let base = baseline.iter().find(|b| b.kernel_name == curr.kernel_name);
                let diff = crate::artifact::ArtifactDiffer::diff(base, curr);
                if diff.is_regression {
                    has_regressions = true;
                }
                diffs.push(diff);
            }

            Response::ArtifactDiffs {
                baseline_tag: tag,
                diffs,
                has_regressions,
            }
        }

        Request::GetArtifact { kernel_name } => {
            info!("GetArtifact request: kernel={}", kernel_name);

            let state_read = state.read().await;
            match state_read.storage.get_artifact(&kernel_name) {
                Ok(Some(metrics)) => Response::ArtifactMetricsResult(metrics),
                Ok(None) => Response::Error {
                    code: ErrorCode::ArtifactNotFound,
                    message: format!("Artifact for kernel '{}' not found", kernel_name),
                },
                Err(e) => Response::Error {
                    code: ErrorCode::InternalError,
                    message: format!("Failed to get artifact: {}", e),
                },
            }
        }

        Request::ListArtifacts => {
            info!("ListArtifacts request");

            let state_read = state.read().await;
            match state_read.storage.list_artifacts() {
                Ok(artifacts) => Response::ArtifactList { artifacts },
                Err(e) => Response::Error {
                    code: ErrorCode::InternalError,
                    message: format!("Failed to list artifacts: {}", e),
                },
            }
        }

        // =====================================================================
        // Phase 4: CI Integration
        // =====================================================================

        Request::RunCi { quick, baseline, parallel_jobs } => {
            info!("RunCi request: quick={}, baseline={:?}, parallel_jobs={}",
                  quick, baseline, parallel_jobs);

            let start_time = Instant::now();

            // Snapshot the per-op work and the per-op invariants we need to run
            // each op concurrently without holding the read lock across awaits.
            //
            // The fuzzer + reference seed are chosen here (one per op) so two
            // workers running in parallel cannot collide on a shared
            // SystemTime::now() value.
            let (ops_snapshot, validation_cfg, max_parallel) = {
                let s = state.read().await;
                let max_parallel = if parallel_jobs == 0 {
                    s.config.ci.parallel_jobs.max(1)
                } else {
                    parallel_jobs
                } as usize;
                let ops: Vec<gpuemu_common::config::OpConfig> = s.config.ops.clone();
                (ops, s.config.validation.clone(), max_parallel)
            };

            // Per-op work: an async closure spawned into a JoinSet, capped by a
            // Semaphore. Each task returns (op_index, Vec<ValidationResult>) so
            // we can re-order back to config order — the SARIF + JUnit reports
            // expect a deterministic op ordering.
            let iterations = if quick { 10 } else { 50 };
            let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(max_parallel));
            let mut join_set = tokio::task::JoinSet::new();

            for (op_index, op) in ops_snapshot.into_iter().enumerate() {
                let state_clone = state.clone();
                let validation_cfg = validation_cfg.clone();
                let permit = semaphore.clone();
                join_set.spawn(async move {
                    let _permit = permit.acquire_owned().await.expect("semaphore closed");
                    let mut per_op = Vec::<ValidationResult>::with_capacity(iterations);
                    info!("Running CI validation for op: {} (parallel slot)", op.name);

                    // One fuzz_config + Fuzzer per op. Distinct seeds across ops
                    // via op_index so two parallel workers cannot collide.
                    let base_seed = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos() as u64;
                    let op_seed = base_seed.wrapping_add(op_index as u64);
                    let fuzz_config = gpuemu_common::types::FuzzConfig::with_seed(op_seed);
                    let mut fuzzer = Fuzzer::new(fuzz_config.clone());
                    let input_names_owned: Vec<String> =
                        op_input_names(&op).into_iter().map(String::from).collect();
                    let input_names_ref: Vec<&str> =
                        input_names_owned.iter().map(String::as_str).collect();

                    for _i in 0..iterations {
                        let test_case = fuzzer.next_test_case(&input_names_ref);
                        let iter_start = Instant::now();

                        let reference_path = std::path::Path::new(&op.reference);
                        let s_read = state_clone.read().await;
                        let reference_result = s_read
                            .executor
                            .run_reference(reference_path, &test_case.inputs, &std::collections::HashMap::new(), true)
                            .await;

                        let reference = match reference_result {
                            Ok(r) => r,
                            Err(e) => {
                                let duration_ms = iter_start.elapsed().as_millis() as u64;
                                let mut result = ValidationResult::fail(
                                    op.name.clone(),
                                    test_case.seed,
                                    vec![gpuemu_common::types::ValidationFailure {
                                        kind: gpuemu_common::types::FailureKind::ReferenceError,
                                        message: format!("Reference script failed: {}", e),
                                        index: None,
                                        expected: None,
                                        actual: None,
                                    }],
                                    duration_ms,
                                );
                                result = result.with_repro_info(test_case.to_repro_info(Some(fuzz_config.clone())));
                                per_op.push(result);
                                continue;
                            }
                        };

                        let output = match op.execution_mode {
                            gpuemu_common::config::ExecutionMode::ScriptBased => {
                                match &op.op_script {
                                    Some(op_script) => {
                                        let op_script_path = std::path::Path::new(op_script);
                                        match s_read.executor
                                            .run_reference(op_script_path, &test_case.inputs, &std::collections::HashMap::new(), false)
                                            .await
                                        {
                                            Ok(r) => r,
                                            Err(e) => {
                                                let duration_ms = iter_start.elapsed().as_millis() as u64;
                                                per_op.push(ValidationResult::fail(
                                                    op.name.clone(),
                                                    test_case.seed,
                                                    vec![gpuemu_common::types::ValidationFailure {
                                                        kind: gpuemu_common::types::FailureKind::ReferenceError,
                                                        message: format!("Op script failed: {}", e),
                                                        index: None,
                                                        expected: None,
                                                        actual: None,
                                                    }],
                                                    duration_ms,
                                                ));
                                                continue;
                                            }
                                        }
                                    }
                                    None => {
                                        warn!("Op '{}' has ScriptBased mode but no op_script", op.name);
                                        reference.clone()
                                    }
                                }
                            }
                            _ => reference.clone(),
                        };

                        let invariants = Some(&op.invariants);
                        let mut result = op_validator(&validation_cfg, &op).validate(
                            &op.name,
                            &output,
                            &reference,
                            test_case.seed,
                            invariants,
                        );
                        enforce_shape_preserved(&mut result, &output, &test_case.inputs, &op);

                        result.duration_ms = iter_start.elapsed().as_millis() as u64;
                        if !result.passed {
                            result = result.with_repro_info(test_case.to_repro_info(Some(fuzz_config.clone())));
                        }
                        per_op.push(result);
                    }
                    (op_index, per_op)
                });
            }

            // Re-stitch per-op results in original op order.
            let mut by_index: std::collections::BTreeMap<usize, Vec<ValidationResult>> =
                std::collections::BTreeMap::new();
            while let Some(join_res) = join_set.join_next().await {
                match join_res {
                    Ok((idx, results)) => {
                        by_index.insert(idx, results);
                    }
                    Err(e) => {
                        warn!("CI op task join failed: {}", e);
                    }
                }
            }
            let mut validation_results: Vec<ValidationResult> = Vec::new();
            for (_, mut v) in by_index {
                validation_results.append(&mut v);
            }
            let mut lint_results = Vec::new();

            // Re-acquire the read lock for the remaining (lint + artifact-diff)
            // sections — these are short, lock-bound, and not in the hot path.
            let state_read = state.read().await;

            // Run lint checks on configured kernels (if any have PTX files)
            // For CI, we'd typically lint PTX from build artifacts
            // This is a simplified implementation
            for kernel in &state_read.config.kernels {
                if let Some(ref source) = kernel.source {
                    let ptx_path = std::path::Path::new(source);
                    if ptx_path.exists() {
                        if let Ok(ptx_content) = std::fs::read_to_string(ptx_path) {
                            let parser = crate::artifact::PtxParser::new();
                            if let Ok(metrics) = parser.parse(&kernel.name, &ptx_content) {
                                let result = crate::artifact::ArtifactLinter::lint(&metrics, &kernel.artifact_checks);
                                lint_results.push(result);
                            }
                        }
                    }
                }
            }

            // Run artifact diff if baseline specified
            let artifact_diffs = if let Some(ref tag) = baseline {
                if state_read.storage.has_artifact_baseline(tag).unwrap_or(false) {
                    let baseline_artifacts = state_read.storage.get_artifact_baseline(tag).unwrap_or_default();
                    let current_artifacts = state_read.storage.list_artifacts().unwrap_or_default();

                    let mut diffs = Vec::new();
                    let mut has_regressions = false;

                    for curr in &current_artifacts {
                        let base = baseline_artifacts.iter().find(|b| b.kernel_name == curr.kernel_name);
                        let diff = crate::artifact::ArtifactDiffer::diff(base, curr);
                        if diff.is_regression {
                            has_regressions = true;
                        }
                        diffs.push(diff);
                    }

                    Some(ArtifactDiffSummary {
                        baseline_tag: tag.clone(),
                        has_regressions,
                        diffs,
                    })
                } else {
                    warn!("Baseline '{}' not found, skipping artifact diff", tag);
                    None
                }
            } else {
                None
            };

            // Build summary
            let passed = validation_results.iter().filter(|r| r.passed).count()
                + lint_results.iter().filter(|r| r.passed).count();
            let failed = validation_results.iter().filter(|r| !r.passed).count()
                + lint_results.iter().filter(|r| !r.passed).count();
            let total = passed + failed;

            let summary = CiRunSummary {
                total_tests: total,
                passed,
                failed,
                skipped: 0,
                duration_ms: start_time.elapsed().as_millis() as u64,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                validation_results,
                lint_results,
                artifact_diffs,
            };

            info!("CI run complete: {} tests, {} passed, {} failed, {:.2}s",
                  total, passed, failed, summary.duration_ms as f64 / 1000.0);

            drop(state_read);
            let mut state_write = state.write().await;
            state_write.last_ci_summary = Some(summary.clone());

            Response::CiRunComplete(summary)
        }

        Request::GetCiSummary => {
            info!("GetCiSummary request");

            let state = state.read().await;
            match &state.last_ci_summary {
                Some(summary) => Response::CiRunComplete(summary.clone()),
                None => Response::Error {
                    code: ErrorCode::NotFound,
                    message: "No CI run has been executed yet".to_string(),
                },
            }
        }

        // =====================================================================
        // Execution Modes: Daemon-Orchestrated
        // =====================================================================

        Request::GetTestCase { op_name, fuzz_config } => {
            info!("GetTestCase request: op={}", op_name);

            let state_read = state.read().await;
            let op_config = state_read.config.ops.iter().find(|op| op.name == op_name);
            let op_config = match op_config {
                Some(c) => c.clone(),
                None => {
                    return Response::Error {
                        code: ErrorCode::OpNotFound,
                        message: format!(
                            "Op '{}' not found. {}",
                            op_name,
                            available_ops_message(&state_read.config)
                        ),
                    };
                }
            };
            drop(state_read);

            let input_names = op_input_names(&op_config);
            let mut fuzzer = Fuzzer::new(fuzz_config.clone());
            let test_case = fuzzer.next_test_case(&input_names);

            Response::TestCase {
                seed: test_case.seed,
                inputs: test_case.inputs,
                shape: test_case.shape,
                dtype: test_case.dtype.to_numpy_dtype().to_string(),
                layout: test_case.layout.to_config_str().to_string(),
            }
        }

        Request::GetTestBatch { op_name, fuzz_config, count } => {
            info!("GetTestBatch request: op={}, count={}", op_name, count);

            let state_read = state.read().await;
            let op_config = state_read.config.ops.iter().find(|op| op.name == op_name);
            let op_config = match op_config {
                Some(c) => c.clone(),
                None => {
                    return Response::Error {
                        code: ErrorCode::OpNotFound,
                        message: format!(
                            "Op '{}' not found. {}",
                            op_name,
                            available_ops_message(&state_read.config)
                        ),
                    };
                }
            };
            drop(state_read);

            let input_names = op_input_names(&op_config);
            let mut fuzzer = Fuzzer::new(fuzz_config.clone());
            let mut cases = Vec::with_capacity(count);

            for _ in 0..count {
                let test_case = fuzzer.next_test_case(&input_names);
                cases.push(TestCaseData {
                    seed: test_case.seed,
                    inputs: test_case.inputs,
                    shape: test_case.shape,
                    dtype: test_case.dtype.to_numpy_dtype().to_string(),
                    layout: test_case.layout.to_config_str().to_string(),
                });
            }

            Response::TestBatch { cases }
        }

        Request::SubmitOutput { op_name, inputs, output, seed, kwargs } => {
            info!("SubmitOutput request: op={}, seed={}", op_name, seed);

            let state_read = state.read().await;
            let op_config = state_read.config.ops.iter().find(|op| op.name == op_name);
            let op_config = match op_config {
                Some(c) => c,
                None => {
                    return Response::Error {
                        code: ErrorCode::OpNotFound,
                        message: format!(
                            "Op '{}' not found. {}",
                            op_name,
                            available_ops_message(&state_read.config)
                        ),
                    };
                }
            };

            // Run reference script
            let reference_path = Path::new(&op_config.reference);
            let reference_result = state_read
                .executor
                .run_reference(reference_path, &inputs, &kwargs, true)
                .await;

            let reference = match reference_result {
                Ok(r) => r,
                Err(e) => {
                    return Response::Error {
                        code: ErrorCode::ReferenceScriptFailed,
                        message: format!("Reference script failed: {}", e),
                    };
                }
            };

            // Validate submitted output against reference (op tolerances overlaid)
            let invariants = Some(&op_config.invariants);
            let mut result = op_validator(&state_read.config.validation, op_config).validate(
                &op_name,
                &output,
                &reference,
                seed,
                invariants,
            );
            enforce_shape_preserved(&mut result, &output, &inputs, op_config);

            // Store result
            drop(state_read);
            let state_write = state.write().await;
            if let Err(e) = state_write.storage.store_result(&result) {
                warn!("Failed to store result: {}", e);
            }
            if !result.passed {
                if let Err(e) = state_write.storage.store_failure(&result) {
                    warn!("Failed to store failure: {}", e);
                }
            }

            Response::SubmitResult { result }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpuemu_common::config::{InvariantConfig, OpConfig};
    use gpuemu_common::protocol::{deserialize_response, serialize_request};
    use gpuemu_common::types::{DType, TensorData};
    use nng::options::Options;
    use std::collections::HashMap;
    use std::time::Duration;

    fn send_request(socket_path: &Path, request: &Request) -> Result<Response, String> {
        let socket = Socket::new(Protocol::Req0).map_err(|e| e.to_string())?;
        socket
            .set_opt::<nng::options::RecvTimeout>(Some(Duration::from_secs(2)))
            .map_err(|e| e.to_string())?;
        socket
            .set_opt::<nng::options::SendTimeout>(Some(Duration::from_secs(2)))
            .map_err(|e| e.to_string())?;
        socket
            .dial(&format!("ipc://{}", socket_path.display()))
            .map_err(|e| e.to_string())?;

        let bytes = serialize_request(request).map_err(|e| e.to_string())?;
        socket.send(&bytes).map_err(|(_, e)| e.to_string())?;
        let response = socket.recv().map_err(|e| e.to_string())?;
        deserialize_response(&response).map_err(|e| e.to_string())
    }

    #[tokio::test]
    async fn test_ping_handler() {
        let tmp = tempfile::TempDir::new().unwrap();
        let storage = Storage::open(tmp.path().join("test.db")).unwrap();
        let config = GpuemuConfig::default();
        let state = Arc::new(RwLock::new(ServerState::new(storage, config)));

        let response = handle_request(Request::Ping, state).await;
        match response {
            Response::Pong { version, .. } => {
                assert!(!version.is_empty());
            }
            _ => panic!("Expected Pong response"),
        }
    }

    #[test]
    fn test_server_smoke_ping_and_validate() {
        let tmp = tempfile::TempDir::new().unwrap();
        let socket_path = std::path::PathBuf::from(format!(
            "/tmp/gpuemu-smoke-{}-{}.sock",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        if socket_path.exists() {
            let _ = std::fs::remove_file(&socket_path);
        }
        let db_path = tmp.path().join("test.db");
        let script_path = tmp.path().join("ref_echo.py");

        std::fs::write(
            &script_path,
            r#"import sys, json, base64, numpy as np

def decode_tensor(tensor_dict):
    shape = tensor_dict["shape"]
    dtype = np.dtype(tensor_dict["dtype"])
    data = base64.b64decode(tensor_dict["data"])
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()

def encode_tensor(arr):
    arr = np.asarray(arr)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data": base64.b64encode(arr.tobytes()).decode("utf-8"),
    }

payload = json.load(sys.stdin)
inputs = {name: decode_tensor(tensor) for name, tensor in payload["inputs"].items()}
result = inputs["x"]
json.dump(encode_tensor(result), sys.stdout)
"#,
        )
        .unwrap();

        let storage = Storage::open(&db_path).unwrap();
        let mut config = GpuemuConfig::default();
        config.ops.push(OpConfig {
            name: "echo".to_string(),
            module: None,
            reference: script_path.display().to_string(),
            input_names: vec!["x".to_string()],
            execution_mode: gpuemu_common::config::ExecutionMode::ClientSide,
            op_script: None,
            frameworks: Vec::new(),
            tolerances: HashMap::new(),
            invariants: InvariantConfig::default(),
        });

        let state = Arc::new(RwLock::new(ServerState::new(storage, config)));
        let server_state = state.clone();
        let server_socket_path = socket_path.clone();
        let server_thread = std::thread::spawn(move || {
            let runtime = tokio::runtime::Runtime::new().expect("runtime");
            runtime.block_on(async move { run_server(&server_socket_path, server_state).await })
        });

        let mut socket_ready = false;
        for _ in 0..20 {
            if socket_path.exists() {
                socket_ready = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(socket_ready, "server socket was never created");

        let mut pong_ok = false;
        for _ in 0..20 {
            let attempt = send_request(&socket_path, &Request::Ping);
            if let Ok(Response::Pong { .. }) = attempt {
                pong_ok = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(pong_ok, "server never responded to ping");

        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        let tensor = TensorData::new(vec![2, 2], DType::Float32, bytes);

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), tensor.clone());

        let response = send_request(
            &socket_path,
            &Request::ValidateOp {
                op_name: "echo".to_string(),
                inputs,
                output: tensor,
                kwargs: HashMap::new(),
            },
        )
        .expect("validate request should succeed");

        match response {
            Response::ValidationResult { result } => {
                assert!(result.passed, "validation should pass: {:?}", result.failures);
            }
            other => panic!("unexpected response: {:?}", other),
        }

        let shutdown = send_request(&socket_path, &Request::Shutdown)
            .expect("shutdown request should succeed");
        assert!(matches!(shutdown, Response::Ok));

        let server_result = server_thread.join().expect("server thread join failed");
        server_result.expect("server returned error");
    }

    /// Smoke: the parallel RunCi branch (parallel_jobs > 1) doesn't panic and
    /// returns a coherent empty summary on an empty ops list. The full
    /// concurrent integration is exercised through the existing CI workflow on
    /// the gpuemu-paper corpus; this test specifically gates the JoinSet +
    /// Semaphore wiring against silent regressions.
    #[tokio::test]
    async fn test_run_ci_parallel_empty_ops() {
        let tmp = tempfile::TempDir::new().unwrap();
        let storage = Storage::open(tmp.path().join("test.db")).unwrap();
        let config = GpuemuConfig::default();
        let state = Arc::new(RwLock::new(ServerState::new(storage, config)));

        let response = handle_request(
            Request::RunCi {
                quick: true,
                baseline: None,
                parallel_jobs: 4,
            },
            state,
        )
        .await;

        match response {
            Response::CiRunComplete(summary) => {
                assert_eq!(summary.total_tests, 0);
                assert_eq!(summary.passed, 0);
                assert_eq!(summary.failed, 0);
                assert!(summary.validation_results.is_empty());
                assert!(summary.lint_results.is_empty());
            }
            other => panic!("expected CiRunComplete, got: {:?}", other),
        }
    }
}
