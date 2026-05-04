//! NNG-based IPC server for the gpuemu daemon.

use crate::executor::{Executor, ExecutorConfig};
use crate::fuzzer::{Fuzzer, regenerate_test_case};
use crate::storage::Storage;
use crate::validator::Validator;
use anyhow::{Context, Result};
use gpuemu_common::config::GpuemuConfig;
use gpuemu_common::protocol::{
    deserialize_request, serialize_response, ErrorCode, Request, Response,
};
use gpuemu_common::types::ValidationResult;
use nng::{Protocol, Socket};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Server state shared across handlers.
pub struct ServerState {
    /// Storage for validation results.
    pub storage: Storage,
    /// Reference script executor.
    pub executor: Executor,
    /// Validator for comparing outputs.
    pub validator: Validator,
    /// Configuration.
    pub config: GpuemuConfig,
    /// Server start time.
    pub start_time: Instant,
}

impl ServerState {
    /// Create new server state.
    pub fn new(storage: Storage, config: GpuemuConfig) -> Self {
        let executor = Executor::new(ExecutorConfig::default());
        let validator = Validator::new(config.validation.clone());

        Self {
            storage,
            executor,
            validator,
            config,
            start_time: Instant::now(),
        }
    }
}

/// Run the NNG server.
pub async fn run_server(socket_path: &Path, state: Arc<RwLock<ServerState>>) -> Result<()> {
    // Ensure parent directory exists
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Remove old socket file if it exists
    if socket_path.exists() {
        std::fs::remove_file(socket_path)?;
    }

    let socket_url = format!("ipc://{}", socket_path.display());
    info!("Starting server on {}", socket_url);

    let socket = Socket::new(Protocol::Rep0).context("Failed to create NNG socket")?;

    socket
        .listen(&socket_url)
        .with_context(|| format!("Failed to listen on {}", socket_url))?;

    info!("Server listening on {}", socket_url);

    loop {
        // Receive message
        let msg = match socket.recv() {
            Ok(msg) => msg,
            Err(e) => {
                error!("Failed to receive message: {}", e);
                continue;
            }
        };

        debug!("Received message ({} bytes)", msg.len());

        // Deserialize request
        let request = match deserialize_request(&msg) {
            Ok(req) => req,
            Err(e) => {
                error!("Failed to deserialize request: {}", e);
                let response = Response::Error {
                    code: ErrorCode::InvalidRequest,
                    message: format!("Invalid request: {}", e),
                };
                if let Ok(bytes) = serialize_response(&response) {
                    let _ = socket.send(&bytes);
                }
                continue;
            }
        };

        // Handle request
        let response = handle_request(request, state.clone()).await;

        // Send response
        match serialize_response(&response) {
            Ok(bytes) => {
                if let Err((_, e)) = socket.send(&bytes) {
                    error!("Failed to send response: {}", e);
                }
            }
            Err(e) => {
                error!("Failed to serialize response: {:?}", e);
            }
        }
    }
}

/// Handle a single request.
async fn handle_request(request: Request, state: Arc<RwLock<ServerState>>) -> Response {
    match request {
        Request::Ping => {
            let state = state.read().await;
            Response::Pong {
                version: gpuemu_common::VERSION.to_string(),
                uptime_secs: state.start_time.elapsed().as_secs(),
            }
        }

        Request::Shutdown => {
            info!("Received shutdown request");
            // In a real implementation, we'd trigger graceful shutdown
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
                        message: format!("Op '{}' not found in configuration", op_name),
                    };
                }
            };

            // Run reference script
            let reference_path = Path::new(&op_config.reference);
            let reference_result = state_read
                .executor
                .run_reference(reference_path, &inputs, &kwargs)
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

            // Validate
            let invariants = Some(&op_config.invariants);
            let result = state_read
                .validator
                .validate(&op_name, &output, &reference, seed, invariants);

            // Store result
            drop(state_read);
            let state_write = state.write().await;
            if let Err(e) = state_write.storage.store_result(&result) {
                warn!("Failed to store result: {}", e);
            }

            Response::ValidationResult(result)
        }

        Request::GetResult { seed } => {
            let state = state.read().await;
            match state.storage.get_result(seed) {
                Ok(Some(result)) => Response::ValidationResult(result),
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
                Ok(results) => Response::Results(results),
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
                Ok(results) => Response::Results(results),
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
                        message: format!("Op '{}' not found in configuration", op_name),
                    };
                }
            };

            // Store the fuzz config for later reproduction
            if let Err(e) = state_read.storage.store_fuzz_config(fuzz_config.seed, &fuzz_config) {
                warn!("Failed to store fuzz config: {}", e);
            }

            drop(state_read);

            // Create fuzzer
            let mut fuzzer = Fuzzer::new(fuzz_config.clone());
            let input_names: Vec<&str> = vec!["input"]; // TODO: Get from op config

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
                    .run_reference(reference_path, &test_case.inputs, &std::collections::HashMap::new())
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

                // For now, use the first input as the "output" to validate
                // In a real implementation, we'd run the actual op
                let output = test_case.inputs.get("input").cloned().unwrap_or_else(|| reference.clone());

                // Validate
                let invariants = Some(&op_config.invariants);
                let mut result = state_read.validator.validate(
                    &op_name,
                    &output,
                    &reference,
                    test_case.seed,
                    invariants,
                );

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

            // Regenerate test case
            let input_names: Vec<&str> = vec!["input"];
            let test_case = regenerate_test_case(repro, &input_names);

            Response::ReproduceResult {
                result: failure,
                inputs: test_case.inputs,
            }
        }

        Request::Minimize { seed, strategy: _, max_iters: _ } => {
            info!("Minimize request: seed={}", seed);

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

            // For now, just return the original failure
            // TODO: Implement actual minimization with binary search
            let repro = failure.repro_info.as_ref();
            let minimized_shape = repro.map(|r| r.shape.clone()).unwrap_or_default();

            Response::MinimizeResult {
                original_seed: seed,
                minimized_seed: seed,
                minimized_shape,
                result: failure,
            }
        }

        Request::ListFailures { limit } => {
            info!("ListFailures request: limit={}", limit);

            let state_read = state.read().await;

            match state_read.storage.list_failures(limit) {
                Ok(failures) => Response::Results(failures),
                Err(e) => Response::Error {
                    code: ErrorCode::InternalError,
                    message: format!("Failed to list failures: {}", e),
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
