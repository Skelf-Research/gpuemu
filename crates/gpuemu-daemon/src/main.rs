//! gpuemu-daemon: GPU-less validation daemon for deep learning kernels.

mod artifact;
mod executor;
mod fuzzer;
mod server;
mod storage;
mod validator;

use anyhow::Result;
use gpuemu_common::config::GpuemuConfig;
use gpuemu_common::{default_db_path, default_socket_path, ensure_gpuemu_dir};
use server::{run_server, ServerState};
use std::sync::Arc;
use storage::Storage;
use tokio::sync::RwLock;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

fn validate_startup(config: &GpuemuConfig) {
    let mut issues = 0u32;

    // Check Python interpreter
    let python = "python3";
    match std::process::Command::new(python).arg("--version").output() {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
            info!("Python interpreter: {} ({})", python, version);
        }
        _ => {
            warn!("Python interpreter '{}' not found or not working", python);
            issues += 1;
        }
    }

    // Check reference scripts exist for configured ops
    for op in &config.ops {
        let ref_path = std::path::Path::new(&op.reference);
        if !ref_path.exists() {
            warn!("Op '{}': reference script '{}' not found", op.name, op.reference);
            issues += 1;
        } else {
            info!("Op '{}': reference script '{}' OK", op.name, op.reference);
        }
    }

    // Check kernel source files exist
    for kernel in &config.kernels {
        if let Some(ref source) = kernel.source {
            let source_path = std::path::Path::new(source);
            if !source_path.exists() {
                warn!("Kernel '{}': source file '{}' not found", kernel.name, source);
                issues += 1;
            }
        }
    }

    if issues > 0 {
        warn!("Startup validation found {} issue(s). Some features may not work.", issues);
    } else {
        info!("Startup validation passed");
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .init();

    info!("gpuemu-daemon v{}", gpuemu_common::VERSION);

    // Ensure gpuemu directory exists
    ensure_gpuemu_dir()?;

    // Load configuration (optional, use defaults if not found)
    let config = GpuemuConfig::find_and_load().unwrap_or_else(|_| {
        info!("No gpuemu.toml found, using default configuration");
        GpuemuConfig::default()
    });

    // Validate startup conditions
    validate_startup(&config);

    // Open storage
    let db_path = default_db_path();
    info!("Opening database at {:?}", db_path);
    let storage = Storage::open(&db_path)?;

    // Create server state
    let state = Arc::new(RwLock::new(ServerState::new(storage, config)));

    let signal_state = state.clone();
    tokio::spawn(async move {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};

            let mut sigterm = signal(SignalKind::terminate()).ok();
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = async {
                    if let Some(ref mut sigterm) = sigterm {
                        sigterm.recv().await;
                    } else {
                        std::future::pending::<()>().await;
                    }
                } => {}
            }
        }

        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
        }

        info!("Shutdown signal received");
        signal_state.write().await.shutdown_requested = true;
    });

    // Run server
    let socket_path = default_socket_path();
    info!("Starting server...");
    run_server(&socket_path, state).await?;

    Ok(())
}
