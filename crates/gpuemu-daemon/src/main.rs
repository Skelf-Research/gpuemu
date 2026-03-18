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
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
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

    // Open storage
    let db_path = default_db_path();
    info!("Opening database at {:?}", db_path);
    let storage = Storage::open(&db_path)?;

    // Create server state
    let state = Arc::new(RwLock::new(ServerState::new(storage, config)));

    // Run server
    let socket_path = default_socket_path();
    info!("Starting server...");
    run_server(&socket_path, state).await?;

    Ok(())
}
