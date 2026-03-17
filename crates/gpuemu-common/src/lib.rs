//! gpuemu-common: Shared types, protocols, and configuration for gpuemu.
//!
//! This crate provides the core types used across the gpuemu daemon, CLI, and clients.

pub mod config;
pub mod protocol;
pub mod rng;
pub mod types;

pub use config::{GpuemuConfig, OpConfig, ValidationConfig};
pub use rng::SeededRng;
pub use protocol::{Request, Response, ErrorCode};
pub use types::{
    DType, TensorData, ValidationResult, ValidationFailure, FailureKind,
    FuzzConfig, ShapeOptions, LayoutType, ReproductionInfo,
};

/// Default socket path for the daemon.
pub fn default_socket_path() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".gpuemu")
        .join("gpuemu.sock")
}

/// Default database path for sled.
pub fn default_db_path() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".gpuemu")
        .join("db")
}

/// Default log directory.
pub fn default_log_path() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".gpuemu")
        .join("logs")
}

/// Ensure the gpuemu home directory exists.
pub fn ensure_gpuemu_dir() -> std::io::Result<std::path::PathBuf> {
    let gpuemu_dir = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".gpuemu");
    std::fs::create_dir_all(&gpuemu_dir)?;
    Ok(gpuemu_dir)
}

/// Package version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
