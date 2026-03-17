//! gpuemu CLI: Command-line interface for GPU-less validation.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use nng::options::Options;
use gpuemu_common::config::GpuemuConfig;
use gpuemu_common::protocol::{deserialize_response, serialize_request, Request, Response};
use gpuemu_common::{default_socket_path, ensure_gpuemu_dir};
use nng::{Protocol, Socket};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;
use tracing::{error, info};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser)]
#[command(name = "gpuemu")]
#[command(about = "GPU-less validation for deep learning kernels")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Manage the validation daemon
    Daemon {
        #[command(subcommand)]
        action: DaemonAction,
    },

    /// Initialize a new gpuemu project
    Init {
        /// Project name
        #[arg(short, long)]
        name: Option<String>,

        /// Target framework (pytorch, jax, tensorflow)
        #[arg(short, long)]
        framework: Option<String>,
    },

    /// Run validation tests
    Test {
        /// Run quick validation (subset of shapes/dtypes)
        #[arg(long)]
        quick: bool,

        /// Run thorough validation
        #[arg(long)]
        thorough: bool,

        /// Specific seed to use
        #[arg(long)]
        seed: Option<u64>,
    },

    /// Check daemon status
    Status,

    /// Show version information
    Version,
}

#[derive(Subcommand)]
enum DaemonAction {
    /// Start the daemon
    Start {
        /// Run in background
        #[arg(long)]
        background: bool,
    },

    /// Stop the daemon
    Stop,

    /// Check daemon status
    Status,

    /// Show daemon logs
    Logs {
        /// Number of lines to show
        #[arg(short, long, default_value = "50")]
        lines: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let level = if cli.verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };

    FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .init();

    match cli.command {
        Commands::Daemon { action } => handle_daemon(action),
        Commands::Init { name, framework } => handle_init(name, framework),
        Commands::Test { quick, thorough, seed } => handle_test(quick, thorough, seed),
        Commands::Status => handle_status(),
        Commands::Version => handle_version(),
    }
}

fn handle_daemon(action: DaemonAction) -> Result<()> {
    match action {
        DaemonAction::Start { background } => {
            ensure_gpuemu_dir()?;

            if background {
                info!("Starting daemon in background...");

                // Find the daemon binary
                let daemon_path = std::env::current_exe()?
                    .parent()
                    .context("No parent directory")?
                    .join("gpuemu-daemon");

                if !daemon_path.exists() {
                    // Try cargo target directory
                    let alt_path = PathBuf::from("target/debug/gpuemu-daemon");
                    if alt_path.exists() {
                        start_daemon_background(&alt_path)?;
                    } else {
                        anyhow::bail!(
                            "Daemon binary not found. Run 'cargo build' first or use 'cargo run --bin gpuemu-daemon'"
                        );
                    }
                } else {
                    start_daemon_background(&daemon_path)?;
                }

                // Wait a moment for daemon to start
                std::thread::sleep(Duration::from_millis(500));

                // Check if daemon is running
                if check_daemon_running() {
                    println!("Daemon started successfully");
                    println!("Socket: {:?}", default_socket_path());
                } else {
                    println!("Warning: Daemon may not have started correctly");
                }
            } else {
                info!("Starting daemon in foreground...");
                println!("Starting daemon in foreground. Press Ctrl+C to stop.");

                // Run daemon directly
                let status = Command::new("cargo")
                    .args(["run", "--bin", "gpuemu-daemon"])
                    .status()?;

                if !status.success() {
                    anyhow::bail!("Daemon exited with error");
                }
            }
            Ok(())
        }

        DaemonAction::Stop => {
            info!("Stopping daemon...");

            if !check_daemon_running() {
                println!("Daemon is not running");
                return Ok(());
            }

            // Send shutdown request
            match send_request(Request::Shutdown) {
                Ok(_) => println!("Daemon stopped"),
                Err(e) => println!("Failed to stop daemon: {}", e),
            }

            // Remove socket file
            let socket_path = default_socket_path();
            if socket_path.exists() {
                std::fs::remove_file(&socket_path)?;
            }

            Ok(())
        }

        DaemonAction::Status => handle_status(),

        DaemonAction::Logs { lines } => {
            let log_path = gpuemu_common::default_log_path().join("daemon.log");
            if log_path.exists() {
                let output = Command::new("tail")
                    .args(["-n", &lines.to_string()])
                    .arg(&log_path)
                    .output()?;
                print!("{}", String::from_utf8_lossy(&output.stdout));
            } else {
                println!("No log file found at {:?}", log_path);
            }
            Ok(())
        }
    }
}

fn handle_init(name: Option<String>, framework: Option<String>) -> Result<()> {
    let config_path = PathBuf::from("gpuemu.toml");

    if config_path.exists() {
        anyhow::bail!("gpuemu.toml already exists");
    }

    let mut template = GpuemuConfig::default_template();

    // Replace placeholders if provided
    if let Some(n) = name {
        template = template.replace("my-project", &n);
    }
    if let Some(f) = framework {
        template = template.replace("pytorch", &f);
    }

    std::fs::write(&config_path, template)?;
    println!("Created gpuemu.toml");
    println!("\nNext steps:");
    println!("  1. Edit gpuemu.toml to configure your ops/kernels");
    println!("  2. Create reference scripts in scripts/");
    println!("  3. Run 'gpuemu daemon start' to start the daemon");
    println!("  4. Run 'gpuemu test' to validate");

    Ok(())
}

fn handle_test(quick: bool, thorough: bool, seed: Option<u64>) -> Result<()> {
    // Check daemon is running
    if !check_daemon_running() {
        println!("Daemon is not running. Start it with: gpuemu daemon start");
        return Ok(());
    }

    // Load config
    let config = GpuemuConfig::find_and_load()?;

    let mode = if quick {
        "quick"
    } else if thorough {
        "thorough"
    } else {
        "standard"
    };

    println!("Running {} validation...", mode);
    println!("Ops to validate: {}", config.ops.len());
    println!("Kernels to validate: {}", config.kernels.len());

    if config.ops.is_empty() && config.kernels.is_empty() {
        println!("\nNo ops or kernels configured. Add them to gpuemu.toml");
        return Ok(());
    }

    // For MVP, just ping the daemon to show it's working
    match send_request(Request::Ping) {
        Ok(Response::Pong { version, uptime_secs }) => {
            println!("\nDaemon v{} (uptime: {}s)", version, uptime_secs);
            println!("Validation would run here in full implementation");
        }
        Ok(other) => {
            println!("Unexpected response: {:?}", other);
        }
        Err(e) => {
            error!("Failed to communicate with daemon: {}", e);
        }
    }

    Ok(())
}

fn handle_status() -> Result<()> {
    let socket_path = default_socket_path();

    print!("Socket: {:?} ", socket_path);
    if socket_path.exists() {
        println!("(exists)");
    } else {
        println!("(not found)");
    }

    print!("Daemon: ");
    if check_daemon_running() {
        match send_request(Request::Ping) {
            Ok(Response::Pong { version, uptime_secs }) => {
                println!("running (v{}, uptime {}s)", version, uptime_secs);
            }
            _ => {
                println!("running (unable to get details)");
            }
        }
    } else {
        println!("not running");
    }

    Ok(())
}

fn handle_version() -> Result<()> {
    println!("gpuemu {}", gpuemu_common::VERSION);
    println!("  daemon: gpuemu-daemon");
    println!("  client: gpuemu-py");
    Ok(())
}

/// Check if daemon is running by attempting to connect.
fn check_daemon_running() -> bool {
    let socket_path = default_socket_path();
    if !socket_path.exists() {
        return false;
    }

    // Try to connect
    let socket_url = format!("ipc://{}", socket_path.display());
    match Socket::new(Protocol::Req0) {
        Ok(socket) => {
            socket
                .set_opt::<nng::options::RecvTimeout>(Some(Duration::from_millis(500)))
                .ok();
            socket.dial(&socket_url).is_ok()
        }
        Err(_) => false,
    }
}

/// Send a request to the daemon and return the response.
fn send_request(request: Request) -> Result<Response> {
    let socket_path = default_socket_path();
    let socket_url = format!("ipc://{}", socket_path.display());

    let socket = Socket::new(Protocol::Req0).context("Failed to create socket")?;

    socket
        .set_opt::<nng::options::RecvTimeout>(Some(Duration::from_secs(30)))
        .context("Failed to set timeout")?;

    socket
        .dial(&socket_url)
        .with_context(|| format!("Failed to connect to {}", socket_url))?;

    let bytes = serialize_request(&request).map_err(|e| anyhow::anyhow!("{:?}", e))?;

    socket.send(&bytes).map_err(|(_, e)| anyhow::anyhow!("Send failed: {}", e))?;

    let response_bytes = socket.recv().context("Failed to receive response")?;

    deserialize_response(&response_bytes).context("Failed to deserialize response")
}

/// Start daemon in background.
fn start_daemon_background(daemon_path: &PathBuf) -> Result<()> {
    Command::new(daemon_path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("Failed to spawn daemon")?;
    Ok(())
}
