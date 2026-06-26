//! gpuemu CLI: Command-line interface for GPU-less validation.

mod debug;
mod init;
mod report;
mod signed_report;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use gpuemu_common::config::GpuemuConfig;
use gpuemu_common::protocol::{
    deserialize_response, serialize_request, MinimizeStrategy, Request, Response,
};
use gpuemu_common::types::{parse_dtypes, FuzzConfig, LayoutType, ShapeOptions, ValueDistribution};
use gpuemu_common::{default_socket_path, ensure_gpuemu_dir};
use nng::options::Options;
use nng::{Protocol, Socket};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;
use tracing::info;
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

        /// Include example ops and tests
        #[arg(long)]
        with_examples: bool,

        /// CI platform to set up (github, gitlab)
        #[arg(long)]
        ci: Option<String>,

        /// Target directory (defaults to current directory)
        #[arg(short, long)]
        target_dir: Option<PathBuf>,
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

    // =========================================================================
    // Phase 2: Fuzzing and Reproducibility
    // =========================================================================
    /// Fuzz test ops with random inputs
    Fuzz {
        /// Op name to fuzz (or omit for all ops)
        #[arg(short, long)]
        op: Option<String>,

        /// Number of test iterations
        #[arg(short, long, default_value = "100")]
        iterations: usize,

        /// Master seed for reproducibility
        #[arg(long)]
        seed: Option<u64>,

        /// Stop on first failure
        #[arg(long)]
        fail_fast: bool,
    },

    /// Reproduce a failing test case by seed
    Reproduce {
        /// Seed of the failing test case
        seed: u64,

        /// Show verbose output including input values
        #[arg(short, long)]
        verbose: bool,

        /// Emit a self-contained reproducer JSON (seed + shape/dtype/layout +
        /// base64 input snapshot) for external tools to consume.
        #[arg(long)]
        reproducer: bool,
    },

    /// Minimize a failing test case
    Minimize {
        /// Seed of the failing test case
        seed: u64,

        /// Minimization strategy
        #[arg(long, value_enum, default_value = "binary-search-dims")]
        strategy: MinimizeStrategyArg,

        /// Maximum iterations
        #[arg(long, default_value = "100")]
        max_iters: usize,
    },

    /// List stored failures
    Failures {
        /// Number of failures to show
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },

    // =========================================================================
    // Phase 3: Artifact Inspection
    // =========================================================================
    /// Lint kernel artifacts against policy rules
    Lint {
        /// Kernel name to lint (or omit for auto-detect from PTX)
        #[arg(short, long)]
        kernel: Option<String>,

        /// Path to PTX file
        #[arg(short, long)]
        ptx: PathBuf,

        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Store current artifacts as a baseline
    Baseline {
        /// Tag name for the baseline
        tag: String,
    },

    /// Compare current artifacts against a baseline
    Diff {
        /// Baseline tag to compare against
        #[arg(long)]
        baseline: String,

        /// Fail with exit code 1 on any regression
        #[arg(long)]
        fail_on_regression: bool,

        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Show artifact metrics for kernels
    Artifacts {
        /// Kernel name (or omit for all)
        #[arg(short, long)]
        kernel: Option<String>,
    },

    // =========================================================================
    // Phase 4: CI Integration
    // =========================================================================
    /// Run CI validation suite (combines fuzz, lint, diff)
    Ci {
        /// Run quick validation only (fewer dtypes, smaller shapes)
        #[arg(long)]
        quick: bool,

        /// Baseline tag to compare artifacts against
        #[arg(long)]
        baseline: Option<String>,

        /// Number of parallel validation jobs (0 = auto from config)
        #[arg(long, default_value = "0")]
        parallel: u32,

        /// Output format (text, json, junit, sarif, pr-comment)
        #[arg(long, default_value = "text")]
        format: String,

        /// Output file (stdout if not specified)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Emit a correctness-coverage report (which configured ops have validation
    /// results) in a format consumable by Codecov / Sonar / similar dashboards.
    Coverage {
        /// Output format: codecov | json | text.
        #[arg(long, default_value = "codecov")]
        format: String,

        /// Output file (stdout if not specified).
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Generate report from stored validation results
    Report {
        /// Output format (text, json, junit, sarif, pr-comment, html)
        #[arg(long, default_value = "text")]
        format: String,

        /// Output file (stdout if not specified)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Include only results from last N hours
        #[arg(long)]
        since_hours: Option<u64>,

        /// Include lint results
        #[arg(long)]
        include_lint: bool,

        /// Include artifact diff results against baseline
        #[arg(long)]
        include_artifacts: Option<String>,

        /// Sign the report (HTML format only) with the user's ed25519 key.
        /// Reads/creates ~/.gpuemu/sign-ed25519.{pub,sec}. Embeds SHA-256 of
        /// the unsigned report + ed25519 signature + public-key fingerprint.
        #[arg(long)]
        signed: bool,
    },

    // =========================================================================
    // Phase 6: Developer UX
    // =========================================================================
    /// Interactive debugging mode
    Debug {
        /// Specific seed to investigate
        #[arg(long)]
        seed: Option<u64>,

        /// Use REPL mode (default)
        #[arg(long)]
        repl: bool,

        /// Filter by op name
        #[arg(long)]
        op: Option<String>,
    },
}

/// Minimization strategy argument for CLI.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum MinimizeStrategyArg {
    /// Binary search on tensor dimensions
    #[value(name = "binary-search-dims")]
    BinarySearchDims,
    /// Binary search on tensor values
    #[value(name = "binary-search-values")]
    BinarySearchValues,
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
        Commands::Init {
            name,
            framework,
            with_examples,
            ci,
            target_dir,
        } => handle_init(name, framework, with_examples, ci, target_dir),
        Commands::Test {
            quick,
            thorough,
            seed,
        } => handle_test(quick, thorough, seed),
        Commands::Status => handle_status(),
        Commands::Version => handle_version(),
        Commands::Fuzz {
            op,
            iterations,
            seed,
            fail_fast,
        } => handle_fuzz(op, iterations, seed, fail_fast),
        Commands::Reproduce {
            seed,
            verbose,
            reproducer,
        } => handle_reproduce(seed, verbose, reproducer),
        Commands::Minimize {
            seed,
            strategy,
            max_iters,
        } => handle_minimize(seed, strategy, max_iters),
        Commands::Failures { limit } => handle_failures(limit),
        Commands::Lint {
            kernel,
            ptx,
            format,
        } => handle_lint(kernel, ptx, format),
        Commands::Baseline { tag } => handle_baseline(tag),
        Commands::Diff {
            baseline,
            fail_on_regression,
            format,
        } => handle_diff(baseline, fail_on_regression, format),
        Commands::Artifacts { kernel } => handle_artifacts(kernel),
        Commands::Ci {
            quick,
            baseline,
            parallel,
            format,
            output,
        } => handle_ci(quick, baseline, parallel, format, output),
        Commands::Report {
            format,
            output,
            since_hours,
            include_lint,
            include_artifacts,
            signed,
        } => handle_report(format, output, since_hours, include_lint, include_artifacts, signed),
        Commands::Coverage { format, output } => handle_coverage(format, output),
        Commands::Debug { seed, repl, op } => handle_debug(seed, repl, op),
    }
}

fn handle_debug(seed: Option<u64>, _repl: bool, op: Option<String>) -> Result<()> {
    let config = debug::DebugConfig {
        seed,
        op,
        repl_mode: true, // TUI mode not yet implemented
    };

    debug::start_debug(config)
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

fn handle_init(
    name: Option<String>,
    framework: Option<String>,
    with_examples: bool,
    ci: Option<String>,
    target_dir: Option<PathBuf>,
) -> Result<()> {
    let config = init::InitConfig {
        name: name.unwrap_or_else(|| "my-project".to_string()),
        framework: framework.unwrap_or_else(|| "pytorch".to_string()),
        with_examples,
        ci,
        target_dir: target_dir.unwrap_or_else(|| PathBuf::from(".")),
    };

    let result = init::init_project(&config)?;

    result.print_summary();
    result.print_next_steps(&config);

    Ok(())
}

fn handle_test(quick: bool, thorough: bool, seed: Option<u64>) -> Result<()> {
    if !check_daemon_running() {
        println!("Daemon is not running. Start it with: gpuemu daemon start");
        return Ok(());
    }

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

    let iterations = match mode {
        "quick" => 10,
        "thorough" => 100,
        _ => 50,
    };

    let mut total_passed = 0;
    let mut total_failed = 0;

    for op in &config.ops {
        println!("\nValidating op: {}", op.name);

        let fuzz_config = FuzzConfig {
            seed: seed.unwrap_or_else(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64
            }),
            shape_options: ShapeOptions::default(),
            dtypes: parse_dtypes(
                &op.tolerances
                    .keys()
                    .cloned()
                    .chain(config.validation.dtypes.iter().cloned())
                    .collect::<Vec<_>>(),
            ),
            layouts: vec![LayoutType::Contiguous, LayoutType::Strided],
            op_schema: None,
            value_distribution: ValueDistribution::default(),
        };

        let request = Request::FuzzOp {
            op_name: op.name.clone(),
            fuzz_config,
            iterations,
            fail_fast: false,
        };

        match send_request(request) {
            Ok(Response::FuzzResults {
                seed: _,
                total,
                passed,
                failed,
                failures,
            }) => {
                println!("  Total: {}, Passed: {}, Failed: {}", total, passed, failed);
                total_passed += passed;
                total_failed += failed;

                if !failures.is_empty() {
                    println!("  Failures:");
                    for f in failures.iter().take(3) {
                        let msg = f
                            .failures
                            .first()
                            .map(|f| f.message.as_str())
                            .unwrap_or("unknown");
                        println!("    seed={}: {}", f.seed, msg);
                    }
                }
            }
            Ok(Response::Error { code, message }) => {
                println!("  Error ({:?}): {}", code, message);
            }
            Ok(other) => {
                println!("  Unexpected response: {:?}", other);
            }
            Err(e) => {
                println!("  Communication error: {}", e);
            }
        }
    }

    println!(
        "\nValidation complete: {} passed, {} failed",
        total_passed, total_failed
    );
    if total_failed > 0 {
        println!("Run 'gpuemu failures' to see stored failures");
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
            Ok(Response::Pong {
                version,
                uptime_secs,
                ..
            }) => {
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

// =============================================================================
// Phase 2: Fuzzing and Reproducibility Handlers
// =============================================================================

fn handle_fuzz(
    op: Option<String>,
    iterations: usize,
    seed: Option<u64>,
    fail_fast: bool,
) -> Result<()> {
    // Check daemon is running
    if !check_daemon_running() {
        println!("Daemon is not running. Start it with: gpuemu daemon start");
        return Ok(());
    }

    // Load config to get op names
    let config = GpuemuConfig::find_and_load()?;

    // Determine which ops to fuzz
    let ops_to_fuzz: Vec<String> = match op {
        Some(name) => {
            if config.ops.iter().any(|o| o.name == name) {
                vec![name]
            } else {
                anyhow::bail!("Op '{}' not found in configuration", name);
            }
        }
        None => config.ops.iter().map(|o| o.name.clone()).collect(),
    };

    if ops_to_fuzz.is_empty() {
        println!("No ops configured. Add them to gpuemu.toml");
        return Ok(());
    }

    // Generate seed if not provided
    let master_seed = seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    });

    println!(
        "Fuzzing {} op(s) with {} iterations each",
        ops_to_fuzz.len(),
        iterations
    );
    println!("Master seed: {}", master_seed);
    println!();

    let mut total_passed = 0;
    let mut total_failed = 0;

    for op_name in &ops_to_fuzz {
        println!("Fuzzing op: {}", op_name);

        let op_config = config.ops.iter().find(|op| &op.name == op_name);
        let dtypes = op_config
            .map(|op| {
                parse_dtypes(
                    &op.tolerances
                        .keys()
                        .cloned()
                        .chain(config.validation.dtypes.iter().cloned())
                        .collect::<Vec<_>>(),
                )
            })
            .unwrap_or_else(|| parse_dtypes(&config.validation.dtypes));
        let layouts = vec![LayoutType::Contiguous, LayoutType::Strided];

        let fuzz_config = FuzzConfig {
            seed: master_seed,
            shape_options: ShapeOptions::default(),
            dtypes,
            layouts,
            op_schema: None,
            value_distribution: ValueDistribution::default(),
        };

        let request = Request::FuzzOp {
            op_name: op_name.clone(),
            fuzz_config,
            iterations,
            fail_fast,
        };

        match send_request(request) {
            Ok(Response::FuzzResults {
                seed: _,
                total,
                passed,
                failed,
                failures,
            }) => {
                println!("  Total: {}, Passed: {}, Failed: {}", total, passed, failed);
                total_passed += passed;
                total_failed += failed;

                if !failures.is_empty() {
                    println!("  Failures:");
                    for (i, f) in failures.iter().take(5).enumerate() {
                        println!(
                            "    [{}] seed={}: {}",
                            i + 1,
                            f.seed,
                            f.failures
                                .first()
                                .map(|f| f.message.as_str())
                                .unwrap_or("unknown")
                        );
                    }
                    if failures.len() > 5 {
                        println!("    ... and {} more", failures.len() - 5);
                    }
                }
            }
            Ok(Response::Error { code, message }) => {
                println!("  Error ({:?}): {}", code, message);
            }
            Ok(other) => {
                println!("  Unexpected response: {:?}", other);
            }
            Err(e) => {
                println!("  Communication error: {}", e);
            }
        }
        println!();
    }

    println!("Summary: {} passed, {} failed", total_passed, total_failed);
    if total_failed > 0 {
        println!("Run 'gpuemu failures' to see stored failures");
        println!("Run 'gpuemu reproduce <seed>' to reproduce a failure");
    }

    Ok(())
}

fn handle_reproduce(seed: u64, verbose: bool, reproducer: bool) -> Result<()> {
    // Check daemon is running
    if !check_daemon_running() {
        println!("Daemon is not running. Start it with: gpuemu daemon start");
        return Ok(());
    }

    let request = Request::Reproduce { seed };

    // --reproducer: emit only the machine-readable JSON, nothing else.
    if reproducer {
        return match send_request(request) {
            Ok(Response::ReproduceResult { result, .. }) => {
                let repro = gpuemu_common::replay::Reproducer::from_result(&result);
                println!("{}", repro.to_json()?);
                Ok(())
            }
            Ok(Response::Error { code, message }) => {
                anyhow::bail!("Error ({:?}): {}", code, message)
            }
            Ok(other) => anyhow::bail!("Unexpected response: {:?}", other),
            Err(e) => Err(e),
        };
    }

    println!("Reproducing failure with seed: {}", seed);

    match send_request(request) {
        Ok(Response::ReproduceResult { result, inputs }) => {
            println!();
            println!("Op: {}", result.op_name);
            println!("Passed: {}", result.passed);
            println!("Seed: {}", result.seed);

            if let Some(repro) = &result.repro_info {
                println!("Shape: {:?}", repro.shape);
                println!("DType: {:?}", repro.dtype);
                println!("Layout: {:?}", repro.layout);
            }

            if !result.failures.is_empty() {
                println!("\nFailures:");
                for f in &result.failures {
                    println!("  - {:?}: {}", f.kind, f.message);
                    if let (Some(exp), Some(act)) = (f.expected, f.actual) {
                        println!("    Expected: {}, Actual: {}", exp, act);
                    }
                }
            }

            if verbose {
                println!("\nInputs:");
                for (name, tensor) in &inputs {
                    println!(
                        "  {}: shape={:?}, dtype={:?}, {} bytes",
                        name,
                        tensor.shape,
                        tensor.dtype,
                        tensor.data.len()
                    );
                }
            }
        }
        Ok(Response::Error { code, message }) => {
            println!("Error ({:?}): {}", code, message);
        }
        Ok(other) => {
            println!("Unexpected response: {:?}", other);
        }
        Err(e) => {
            println!("Communication error: {}", e);
        }
    }

    Ok(())
}

fn handle_minimize(seed: u64, strategy: MinimizeStrategyArg, max_iters: usize) -> Result<()> {
    // Check daemon is running
    if !check_daemon_running() {
        println!("Daemon is not running. Start it with: gpuemu daemon start");
        return Ok(());
    }

    println!("Minimizing failure with seed: {}", seed);
    println!("Strategy: {:?}", strategy);
    println!("Max iterations: {}", max_iters);

    let proto_strategy = match strategy {
        MinimizeStrategyArg::BinarySearchDims => MinimizeStrategy::BinarySearchDims,
        MinimizeStrategyArg::BinarySearchValues => MinimizeStrategy::BinarySearchValues,
    };

    let request = Request::Minimize {
        seed,
        strategy: proto_strategy,
        max_iters,
    };

    match send_request(request) {
        Ok(Response::MinimizeResult {
            original_seed,
            minimized_seed,
            minimized_shape,
            result,
        }) => {
            println!();
            println!("Original seed: {}", original_seed);
            println!("Minimized seed: {}", minimized_seed);
            println!("Minimized shape: {:?}", minimized_shape);
            println!();
            println!("Minimized result:");
            println!("  Op: {}", result.op_name);
            println!("  Passed: {}", result.passed);

            if !result.failures.is_empty() {
                println!("  Failures:");
                for f in &result.failures {
                    println!("    - {:?}: {}", f.kind, f.message);
                }
            }
        }
        Ok(Response::Error { code, message }) => {
            println!("Error ({:?}): {}", code, message);
        }
        Ok(other) => {
            println!("Unexpected response: {:?}", other);
        }
        Err(e) => {
            println!("Communication error: {}", e);
        }
    }

    Ok(())
}

fn handle_failures(limit: usize) -> Result<()> {
    // Check daemon is running
    if !check_daemon_running() {
        println!("Daemon is not running. Start it with: gpuemu daemon start");
        return Ok(());
    }

    println!("Listing up to {} recent failures...\n", limit);

    let request = Request::ListFailures { limit };

    match send_request(request) {
        Ok(Response::Results { results: failures }) => {
            if failures.is_empty() {
                println!("No failures stored.");
            } else {
                println!(
                    "{:<20} {:<15} {:<10} {}",
                    "SEED", "OP", "PASSED", "FIRST FAILURE"
                );
                println!("{}", "-".repeat(70));

                for f in &failures {
                    let first_failure = f
                        .failures
                        .first()
                        .map(|f| f.message.chars().take(30).collect::<String>())
                        .unwrap_or_else(|| "-".to_string());
                    println!(
                        "{:<20} {:<15} {:<10} {}",
                        f.seed,
                        f.op_name.chars().take(15).collect::<String>(),
                        f.passed,
                        first_failure
                    );
                }

                println!();
                println!("Total: {} failures", failures.len());
                println!("\nTo reproduce a failure: gpuemu reproduce <seed>");
            }
        }
        Ok(Response::Error { code, message }) => {
            println!("Error ({:?}): {}", code, message);
        }
        Ok(other) => {
            println!("Unexpected response: {:?}", other);
        }
        Err(e) => {
            println!("Communication error: {}", e);
        }
    }

    Ok(())
}

// =============================================================================
// Phase 3: Artifact Inspection Handlers
// =============================================================================

fn handle_lint(kernel: Option<String>, ptx: PathBuf, format: String) -> Result<()> {
    // Check daemon is running
    if !check_daemon_running() {
        println!("Daemon is not running. Start it with: gpuemu daemon start");
        return Ok(());
    }

    // Read PTX file
    let ptx_content = std::fs::read_to_string(&ptx)
        .with_context(|| format!("Failed to read PTX file: {:?}", ptx))?;

    println!("Linting PTX file: {:?}", ptx);
    if let Some(ref k) = kernel {
        println!("Kernel: {}", k);
    }
    println!();

    let request = Request::LintKernel {
        kernel_name: kernel,
        ptx_content,
    };

    match send_request(request) {
        Ok(Response::LintResults { results }) => {
            if format == "json" {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&results).unwrap_or_default()
                );
            } else {
                let mut all_passed = true;
                for result in &results {
                    let status = if result.passed { "PASS" } else { "FAIL" };
                    all_passed &= result.passed;

                    println!("[{}] {}", status, result.kernel_name);
                    println!("  Registers: {}", result.metrics.register_count);
                    println!("  Spills: {}", result.metrics.spill_count);
                    println!(
                        "  Local memory: {} bytes",
                        result.metrics.local_memory_bytes
                    );
                    println!(
                        "  Shared memory: {} bytes",
                        result.metrics.shared_memory_bytes
                    );
                    println!("  Instructions: {}", result.metrics.instruction_count);

                    if !result.metrics.patterns_found.is_empty() {
                        println!("  Patterns: {}", result.metrics.patterns_found.join(", "));
                    }

                    if !result.violations.is_empty() {
                        println!("  Violations:");
                        for v in &result.violations {
                            println!("    - {:?}: {}", v.kind, v.message);
                        }
                    }
                    println!();
                }

                if all_passed {
                    println!("All kernels passed lint checks.");
                } else {
                    println!("Some kernels failed lint checks.");
                    std::process::exit(1);
                }
            }
        }
        Ok(Response::Error { code, message }) => {
            println!("Error ({:?}): {}", code, message);
            std::process::exit(1);
        }
        Ok(other) => {
            println!("Unexpected response: {:?}", other);
        }
        Err(e) => {
            println!("Communication error: {}", e);
        }
    }

    Ok(())
}

fn handle_baseline(tag: String) -> Result<()> {
    // Check daemon is running
    if !check_daemon_running() {
        println!("Daemon is not running. Start it with: gpuemu daemon start");
        return Ok(());
    }

    println!("Storing current artifacts as baseline: {}", tag);

    let request = Request::StoreArtifactBaseline { tag: tag.clone() };

    match send_request(request) {
        Ok(Response::Ok) => {
            println!("Baseline '{}' stored successfully.", tag);
        }
        Ok(Response::Error { code, message }) => {
            println!("Error ({:?}): {}", code, message);
            std::process::exit(1);
        }
        Ok(other) => {
            println!("Unexpected response: {:?}", other);
        }
        Err(e) => {
            println!("Communication error: {}", e);
        }
    }

    Ok(())
}

fn handle_diff(baseline: String, fail_on_regression: bool, format: String) -> Result<()> {
    // Check daemon is running
    if !check_daemon_running() {
        println!("Daemon is not running. Start it with: gpuemu daemon start");
        return Ok(());
    }

    println!("Comparing current artifacts against baseline: {}", baseline);
    println!();

    let request = Request::DiffArtifactBaseline {
        tag: baseline.clone(),
    };

    match send_request(request) {
        Ok(Response::ArtifactDiffs {
            baseline_tag,
            diffs,
            has_regressions,
        }) => {
            if format == "json" {
                let json_output = serde_json::json!({
                    "baseline_tag": baseline_tag,
                    "has_regressions": has_regressions,
                    "diffs": diffs,
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json_output).unwrap_or_default()
                );
            } else {
                if diffs.is_empty() {
                    println!("No artifacts to compare.");
                    return Ok(());
                }

                println!(
                    "{:<30} {:>10} {:>10} {:>12} {:>10} {}",
                    "KERNEL", "REGS", "SPILLS", "LOCAL_MEM", "INSTRS", "STATUS"
                );
                println!("{}", "-".repeat(85));

                for diff in &diffs {
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

                    println!(
                        "{:<30} {:>10} {:>10} {:>12} {:>10} {}",
                        diff.kernel_name.chars().take(30).collect::<String>(),
                        format_delta(diff.register_delta),
                        format_delta(diff.spill_delta),
                        format_delta(diff.local_memory_delta),
                        format_delta(diff.instruction_delta),
                        status
                    );
                }

                println!();
                if has_regressions {
                    println!("Regressions detected!");
                    if fail_on_regression {
                        std::process::exit(1);
                    }
                } else {
                    println!("No regressions detected.");
                }
            }
        }
        Ok(Response::Error { code, message }) => {
            println!("Error ({:?}): {}", code, message);
            std::process::exit(1);
        }
        Ok(other) => {
            println!("Unexpected response: {:?}", other);
        }
        Err(e) => {
            println!("Communication error: {}", e);
        }
    }

    Ok(())
}

fn handle_artifacts(kernel: Option<String>) -> Result<()> {
    // Check daemon is running
    if !check_daemon_running() {
        println!("Daemon is not running. Start it with: gpuemu daemon start");
        return Ok(());
    }

    match kernel {
        Some(name) => {
            println!("Fetching artifact metrics for kernel: {}", name);
            println!();

            let request = Request::GetArtifact {
                kernel_name: name.clone(),
            };

            match send_request(request) {
                Ok(Response::ArtifactMetricsResult(metrics)) => {
                    println!("Kernel: {}", metrics.kernel_name);
                    println!("  Source: {:?}", metrics.source);
                    println!("  Registers: {}", metrics.register_count);
                    println!("  Spills: {}", metrics.spill_count);
                    println!("  Local memory: {} bytes", metrics.local_memory_bytes);
                    println!("  Shared memory: {} bytes", metrics.shared_memory_bytes);
                    println!("  Instructions: {}", metrics.instruction_count);
                    if !metrics.patterns_found.is_empty() {
                        println!("  Patterns: {}", metrics.patterns_found.join(", "));
                    }
                }
                Ok(Response::Error { code, message }) => {
                    println!("Error ({:?}): {}", code, message);
                    std::process::exit(1);
                }
                Ok(other) => {
                    println!("Unexpected response: {:?}", other);
                }
                Err(e) => {
                    println!("Communication error: {}", e);
                }
            }
        }
        None => {
            println!("Listing all stored artifacts...");
            println!();

            let request = Request::ListArtifacts;

            match send_request(request) {
                Ok(Response::ArtifactList { artifacts }) => {
                    if artifacts.is_empty() {
                        println!("No artifacts stored.");
                        println!("Run 'gpuemu lint --ptx <file>' to analyze and store artifacts.");
                    } else {
                        println!(
                            "{:<30} {:>8} {:>8} {:>12} {:>10}",
                            "KERNEL", "REGS", "SPILLS", "LOCAL_MEM", "INSTRS"
                        );
                        println!("{}", "-".repeat(75));

                        for m in &artifacts {
                            println!(
                                "{:<30} {:>8} {:>8} {:>12} {:>10}",
                                m.kernel_name.chars().take(30).collect::<String>(),
                                m.register_count,
                                m.spill_count,
                                m.local_memory_bytes,
                                m.instruction_count
                            );
                        }

                        println!();
                        println!("Total: {} artifacts", artifacts.len());
                    }
                }
                Ok(Response::Error { code, message }) => {
                    println!("Error ({:?}): {}", code, message);
                    std::process::exit(1);
                }
                Ok(other) => {
                    println!("Unexpected response: {:?}", other);
                }
                Err(e) => {
                    println!("Communication error: {}", e);
                }
            }
        }
    }

    Ok(())
}

// =============================================================================
// Phase 4: CI Integration Handlers
// =============================================================================

fn handle_ci(
    quick: bool,
    baseline: Option<String>,
    parallel: u32,
    format: String,
    output: Option<PathBuf>,
) -> Result<()> {
    // Check daemon is running
    if !check_daemon_running() {
        println!("Daemon is not running. Start it with: gpuemu daemon start");
        return Ok(());
    }

    let output_format =
        report::OutputFormat::from_str(&format).unwrap_or(report::OutputFormat::Text);

    println!("Running CI validation suite...");
    if quick {
        println!("Mode: quick");
    }
    if let Some(ref b) = baseline {
        println!("Baseline: {}", b);
    }
    println!();

    let request = Request::RunCi {
        quick,
        baseline,
        parallel_jobs: parallel,
    };

    match send_request(request) {
        Ok(Response::CiRunComplete(summary)) => {
            let report_content = report::generate_report(&summary, output_format);

            // Output to file or stdout
            if let Some(path) = output {
                std::fs::write(&path, &report_content)
                    .with_context(|| format!("Failed to write report to {:?}", path))?;
                println!("Report written to {:?}", path);
            } else {
                println!("{}", report_content);
            }

            // Exit with appropriate code
            std::process::exit(summary.exit_code());
        }
        Ok(Response::Error { code, message }) => {
            println!("Error ({:?}): {}", code, message);
            std::process::exit(1);
        }
        Ok(other) => {
            println!("Unexpected response: {:?}", other);
            std::process::exit(1);
        }
        Err(e) => {
            println!("Communication error: {}", e);
            std::process::exit(1);
        }
    }
}

fn handle_report(
    format: String,
    output: Option<PathBuf>,
    since_hours: Option<u64>,
    include_lint: bool,
    include_artifacts: Option<String>,
    signed: bool,
) -> Result<()> {
    // Check daemon is running
    if !check_daemon_running() {
        println!("Daemon is not running. Start it with: gpuemu daemon start");
        return Ok(());
    }

    let output_format =
        report::OutputFormat::from_str(&format).unwrap_or(report::OutputFormat::Text);

    println!("Generating report...");
    println!();

    // Fetch validation results
    let validation_results = match send_request(Request::ListResults { limit: 1000 }) {
        Ok(Response::Results { results }) => {
            // Filter by time if specified
            if let Some(hours) = since_hours {
                let cutoff = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
                    .saturating_sub(hours * 3600);

                results
                    .into_iter()
                    .filter(|r| r.timestamp >= cutoff)
                    .collect()
            } else {
                results
            }
        }
        _ => Vec::new(),
    };

    // Fetch lint results if requested
    let lint_results: Vec<gpuemu_common::types::LintResult> = if include_lint {
        // For now, return empty - would need stored lint results
        Vec::new()
    } else {
        Vec::new()
    };

    // Fetch artifact diffs if baseline specified
    let artifact_diffs = if let Some(ref tag) = include_artifacts {
        match send_request(Request::DiffArtifactBaseline { tag: tag.clone() }) {
            Ok(Response::ArtifactDiffs {
                baseline_tag,
                diffs,
                has_regressions,
            }) => Some(gpuemu_common::types::ArtifactDiffSummary {
                baseline_tag,
                has_regressions,
                diffs,
            }),
            _ => None,
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

    let summary = gpuemu_common::types::CiRunSummary {
        total_tests: total,
        passed,
        failed,
        skipped: 0,
        duration_ms: 0, // Unknown for stored results
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        validation_results,
        lint_results,
        artifact_diffs,
    };

    // HTML / signed HTML is handled separately — it isn't a CiRunSummary
    // serialisation but a customer-facing artefact with embedded styling and
    // (optionally) an ed25519 signature footer.
    let report_content = if format.to_lowercase() == "html" || signed {
        let unsigned = signed_report::render_html(&summary);
        if signed {
            let key = signed_report::load_or_generate_keypair()?;
            signed_report::sign_html(&unsigned, &key)?
        } else {
            unsigned
        }
    } else {
        report::generate_report(&summary, output_format)
    };

    // Output to file or stdout
    if let Some(path) = output {
        std::fs::write(&path, &report_content)
            .with_context(|| format!("Failed to write report to {:?}", path))?;
        println!("Report written to {:?}", path);
        if signed {
            let pub_path = dirs::home_dir()
                .unwrap_or_default()
                .join(".gpuemu/sign-ed25519.pub");
            println!("Share the public key for verification: {}", pub_path.display());
        }
    } else {
        println!("{}", report_content);
    }

    Ok(())
}

/// Emit a correctness-coverage report.
///
/// "Coverage" here means *which configured ops have a recent validation result*
/// — the kernel-correctness analogue of line coverage. A project with 50 ops
/// in `gpuemu.toml` and only 40 of them seen by `gpuemu ci` has 80 % coverage;
/// the remaining 10 are still on the legacy `torch.testing.assert_close`
/// oracle and represent uncovered surface.
///
/// Supports three output formats:
///   - `codecov`: Codecov-compatible JSON ({ coverage: { <file>: <line>: 1|0 }})
///     using each op_name as a synthetic file. Codecov happily ingests this
///     even though there are no source lines per se.
///   - `json`: structured `{ covered: [...], uncovered: [...], percent: ... }`
///     for scripted ingestion.
///   - `text`: human-readable summary.
fn handle_coverage(format: String, output: Option<PathBuf>) -> Result<()> {
    if !check_daemon_running() {
        println!("Daemon is not running. Start it with: gpuemu daemon start");
        return Ok(());
    }

    // Fetch the configured op list from the loaded config (the daemon doesn't
    // expose this directly today; we read it from the same gpuemu.toml the
    // daemon was started against).
    let config_path = std::env::current_dir()
        .unwrap_or_default()
        .join("gpuemu.toml");
    let configured_ops: Vec<String> = if config_path.exists() {
        match std::fs::read_to_string(&config_path) {
            Ok(text) => match toml::from_str::<toml::Value>(&text) {
                Ok(toml::Value::Table(tbl)) => tbl
                    .get("ops")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|op| {
                                op.get("name").and_then(|n| n.as_str()).map(String::from)
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
                _ => Vec::new(),
            },
            Err(_) => Vec::new(),
        }
    } else {
        Vec::new()
    };

    // Fetch validated ops from the daemon's result store.
    let validated: std::collections::BTreeSet<String> =
        match send_request(Request::ListResults { limit: 10_000 }) {
            Ok(Response::Results { results }) => {
                results.into_iter().map(|r| r.op_name).collect()
            }
            _ => Default::default(),
        };

    let configured: std::collections::BTreeSet<String> =
        configured_ops.into_iter().collect();
    let covered: Vec<&String> = configured.intersection(&validated).collect();
    let uncovered: Vec<&String> = configured.difference(&validated).collect();
    let percent = if configured.is_empty() {
        0.0
    } else {
        covered.len() as f64 / configured.len() as f64 * 100.0
    };

    let body = match format.to_lowercase().as_str() {
        "codecov" => {
            // Codecov consumes a JSON of the form
            //   { "coverage": { "<file>": { "<line>": <hit-count> } } }
            // We model each op as a synthetic file with a single line; hit
            // count = 1 if validated, 0 if not. The frontend renders this as
            // a tree under gpuemu/.
            let mut coverage = serde_json::Map::new();
            for op in &configured {
                let key = format!("gpuemu/{}.op", op);
                let hit = if validated.contains(op) { 1 } else { 0 };
                let mut lines = serde_json::Map::new();
                lines.insert("1".into(), serde_json::Value::from(hit));
                coverage.insert(key, serde_json::Value::Object(lines));
            }
            serde_json::to_string_pretty(&serde_json::json!({
                "coverage": coverage,
                "messages": {
                    "gpuemu_summary": format!(
                        "{:.1}% kernel-correctness coverage ({} of {} ops validated; \
                         {} still on torch.allclose)",
                        percent, covered.len(), configured.len(), uncovered.len()
                    )
                }
            }))?
        }
        "json" => serde_json::to_string_pretty(&serde_json::json!({
            "covered": covered,
            "uncovered": uncovered,
            "configured_total": configured.len(),
            "percent": percent,
        }))?,
        _ => {
            let mut t = String::new();
            t.push_str(&format!(
                "gpuemu correctness coverage: {:.1}%  ({} / {} ops validated)\n\n",
                percent,
                covered.len(),
                configured.len(),
            ));
            t.push_str("Covered:\n");
            for op in &covered {
                t.push_str(&format!("  ✓ {}\n", op));
            }
            t.push_str("\nUncovered:\n");
            for op in &uncovered {
                t.push_str(&format!("  ✗ {}  (still on legacy oracle)\n", op));
            }
            t
        }
    };

    if let Some(path) = output {
        std::fs::write(&path, &body)
            .with_context(|| format!("writing {:?}", path))?;
        println!("Coverage report written to {:?}", path);
    } else {
        println!("{}", body);
    }
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
    send_request_to(&socket_path, request)
}

fn send_request_to(socket_path: &std::path::PathBuf, request: Request) -> Result<Response> {
    let socket_url = format!("ipc://{}", socket_path.display());

    let socket = Socket::new(Protocol::Req0).context("Failed to create socket")?;

    socket
        .set_opt::<nng::options::RecvTimeout>(Some(Duration::from_secs(30)))
        .context("Failed to set timeout")?;

    socket
        .dial(&socket_url)
        .with_context(|| format!("Failed to connect to {}", socket_url))?;

    let bytes = serialize_request(&request).map_err(|e| anyhow::anyhow!("{:?}", e))?;

    socket
        .send(&bytes)
        .map_err(|(_, e)| anyhow::anyhow!("Send failed: {}", e))?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use gpuemu_common::protocol::{serialize_response, PROTOCOL_VERSION};

    #[test]
    fn test_pong_response_parsing() {
        let response = Response::Pong {
            version: "0.1.0".to_string(),
            protocol_version: PROTOCOL_VERSION,
            uptime_secs: 42,
        };
        let bytes = serialize_response(&response).unwrap();
        let decoded = deserialize_response(&bytes).unwrap();
        match decoded {
            Response::Pong {
                version,
                protocol_version,
                uptime_secs,
                ..
            } => {
                assert_eq!(version, "0.1.0");
                assert_eq!(protocol_version, PROTOCOL_VERSION);
                assert_eq!(uptime_secs, 42);
            }
            _ => panic!("Expected Pong response"),
        }
    }

    #[test]
    fn test_request_serialization_for_ping() {
        let request = Request::Ping;
        let bytes = serialize_request(&request).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["type"], "Ping");
    }
}
