//! REPL mode for interactive debugging.

use anyhow::{Context, Result};
use gpuemu_common::protocol::{
    deserialize_response, serialize_request, MinimizeStrategy, Request, Response,
};
use gpuemu_common::{default_socket_path, types::ValidationResult};
use nng::{options::Options, Protocol, Socket};
use std::io::{self, BufRead, Write};
use std::time::Duration;

use super::DebugConfig;

/// Start the REPL debug session.
pub fn start_repl(config: DebugConfig) -> Result<()> {
    println!("gpuemu debug REPL");
    println!("Type 'help' for available commands, 'quit' to exit.\n");

    // Check daemon is running
    if !check_daemon_running() {
        println!("Warning: Daemon is not running. Start with: gpuemu daemon start");
        println!();
    }

    // Load initial failures
    let mut failures = load_failures()?;
    let mut current_seed: Option<u64> = config.seed;

    // Apply filter if specified
    if let Some(ref op) = config.op {
        failures.retain(|f| f.op_name == *op);
        println!("Filtered to {} failures for op '{}'", failures.len(), op);
    }

    // If seed specified, show it
    if let Some(seed) = current_seed {
        show_failure_details(seed)?;
    } else if !failures.is_empty() {
        println!("Loaded {} failures.", failures.len());
        println!("Use 'list' to see failures or 'show <seed>' to inspect one.\n");
    }

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        // Print prompt
        if let Some(seed) = current_seed {
            print!("gpuemu [{}]> ", seed);
        } else {
            print!("gpuemu> ");
        }
        stdout.flush()?;

        // Read command
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            // EOF
            break;
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Parse and execute command
        let parts: Vec<&str> = line.split_whitespace().collect();
        let cmd = parts[0].to_lowercase();
        let args = &parts[1..];

        match cmd.as_str() {
            "help" | "h" | "?" => print_help(),
            "quit" | "exit" | "q" => {
                println!("Goodbye!");
                break;
            }
            "list" | "ls" => {
                let limit = args.first().and_then(|s| s.parse().ok()).unwrap_or(20);
                list_failures(limit)?;
            }
            "show" | "s" => {
                if let Some(seed_str) = args.first() {
                    if let Ok(seed) = seed_str.parse::<u64>() {
                        show_failure_details(seed)?;
                        current_seed = Some(seed);
                    } else {
                        println!("Invalid seed: {}", seed_str);
                    }
                } else if let Some(seed) = current_seed {
                    show_failure_details(seed)?;
                } else {
                    println!("Usage: show <seed>");
                }
            }
            "minimize" | "min" => {
                let seed = args.first().and_then(|s| s.parse().ok()).or(current_seed);

                if let Some(seed) = seed {
                    minimize_failure(seed)?;
                } else {
                    println!("Usage: minimize <seed>");
                }
            }
            "reproduce" | "repro" | "r" => {
                let seed = args.first().and_then(|s| s.parse().ok()).or(current_seed);

                if let Some(seed) = seed {
                    reproduce_failure(seed)?;
                } else {
                    println!("Usage: reproduce <seed>");
                }
            }
            "export" | "e" => {
                let seed = args.first().and_then(|s| s.parse().ok()).or(current_seed);

                if let Some(seed) = seed {
                    export_reproducer(seed)?;
                } else {
                    println!("Usage: export <seed>");
                }
            }
            "tensor" | "t" => {
                if args.is_empty() {
                    println!("Usage: tensor <name>[start:end]");
                } else {
                    let tensor_spec = args.join(" ");
                    inspect_tensor(&tensor_spec, current_seed)?;
                }
            }
            "refresh" => {
                failures = load_failures()?;
                println!("Reloaded {} failures.", failures.len());
            }
            "status" => {
                check_status()?;
            }
            "clear" => {
                // ANSI escape to clear screen
                print!("\x1b[2J\x1b[H");
                stdout.flush()?;
            }
            _ => {
                println!(
                    "Unknown command: {}. Type 'help' for available commands.",
                    cmd
                );
            }
        }

        println!();
    }

    Ok(())
}

fn print_help() {
    println!("Available commands:");
    println!();
    println!("  list [limit]       - List recent failures (default: 20)");
    println!("  show <seed>        - Show details for a failure");
    println!("  reproduce <seed>   - Re-run validation for a failure");
    println!("  minimize <seed>    - Minimize failing test case");
    println!("  export <seed>      - Export reproducer script");
    println!("  tensor <name>      - Inspect tensor values");
    println!("  refresh            - Reload failures from daemon");
    println!("  status             - Check daemon status");
    println!("  clear              - Clear screen");
    println!("  help               - Show this help");
    println!("  quit               - Exit debug mode");
    println!();
    println!("Tip: If a seed is selected, commands use it by default.");
}

fn load_failures() -> Result<Vec<ValidationResult>> {
    let request = Request::ListFailures { limit: 100 };

    match send_request(request) {
        Ok(Response::Results { results: failures }) => Ok(failures),
        Ok(Response::Error { code, message }) => {
            anyhow::bail!("Error ({:?}): {}", code, message);
        }
        Ok(_) => Ok(Vec::new()),
        Err(e) => {
            // Daemon might not be running
            eprintln!("Warning: Could not load failures: {}", e);
            Ok(Vec::new())
        }
    }
}

fn list_failures(limit: usize) -> Result<()> {
    let request = Request::ListFailures { limit };

    match send_request(request)? {
        Response::Results { results: failures } => {
            if failures.is_empty() {
                println!("No failures stored.");
            } else {
                println!("{:<20} {:<15} {}", "SEED", "OP", "FIRST FAILURE");
                println!("{}", "-".repeat(60));

                for f in &failures {
                    let first_failure = f
                        .failures
                        .first()
                        .map(|f| f.message.chars().take(30).collect::<String>())
                        .unwrap_or_else(|| "-".to_string());
                    println!("{:<20} {:<15} {}", f.seed, f.op_name, first_failure);
                }

                println!();
                println!("Total: {} failures", failures.len());
            }
        }
        Response::Error { code, message } => {
            println!("Error ({:?}): {}", code, message);
        }
        _ => {
            println!("Unexpected response");
        }
    }

    Ok(())
}

fn show_failure_details(seed: u64) -> Result<()> {
    println!("Fetching details for seed {}...", seed);

    let request = Request::Reproduce { seed };

    match send_request(request)? {
        Response::ReproduceResult { result, inputs } => {
            println!();
            println!("Op: {}", result.op_name);
            println!("Seed: {}", result.seed);
            println!("Passed: {}", result.passed);

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
                        println!(
                            "    Expected: {:.6}, Actual: {:.6}, Diff: {:.6e}",
                            exp,
                            act,
                            (exp - act).abs()
                        );
                    }
                    if let Some(idx) = f.index {
                        println!("    At index: {}", idx);
                    }
                }
            }

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
        Response::Error { code, message } => {
            println!("Error ({:?}): {}", code, message);
        }
        _ => {
            println!("Unexpected response");
        }
    }

    Ok(())
}

fn reproduce_failure(seed: u64) -> Result<()> {
    println!("Reproducing failure with seed {}...", seed);

    let request = Request::Reproduce { seed };

    match send_request(request)? {
        Response::ReproduceResult { result, .. } => {
            if result.passed {
                println!("Test PASSED on reproduction.");
                println!("The failure may have been intermittent or fixed.");
            } else {
                println!("Test FAILED on reproduction.");
                println!("Failures:");
                for f in &result.failures {
                    println!("  - {:?}: {}", f.kind, f.message);
                }
            }
        }
        Response::Error { code, message } => {
            println!("Error ({:?}): {}", code, message);
        }
        _ => {
            println!("Unexpected response");
        }
    }

    Ok(())
}

fn minimize_failure(seed: u64) -> Result<()> {
    println!("Minimizing failure with seed {}...", seed);
    println!("This may take a while...");

    let request = Request::Minimize {
        seed,
        strategy: MinimizeStrategy::BinarySearchDims,
        max_iters: 100,
    };

    match send_request(request)? {
        Response::MinimizeResult {
            original_seed,
            minimized_seed,
            minimized_shape,
            result,
        } => {
            println!();
            println!("Minimization complete!");
            println!("Original seed: {}", original_seed);
            println!("Minimized seed: {}", minimized_seed);
            println!("Minimized shape: {:?}", minimized_shape);

            if !result.failures.is_empty() {
                println!("\nMinimized failure:");
                for f in &result.failures {
                    println!("  - {:?}: {}", f.kind, f.message);
                }
            }
        }
        Response::Error { code, message } => {
            println!("Error ({:?}): {}", code, message);
        }
        _ => {
            println!("Unexpected response");
        }
    }

    Ok(())
}

fn export_reproducer(seed: u64) -> Result<()> {
    println!("Exporting reproducer script for seed {}...", seed);

    let request = Request::Reproduce { seed };

    match send_request(request)? {
        Response::ReproduceResult { result, inputs } => {
            let script = generate_reproducer_script(&result, &inputs);

            let filename = format!("reproduce_{}.py", seed);
            std::fs::write(&filename, &script)?;

            println!("Reproducer script written to: {}", filename);
            println!("\nRun with: python {}", filename);
        }
        Response::Error { code, message } => {
            println!("Error ({:?}): {}", code, message);
        }
        _ => {
            println!("Unexpected response");
        }
    }

    Ok(())
}

fn generate_reproducer_script(
    result: &ValidationResult,
    inputs: &std::collections::HashMap<String, gpuemu_common::types::TensorData>,
) -> String {
    let mut script = String::new();

    script.push_str("#!/usr/bin/env python3\n");
    script.push_str(&format!(
        "\"\"\"Reproducer for failure seed {}\n",
        result.seed
    ));
    script.push_str(&format!("Op: {}\n", result.op_name));
    script.push_str("\"\"\"\n\n");

    script.push_str("import numpy as np\n");
    script.push_str("import torch  # Adjust for your framework\n\n");

    // Generate input data
    script.push_str("# Input data\n");
    for (name, tensor) in inputs {
        let shape_str = tensor
            .shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        script.push_str(&format!(
            "# {}: shape=[{}], dtype={:?}\n",
            name, shape_str, tensor.dtype
        ));
        script.push_str(&format!("{}_data = np.frombuffer(\n", name));
        script.push_str(&format!(
            "    bytes.fromhex('{}'),\n",
            tensor
                .data
                .iter()
                .map(|b| format!("{:02x}", b))
                .collect::<String>()
        ));
        script.push_str(&format!("    dtype=np.{}\n", tensor.dtype.to_numpy_dtype()));
        script.push_str(&format!(").reshape({})\n", shape_str));
        script.push_str(&format!("{} = torch.from_numpy({}_data)\n\n", name, name));
    }

    // Generate test
    script.push_str("# Run the op\n");
    script.push_str(&format!("# result = your_{}(**inputs)\n\n", result.op_name));

    script.push_str("# Expected failures:\n");
    for f in &result.failures {
        script.push_str(&format!("# - {:?}: {}\n", f.kind, f.message));
    }

    script
}

fn inspect_tensor(spec: &str, current_seed: Option<u64>) -> Result<()> {
    let seed = current_seed
        .ok_or_else(|| anyhow::anyhow!("No seed selected. Use 'show <seed>' first."))?;

    // Parse tensor spec like "output[0:10]" or "input"
    let (name, slice) = if let Some(bracket_pos) = spec.find('[') {
        let name = &spec[..bracket_pos];
        let slice_str = &spec[bracket_pos + 1..spec.len() - 1];
        (name, Some(slice_str))
    } else {
        (spec, None)
    };

    let request = Request::Reproduce { seed };

    match send_request(request)? {
        Response::ReproduceResult { inputs, .. } => {
            if let Some(tensor) = inputs.get(name) {
                println!(
                    "Tensor '{}': shape={:?}, dtype={:?}",
                    name, tensor.shape, tensor.dtype
                );

                // Show slice of data
                let start = 0;
                let end = std::cmp::min(10, tensor.data.len() / 4);

                if let Some(_slice_str) = slice {
                    println!(
                        "Slice parsing not yet implemented. Showing first {} values:",
                        end
                    );
                }

                // Interpret as float32 for now
                let floats: Vec<f32> = tensor
                    .data
                    .chunks_exact(4)
                    .take(end)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();

                for (i, v) in floats.iter().enumerate() {
                    println!("  [{}]: {:.6}", start + i, v);
                }

                if tensor.data.len() / 4 > end {
                    println!("  ... ({} more values)", tensor.data.len() / 4 - end);
                }
            } else {
                println!(
                    "Tensor '{}' not found. Available: {:?}",
                    name,
                    inputs.keys().collect::<Vec<_>>()
                );
            }
        }
        Response::Error { code, message } => {
            println!("Error ({:?}): {}", code, message);
        }
        _ => {
            println!("Unexpected response");
        }
    }

    Ok(())
}

fn check_status() -> Result<()> {
    if check_daemon_running() {
        match send_request(Request::Ping)? {
            Response::Pong {
                version,
                uptime_secs,
                ..
            } => {
                println!("Daemon: running (v{}, uptime {}s)", version, uptime_secs);
            }
            _ => {
                println!("Daemon: running (unable to get details)");
            }
        }
    } else {
        println!("Daemon: not running");
    }
    Ok(())
}

fn check_daemon_running() -> bool {
    let socket_path = default_socket_path();
    if !socket_path.exists() {
        return false;
    }

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

    socket
        .send(&bytes)
        .map_err(|(_, e)| anyhow::anyhow!("Send failed: {}", e))?;

    let response_bytes = socket.recv().context("Failed to receive response")?;

    deserialize_response(&response_bytes).context("Failed to deserialize response")
}
