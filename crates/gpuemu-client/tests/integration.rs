//! End-to-end test of the Rust client against a real daemon.
//!
//! Skips (passes trivially) when the `gpuemu-daemon` binary isn't built or
//! `python3` isn't on PATH, so it never fails a minimal environment. When both
//! are present it launches an isolated daemon, fuzzes an `identity` op via the
//! client-side closure path, and asserts the run validates.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use gpuemu_client::Client;
use gpuemu_common::types::{FuzzConfig, ValueDistribution};

fn daemon_binary() -> Option<PathBuf> {
    // crates/gpuemu-client -> workspace root -> target/<profile>/gpuemu-daemon
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest.parent()?.parent()?;
    for profile in ["debug", "release"] {
        let p = root.join("target").join(profile).join("gpuemu-daemon");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn have_python3() -> bool {
    Command::new("python3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// A minimal daemon launched in an isolated HOME with a temp project dir.
struct TestDaemon {
    child: Child,
    socket: PathBuf,
    _work: PathBuf,
    _home: PathBuf,
}

impl Drop for TestDaemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn unique_dir(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("gpuemu-client-it-{}-{}", tag, std::process::id()));
    fs::create_dir_all(&p).unwrap();
    p
}

fn start_daemon(bin: &PathBuf) -> Option<TestDaemon> {
    let work = unique_dir("work");
    let home = unique_dir("home");
    // identity reference: echo the single named input back.
    let ref_script = "import base64, json, sys\n\
import numpy as np\n\
p = json.load(sys.stdin)\n\
t = p[\"inputs\"][\"input\"]\n\
arr = np.frombuffer(base64.b64decode(t[\"data\"]), dtype=np.dtype(t[\"dtype\"])).reshape(t[\"shape\"])\n\
print(json.dumps({\"shape\": list(arr.shape), \"dtype\": str(arr.dtype), \
\"data\": base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()}))\n";
    fs::write(work.join("ref.py"), ref_script).unwrap();
    fs::write(
        work.join("gpuemu.toml"),
        "[project]\nname = \"client-it\"\n\n\
[validation]\ntolerances = { float32 = 1e-5 }\n\n\
[[ops]]\nname = \"identity\"\nreference = \"ref.py\"\n\
input_names = [\"input\"]\nexecution_mode = \"client_side\"\n",
    )
    .unwrap();

    let gpuemu_dir = home.join(".gpuemu");
    fs::create_dir_all(&gpuemu_dir).unwrap();
    let socket = gpuemu_dir.join("gpuemu.sock");

    let child = Command::new(bin)
        .current_dir(&work)
        .env("HOME", &home)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;

    let mut daemon = TestDaemon {
        child,
        socket,
        _work: work,
        _home: home,
    };

    // Poll for readiness.
    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline {
        if let Ok(client) = Client::connect_to(&daemon.socket, Duration::from_millis(1500)) {
            if client.ping().is_ok() {
                return Some(daemon);
            }
        }
        std::thread::sleep(Duration::from_millis(300));
    }
    let _ = daemon.child.kill();
    None
}

#[test]
fn fuzz_identity_end_to_end() {
    let Some(bin) = daemon_binary() else {
        eprintln!("skip: gpuemu-daemon binary not built");
        return;
    };
    if !have_python3() {
        eprintln!("skip: python3 not available");
        return;
    }
    let Some(daemon) = start_daemon(&bin) else {
        eprintln!("skip: daemon did not become ready");
        return;
    };

    let client = Client::connect_to(&daemon.socket, Duration::from_secs(10)).unwrap();

    // identity op: the closure echoes the input back, so every case must pass.
    let mut cfg = FuzzConfig::with_seed(123);
    cfg.value_distribution = ValueDistribution::Regular;
    let summary = client
        .fuzz_op_client_side("identity", cfg, 8, |inputs: &HashMap<_, _>| {
            inputs.get("input").cloned().expect("input present")
        })
        .expect("fuzz run");

    assert_eq!(summary.total, 8);
    assert_eq!(summary.passed, 8, "failures: {:?}", summary.failures);
}
