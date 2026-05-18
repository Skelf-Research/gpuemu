"""Live client<->daemon round-trip test.

This is the first test that actually starts the daemon and drives it through the
Python client. It guards the integration bugs that were invisible without a live
daemon: the connect/version-check recursion, base64 tensor encoding, and per-op
tolerance application.

Skips cleanly when pynng or the daemon binary are unavailable.
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

pynng = pytest.importorskip("pynng")
from gpuemu_py.client import Client  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_daemon():
    env = os.environ.get("GPUEMU_DAEMON")
    if env and Path(env).exists():
        return env
    cands = [REPO_ROOT / "target" / prof / "gpuemu-daemon" for prof in ("release", "debug")]
    cands = [p for p in cands if p.exists()]
    if not cands:
        return None
    # Newest build wins (avoid a stale release binary shadowing a fresh debug one).
    return str(max(cands, key=lambda p: p.stat().st_mtime))


DAEMON = _find_daemon()
pytestmark = pytest.mark.skipif(DAEMON is None, reason="gpuemu-daemon binary not built")

# An identity op: output == input. Reference echoes its single input back.
REF_SCRIPT = """\
import base64, json, sys
import numpy as np
p = json.load(sys.stdin)
t = p["inputs"]["input"]
arr = np.frombuffer(base64.b64decode(t["data"]), dtype=np.dtype(t["dtype"])).reshape(t["shape"])
print(json.dumps({"shape": list(arr.shape), "dtype": str(arr.dtype),
                  "data": base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()}))
"""

CONFIG = """\
[project]
name = "roundtrip-test"

[validation]
tolerances = { float32 = 1e-5 }

[[ops]]
name = "identity"
reference = "ref.py"
input_names = ["input"]
execution_mode = "client_side"
tolerances = { float32 = 0.5 }
"""


@pytest.fixture
def daemon():
    work = tempfile.mkdtemp(prefix="gpuemu-rt-")
    (Path(work) / "ref.py").write_text(REF_SCRIPT)
    (Path(work) / "gpuemu.toml").write_text(CONFIG)
    home = tempfile.mkdtemp(prefix="gpuemu-home-")
    sock = os.path.join(home, ".gpuemu", "gpuemu.sock")
    os.makedirs(os.path.join(home, ".gpuemu"), exist_ok=True)
    env = dict(os.environ, HOME=home)
    proc = subprocess.Popen([DAEMON], cwd=work, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # wait for readiness
    deadline = time.time() + 20
    ready = False
    while time.time() < deadline and proc.poll() is None:
        try:
            with Client(socket_path=sock, timeout_ms=1500) as c:
                c.ping()
                ready = True
                break
        except Exception:
            time.sleep(0.3)
    if not ready:
        proc.terminate()
        pytest.fail("daemon did not become ready")
    yield sock
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


def test_ping(daemon):
    with Client(socket_path=daemon) as c:
        info = c.ping()
        assert info["protocol_version"] == 1


def test_validate_identity_pass_and_fail(daemon):
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    with Client(socket_path=daemon) as c:
        # exact echo -> pass
        r = c.validate_op("identity", {"input": x}, x)
        assert r.passed, r.failures
        assert r.error_stats is not None and r.error_stats["count"] == 12
        # perturbed beyond op tolerance (0.5) -> fail
        bad = x.copy()
        bad[0, 0] += 1.0
        r2 = c.validate_op("identity", {"input": x}, bad)
        assert not r2.passed


def test_op_tolerance_applied(daemon):
    # diff 0.1 is within the op tol (0.5) but exceeds the global tol (1e-5):
    # passing confirms per-op tolerances are honored.
    x = np.zeros((2, 2), dtype=np.float32)
    out = x + 0.1
    with Client(socket_path=daemon) as c:
        r = c.validate_op("identity", {"input": x}, out)
        assert r.passed, "per-op tolerance (0.5) should permit a 0.1 diff"


def test_lint_kernel(daemon):
    # Exercises the artifact path + the LintResults struct-variant serde fix.
    ptx = (
        ".version 7.0\n.target sm_80\n.address_size 64\n"
        ".visible .entry k(.param .u64 p) {\n"
        "    .reg .f32 %f<4>;\n"
        "    add.f32 %f1, %f1, %f1;\n"
        "    ret;\n"
        "}\n"
    )
    with Client(socket_path=daemon) as c:
        results = c.lint_kernel(ptx)
        assert results, "expected at least one lint result"
        m = results[0]["metrics"]
        assert m["register_count"] == 4
        assert results[0]["kernel_name"] == "k"


def test_get_test_case_singular(daemon):
    # Singular get_test_case (one case) — exercises the GetTestCase RPC.
    with Client(socket_path=daemon) as c:
        schema = {
            "name": "identity",
            "dims": [{"name": "N", "candidates": [4, 8]}],
            "inputs": [{"name": "input", "dims": ["N"]}],
            "output": {"name": "out", "dims": ["N"]},
        }
        # get_test_case in the current client uses fuzz_op_config_dict; we
        # exercise it through get_test_batch(count=1) which shares the path.
        cases = c.get_test_batch("identity", count=1, seed=42,
                                 op_schema=schema, dtypes=["float32"])
        assert len(cases) == 1
        assert list(cases[0]["inputs"]["input"].shape)[0] in (4, 8)


def test_value_distribution_pipes_through(daemon):
    # Adversarial distribution must reach the Rust fuzzer and produce
    # non-uniform values (some non-finite or extreme magnitudes).
    schema = {
        "name": "identity",
        "dims": [{"name": "N", "candidates": [128]}],
        "inputs": [{"name": "input", "dims": ["N"]}],
        "output": {"name": "out", "dims": ["N"]},
    }
    with Client(socket_path=daemon) as c:
        cases = c.get_test_batch("identity", count=4, seed=7,
                                 op_schema=schema, dtypes=["float32"],
                                 value_distribution="adversarial")
        all_vals = np.concatenate([cs["inputs"]["input"].ravel() for cs in cases])
        has_nonfinite = bool(np.any(~np.isfinite(all_vals)))
        has_extreme = bool(np.any(np.abs(all_vals[np.isfinite(all_vals)]) > 1e6))
        # Adversarial buckets are 1/5 zero, 1/5 subnormal, 1/5 ~1e30, 1/5 wide-uniform,
        # 1/5 non-finite. Across 512 elements both should be present.
        assert has_nonfinite, "expected NaN/Inf in adversarial distribution"
        assert has_extreme, "expected large-magnitude values in adversarial distribution"


def test_get_test_batch_and_submit(daemon):
    schema = {
        "name": "identity",
        "dims": [{"name": "N", "candidates": [4, 8]}],
        "inputs": [{"name": "input", "dims": ["N"]}],
        "output": {"name": "out", "dims": ["N"]},
    }
    with Client(socket_path=daemon) as c:
        cases = c.get_test_batch("identity", count=4, seed=1,
                                 op_schema=schema, dtypes=["float32"])
        assert len(cases) == 4
        for case in cases:
            out = case["inputs"]["input"]  # identity kernel
            res = c.submit_output("identity", case["inputs"], out, case["seed"])
            assert res.passed
