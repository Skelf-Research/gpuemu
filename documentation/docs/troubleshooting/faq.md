# Frequently Asked Questions

Answers to the most common questions about gpuemu, organized by topic.

---

## General

??? question "What is gpuemu?"

    gpuemu is a GPU-less validation toolchain for deep learning kernels. It lets you validate the correctness of GPU-targeted operations entirely on CPU, using deterministic reference implementations, numerical tolerance checking, shape/layout fuzzing, artifact linting, and CI integration.

    gpuemu is **not** a cycle-accurate GPU emulator. It does not simulate GPU hardware or measure performance. Its purpose is to catch correctness bugs before code reaches real hardware.

??? question "Do I need a GPU to use gpuemu?"

    No. That is the entire point of gpuemu. All validation runs on CPU. The daemon executes reference scripts in Python, compares outputs numerically, and reports results -- no GPU drivers, no CUDA runtime, and no hardware required.

    This makes gpuemu ideal for:

    - Local development on laptops without GPUs
    - CI pipelines running on CPU-only instances
    - Code review workflows where correctness matters more than performance

??? question "What frameworks are supported?"

    gpuemu provides first-class adapters for three major deep learning frameworks:

    | Framework | Adapter | Install Extra |
    |-----------|---------|---------------|
    | **PyTorch** | `gpuemu_py.frameworks.pytorch` | `pip install ./gpuemu-py[torch]` |
    | **JAX** | `gpuemu_py.frameworks.jax` | `pip install ./gpuemu-py[jax]` |
    | **TensorFlow** | `gpuemu_py.frameworks.tensorflow` | `pip install ./gpuemu-py[tensorflow]` |

    Each adapter handles tensor conversion, dtype mapping, and framework-specific idioms so you can validate ops using native framework types.

??? question "What platforms are supported?"

    | Platform | Status | Notes |
    |----------|--------|-------|
    | **Linux** | Primary | Full workflow including artifact inspection (PTX/SASS analysis via `cuobjdump`) |
    | **macOS** | Core validation | CPU validation works fully. Artifact inspection is optional and skipped if `cuobjdump` is not available. |
    | **Windows** | Future | Not currently targeted. Contributions welcome. |

---

## Validation

??? question "How does validation work?"

    When you submit an op for validation, the following pipeline executes:

    1. **Your code** computes the op output (on CPU or GPU).
    2. **The daemon** spawns a CPU reference script that computes the expected output for the same inputs.
    3. **The validator** compares the two outputs element-by-element using per-dtype absolute and relative tolerances.
    4. **Invariant checks** verify structural properties (no NaN, no Inf, shape preserved, etc.).
    5. **The result** is stored in the daemon's sled database and returned to the client.

    ```bash
    # Run validation for all configured ops
    gpuemu test
    ```

    The daemon handles execution, comparison, and storage. You only need to provide the op output and a reference script.

??? question "What are tolerances?"

    Tolerances define the acceptable numerical difference between your op's output and the reference output. Floating-point arithmetic is inherently imprecise, and different implementations (GPU vs. CPU, different reduction orders) produce slightly different results. Tolerances account for this.

    gpuemu uses two tolerance values per dtype:

    | Parameter | Meaning |
    |-----------|---------|
    | `atol` | **Absolute tolerance** -- maximum allowed absolute difference (`|actual - expected|`) |
    | `rtol` | **Relative tolerance** -- maximum allowed relative difference (`|actual - expected| / |expected|`) |

    Default tolerances by dtype:

    | Dtype | `atol` | `rtol` |
    |-------|--------|--------|
    | `float32` | `1e-5` | `1e-5` |
    | `float16` | `1e-2` | `1e-2` |
    | `bfloat16` | `1e-2` | `1e-2` |
    | `float64` | `1e-10` | `1e-10` |

    A validation passes if, for every element, the difference satisfies: `|actual - expected| <= atol + rtol * |expected|`.

??? question "Can I customize tolerances?"

    Yes. Tolerances can be customized at multiple levels:

    **Per-dtype globally** in `gpuemu.toml`:

    ```toml
    [validation.tolerances]
    float32 = { atol = 1e-5, rtol = 1e-5 }
    float16 = { atol = 1e-3, rtol = 1e-3 }
    ```

    **Per-op** in `gpuemu.toml` (overrides global defaults):

    ```toml
    [[ops]]
    name = "my_op"
    reference = "scripts/my_op_ref.py"

    [ops.tolerances]
    float32 = { atol = 1e-4, rtol = 1e-4 }
    ```

    **Programmatically** in Python using calibration:

    ```python
    from gpuemu_py.tolerances import calibrate_tolerance

    # Run multiple iterations and find the tightest tolerance that passes
    recommended = calibrate_tolerance(client, "my_op", dtype="float32", iterations=100)
    print(recommended)  # {"atol": 2.5e-6, "rtol": 3.1e-6}
    ```

??? question "What are invariants?"

    Invariants are structural checks applied to op outputs, independent of numerical value comparison. They catch categories of bugs that tolerances alone would miss.

    | Invariant | What it checks |
    |-----------|----------------|
    | `shape_preserved` | Output tensor shape matches the reference output shape exactly |
    | `non_negative` | All output elements are >= 0 (useful for ReLU, softmax, etc.) |
    | `finite` | All output elements are finite (no NaN, no Inf) |
    | `symmetric` | Output matrix is symmetric (for square matrix outputs) |
    | `normalized` | Values sum to 1 along the last axis (for probability distributions) |

    Configure invariants per-op in `gpuemu.toml`:

    ```toml
    [[ops]]
    name = "softmax"
    reference = "scripts/softmax_ref.py"
    invariants = ["shape_preserved", "non_negative", "normalized"]
    ```

??? question "What is a seed?"

    A seed is a deterministic random number used to generate test inputs. When gpuemu creates input tensors for validation or fuzzing, it uses a seeded pseudorandom number generator (xorshift128+ with Blake2b seed derivation). The same seed always produces the same inputs.

    This guarantees **reproducible tests**:

    ```bash
    # First run discovers a failure at seed 98765
    gpuemu fuzz --op matmul --iterations 100
    # ...FAIL at seed 98765

    # Reproduce the exact same failure
    gpuemu test --seed 98765
    ```

    The cross-language RNG implementation (Rust and Python produce identical sequences) means you can reproduce failures regardless of which client triggered them.

---

## Fuzzing

??? question "What does fuzzing test?"

    Fuzzing automatically generates randomized test inputs to stress-test your op across a wide range of conditions. The fuzzer varies:

    - **Shapes** -- batch sizes, sequence lengths, hidden dimensions, edge cases like size-0 and size-1 dimensions
    - **Dtypes** -- all configured dtypes (`float32`, `float16`, `bfloat16`, `float64`)
    - **Memory layouts** -- contiguous, strided, transposed, and non-contiguous tensor layouts
    - **Value ranges** -- normal values, very small values (subnormals), very large values, mixed signs

    ```bash
    gpuemu fuzz --op matmul --iterations 100
    ```

    Each iteration uses a unique deterministic seed, so any failure can be reproduced exactly.

??? question "How many fuzzing iterations should I run?"

    It depends on your goals:

    | Scenario | Recommended Iterations |
    |----------|----------------------|
    | Quick sanity check during development | 50--100 |
    | Pre-commit or pull request CI gate | 100--500 |
    | Thorough nightly testing | 1,000+ |
    | Initial validation of a new op | 5,000--10,000 |

    More iterations cover more of the input space but take longer. A good strategy is to run a small number on every commit and a large number on a nightly schedule:

    ```bash
    # Fast check (CI on every push)
    gpuemu fuzz --op matmul --iterations 100

    # Thorough check (nightly)
    gpuemu fuzz --op matmul --iterations 5000
    ```

??? question "What is test case minimization?"

    When fuzzing discovers a failure, the failing input may be large and complex (e.g., a 128x256 matrix). Test case minimization automatically searches for the **smallest input that still triggers the same failure**.

    The minimizer uses binary search on tensor dimensions and values to shrink the reproducer:

    ```bash
    # Minimize a failure found at seed 98765
    gpuemu minimize --op matmul --seed 98765
    ```

    Example output:

    ```
    Original: shape (128, 256) x (256, 64) -- FAIL (max diff: 2.3e-4)
    Minimized: shape (2, 3) x (3, 2) -- FAIL (max diff: 1.8e-4)
    Minimal reproducer saved. Seed: 98765, shapes: [(2, 3), (3, 2)]
    ```

    A smaller reproducer is easier to debug and makes a better regression test.

---

## Architecture

??? question "What is the daemon?"

    The gpuemu daemon (`gpuemu-daemon`) is a long-running background Rust process that serves as the validation engine. It handles:

    - **IPC** -- Listens on a Unix domain socket (`~/.gpuemu/gpuemu.sock`) via NNG REP/REQ protocol
    - **Execution** -- Spawns Python reference scripts as subprocesses to compute expected outputs
    - **Validation** -- Compares op outputs against reference outputs with per-dtype tolerances
    - **Fuzzing** -- Generates randomized test cases with deterministic seeds
    - **Storage** -- Persists results, failures, baselines, and artifact metrics in a sled embedded database
    - **Artifact analysis** -- Parses PTX/SASS output and lints against configurable policies

    ```bash
    # Start the daemon
    gpuemu daemon start --background

    # Check status
    gpuemu daemon status

    # Stop it
    gpuemu daemon stop
    ```

    The CLI, Python client, and VS Code extension all communicate with the daemon over IPC. They do not perform validation themselves.

??? question "Where is data stored?"

    All gpuemu runtime data is stored under the `~/.gpuemu/` directory:

    ```
    ~/.gpuemu/
    ├── gpuemu.sock       # Unix domain socket (IPC endpoint)
    ├── bin/              # CLI binary (if installed here)
    ├── db/               # sled embedded database
    │   ├── results/      # Validation results
    │   ├── failures/     # Recorded failures
    │   ├── baselines/    # Named baseline snapshots
    │   └── artifacts/    # Artifact metrics and baselines
    └── logs/             # Daemon log files
    ```

    The sled database uses `rkyv` for zero-copy deserialization, making lookups fast even with thousands of stored results. Data persists across daemon restarts.

??? question "What is a reference script?"

    A reference script is a standalone Python program that provides the canonical CPU implementation of an operation. It follows a strict protocol:

    1. Read JSON with base64-encoded tensor data from **stdin**
    2. Compute the expected output using standard CPU libraries (NumPy, etc.)
    3. Write JSON with base64-encoded result tensors to **stdout**

    ```python title="scripts/matmul_ref.py"
    import json, base64, sys
    import numpy as np

    def decode_tensor(encoded):
        data = base64.b64decode(encoded["data"])
        return np.frombuffer(data, dtype=np.dtype(encoded["dtype"])).reshape(encoded["shape"])

    def encode_tensor(arr):
        return {
            "data": base64.b64encode(arr.tobytes()).decode("ascii"),
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
        }

    request = json.loads(sys.stdin.read())
    a = decode_tensor(request["inputs"]["a"])
    b = decode_tensor(request["inputs"]["b"])
    result = np.matmul(a, b)
    json.dump({"outputs": {"result": encode_tensor(result)}}, sys.stdout)
    ```

    Reference scripts must be deterministic, side-effect-free, and must not write anything to stdout except the JSON response.

---

## CI

??? question "Can I use gpuemu in CI without a GPU?"

    Yes. Running in CI without a GPU is a core design goal of gpuemu. The daemon, CLI, and Python client all run on CPU-only machines. A typical CI workflow:

    ```yaml title=".github/workflows/gpuemu.yml"
    jobs:
      validate:
        runs-on: ubuntu-latest  # No GPU needed
        steps:
          - uses: actions/checkout@v4
          - name: Install gpuemu
            run: |
              cargo build --release
              pip install ./gpuemu-py
          - name: Start daemon
            run: ./target/release/gpuemu daemon start --background
          - name: Run tests
            run: ./target/release/gpuemu test --format junit > results.xml
          - name: Upload results
            uses: actions/upload-artifact@v4
            with:
              name: test-results
              path: results.xml
    ```

    No GPU runners, no CUDA installation, no special hardware. Standard CI infrastructure is sufficient.

??? question "What output formats does CI support?"

    gpuemu supports three output formats for CI integration:

    | Format | Flag | Use Case |
    |--------|------|----------|
    | **Text** | `--format text` (default) | Human-readable console output |
    | **JSON** | `--format json` | Machine-readable structured output for custom tooling |
    | **JUnit XML** | `--format junit` | Standard test report format consumed by CI platforms (GitHub Actions, GitLab CI, Jenkins, etc.) |

    ```bash
    # Human-readable output
    gpuemu test --format text

    # JSON for scripting
    gpuemu test --format json > results.json

    # JUnit XML for CI platforms
    gpuemu test --format junit > results.xml
    ```

    The JSON format includes all validation details (seed, max diff, tolerances used, failure reasons) and is suitable for building dashboards or custom reporting.

---

## Next Steps

- [Common Issues](common-issues.md) -- Solutions to specific error messages and problems.
- [Configuration](../getting-started/configuration.md) -- Full reference for `gpuemu.toml` settings.
- [Architecture](../concepts/architecture.md) -- Understand how the components fit together.
- [CI Integration Tutorial](../tutorials/ci-integration.md) -- Step-by-step CI setup guide.
