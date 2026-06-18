# Common Issues

This page covers the most frequently encountered problems when using gpuemu, along with their causes and solutions.

---

## 1. Daemon Won't Start

!!! bug "Symptom"

    Running `gpuemu daemon start` hangs, exits silently, or prints an error about the socket file.

!!! tip "Solution"

    **Check if a socket file already exists** from a previous crashed daemon:

    ```bash
    ls -la ~/.gpuemu/gpuemu.sock
    ```

    If the file exists but no daemon is running, remove it manually:

    ```bash
    rm ~/.gpuemu/gpuemu.sock
    gpuemu daemon start --background
    ```

    **Check if another daemon instance is already running:**

    ```bash
    ps aux | grep gpuemu-daemon
    ```

    If a daemon process exists, either use it or kill it before starting a new one.

    **Check the daemon logs for errors:**

    ```bash
    gpuemu daemon logs
    ```

    Common log errors include port conflicts, permission issues on `~/.gpuemu/`, or corrupted database files.

---

## 2. Connection Refused / Socket Not Found

!!! bug "Symptom"

    CLI or Python client reports `ConnectionRefused`, `Socket not found`, or `Failed to connect to daemon`.

!!! tip "Solution"

    The daemon is not running. Start it:

    ```bash
    gpuemu daemon start --background
    ```

    Verify it is listening:

    ```bash
    gpuemu daemon status
    ```

    If you are using a custom socket path, make sure the `GPUEMU_SOCKET` environment variable is set correctly in both the daemon and client:

    ```bash
    export GPUEMU_SOCKET=/path/to/custom.sock
    gpuemu daemon start --background
    gpuemu daemon status  # Should connect to the custom socket
    ```

    If the daemon is running but the client still cannot connect, check that the socket file exists and has the correct permissions:

    ```bash
    ls -la ~/.gpuemu/gpuemu.sock
    ```

---

## 3. Reference Script Fails

!!! bug "Symptom"

    Validation returns `ReferenceScriptFailed` with the reference script's stderr output.

!!! tip "Solution"

    Reference scripts must follow a strict protocol: read JSON from **stdin**, write JSON to **stdout**. The most common causes of failure:

    **Stray `print()` statements** that corrupt stdout:

    ```python
    # BAD -- this output will be mixed into the JSON response
    print("Debug: processing inputs")

    # GOOD -- use stderr for debug output
    import sys
    print("Debug: processing inputs", file=sys.stderr)
    ```

    **Test the script manually** to verify it works:

    ```bash
    echo '{"inputs": {"a": {"data": "AAAAAAAAAIA/", "dtype": "float32", "shape": [1, 1]}}, "kwargs": {}}' | python scripts/my_ref.py
    ```

    You should see clean JSON output with no extra text.

    **Missing dependencies**: Ensure the reference script's dependencies (NumPy, etc.) are installed in the same Python environment that the daemon uses.

    **Syntax errors or exceptions**: Run the script directly to see the full traceback:

    ```bash
    echo '{}' | python scripts/my_ref.py
    ```

---

## 4. Tolerance Exceeded Warnings

!!! bug "Symptom"

    Validation fails with messages like `max absolute diff 2.3e-4 exceeds tolerance 1e-5` or you see warnings about tolerance consumption exceeding the warning threshold.

!!! tip "Solution"

    The configured tolerance is too tight for your dtype and operation. Some operations (e.g., reductions, softmax, layer normalization) accumulate more floating-point error than element-wise ops.

    **Use recommended tolerances** based on empirical measurement:

    ```python
    from gpuemu.tolerances import get_recommended_tolerance

    rec = get_recommended_tolerance("softmax", dtype="float32")
    print(rec)  # {"atol": 1e-4, "rtol": 1e-4}
    ```

    **Calibrate tolerances** by running multiple iterations and finding the minimum safe threshold:

    ```python
    from gpuemu.tolerances import calibrate_tolerance

    result = calibrate_tolerance(client, "softmax", dtype="float32", iterations=200)
    print(result)  # {"atol": 5.2e-5, "rtol": 4.8e-5}
    ```

    **Apply operation-specific multipliers** in `gpuemu.toml`:

    ```toml
    [[ops]]
    name = "softmax"
    reference = "scripts/softmax_ref.py"

    [ops.tolerances]
    float32 = { atol = 1e-4, rtol = 1e-4 }   # Looser than global default
    float16 = { atol = 5e-2, rtol = 5e-2 }
    ```

    As a general rule: reductions, normalizations, and operations involving `exp()` or `log()` need looser tolerances than element-wise operations.

---

## 5. NaN/Inf Detected in Output

!!! bug "Symptom"

    Validation fails with `NaN detected in output` or `Inf detected in output`.

!!! tip "Solution"

    Check your reference implementation for common numerical hazards:

    - **Division by zero** -- Add epsilon to denominators: `x / (y + 1e-8)` instead of `x / y`
    - **Log of zero** -- Clamp inputs: `np.log(np.maximum(x, 1e-8))`
    - **Exp overflow** -- Use numerically stable implementations: `log_softmax` instead of `log(softmax(x))`
    - **Large intermediate values** -- Consider computing in higher precision, then casting down

    If NaN or Inf values are **expected behavior** for your operation (e.g., an op that intentionally produces Inf for certain inputs), disable the checks for that specific op:

    ```toml
    [[ops]]
    name = "my_op"
    reference = "scripts/my_op_ref.py"

    # Disable NaN/Inf checks for this op only
    check_nan = false
    check_inf = false
    ```

    Or disable globally in `[validation]` (not recommended):

    ```toml
    [validation]
    check_nan = false
    check_inf = false
    ```

---

## 6. Shape Mismatch Errors

!!! bug "Symptom"

    Validation fails with `Shape mismatch: expected (4, 8) but got (8, 4)` or similar.

!!! tip "Solution"

    The output shape from your op must exactly match the output shape from the reference script. Common causes:

    **Transposed output**: Your op returns a transposed result compared to the reference. Check the dimension ordering in both implementations.

    **Missing batch dimension**: Your reference script may not handle batched inputs correctly. Verify the reference works for both single and batched inputs.

    **Squeeze/unsqueeze mismatch**: One implementation drops a size-1 dimension that the other preserves. Be explicit about shapes:

    ```python
    # In your reference script, be explicit about output shape
    result = np.matmul(a, b)
    assert result.shape == expected_shape, f"Expected {expected_shape}, got {result.shape}"
    ```

    Debug by printing shapes to stderr in your reference script:

    ```python
    import sys
    print(f"Input a: {a.shape}, Input b: {b.shape}, Output: {result.shape}", file=sys.stderr)
    ```

---

## 7. pynng Import Errors

!!! bug "Symptom"

    ```
    ImportError: No module named 'pynng'
    ```

    or build errors when installing pynng.

!!! tip "Solution"

    Install pynng with the correct version:

    ```bash
    pip install pynng>=0.8.0
    ```

    Or install the full gpuemu Python package which includes it as a dependency:

    ```bash
    pip install ./gpuemu-py
    ```

    **On macOS**, pynng compiles native NNG bindings and may require CMake:

    ```bash
    brew install cmake
    pip install pynng>=0.8.0
    ```

    **In virtual environments**, make sure you are installing into the correct environment:

    ```bash
    which python   # Verify this points to your venv
    pip list | grep pynng
    ```

---

## 8. Slow Validation

!!! bug "Symptom"

    Validation takes many seconds per op, or fuzzing runs are unacceptably slow.

!!! tip "Solution"

    Reference scripts are spawned as Python subprocesses for each validation call, which includes Python interpreter startup time. To improve performance:

    **Use `--quick` mode** for faster iteration during development:

    ```bash
    gpuemu test --quick
    ```

    This tests only the primary dtype (typically `float32`) instead of all configured dtypes.

    **Keep reference scripts simple**: Minimize imports and avoid heavy initialization. NumPy is usually sufficient -- do not import PyTorch or TensorFlow in reference scripts.

    ```python
    # SLOW -- imports a large framework
    import torch
    result = torch.matmul(torch.from_numpy(a), torch.from_numpy(b)).numpy()

    # FAST -- uses NumPy directly
    import numpy as np
    result = np.matmul(a, b)
    ```

    **Use daemon-orchestrated mode** for batch runs and CI. The daemon keeps reference scripts warm and batches invocations:

    ```toml
    [[ops]]
    name = "matmul"
    reference = "scripts/matmul_ref.py"
    execution_mode = "daemon_orchestrated"
    ```

    **Increase parallel jobs** in `gpuemu.toml` for CI:

    ```toml
    [ci]
    parallel_jobs = 4
    ```

---

## 9. Artifact Linting Fails

!!! bug "Symptom"

    `gpuemu lint` reports errors about `cuobjdump not available` or fails with PTX parse errors.

!!! tip "Solution"

    **`cuobjdump` is not available**: This tool is part of the CUDA toolkit and is only available on Linux with CUDA installed. On macOS, artifact inspection is optional and can be skipped:

    ```bash
    # Skip artifact checks -- validation still works
    gpuemu test  # Artifact linting is separate from validation
    ```

    Artifact linting is invoked explicitly via `gpuemu lint` or through kernel definitions in `gpuemu.toml`. If you do not have CUDA tools, you can safely omit `[[kernels]]` sections from your configuration.

    **PTX parse errors**: Ensure the PTX content is valid and complete. The parser uses regex-based extraction and may fail on unusual PTX formatting. Check the PTX file manually:

    ```bash
    # Verify PTX is well-formed
    head -20 kernels/my_kernel.ptx
    ```

    **Adjusting artifact limits**: If linting fails because your kernel exceeds resource limits, adjust the thresholds to match your target architecture:

    ```toml
    [[kernels]]
    name = "my_kernel"
    source = "kernels/my_kernel.cu"

    [kernels.artifact_checks]
    max_registers = 128     # Increase for complex kernels
    max_spills = 4          # Allow some spills if necessary
    max_local_memory_bytes = 1024
    ```

---

## 10. VS Code Extension Not Activating

!!! bug "Symptom"

    The gpuemu VS Code extension does not show diagnostics, the status bar item is missing, or commands are not available.

!!! tip "Solution"

    **The extension requires a `gpuemu.toml` file in the workspace root** to activate. Create one or open a directory that contains one:

    ```bash
    gpuemu init --name my-project --framework pytorch
    ```

    **Check the `gpuemu.binaryPath` setting**: The extension needs to find the `gpuemu` CLI binary. Open VS Code settings and verify:

    ```json
    {
        "gpuemu.binaryPath": "gpuemu"
    }
    ```

    If the binary is not on your `PATH`, provide the full path:

    ```json
    {
        "gpuemu.binaryPath": "/home/user/.gpuemu/bin/gpuemu"
    }
    ```

    **Check the extension is installed and enabled**: Open the Extensions panel (++ctrl+shift+x++ / ++cmd+shift+x++) and search for "gpuemu". Ensure it is installed and not disabled.

    **Check the Output panel for errors**: Open View > Output, select "gpuemu" from the dropdown, and look for error messages.

---

## 11. Fuzzing Finds Too Many Failures

!!! bug "Symptom"

    A fuzz run produces hundreds of failures, making it difficult to know where to start.

!!! tip "Solution"

    **Start with fewer iterations** to get a manageable number of failures:

    ```bash
    gpuemu fuzz --op my_op --iterations 10
    ```

    **Tighten one issue at a time**. Look at the first few failures and categorize them:

    - Are they all the same dtype? Fix that dtype's tolerance first.
    - Are they all large shapes? You may have an accumulation error that scales with input size.
    - Are they all a specific layout? Check non-contiguous tensor handling.

    **Use minimization** to find the simplest reproducer for each unique failure mode:

    ```bash
    gpuemu minimize --op my_op --seed <first-failing-seed>
    ```

    **Review the failure summary** to identify patterns:

    ```bash
    gpuemu failures --op my_op
    ```

    Fix the root cause, then re-run fuzzing to confirm the fix and discover the next category of issues.

---

## 12. Baseline Comparison Fails

!!! bug "Symptom"

    Running `gpuemu diff` or `gpuemu test` with `fail_on_regression = true` reports baseline comparison failures even though the op is correct.

!!! tip "Solution"

    **Ensure a baseline was stored** before running the comparison. Baselines are created explicitly:

    ```bash
    # Store the current results as a named baseline
    gpuemu baseline store v1.0

    # Later, compare against the stored baseline
    gpuemu baseline diff v1.0
    ```

    If no baseline exists for the tag you are comparing against, the diff will fail with `BaselineNotFound`.

    **Update the baseline** after intentional changes to your op. If you have changed the implementation and the new results are correct, store a new baseline:

    ```bash
    gpuemu test                     # Verify the new results are correct
    gpuemu baseline store v1.1      # Store updated baseline
    ```

    **Check that the seed matches**: Baselines are seed-specific. If you changed the global seed in `gpuemu.toml`, previous baselines may no longer be comparable. Use explicit seeds for baseline comparisons:

    ```bash
    gpuemu test --seed 42
    gpuemu baseline store v1.0 --seed 42
    gpuemu baseline diff v1.0 --seed 42
    ```

---

## Still Stuck?

If your issue is not covered here:

- Check the [FAQ](faq.md) for general questions.
- Review the [Configuration](../getting-started/configuration.md) reference to ensure your `gpuemu.toml` is correct.
- Inspect daemon logs with `gpuemu daemon logs` for detailed error information.
- Search existing issues in the [GitHub repository](https://github.com/example/gpuemu/issues).
