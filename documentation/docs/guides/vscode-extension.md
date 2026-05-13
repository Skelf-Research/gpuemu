# VS Code Extension

The gpuemu VS Code extension brings live validation diagnostics, code actions, test explorer integration, and on-save workflows directly into your editor.

---

## Installation

### Option A: Install from VSIX

If a pre-built `.vsix` file is available (from a release or CI artifact):

1. Open VS Code.
2. Open the Command Palette (++ctrl+shift+p++ on Linux, ++cmd+shift+p++ on macOS).
3. Type **"Extensions: Install from VSIX..."** and select it.
4. Browse to the `.vsix` file and install.

### Option B: Build from source

```bash
cd vscode-gpuemu/
npm install
npm run compile
```

Then package and install:

```bash
npx vsce package
# Produces gpuemu-0.1.0.vsix
```

Install the generated `.vsix` via the Command Palette as described above.

!!! tip "Development mode"

    For extension development, open the `vscode-gpuemu/` folder in VS Code and press ++f5++ to launch a development Extension Host with the extension loaded.

### Activation

The extension activates automatically when a `gpuemu.toml` file is found in your workspace. No manual activation is needed.

!!! note "Requirements"

    - **VS Code 1.85+**
    - **Node.js 18+** and **npm** (only for building from source)
    - The `gpuemu` CLI binary must be available on your `PATH` or configured via the `gpuemu.binaryPath` setting

---

## Settings

Configure the extension in your VS Code settings (`.vscode/settings.json` or the Settings UI):

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `gpuemu.binaryPath` | `string` | `""` (auto-detect) | Path to the `gpuemu` CLI binary. When empty, the extension searches `~/.gpuemu/bin/gpuemu`, `/usr/local/bin/gpuemu`, and `PATH`. |
| `gpuemu.autoStartDaemon` | `boolean` | `true` | Automatically start the gpuemu daemon when the workspace contains `gpuemu.toml`. |
| `gpuemu.showStatusBar` | `boolean` | `true` | Show the daemon status indicator in the status bar. |

### Example settings

```json
{
    "gpuemu.binaryPath": "/home/user/.gpuemu/bin/gpuemu",
    "gpuemu.autoStartDaemon": true,
    "gpuemu.showStatusBar": true
}
```

---

## Commands

All commands are available via the Command Palette (++ctrl+shift+p++ on Linux, ++cmd+shift+p++ on macOS). Type `gpuemu` to filter.

| Command | Palette Name | Description |
|---------|-------------|-------------|
| `gpuemu.init` | **gpuemu: Initialize Project** | Create a new `gpuemu.toml` with framework selection (PyTorch, JAX, TensorFlow) |
| `gpuemu.startDaemon` | **gpuemu: Start Daemon** | Start the gpuemu daemon in the background |
| `gpuemu.stopDaemon` | **gpuemu: Stop Daemon** | Stop the running daemon |
| `gpuemu.runTests` | **gpuemu: Run Validation Tests** | Run full validation for all ops in `gpuemu.toml` |
| `gpuemu.runQuickTests` | **gpuemu: Run Quick Validation** | Run validation with `--quick` flag for fast feedback |
| `gpuemu.fuzz` | **gpuemu: Run Fuzz Tests** | Run fuzz testing with configurable iteration count |
| `gpuemu.reproduce` | **gpuemu: Reproduce Failure** | Reproduce a specific failure by entering its seed |
| `gpuemu.showFailures` | **gpuemu: Show Failures** | Open and focus the Failures tree view |

---

## Status Bar

When `gpuemu.showStatusBar` is enabled, a status indicator appears in the bottom status bar.

### Daemon running

> :material-check: **gpuemu v0.1.0** -- "gpuemu daemon is running. Click to stop."

The indicator shows a checkmark icon with the daemon version. Clicking it runs `gpuemu.stopDaemon`.

### Daemon stopped

> :material-close-circle: **gpuemu** -- "gpuemu daemon is not running. Click to start."

The indicator shows a circle-slash icon with a warning background. Clicking it runs `gpuemu.startDaemon`.

!!! tip "Quick toggle"

    Click the status bar item to toggle the daemon on and off without opening the Command Palette.

---

## Problems Panel

Validation failures from the daemon appear as diagnostics in the VS Code **Problems** panel (++ctrl+shift+m++ on Linux, ++cmd+shift+m++ on macOS).

### Severity mapping

Failure types are mapped to VS Code diagnostic severities:

| Failure Kind | VS Code Severity | Appearance |
|-------------|-----------------|------------|
| `ReferenceError` | **Error** | Red underline, error icon |
| `ShapeMismatch` | **Error** | Red underline, error icon |
| `NaN detected` | **Error** | Red underline, error icon |
| `Inf` | **Error** | Red underline, error icon |
| `ToleranceExceeded` | **Warning** | Yellow underline, warning icon |

### Diagnostic format

Each diagnostic includes:

- **Op name** in brackets: `[flash_attention]`
- **Failure message**: what went wrong
- **Dtype and shape** when available: `(dtype: float16) (shape: [2, 128, 8, 64])`
- **Max diff** for tolerance failures: `max_diff: 1.234e-02`
- **Seed** in related information for reproduction

### Example diagnostic

```
[flash_attention] Tolerance exceeded (dtype: float16) (shape: [2, 128, 8, 64]) max_diff: 1.234e-02
  Reproducible with seed 8374629105
```

Diagnostics refresh automatically after running tests, fuzzing, or on-save validation. You can also manually refresh with the `gpuemu.refreshDiagnostics` command.

---

## Code Actions

Right-click on a gpuemu diagnostic in the editor to access code actions. These appear in the Quick Fix lightbulb menu.

### Available actions

| Action | Description | Trigger |
|--------|-------------|---------|
| **Reproduce Failure** | Runs `gpuemu reproduce <seed>` to regenerate the exact inputs and display a detailed failure report | Available when the diagnostic contains a seed |
| **Minimize Test Case** | Runs `gpuemu minimize <seed>` to find the smallest input that still triggers the failure | Available when the diagnostic contains a seed |
| **Fuzz Operation** | Runs `gpuemu fuzz --op <op_name> --iterations 50` to run additional fuzz iterations on the failing op | Available when the diagnostic contains an op name |

### How it works

1. gpuemu diagnostics include structured data: the op name (extracted from `[op_name]` in the message) and the seed (extracted from `seed: N`).
2. When you right-click a diagnostic, the code action provider parses these fields and offers the appropriate actions.
3. **Reproduce** and **Minimize** open results in a new text document. **Fuzz** opens a terminal.

---

## Test Explorer

Ops registered in `gpuemu.toml` appear as test items in the VS Code **Testing** sidebar (the beaker icon in the Activity Bar).

### How it works

1. The extension reads `gpuemu.toml` and extracts all `[[ops]]` entries by name.
2. Each op appears as a test item with the tag `gpuemu`.
3. Click the **Run** button next to an op to validate it.
4. Results are mapped to pass/fail/error states in the sidebar.

### Running tests

- **Run all**: Click the play button at the top of the Testing sidebar to validate all ops.
- **Run one**: Click the play button next to a specific op to validate only that op.
- **Refresh**: Click the refresh button to re-read `gpuemu.toml` and discover new ops.

### Run profile

The extension registers a **Validate** run profile. This is the default profile used when you click Run in the Testing sidebar.

---

## Failures Tree View

The **gpuemu Failures** tree view appears in the Explorer sidebar when a `gpuemu.toml` is present in the workspace.

### Features

- Lists all stored failures from the daemon, sorted by recency
- Each entry shows: **op name**, **seed**, and a truncated failure message
- Entries have an error icon for visual distinction
- Click a failure to **reproduce** it (runs `gpuemu.reproduce` with the failure's seed)
- The title bar has a **Run Validation Tests** button to re-run all tests

### Refreshing

The failures list refreshes automatically after:

- Running tests (`gpuemu.runTests` or `gpuemu.runQuickTests`)
- Running fuzz tests (`gpuemu.fuzz`)
- On-save validation triggers

You can also refresh manually by running `gpuemu.showFailures` from the Command Palette.

---

## On-Save Validation

The extension watches for file saves and triggers validation or linting automatically.

### Python reference scripts

Saving a `.py` file inside the `scripts/` directory triggers validation for the corresponding op. The op name is inferred from the filename convention:

```
scripts/ref_<op_name>.py  -->  validates op: <op_name>
```

For example, saving `scripts/ref_flash_attention.py` triggers validation of the `flash_attention` op.

!!! note "Status bar feedback"

    During on-save validation, the status bar shows a spinning indicator:

    > :material-sync: gpuemu: validating flash_attention...

    When complete:

    > :material-check: gpuemu: flash_attention validated

### CUDA kernel source files

Saving a `.cu` or `.cuh` file triggers artifact linting via `gpuemu lint`. This checks register pressure, spills, and pattern matching against the configured artifact checks.

### Configuration changes

Saving `gpuemu.toml` triggers:

1. Re-validation of the config file (see [Config Validation](#config-validation) below)
2. A diagnostics refresh to pick up any changes to op definitions or tolerances

---

## Config Validation

The extension validates `gpuemu.toml` live as you edit it. Diagnostics appear inline in the editor and in the Problems panel.

### What is validated

| Check | Severity | Example |
|-------|----------|---------|
| Invalid dtype names | Warning | `float33 = 1e-5` -- "Invalid dtype. Valid: float16, bfloat16, float32, ..." |
| Invalid execution modes | Error | `execution_mode = "gpu_only"` -- "Invalid execution_mode. Valid: client_side, daemon_orchestrated, script_based" |
| Invalid layout names | Warning | `layouts = ["rowmajor"]` -- not a recognized layout |
| Malformed TOML syntax | Error | Syntax errors in the TOML file |

### When validation runs

Config validation runs on:

- **Open**: When `gpuemu.toml` is opened in the editor
- **Save**: When `gpuemu.toml` is saved

---

## Code Snippets

The extension provides TOML code snippets for common gpuemu configuration patterns. These are available when editing `gpuemu.toml` or any `.toml` file.

### Available snippets

| Prefix | Name | Description |
|--------|------|-------------|
| `op` or `gpuemu-op` | Op configuration | Scaffold a new `[[ops]]` entry with name, module, reference, tolerances, and invariants |
| `kernel` or `gpuemu-kernel` | Kernel configuration | Scaffold a new `[[kernels]]` entry with name, source, reference, tolerances, and artifact checks |
| `tolerances` or `gpuemu-tol` | Tolerance block | Add a `[ops.tolerances]` or `[kernels.tolerances]` block with float32, float16, bfloat16 |
| `invariants` or `gpuemu-inv` | Invariants block | Add an invariants block with no_nan, no_inf, non_negative, shape_preserved |

### Example: using the `op` snippet

Type `op` in `gpuemu.toml` and press ++tab++. The following scaffold is inserted with tab stops for each value:

```toml
[[ops]]
name = "op_name"
module = "module.name"
reference = "scripts/ref_op_name.py"

[ops.tolerances]
float32 = 1e-5
float16 = 1e-3

[ops.invariants]
no_nan = true
no_inf = true
```

---

## Workflow Example

This section walks through a complete workflow: writing a reference script, validating it, finding a failure, and fixing it -- all from within VS Code.

### Step 1: Initialize the project

Open the Command Palette (++cmd+shift+p++) and run **gpuemu: Initialize Project**. Select your framework (e.g., PyTorch). This creates `gpuemu.toml` and the daemon starts automatically.

The status bar shows:

> :material-check: **gpuemu v0.1.0**

### Step 2: Add an op using snippets

Open `gpuemu.toml`, type `op`, and press ++tab++. Fill in the op details:

```toml
[[ops]]
name = "fused_silu"
module = "my_kernels"
reference = "scripts/ref_fused_silu.py"

[ops.tolerances]
float32 = 1e-5
float16 = 1e-3

[ops.invariants]
no_nan = true
no_inf = true
```

Save the file. The extension validates the config -- if there are errors, they appear in the Problems panel.

### Step 3: Write the reference script

Create `scripts/ref_fused_silu.py`:

```python
import numpy as np

def reference(inputs: dict, **kwargs) -> np.ndarray:
    x = inputs["x"]
    return x * (1.0 / (1.0 + np.exp(-x)))  # SiLU = x * sigmoid(x)
```

Save the file. The extension detects that `scripts/ref_fused_silu.py` matches the `ref_<op_name>.py` convention and triggers validation for the `fused_silu` op.

### Step 4: Run fuzz tests

Open the Command Palette and run **gpuemu: Run Fuzz Tests**. Enter `200` iterations. A terminal opens running:

```bash
gpuemu fuzz --iterations 200
```

### Step 5: Inspect failures

After fuzzing completes, check the **Problems** panel. Suppose you see:

```
[fused_silu] Tolerance exceeded (dtype: float16) (shape: [32, 1024, 512]) max_diff: 3.456e-02
  Reproducible with seed 7291038456
```

The failure also appears in the **gpuemu Failures** tree view in the Explorer sidebar.

### Step 6: Reproduce the failure

Right-click the diagnostic and select **Reproduce Failure**. This runs `gpuemu reproduce 7291038456` and opens a detailed report showing the exact inputs, expected output, actual output, and element-wise differences.

### Step 7: Minimize the test case

Right-click again and select **Minimize Test Case**. gpuemu binary-searches the input dimensions to find the smallest shape that still triggers the failure.

### Step 8: Fix and re-validate

Based on the reproduction, you discover that the `float16` tolerance is too tight for large tensors. Update `gpuemu.toml`:

```toml
[ops.tolerances]
float32 = 1e-5
float16 = 5e-3  # Relaxed from 1e-3
```

Save the file. The config is re-validated. Run **gpuemu: Run Quick Validation** to confirm the fix. The Problems panel clears.

### Step 9: Verify in the Testing sidebar

Open the Testing sidebar (beaker icon). You see `fused_silu` listed as a test item. Click Run. It passes with a green checkmark.

---

## Next Steps

- [Installation Guide](../getting-started/installation.md) -- Installing the CLI and Python client
- [Model Developer Guide](model-developer.md) -- Validation workflows for ML engineers
- [Kernel Author Guide](kernel-author.md) -- On-save linting for `.cu` files
- [CLI Reference](../reference/cli.md) -- Full command documentation
