# Quick Start

This guide will get you validating your first op in under 5 minutes.

## 1. Initialize Project

```bash
# Create a new gpuemu project
gpuemu init --name my-ops --framework pytorch --with-examples

# This creates:
# - gpuemu.toml        # Configuration file
# - scripts/           # Reference implementations
# - tests/             # Example tests
# - .gpuemu/           # Local state directory
```

## 2. Start the Daemon

```bash
# Start in background
gpuemu daemon start --background

# Verify it's running
gpuemu status
```

## 3. Run Validation

```bash
# Quick validation (fewer test cases)
gpuemu test --quick

# Full validation
gpuemu test
```

## 4. Fuzz Testing

```bash
# Run 100 random test cases
gpuemu fuzz --iterations 100

# Fuzz a specific op
gpuemu fuzz --op my_custom_op --iterations 1000
```

## 5. Investigate Failures

```bash
# List recent failures
gpuemu failures

# Reproduce a specific failure
gpuemu reproduce <seed>

# Minimize to smallest failing case
gpuemu minimize <seed>
```

## 6. Interactive Debugging

```bash
# Start interactive debug mode
gpuemu debug

# Available commands:
# list          - Show failures
# show <seed>   - Inspect failure details
# tensor <name> - View tensor values
# export <seed> - Generate reproducer script
```

## Example: Adding Your Own Op

### 1. Create a reference implementation

```python
# scripts/ref_my_softmax.py
#!/usr/bin/env python3
import sys
import pickle
import torch

def reference(**inputs):
    x = inputs["x"]
    return torch.softmax(x, dim=-1)

if __name__ == "__main__":
    inputs = pickle.load(sys.stdin.buffer)
    result = reference(**inputs)
    pickle.dump(result.cpu(), sys.stdout.buffer)
```

### 2. Add to gpuemu.toml

```toml
[[ops]]
name = "my_softmax"
module = "my_module.softmax"
reference = "scripts/ref_my_softmax.py"

[ops.tolerances]
float32 = 1e-5
float16 = 1e-3

[ops.invariants]
no_nan = true
no_inf = true
```

### 3. Validate

```bash
gpuemu fuzz --op my_softmax --iterations 100
```

## Next Steps

- [Configuration Reference](./configuration.md) - Full config options
- [PyTorch Tutorial](../tutorials/pytorch-validation.md) - Deep dive into PyTorch validation
- [CI Integration](../tutorials/ci-integration.md) - Set up continuous validation
