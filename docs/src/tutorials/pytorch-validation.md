# PyTorch Validation Tutorial

Step-by-step guide to validating PyTorch ops with gpuemu.

## Setup

```bash
gpuemu init --name pytorch-validation --framework pytorch
pip install gpuemu[pytorch]
```

## Example: Validating a Custom Softmax

### 1. Reference Implementation

```python
# scripts/ref_my_softmax.py
import torch
import pickle
import sys

def reference(**inputs):
    x = inputs["x"]
    return torch.softmax(x, dim=-1)

if __name__ == "__main__":
    inputs = pickle.load(sys.stdin.buffer)
    result = reference(**inputs)
    pickle.dump(result.cpu(), sys.stdout.buffer)
```

### 2. Configuration

```toml
[[ops]]
name = "my_softmax"
module = "my_module.softmax"
reference = "scripts/ref_my_softmax.py"

[ops.tolerances]
float32 = 1e-6
float16 = 1e-3

[ops.invariants]
no_nan = true
no_inf = true
```

### 3. Run Validation

```bash
gpuemu daemon start --background
gpuemu fuzz --op my_softmax --iterations 1000
```

## Handling Failures

See the [debugging guide](../guides/model-developer.md#debugging-failures).
