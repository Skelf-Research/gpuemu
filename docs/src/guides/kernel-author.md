# Kernel Author Guide

Guide for CUDA kernel developers using gpuemu for validation and artifact checking.

## Kernel Validation

### Reference Script

```python
# scripts/ref_fused_softmax.py
import torch

def reference(**inputs):
    x = inputs["x"]
    return torch.softmax(x, dim=-1)
```

### Configuration

```toml
[[kernels]]
name = "fused_softmax"
source = "kernels/fused_softmax.cu"
reference = "scripts/ref_fused_softmax.py"

[kernels.artifact_checks]
max_registers = 48
max_spills = 0
max_local_memory = 0
```

## Artifact Checking

gpuemu analyzes compiled PTX/SASS for:

- **Register usage** - High register count reduces occupancy
- **Spills** - Memory spills hurt performance
- **Local memory** - Stack usage
- **Instruction patterns** - Detect known slow patterns

```bash
gpuemu lint --ptx kernel.ptx
gpuemu baseline v1.0
# ... make changes ...
gpuemu diff --baseline v1.0 --fail-on-regression
```
