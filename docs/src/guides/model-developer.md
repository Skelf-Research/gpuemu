# Model Developer Guide

This guide is for ML engineers who want to validate custom ops in their models.

## Overview

As a model developer, you may be using custom ops from libraries or implementing your own. gpuemu helps you ensure these ops behave correctly across different:

- Input shapes (batch sizes, sequence lengths)
- Data types (float32, float16, bfloat16)
- Memory layouts (contiguous, strided)

## Setting Up Validation

### 1. Identify Critical Ops

Focus validation on:
- Custom CUDA kernels
- Fused operations
- Operations with numerical precision concerns
- Operations that differ between training and inference

### 2. Write Reference Implementations

For each op, write a reference implementation in Python:

```python
# scripts/ref_custom_attention.py
import torch

def reference(**inputs):
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    
    # Standard attention (known correct implementation)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)
```

### 3. Configure Tolerances

Different dtypes need different tolerances:

```toml
[ops.tolerances]
float32 = 1e-5   # High precision
float16 = 1e-3   # Lower precision acceptable
bfloat16 = 1e-2  # Even lower precision
```

## Debugging Failures

When validation fails:

1. **Check the failure message** - Is it tolerance, NaN, or shape mismatch?
2. **Reproduce the failure** - `gpuemu reproduce <seed>`
3. **Minimize** - `gpuemu minimize <seed>` to find smallest failing case
4. **Export reproducer** - `gpuemu debug` then `export <seed>`

## Best Practices

1. **Start with float32** - Get correctness first, then test lower precision
2. **Include edge cases** - Empty tensors, size-1 dimensions
3. **Test gradient computation** - If op is used in training
4. **Run in CI** - Catch regressions early
