# Framework Integrations

This document describes how gpuemu integrates with major deep learning frameworks via the daemon + client architecture.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     gpuemu daemon (Rust)                    │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Validation  │  │  Reference  │  │  Storage (sled)     │  │
│  │ Engine      │  │  Executor   │  │  - results          │  │
│  └─────────────┘  └─────────────┘  │  - baselines        │  │
│                                    └─────────────────────┘  │
│                          │                                  │
│                  ┌───────▼───────┐                          │
│                  │  IPC (nng)    │                          │
│                  └───────┬───────┘                          │
└──────────────────────────┼──────────────────────────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
┌──────▼──────┐     ┌──────▼──────┐     ┌──────▼──────┐
│  PyTorch    │     │  JAX        │     │ TensorFlow  │
│  gpuemu-py  │     │  gpuemu-py  │     │ gpuemu-py   │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Installation

```bash
# Install daemon (Rust)
cargo install gpuemu

# Install Python client
pip install gpuemu-py

# Start daemon
gpuemu daemon start
```

## PyTorch

PyTorch custom CUDA extensions are a primary use case for gpuemu.

### Configuration

```toml
# gpuemu.toml
[project]
name = "my-pytorch-project"
framework = "pytorch"

[[ops]]
name = "custom_attention"
module = "my_module.custom_attention"
reference = "scripts/ref_attention.py"

[ops.tolerances]
float32 = 1e-5
float16 = 1e-3
```

### Client Usage

```python
import torch
from gpuemu_py import Client, validate, validate_op

# Connect to daemon
client = Client()

# Validate a model
def test_model():
    model = MyModel()
    with validate(client, model):
        output = model(torch.randn(2, 128, 512))

# Validate a custom op
def test_custom_op():
    from my_module import custom_attention
    q, k, v = torch.randn(2, 8, 128, 64), torch.randn(2, 8, 128, 64), torch.randn(2, 8, 128, 64)

    with validate_op(client, "custom_attention", inputs={"q": q, "k": k, "v": v}):
        output = custom_attention(q, k, v)
```

### Reference Implementation

```python
# scripts/ref_attention.py
import sys
import pickle
import torch

def reference(q, k, v):
    """CPU reference implementation."""
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)

if __name__ == "__main__":
    inputs = pickle.load(sys.stdin.buffer)
    result = reference(**inputs)
    pickle.dump(result, sys.stdout.buffer)
```

### Validation Targets

- Shape fuzzing with dynamic batch/sequence dimensions
- Layout validation (contiguous vs strided views)
- Dtype transitions (fp16/bf16/fp32 accumulation)
- Autograd correctness for backward passes

## JAX

JAX custom primitives and XLA custom calls can be validated with gpuemu.

### Configuration

```toml
# gpuemu.toml
[project]
name = "my-jax-project"
framework = "jax"

[[ops]]
name = "custom_primitive"
module = "my_module.custom_prim"
reference = "scripts/ref_primitive.py"
```

### Client Usage

```python
import jax
import jax.numpy as jnp
from gpuemu_py import Client, validate_op

client = Client()

def test_custom_primitive():
    from my_module import custom_prim

    x = jnp.ones((128, 256))
    with validate_op(client, "custom_primitive", inputs={"x": x}):
        output = custom_prim(x)

def test_vmap_compatibility():
    from my_module import custom_prim

    x = jnp.ones((32, 128, 256))  # batched
    with validate_op(client, "custom_primitive", inputs={"x": x}):
        output = jax.vmap(custom_prim)(x)
```

### Reference Implementation

```python
# scripts/ref_primitive.py
import sys
import pickle
import jax.numpy as jnp

def reference(x):
    """CPU reference using JAX."""
    return jnp.sin(x) + jnp.cos(x)

if __name__ == "__main__":
    inputs = pickle.load(sys.stdin.buffer)
    result = reference(**inputs)
    pickle.dump(result, sys.stdout.buffer)
```

### Validation Targets

- Shape polymorphism with dynamic dimensions
- Batching rules (`vmap` compatibility)
- Gradient rules (`jvp`/`vjp` correctness)
- Sharding behavior validation

## TensorFlow

TensorFlow custom ops can be validated for correctness without GPU execution.

### Configuration

```toml
# gpuemu.toml
[project]
name = "my-tf-project"
framework = "tensorflow"

[[ops]]
name = "custom_op"
module = "my_module.custom_op"
reference = "scripts/ref_custom_op.py"
```

### Client Usage

```python
import tensorflow as tf
from gpuemu_py import Client, validate_op

client = Client()

def test_custom_op():
    from my_module import custom_op

    x = tf.random.normal((128, 256))
    with validate_op(client, "custom_op", inputs={"x": x}):
        output = custom_op(x)

def test_gradient():
    from my_module import custom_op

    x = tf.Variable(tf.random.normal((128, 256)))
    with validate_op(client, "custom_op", inputs={"x": x}, check_gradient=True):
        with tf.GradientTape() as tape:
            output = custom_op(x)
            loss = tf.reduce_sum(output)
        grad = tape.gradient(loss, x)
```

### Validation Targets

- Dynamic shape handling with `None` dimensions
- Dtype casting and promotion rules
- Gradient correctness for trainable ops
- SavedModel compatibility

## Common Patterns

### Kernel Structure Contract

All frameworks benefit from structuring kernels to separate concerns:

```
┌─────────────────────────────────────────┐
│  Math Core (__host__ __device__)        │
│  - Pure computation logic               │
│  - No thread/block assumptions          │
└─────────────────────────────────────────┘
          ↓                    ↓
┌─────────────────┐   ┌─────────────────┐
│  GPU Launcher   │   │  CPU Reference  │
│  - Grid/block   │   │  (Python script │
│    mapping      │   │   or executable)│
└─────────────────┘   └─────────────────┘
```

### Reference Script Protocol

gpuemu invokes reference scripts via stdin/stdout with pickle serialization:

```python
# Standard reference script structure
import sys
import pickle

def reference(**inputs):
    # Implementation here
    return output

if __name__ == "__main__":
    inputs = pickle.load(sys.stdin.buffer)
    result = reference(**inputs)
    pickle.dump(result, sys.stdout.buffer)
```

### CI Configuration

```yaml
# .github/workflows/gpuemu.yml
name: GPU-less Validation
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install gpuemu
        run: |
          curl -sSL https://gpuemu.dev/install.sh | sh
          pip install gpuemu-py

      - name: Start daemon
        run: gpuemu daemon start --background

      - name: Run validation
        run: gpuemu ci

      - name: Store results
        uses: actions/upload-artifact@v4
        with:
          name: validation-results
          path: .gpuemu/results/
```

### Cross-Framework Testing

When ops are shared across frameworks:

```toml
# gpuemu.toml
[[ops]]
name = "shared_attention"
frameworks = ["pytorch", "jax", "tensorflow"]
reference = "scripts/ref_attention.py"

# Framework-specific tolerance overrides
[ops.tolerances.pytorch]
float16 = 1e-3

[ops.tolerances.jax]
float16 = 1e-2  # JAX may have slightly different numerics
```
