# Framework Integrations

How gpuemu integrates with PyTorch, JAX, and TensorFlow — from single-shot validation to full fuzzing with framework-specific checks.

## How It Works

```
  Your GPU Code                    gpuemu Daemon
  ┌──────────────┐                 ┌──────────────────┐
  │ run_op()     │                 │ Fuzzer generates │
  │ (on GPU)     │  submit_output  │ random inputs    │
  │              │───────────────> │                  │
  │              │                 │ Runs reference   │
  │              │  pass/fail      │ script on CPU    │
  │              │<─────────────── │                  │
  └──────────────┘                 │ Compares output │ │
                                   │ Stores result   │
                                   └──────────────────┘
```

**You write a CPU reference script. You run your GPU op. gpuemu compares them.**

## Quick Reference

| What you want | How to do it |
|---|---|
| Validate one output | `client.validate_op(name, inputs, output)` |
| Validate with framework adapter | `with validate_pytorch(client, name, inputs) as ctx` |
| Fuzz with GPU op | `fuzz_pytorch_op(client, name, run_op=lambda i: my_gpu_kernel(i))` |
| Fetch test cases yourself | `cases = client.get_test_batch(name, count=50)` |
| Submit one result | `client.submit_output(name, inputs, output, seed)` |

---

## PyTorch

### Setup

```toml
# gpuemu.toml
[project]
name = "my-pytorch-project"
framework = "pytorch"

[[ops]]
name = "flash_attention"
reference = "scripts/ref_flash_attention.py"
input_names = ["q", "k", "v"]
execution_mode = "client_side"

[ops.tolerances]
float32 = 1e-5
float16 = 1e-3
bfloat16 = 1e-3

[ops.invariants]
no_nan = true
no_inf = true
shape_preserved = true
```

### Reference Script (runs on CPU, called by daemon)

```python
#!/usr/bin/env python3
"""scripts/ref_flash_attention.py"""
import sys, json, base64
import numpy as np

def decode_tensor(d):
    dtype = np.dtype(d["dtype"])
    data = base64.b64decode(d["data"])
    return np.frombuffer(data, dtype=dtype).reshape(d["shape"]).copy()

def encode_tensor(arr):
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data": base64.b64encode(arr.tobytes()).decode(),
    }

def reference(q, k, v):
    """Scaled dot-product attention (CPU reference)."""
    scale = q.shape[-1] ** -0.5
    scores = np.matmul(q, k.transpose(0, 2, 1)) * scale
    scores -= np.max(scores, axis=-1, keepdims=True)
    attn = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    return np.matmul(attn, v)

if __name__ == "__main__":
    input_json = json.load(sys.stdin)
    inputs = {name: decode_tensor(t) for name, t in input_json["inputs"].items()}
    result = reference(**inputs)
    json.dump(encode_tensor(result), sys.stdout)
```

### Fuzzing (the main path for GPU developers)

```python
import torch
from gpuemu import Client
from gpuemu.frameworks.pytorch import fuzz_pytorch_op

client = Client()

result = fuzz_pytorch_op(
    client,
    "flash_attention",
    run_op=lambda inputs: torch.ops.my_flash_attn(
        inputs["q"].cuda(),
        inputs["k"].cuda(),
        inputs["v"].cuda(),
    ),
    iterations=100,
    seed=42,            # reproducible
    fail_fast=True,     # stop on first failure
    check_backward=True, # also validate gradients
)

print(f"Forward:  {result['passed']}/{result['total']} passed")
print(f"Backward: {len(result['backward_failures'])} failures")

if result['forward_failures']:
    f = result['forward_failures'][0]
    print(f"First failure at seed {f.seed}: {f.failures[0]['message']}")
```

**What `fuzz_pytorch_op` does step by step:**

1. Calls `client.get_test_batch("flash_attention", count=100)` — daemon generates 100 random test cases with varying shapes, dtypes, and layouts
2. For each test case, converts numpy arrays → `torch.Tensor` on GPU (`.cuda()`)
3. Runs your `run_op(inputs)` — this is YOUR kernel executing on the GPU
4. Converts output back to numpy, calls `client.submit_output(...)` — daemon runs the CPU reference script and compares
5. If `check_backward=True`: also validates gradients via finite differences (`check_autograd`)
6. Returns structured result dict

### Single-Shot Validation

For when you have specific inputs you want to test:

```python
import torch
from gpuemu import Client
from gpuemu.frameworks.pytorch import validate_pytorch, PyTorchAdapter

client = Client()
adapter = PyTorchAdapter()

# Option 1: context manager (recommended for test functions)
with validate_pytorch(client, "flash_attention", {"q": q, "k": k, "v": v}) as ctx:
    ctx["output"] = torch.ops.my_flash_attn(q.cuda(), k.cuda(), v.cuda())

# Option 2: manual submit
output = torch.ops.my_flash_attn(q.cuda(), k.cuda(), v.cuda())
np_inputs = {k: adapter.to_numpy(v) for k, v in [("q", q), ("k", k), ("v", v)]}
np_output = adapter.to_numpy(output)
result = client.validate_op("flash_attention", np_inputs, np_output)
```

### Gradient Validation

```python
from gpuemu.frameworks.pytorch import check_autograd, validate_custom_autograd_function

# Validate autograd correctness against numerical finite differences
x = torch.randn(10, requires_grad=True)
assert check_autograd(lambda x: x ** 2, {"x": x})

# Validate a custom torch.autograd.Function
class MyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x * 2
    @staticmethod
    def backward(ctx, grad):
        return grad * 2

result = validate_custom_autograd_function(MyFunc, {"x": x})
assert result["forward_ok"] and result["backward_ok"]
```

### What PyTorch adapter handles automatically

- GPU → CPU transfer (`.detach().cpu().numpy()`)
- dtype mapping (torch.float16 → "float16" for tolerance lookup)
- Gradient detachment before numpy conversion
- Autograd graph preservation when needed

---

## JAX

### Setup

```toml
# gpuemu.toml
[project]
name = "my-jax-project"
framework = "jax"

[[ops]]
name = "custom_primitive"
reference = "scripts/ref_primitive.py"
input_names = ["x"]
execution_mode = "client_side"

[ops.tolerances]
float32 = 1e-5
```

### Reference Script

```python
#!/usr/bin/env python3
"""scripts/ref_primitive.py"""
import sys, json, base64
import numpy as np

def decode_tensor(d):
    dtype = np.dtype(d["dtype"])
    data = base64.b64decode(d["data"])
    return np.frombuffer(data, dtype=dtype).reshape(d["shape"]).copy()

def encode_tensor(arr):
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data": base64.b64encode(arr.tobytes()).decode(),
    }

def reference(x):
    """CPU reference for custom JAX primitive."""
    return np.sin(x) + np.cos(x)

if __name__ == "__main__":
    input_json = json.load(sys.stdin)
    inputs = {name: decode_tensor(t) for name, t in input_json["inputs"].items()}
    result = reference(**inputs)
    json.dump(encode_tensor(result), sys.stdout)
```

### Fuzzing

```python
import jax.numpy as jnp
from gpuemu import Client
from gpuemu.frameworks.jax import fuzz_jax_op

client = Client()

result = fuzz_jax_op(
    client,
    "custom_primitive",
    run_op=lambda x: jnp.sin(x) + jnp.cos(x),  # your JAX op
    iterations=50,
    check_jit=True,    # verify JIT compilation doesn't change results
    check_vmap=True,   # verify vmap batching works correctly
)

print(f"Forward: {result['passed']}/{result['total']} passed")
print(f"JIT failures: {len(result['jit_failures'])}")
print(f"vmap failures: {len(result['vmap_failures'])}")
```

**What `fuzz_jax_op` does step by step:**

1. Calls `client.get_test_batch(...)` — daemon generates random test cases
2. Converts numpy → `jnp.array` for each input
3. Runs your `run_op(**inputs)` — your JAX computation
4. Calls `client.submit_output(...)` — daemon validates against reference
5. If `check_jit=True`: compares `jax.jit(run_op)(**inputs)` against eager output
6. If `check_vmap=True`: verifies vmap produces same result as element-wise execution
7. Returns structured result dict

### Single-Shot Validation

```python
from gpuemu.frameworks.jax import validate_jax, JAXAdapter

client = Client()
adapter = JAXAdapter()

# Context manager
with validate_jax(client, "custom_primitive", {"x": x}) as ctx:
    ctx["output"] = my_custom_op(x)

# Manual
output = my_custom_op(x)
result = client.validate_op("custom_primitive", {"x": adapter.to_numpy(x)}, adapter.to_numpy(output))
```

### JAX-Specific Checks

```python
from gpuemu.frameworks.jax import (
    check_jit_safe,       # JIT doesn't change numerics
    check_vmap_compatible, # vmap works and matches element-wise
    check_pmap_compatible, # pmap works across devices
    check_grad_safe,       # jax.grad produces finite values
    validate_jax_primitive, # full validation of a custom primitive
)

# Check that compilation is safe
assert check_jit_safe(my_op, {"x": jnp.ones(10)})

# Check vmap
x = jnp.ones((4, 10))  # batch of 4
assert check_vmap_compatible(my_op, {"x": x})
```

### What JAX adapter handles automatically

- `jnp.array` ↔ numpy conversion (via `np.asarray`)
- Floating-point detection for gradient compatibility
- Functional gradient computation with `jax.grad`

---

## TensorFlow

### Setup

```toml
# gpuemu.toml
[project]
name = "my-tf-project"
framework = "tensorflow"

[[ops]]
name = "custom_matmul"
reference = "scripts/ref_matmul.py"
input_names = ["x", "w"]
execution_mode = "client_side"

[ops.tolerances]
float32 = 1e-5
float16 = 1e-3
```

### Reference Script

```python
#!/usr/bin/env python3
"""scripts/ref_matmul.py"""
import sys, json, base64
import numpy as np

def decode_tensor(d):
    dtype = np.dtype(d["dtype"])
    data = base64.b64decode(d["data"])
    return np.frombuffer(data, dtype=dtype).reshape(d["shape"]).copy()

def encode_tensor(arr):
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data": base64.b64encode(arr.tobytes()).decode(),
    }

def reference(x, w):
    return np.matmul(x, w)

if __name__ == "__main__":
    input_json = json.load(sys.stdin)
    inputs = {name: decode_tensor(t) for name, t in input_json["inputs"].items()}
    result = reference(**inputs)
    json.dump(encode_tensor(result), sys.stdout)
```

### Fuzzing

```python
import tensorflow as tf
from gpuemu import Client
from gpuemu.frameworks.tensorflow import fuzz_tensorflow_op

client = Client()

result = fuzz_tensorflow_op(
    client,
    "custom_matmul",
    run_op=lambda x, w: tf.matmul(x, w),
    iterations=50,
    check_gradient=True,  # validate gradients via GradientTape
    check_xla=True,       # verify XLA compilation doesn't change results
)

print(f"Forward: {result['passed']}/{result['total']} passed")
print(f"Gradient failures: {len(result['gradient_failures'])}")
print(f"XLA failures: {len(result['xla_failures'])}")
```

**What `fuzz_tensorflow_op` does step by step:**

1. Calls `client.get_test_batch(...)` — daemon generates random test cases
2. Converts numpy → `tf.constant` for each input
3. Runs your `run_op(**inputs)` — your TF computation
4. Calls `client.submit_output(...)` — daemon validates against reference
5. If `check_gradient=True`: records with `GradientTape`, validates gradients against reference
6. If `check_xla=True`: compares `tf.function(jit_compile=True)` against eager output
7. Returns structured result dict

### Single-Shot Validation

```python
from gpuemu.frameworks.tensorflow import validate_tensorflow, TensorFlowAdapter

client = Client()
adapter = TensorFlowAdapter()

# Simple forward
with validate_tensorflow(client, "custom_matmul", {"x": x, "w": w}) as ctx:
    ctx["output"] = tf.matmul(x, w)

# With gradient check
x = tf.Variable(tf.random.normal((32, 128)))
with validate_tensorflow(client, "custom_matmul", {"x": x}, check_gradient=True) as ctx:
    with ctx["tape"]:
        ctx["output"] = my_custom_op(x)
```

### TensorFlow-Specific Checks

```python
from gpuemu.frameworks.tensorflow import (
    check_keras_layer,       # validate a Keras layer's forward+backward
    check_tf_function_safe,  # @tf.function doesn't change results
    check_xla_compatible,    # XLA compilation works and matches
    validate_custom_gradient, # custom gradient vs finite differences
)

# Validate a Keras layer
layer = MyCustomLayer(units=64)
assert check_keras_layer(layer, (32, 128), client, "my_layer")

# Check tf.function safety
assert check_tf_function_safe(my_op, {"x": tf.ones(10)})

# Validate custom gradient
result = validate_custom_gradient(forward_fn, grad_fn, {"x": tf.ones(10)})
assert result["gradient_ok"]
```

### What TensorFlow adapter handles automatically

- `tf.Tensor` ↔ numpy conversion (`.numpy()`)
- `tf.Variable` detection for gradient tracking
- `GradientTape` lifecycle management
- dtype casting with `tf.cast`

---

## Execution Modes

Every op in `gpuemu.toml` has an `execution_mode` that determines how testing works:

### `client_side` (default — recommended)

The client (which has GPU access) runs the op. The daemon only runs the reference and validates.

```python
# PyTorch
result = fuzz_pytorch_op(client, "my_op", run_op=lambda i: my_kernel(i["x"].cuda()))

# JAX
result = fuzz_jax_op(client, "my_op", run_op=lambda x: my_jax_op(x))

# TensorFlow
result = fuzz_tensorflow_op(client, "my_op", run_op=lambda x: tf_op(x))
```

### `daemon_orchestrated`

Fine-grained control: you fetch test cases and submit results one at a time.

```python
cases = client.get_test_batch("my_op", count=50)
for case in cases:
    output = my_gpu_op(case["inputs"])
    result = client.submit_output("my_op", case["inputs"], output, case["seed"])
    if not result.passed:
        break
```

### `script_based`

The daemon runs both the reference and your op script (requires GPU on daemon machine).

```toml
[[ops]]
name = "my_op"
reference = "scripts/ref_my_op.py"
op_script = "scripts/run_my_op.py"
execution_mode = "script_based"
```

The op script receives the same JSON+base64 input format as the reference script and must produce the same JSON+base64 output format.

---

## CI Configuration

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
          pip install gpuemu

      - name: Start daemon
        run: gpuemu daemon start --background

      - name: Run CI validation
        run: gpuemu ci --format json --output report.json

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: validation-results
          path: report.json
```

For GPU-enabled CI runners, use `script_based` execution mode and the daemon will handle everything automatically:

```yaml
- name: Run validation (with GPU)
  run: gpuemu ci
```

---

## Cross-Framework Testing

When ops are shared across frameworks, register them once with multiple framework adapters:

```python
import torch
import jax.numpy as jnp
from gpuemu import Client
from gpuemu.frameworks.pytorch import fuzz_pytorch_op
from gpuemu.frameworks.jax import fuzz_jax_op

client = Client()

# Same op, different frameworks
pytorch_result = fuzz_pytorch_op(
    client, "attention",
    run_op=lambda i: torch.ops.my_attn(i["q"].cuda(), i["k"].cuda(), i["v"].cuda()),
    iterations=50,
)

jax_result = fuzz_jax_op(
    client, "attention",
    run_op=lambda q, k, v: jnp.dot(q, jnp.transpose(k, (0, 2, 1))),
    iterations=50,
    check_jit=True,
)
```

---

## Reference Script Protocol

All reference scripts communicate with the daemon via **JSON + base64** over stdin/stdout:

**Input (daemon → script):**
```json
{
  "inputs": {
    "x": {"shape": [2, 3], "dtype": "float32", "data": "AAAAAB..."},
    "w": {"shape": [3, 4], "dtype": "float32", "data": "BBBBBB..."}
  },
  "kwargs": {}
}
```

**Output (script → daemon):**
```json
{
  "shape": [2, 4],
  "dtype": "float32",
  "data": "CCCCCC..."
}
```

The `data` field is `base64(model_tensor.tobytes())`. This protocol is the same for all three frameworks — only the framework-specific decode/encode logic differs.

---

## API Summary

| Method | Framework | What it does |
|--------|-----------|-------------|
| `fuzz_pytorch_op(client, name, run_op, ...)` | PyTorch | Fuzz with GPU execution + optional gradient check |
| `fuzz_jax_op(client, name, run_op, ...)` | JAX | Fuzz with optional JIT/vmap checks |
| `fuzz_tensorflow_op(client, name, run_op, ...)` | TF | Fuzz with optional gradient/XLA checks |
| `validate_pytorch(client, name, inputs)` | PyTorch | Single-shot context manager |
| `validate_jax(client, name, inputs)` | JAX | Single-shot context manager |
| `validate_tensorflow(client, name, inputs)` | TF | Single-shot context manager |
| `check_autograd(op, inputs)` | PyTorch | Compare analytical vs numerical gradients |
| `check_vmap_compatible(op, inputs)` | JAX | Verify vmap batching correctness |
| `check_jit_safe(op, inputs)` | JAX | Verify JIT doesn't change results |
| `check_keras_layer(layer, shape, client, name)` | TF | Validate Keras layer forward+backward |
| `check_xla_compatible(op, inputs)` | TF | Verify XLA compilation |
| `validate_custom_gradient(func, grad_fn, inputs)` | TF | Custom gradient vs finite differences |