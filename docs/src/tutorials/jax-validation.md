# JAX Validation Tutorial

Validating JAX primitives and custom ops with gpuemu.

## Setup

```bash
gpuemu init --name jax-validation --framework jax
pip install gpuemu-py[jax]
```

## JAX-Specific Features

### vmap Compatibility

```python
from gpuemu_py.frameworks.jax import check_vmap_compatible

def my_op(x):
    return jnp.sin(x)

assert check_vmap_compatible(my_op, {"x": jnp.ones((4, 10))})
```

### JIT Safety

```python
from gpuemu_py.frameworks.jax import check_jit_safe

assert check_jit_safe(my_op, {"x": jnp.ones(10)})
```

## Configuration

```toml
[project]
framework = "jax"

[[ops]]
name = "my_jax_op"
reference = "scripts/ref_my_jax_op.py"
```
