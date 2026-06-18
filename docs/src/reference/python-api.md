# Python API Reference

## gpuemu

### Client

```python
from gpuemu import Client

client = Client()
client.connect()

# Validate an op
result = client.validate_op("my_op", inputs, expected_output)

# Run fuzz tests
results = client.fuzz_op("my_op", iterations=100)

client.close()
```

### Context Managers

```python
from gpuemu import validate_op

with validate_op(client, "my_op", inputs, expected):
    pass  # Validation happens automatically
```

### Framework Adapters

```python
# PyTorch
from gpuemu import get_pytorch_adapter
PyTorchAdapter, validate_pytorch, check_autograd = get_pytorch_adapter()

# JAX
from gpuemu import get_jax_adapter
JAXAdapter, validate_jax, check_vmap, check_jit = get_jax_adapter()

# TensorFlow
from gpuemu import get_tensorflow_adapter
TensorFlowAdapter, validate_tf, check_keras = get_tensorflow_adapter()
```

See framework-specific guides for detailed API documentation.
