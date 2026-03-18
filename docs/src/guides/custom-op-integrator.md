# Custom Op Integrator Guide

Guide for integrating gpuemu validation into custom op libraries.

## Integration Points

### PyTorch Custom Ops

```python
from gpuemu_py import validate_op

class MyCustomOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        with validate_op("my_op", {"x": x}):
            return my_cuda_kernel(x)
```

### Testing Framework

```python
import pytest
from gpuemu_py import Client

@pytest.fixture
def gpuemu_client():
    return Client()

def test_my_op(gpuemu_client):
    x = torch.randn(32, 128)
    result = my_op(x)
    gpuemu_client.validate("my_op", {"x": x}, result)
```

## Automated Validation

Configure CI to run validation on every PR. See [CI Integration](../tutorials/ci-integration.md).
