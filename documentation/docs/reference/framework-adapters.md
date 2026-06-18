# Framework Adapters Reference

gpuemu provides framework adapters for PyTorch, JAX, and TensorFlow. Each adapter
implements a common interface for tensor conversion, gradient computation, and
framework-specific validation checks.

---

## Base Adapter

### `FrameworkAdapter` (ABC)

The abstract base class that all framework adapters implement.

```python
from gpuemu.adapters.base import FrameworkAdapter
```

#### Abstract Methods

These methods must be implemented by every adapter.

##### `to_numpy()`

```python
@abstractmethod
def to_numpy(self, tensor: Any) -> np.ndarray
```

Convert a framework-specific tensor to a NumPy array.

| Parameter | Type | Description |
|-----------|------|-------------|
| `tensor` | `Any` | A framework-native tensor object |

**Returns:** `np.ndarray`

---

##### `from_numpy()`

```python
@abstractmethod
def from_numpy(self, array: np.ndarray, dtype: str | None = None) -> Any
```

Convert a NumPy array to a framework-specific tensor.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `array` | `np.ndarray` | | Source array |
| `dtype` | `str \| None` | `None` | Target dtype. If `None`, inferred from the array. |

**Returns:** Framework-native tensor

---

##### `get_dtype_name()`

```python
@abstractmethod
def get_dtype_name(self, tensor: Any) -> str
```

Return the canonical dtype name for a tensor (e.g., `"float32"`).

| Parameter | Type | Description |
|-----------|------|-------------|
| `tensor` | `Any` | A framework-native tensor |

**Returns:** `str`

---

##### `requires_grad()`

```python
@abstractmethod
def requires_grad(self, tensor: Any, requires: bool = True) -> Any
```

Enable or disable gradient tracking on a tensor.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor` | `Any` | | A framework-native tensor |
| `requires` | `bool` | `True` | Whether to enable gradient tracking |

**Returns:** The tensor with gradient tracking set

---

##### `compute_gradient()`

```python
@abstractmethod
def compute_gradient(
    self,
    fn: Callable,
    inputs: list[Any],
    output_index: int = 0,
) -> list[np.ndarray]
```

Compute gradients of a function with respect to its inputs.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fn` | `Callable` | | The function to differentiate |
| `inputs` | `list[Any]` | | Input tensors |
| `output_index` | `int` | `0` | Index of the output to differentiate |

**Returns:** `list[np.ndarray]` -- Gradients for each input

---

#### Concrete Methods

These methods are provided by the base class.

##### `is_available()`

```python
def is_available(self) -> bool
```

Check whether the framework is importable in the current environment.

**Returns:** `bool`

---

##### `get_framework_name()`

```python
def get_framework_name(self) -> str
```

Return the framework name string (e.g., `"pytorch"`, `"jax"`, `"tensorflow"`).

**Returns:** `str`

---

### `GradientChecker`

Utility class for verifying gradient correctness via finite differences.

```python
from gpuemu.adapters.base import GradientChecker

checker = GradientChecker(adapter=adapter, epsilon=1e-5)
result = checker.check(fn=my_op, inputs=[x], output_index=0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adapter` | `FrameworkAdapter` | | The adapter to use for tensor operations |
| `epsilon` | `float` | `1e-5` | Finite difference step size |

| Method | Description |
|--------|-------------|
| `check(fn, inputs, output_index)` | Compare analytic gradients against finite-difference approximation |

---

## PyTorch Adapter

### `PyTorchAdapter`

```python
from gpuemu.adapters.pytorch import PyTorchAdapter
```

Implements `FrameworkAdapter` for PyTorch tensors (`torch.Tensor`).

#### Methods

All abstract methods from `FrameworkAdapter` are implemented:

=== "to_numpy()"

    ```python
    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray
    ```
    Handles GPU tensors by calling `.detach().cpu().numpy()`.

=== "from_numpy()"

    ```python
    def from_numpy(self, array: np.ndarray, dtype: str | None = None) -> torch.Tensor
    ```
    Creates a `torch.Tensor` with optional dtype conversion via `torch.from_numpy()`.

=== "get_dtype_name()"

    ```python
    def get_dtype_name(self, tensor: torch.Tensor) -> str
    ```
    Maps `torch.dtype` to canonical string (e.g., `torch.float32` to `"float32"`).

=== "requires_grad()"

    ```python
    def requires_grad(self, tensor: torch.Tensor, requires: bool = True) -> torch.Tensor
    ```
    Calls `tensor.requires_grad_(requires)` and returns the tensor.

=== "compute_gradient()"

    ```python
    def compute_gradient(
        self,
        fn: Callable,
        inputs: list[torch.Tensor],
        output_index: int = 0,
    ) -> list[np.ndarray]
    ```
    Uses `torch.autograd.grad()` to compute gradients analytically.

---

### Standalone Functions

#### `validate_pytorch()`

```python
def validate_pytorch(
    op_fn: Callable,
    ref_fn: Callable,
    inputs: dict[str, torch.Tensor],
    tolerance: ToleranceConfig | None = None,
) -> ValidationResult
```

Validate a PyTorch op against a reference function.

| Parameter | Type | Description |
|-----------|------|-------------|
| `op_fn` | `Callable` | The op under test |
| `ref_fn` | `Callable` | The reference implementation |
| `inputs` | `dict[str, torch.Tensor]` | Named input tensors |
| `tolerance` | `ToleranceConfig \| None` | Custom tolerance. Uses dtype default if `None`. |

---

#### `check_autograd()`

```python
def check_autograd(
    op_fn: Callable,
    inputs: list[torch.Tensor],
    epsilon: float = 1e-5,
) -> bool
```

Verify that a PyTorch op has correct autograd gradients using `torch.autograd.gradcheck`.

---

#### `validate_custom_autograd_function()`

```python
def validate_custom_autograd_function(
    autograd_fn: type,
    inputs: list[torch.Tensor],
    ref_fn: Callable | None = None,
) -> ValidationResult
```

Validate a custom `torch.autograd.Function` subclass, checking both forward and backward.

| Parameter | Type | Description |
|-----------|------|-------------|
| `autograd_fn` | `type` | The `torch.autograd.Function` subclass |
| `inputs` | `list[torch.Tensor]` | Input tensors |
| `ref_fn` | `Callable \| None` | Optional reference for forward pass comparison |

---

#### `fuzz_pytorch_op()`

```python
def fuzz_pytorch_op(
    op_fn: Callable,
    ref_fn: Callable,
    config: FuzzConfig | None = None,
) -> FuzzResults
```

Fuzz test a PyTorch op against a reference function.

---

## JAX Adapter

### `JAXAdapter`

```python
from gpuemu.adapters.jax import JAXAdapter
```

Implements `FrameworkAdapter` for JAX arrays (`jax.Array`).

#### Methods

=== "to_numpy()"

    ```python
    def to_numpy(self, tensor: jax.Array) -> np.ndarray
    ```
    Converts via `np.asarray(tensor)`.

=== "from_numpy()"

    ```python
    def from_numpy(self, array: np.ndarray, dtype: str | None = None) -> jax.Array
    ```
    Creates a `jax.Array` via `jnp.array()` with optional dtype.

=== "get_dtype_name()"

    ```python
    def get_dtype_name(self, tensor: jax.Array) -> str
    ```
    Maps `jnp.dtype` to canonical string.

=== "requires_grad()"

    ```python
    def requires_grad(self, tensor: jax.Array, requires: bool = True) -> jax.Array
    ```
    JAX arrays are always differentiable. Returns the tensor unchanged.

=== "compute_gradient()"

    ```python
    def compute_gradient(
        self,
        fn: Callable,
        inputs: list[jax.Array],
        output_index: int = 0,
    ) -> list[np.ndarray]
    ```
    Uses `jax.grad()` to compute gradients.

---

### Standalone Functions

#### `validate_jax()`

```python
def validate_jax(
    op_fn: Callable,
    ref_fn: Callable,
    inputs: dict[str, jax.Array],
    tolerance: ToleranceConfig | None = None,
) -> ValidationResult
```

Validate a JAX op against a reference function.

---

#### `check_vmap_compatible()`

```python
def check_vmap_compatible(op_fn: Callable, sample_inputs: list[jax.Array]) -> bool
```

Check whether an op is compatible with `jax.vmap`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `op_fn` | `Callable` | The function to test |
| `sample_inputs` | `list[jax.Array]` | Sample inputs (a batch dimension is added automatically) |

---

#### `check_jit_safe()`

```python
def check_jit_safe(op_fn: Callable, sample_inputs: list[jax.Array]) -> bool
```

Check whether an op can be safely compiled with `jax.jit`.

---

#### `check_pmap_compatible()`

```python
def check_pmap_compatible(op_fn: Callable, sample_inputs: list[jax.Array]) -> bool
```

Check whether an op is compatible with `jax.pmap` for multi-device execution.

---

#### `check_grad_safe()`

```python
def check_grad_safe(op_fn: Callable, sample_inputs: list[jax.Array]) -> bool
```

Check whether `jax.grad()` can be applied to the op without errors.

---

#### `validate_jax_primitive()`

```python
def validate_jax_primitive(
    primitive: jax.core.Primitive,
    impl_fn: Callable,
    ref_fn: Callable,
    sample_inputs: list[jax.Array],
) -> ValidationResult
```

Validate a custom JAX primitive implementation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `primitive` | `jax.core.Primitive` | The JAX primitive to validate |
| `impl_fn` | `Callable` | The primitive implementation |
| `ref_fn` | `Callable` | The reference implementation |
| `sample_inputs` | `list[jax.Array]` | Sample inputs for testing |

---

#### `fuzz_jax_op()`

```python
def fuzz_jax_op(
    op_fn: Callable,
    ref_fn: Callable,
    config: FuzzConfig | None = None,
) -> FuzzResults
```

Fuzz test a JAX op against a reference function.

---

## TensorFlow Adapter

### `TensorFlowAdapter`

```python
from gpuemu.adapters.tensorflow import TensorFlowAdapter
```

Implements `FrameworkAdapter` for TensorFlow tensors (`tf.Tensor`).

#### Methods

=== "to_numpy()"

    ```python
    def to_numpy(self, tensor: tf.Tensor) -> np.ndarray
    ```
    Converts via `tensor.numpy()`.

=== "from_numpy()"

    ```python
    def from_numpy(self, array: np.ndarray, dtype: str | None = None) -> tf.Tensor
    ```
    Creates a `tf.Tensor` via `tf.constant()` with optional dtype cast.

=== "get_dtype_name()"

    ```python
    def get_dtype_name(self, tensor: tf.Tensor) -> str
    ```
    Maps `tf.DType` to canonical string.

=== "requires_grad()"

    ```python
    def requires_grad(self, tensor: tf.Tensor, requires: bool = True) -> tf.Tensor
    ```
    Returns a `tf.Variable` wrapping the tensor when `requires=True`.

=== "compute_gradient()"

    ```python
    def compute_gradient(
        self,
        fn: Callable,
        inputs: list[tf.Tensor],
        output_index: int = 0,
    ) -> list[np.ndarray]
    ```
    Uses `tf.GradientTape` to compute gradients.

---

### Standalone Functions

#### `validate_tensorflow()`

```python
def validate_tensorflow(
    op_fn: Callable,
    ref_fn: Callable,
    inputs: dict[str, tf.Tensor],
    tolerance: ToleranceConfig | None = None,
) -> ValidationResult
```

Validate a TensorFlow op against a reference function.

---

#### `check_keras_layer()`

```python
def check_keras_layer(
    layer: tf.keras.layers.Layer,
    sample_input: tf.Tensor,
) -> bool
```

Verify that a Keras layer can perform a forward pass without error.

| Parameter | Type | Description |
|-----------|------|-------------|
| `layer` | `tf.keras.layers.Layer` | The Keras layer to check |
| `sample_input` | `tf.Tensor` | A sample input tensor |

---

#### `check_tf_function_safe()`

```python
def check_tf_function_safe(op_fn: Callable, sample_inputs: list[tf.Tensor]) -> bool
```

Check whether an op can be safely wrapped with `@tf.function`.

---

#### `check_xla_compatible()`

```python
def check_xla_compatible(op_fn: Callable, sample_inputs: list[tf.Tensor]) -> bool
```

Check whether an op is compatible with XLA compilation via `tf.function(jit_compile=True)`.

---

#### `validate_custom_gradient()`

```python
def validate_custom_gradient(
    op_fn: Callable,
    grad_fn: Callable,
    sample_inputs: list[tf.Tensor],
    epsilon: float = 1e-5,
) -> bool
```

Validate a custom gradient registered with `@tf.custom_gradient`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `op_fn` | `Callable` | | The forward function |
| `grad_fn` | `Callable` | | The custom gradient function |
| `sample_inputs` | `list[tf.Tensor]` | | Sample inputs |
| `epsilon` | `float` | `1e-5` | Finite difference step size for comparison |

---

#### `fuzz_tensorflow_op()`

```python
def fuzz_tensorflow_op(
    op_fn: Callable,
    ref_fn: Callable,
    config: FuzzConfig | None = None,
) -> FuzzResults
```

Fuzz test a TensorFlow op against a reference function.

---

## Usage Examples

### Cross-Framework Validation

```python
from gpuemu.adapters.pytorch import PyTorchAdapter, validate_pytorch
from gpuemu.adapters.jax import JAXAdapter, validate_jax
from gpuemu.adapters.tensorflow import TensorFlowAdapter, validate_tensorflow
import numpy as np

# Define a reference in NumPy
def softmax_ref(logits):
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

# Validate across all frameworks
x_np = np.random.randn(4, 64).astype(np.float32)
```

=== "PyTorch"

    ```python
    import torch

    adapter = PyTorchAdapter()
    x = adapter.from_numpy(x_np)
    result = validate_pytorch(
        op_fn=lambda logits: torch.softmax(logits, dim=-1),
        ref_fn=softmax_ref,
        inputs={"logits": x},
    )
    print(f"PyTorch: passed={result.passed}, max_diff={result.max_diff}")
    ```

=== "JAX"

    ```python
    import jax
    import jax.numpy as jnp

    adapter = JAXAdapter()
    x = adapter.from_numpy(x_np)
    result = validate_jax(
        op_fn=jax.nn.softmax,
        ref_fn=softmax_ref,
        inputs={"logits": x},
    )
    print(f"JAX: passed={result.passed}, max_diff={result.max_diff}")
    ```

=== "TensorFlow"

    ```python
    import tensorflow as tf

    adapter = TensorFlowAdapter()
    x = adapter.from_numpy(x_np)
    result = validate_tensorflow(
        op_fn=lambda logits: tf.nn.softmax(logits, axis=-1),
        ref_fn=softmax_ref,
        inputs={"logits": x},
    )
    print(f"TensorFlow: passed={result.passed}, max_diff={result.max_diff}")
    ```

### Gradient Checking

```python
from gpuemu.adapters.pytorch import PyTorchAdapter, check_autograd
from gpuemu.adapters.base import GradientChecker
import torch

adapter = PyTorchAdapter()
checker = GradientChecker(adapter=adapter, epsilon=1e-5)

x = torch.randn(4, 64, requires_grad=True, dtype=torch.float64)
result = checker.check(
    fn=lambda t: torch.softmax(t, dim=-1),
    inputs=[x],
    output_index=0,
)
print(f"Gradient check passed: {result}")
```
