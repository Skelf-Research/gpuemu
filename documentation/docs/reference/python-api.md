# Python API Reference

Complete reference for the `gpuemu` Python client library. This package provides
a high-level interface for communicating with the gpuemu daemon, running validations,
fuzz testing, and managing results.

```bash
pip install gpuemu
```

---

## Client

The primary interface for interacting with the gpuemu daemon.

### Constructor

```python
Client(socket_path: str | None = None, timeout_ms: int = 30000)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `socket_path` | `str \| None` | `None` | Path to the daemon Unix socket. Defaults to `~/.gpuemu/gpuemu.sock`. |
| `timeout_ms` | `int` | `30000` | Request timeout in milliseconds |

**Context Manager Support**

The `Client` class supports the context manager protocol for automatic cleanup:

```python
from gpuemu import Client

with Client() as client:
    result = client.ping()
    print(result)
```

### Methods

#### `ping()`

Check connectivity with the daemon.

```python
def ping() -> str
```

Returns `"pong"` if the daemon is reachable.

---

#### `validate_op()`

Validate an operation against its reference implementation.

```python
def validate_op(
    op_name: str,
    inputs: dict[str, np.ndarray],
    output: np.ndarray,
    dtype: str = "float32",
    seed: int | None = None,
) -> ValidationResult
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `op_name` | `str` | Name of the op (must match `gpuemu.toml`) |
| `inputs` | `dict[str, np.ndarray]` | Named input tensors |
| `output` | `np.ndarray` | The output tensor to validate |
| `dtype` | `str` | Data type used for tolerance lookup |
| `seed` | `int \| None` | Optional seed for reproducibility |

Returns a [`ValidationResult`](#validationresult).

---

#### `get_result()`

Retrieve a stored validation result by seed.

```python
def get_result(seed: int) -> ValidationResult
```

---

#### `list_results()`

List all stored validation results.

```python
def list_results() -> list[ValidationResult]
```

---

#### `store_baseline()`

Store current results as a named baseline.

```python
def store_baseline(tag: str) -> None
```

---

#### `fuzz_op()`

Run daemon-side fuzz testing on an operation.

```python
def fuzz_op(
    op_name: str,
    iterations: int = 100,
    seed: int | None = None,
) -> FuzzResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `op_name` | `str` | | Name of the op to fuzz |
| `iterations` | `int` | `100` | Number of fuzz iterations |
| `seed` | `int \| None` | `None` | Fixed seed for reproducibility |

Returns a [`FuzzResults`](#fuzzresults).

---

#### `reproduce()`

Reproduce a specific fuzz failure.

```python
def reproduce(seed: int) -> ReproduceResult
```

Returns a [`ReproduceResult`](#reproduceresult).

---

#### `minimize()`

Minimize a failing test case.

```python
def minimize(
    seed: int,
    strategy: str | None = None,
    max_iters: int = 100,
) -> MinimizeResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | | Seed of the failure to minimize |
| `strategy` | `str \| None` | `None` | `"binary-search-dims"` or `"binary-search-values"` |
| `max_iters` | `int` | `100` | Maximum minimization iterations |

Returns a [`MinimizeResult`](#minimizeresult).

---

#### `list_failures()`

List stored fuzz failures.

```python
def list_failures(limit: int = 20) -> list[ValidationResult]
```

---

#### `get_test_case()`

Retrieve a specific test case from the daemon for client-side execution.

```python
def get_test_case(op_name: str, seed: int) -> dict
```

Returns a dictionary containing the test case inputs and metadata.

---

#### `get_test_batch()`

Retrieve a batch of test cases for client-side execution.

```python
def get_test_batch(op_name: str, seeds: list[int]) -> list[dict]
```

---

#### `submit_output()`

Submit the output of a client-side execution back to the daemon for validation.

```python
def submit_output(
    op_name: str,
    seed: int,
    output: np.ndarray,
) -> ValidationResult
```

---

#### `fuzz_op_client_side()`

Run client-side fuzz testing. The daemon generates test cases, the client executes
them locally, and submits results back for validation.

```python
def fuzz_op_client_side(
    op_name: str,
    op_fn: Callable,
    iterations: int = 100,
    seed: int | None = None,
) -> FuzzResults
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `op_name` | `str` | Name of the op to fuzz |
| `op_fn` | `Callable` | The function under test |
| `iterations` | `int` | Number of fuzz iterations |
| `seed` | `int \| None` | Fixed seed for reproducibility |

---

## Data Classes

### `ValidationResult`

Result of a single validation run.

| Field | Type | Description |
|-------|------|-------------|
| `passed` | `bool` | Whether the validation passed |
| `seed` | `int` | Seed used for this validation |
| `op_name` | `str` | Name of the validated op |
| `max_diff` | `float` | Maximum absolute difference |
| `max_rel_diff` | `float` | Maximum relative difference |
| `failures` | `list[str]` | List of failure descriptions |
| `timestamp` | `str` | ISO 8601 timestamp |
| `duration_ms` | `int` | Validation duration in milliseconds |
| `repro_info` | `ReproductionInfo \| None` | Reproduction information if the test failed |

---

### `FuzzResults`

Aggregated results from a fuzz testing session.

| Field | Type | Description |
|-------|------|-------------|
| `seed` | `int` | Root seed for this fuzz session |
| `total` | `int` | Total number of iterations run |
| `passed` | `int` | Number of passing iterations |
| `failed` | `int` | Number of failing iterations |
| `failures` | `list[ValidationResult]` | Detailed results for each failure |

---

### `ReproduceResult`

Result of reproducing a specific failure.

| Field | Type | Description |
|-------|------|-------------|
| `result` | `ValidationResult` | The validation result of the reproduction |
| `inputs` | `dict[str, np.ndarray]` | The input tensors that triggered the failure |

---

### `MinimizeResult`

Result of minimizing a failing test case.

| Field | Type | Description |
|-------|------|-------------|
| `original_seed` | `int` | The original failure seed |
| `minimized_seed` | `int` | Seed for the minimized test case |
| `minimized_shape` | `tuple[int, ...]` | The minimized input shape |
| `result` | `ValidationResult` | Validation result of the minimized case |

---

### `ReproductionInfo`

Metadata needed to exactly reproduce a test case.

| Field | Type | Description |
|-------|------|-------------|
| `seed` | `int` | RNG seed |
| `shape` | `tuple[int, ...]` | Input tensor shape |
| `strides` | `tuple[int, ...]` | Input tensor strides |
| `dtype` | `str` | Data type string |
| `layout` | `str` | Memory layout descriptor |
| `fuzz_config` | `FuzzConfig` | The fuzz configuration used |
| `input_snapshot` | `dict` | Serialized snapshot of input values |

---

## Validation Utilities

### `validate_op()` Context Manager

A convenience context manager that wraps op execution with automatic validation.

```python
from gpuemu.validation import validate_op

with validate_op("softmax", inputs={"logits": x}) as ctx:
    output = my_softmax(x)
    ctx.set_output(output)

assert ctx.result.passed
```

---

### Fuzz Generators

Generators that yield randomized configurations for fuzz testing.

#### `fuzz_shapes()`

```python
def fuzz_shapes(
    min_dims: int = 1,
    max_dims: int = 4,
    min_size: int = 1,
    max_size: int = 1024,
) -> Iterator[tuple[int, ...]]
```

Yields random tensor shapes.

---

#### `fuzz_dtypes()`

```python
def fuzz_dtypes(
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> Iterator[str]
```

Yields random dtype strings, optionally filtered.

---

#### `fuzz_layouts()`

```python
def fuzz_layouts() -> Iterator[str]
```

Yields random memory layouts (`"contiguous"`, `"strided"`, `"channels_last"`, etc.).

---

#### `fuzz_shapes_seeded()`

```python
def fuzz_shapes_seeded(seed: int, **kwargs) -> Iterator[tuple[int, ...]]
```

Deterministic variant of `fuzz_shapes()` with a fixed seed.

---

#### `fuzz_dtypes_seeded()`

```python
def fuzz_dtypes_seeded(seed: int, **kwargs) -> Iterator[str]
```

Deterministic variant of `fuzz_dtypes()` with a fixed seed.

---

#### `fuzz_layouts_seeded()`

```python
def fuzz_layouts_seeded(seed: int) -> Iterator[str]
```

Deterministic variant of `fuzz_layouts()` with a fixed seed.

---

### `generate_random_tensor()`

Generate a random tensor from a seed and specification.

```python
def generate_random_tensor(
    seed: int,
    shape: tuple[int, ...],
    dtype: str = "float32",
    domain: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | | RNG seed for reproducibility |
| `shape` | `tuple[int, ...]` | | Tensor shape |
| `dtype` | `str` | `"float32"` | NumPy-compatible dtype string |
| `domain` | `tuple[float, float]` | `(-1.0, 1.0)` | Value range `(min, max)` |

---

### `FuzzConfig`

Configuration dataclass for fuzz testing sessions.

```python
@dataclass
class FuzzConfig:
    iterations: int = 100
    seed: int | None = None
    min_dims: int = 1
    max_dims: int = 4
    min_size: int = 1
    max_size: int = 1024
    dtypes: list[str] = field(default_factory=lambda: ["float32"])
    layouts: list[str] = field(default_factory=lambda: ["contiguous"])
```

---

### `SeededFuzzer`

A stateful fuzzer that generates reproducible test cases.

```python
class SeededFuzzer:
    def __init__(self, seed: int, config: FuzzConfig | None = None): ...
    def next_test_case(self) -> TestCase: ...
    def run(self, op_fn: Callable) -> FuzzResults: ...
```

#### `TestCase`

```python
@dataclass
class TestCase:
    seed: int
    shape: tuple[int, ...]
    dtype: str
    layout: str
    inputs: dict[str, np.ndarray]
```

---

## RNG

Deterministic random number generation for reproducible testing.

### `SeededRng`

A portable, seedable RNG that produces identical sequences across Python and Rust.

```python
class SeededRng:
    def __init__(self, seed: int): ...
    def derive(self, domain: str) -> "SeededRng": ...
    def choice(self, items: list[T]) -> T: ...
    def gen_range(self, low: int, high: int) -> int: ...
    def gen_u64(self) -> int: ...
    def gen_f32(self) -> float: ...
    def randn(self, shape: tuple[int, ...]) -> np.ndarray: ...
```

| Method | Description |
|--------|-------------|
| `derive(domain)` | Create a child RNG scoped to a named domain |
| `choice(items)` | Pick a random element from a list |
| `gen_range(low, high)` | Generate an integer in `[low, high)` |
| `gen_u64()` | Generate a random unsigned 64-bit integer |
| `gen_f32()` | Generate a random float in `[0.0, 1.0)` |
| `randn(shape)` | Generate a tensor of normally distributed values |

---

### Standalone Functions

#### `derive_seed()`

```python
def derive_seed(seed: int, domain: str) -> int
```

Derive a new seed by hashing the parent seed with a domain string.

---

#### `generate_seed()`

```python
def generate_seed() -> int
```

Generate a fresh random seed from system entropy.

---

## Tolerances

Utilities for managing numerical comparison tolerances.

### `ToleranceConfig`

Configuration for a single tolerance check.

```python
@dataclass
class ToleranceConfig:
    atol: float  # Absolute tolerance
    rtol: float  # Relative tolerance
```

| Method | Description |
|--------|-------------|
| `for_dtype(dtype: str)` | Return a `ToleranceConfig` appropriate for the given dtype |
| `strict()` | Return a strict tolerance (`atol=1e-7, rtol=1e-7`) |
| `relaxed()` | Return a relaxed tolerance (`atol=1e-3, rtol=1e-3`) |
| `scale(factor: float)` | Return a new config with tolerances scaled by `factor` |

---

### `ToleranceProfile`

Named tolerance profiles for common use cases.

```python
class ToleranceProfile:
    @staticmethod
    def get(name: str) -> ToleranceConfig: ...

    @staticmethod
    def for_testing() -> ToleranceConfig: ...

    @staticmethod
    def for_production() -> ToleranceConfig: ...

    @staticmethod
    def for_cross_framework() -> ToleranceConfig: ...
```

| Profile | Description |
|---------|-------------|
| `for_testing()` | Relaxed tolerances suitable for development |
| `for_production()` | Strict tolerances for production validation |
| `for_cross_framework()` | Tolerances accounting for cross-framework numerical variance |

---

### Standalone Functions

#### `calibrate_tolerance()`

```python
def calibrate_tolerance(
    op_fn: Callable,
    ref_fn: Callable,
    shapes: list[tuple[int, ...]],
    dtype: str = "float32",
    n_samples: int = 100,
) -> ToleranceConfig
```

Empirically determine appropriate tolerances by running both functions on random inputs.

---

#### `get_recommended_tolerance()`

```python
def get_recommended_tolerance(
    dtype: str,
    op_type: str = "elementwise",
) -> ToleranceConfig
```

Return recommended tolerance values based on dtype and operation type.

---

## Auto-generated API Documentation

::: gpuemu.client.Client
