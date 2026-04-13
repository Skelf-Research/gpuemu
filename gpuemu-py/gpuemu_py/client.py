"""Client for communicating with the gpuemu daemon."""

import base64
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROTOCOL_VERSION = 1

try:
    import pynng

    HAS_PYNNG = True
except ImportError:
    pynng = None
    HAS_PYNNG = False


@dataclass
class ReproductionInfo:
    """Information needed to reproduce a failure."""

    seed: int
    shape: List[int]
    strides: List[int]
    dtype: str
    layout: str
    fuzz_config: Optional[Dict[str, Any]] = None
    input_snapshot: Optional[bytes] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReproductionInfo":
        """Create from dictionary."""
        return cls(
            seed=data.get("seed", 0),
            shape=data.get("shape", []),
            strides=data.get("strides", []),
            dtype=data.get("dtype", "Float32"),
            layout=data.get("layout", "Contiguous"),
            fuzz_config=data.get("fuzz_config"),
            input_snapshot=base64.b64decode(data["input_snapshot"])
            if data.get("input_snapshot")
            else None,
        )


@dataclass
class ValidationResult:
    """Result of a validation run."""

    passed: bool
    seed: int
    op_name: str
    max_diff: float
    max_rel_diff: float
    failures: List[Dict[str, Any]]
    timestamp: int
    duration_ms: int
    repro_info: Optional[ReproductionInfo] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationResult":
        """Create from dictionary."""
        repro = None
        if data.get("repro_info"):
            repro = ReproductionInfo.from_dict(data["repro_info"])

        return cls(
            passed=data.get("passed", False),
            seed=data.get("seed", 0),
            op_name=data.get("op_name", ""),
            max_diff=data.get("max_diff", 0.0),
            max_rel_diff=data.get("max_rel_diff", 0.0),
            failures=data.get("failures", []),
            timestamp=data.get("timestamp", 0),
            duration_ms=data.get("duration_ms", 0),
            repro_info=repro,
        )


@dataclass
class FuzzResults:
    """Results of a fuzz testing session."""

    seed: int
    total: int
    passed: int
    failed: int
    failures: List[ValidationResult]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FuzzResults":
        """Create from dictionary."""
        return cls(
            seed=data.get("seed", 0),
            total=data.get("total", 0),
            passed=data.get("passed", 0),
            failed=data.get("failed", 0),
            failures=[ValidationResult.from_dict(f) for f in data.get("failures", [])],
        )


@dataclass
class ReproduceResult:
    """Result of reproducing a failure."""

    result: ValidationResult
    inputs: Dict[str, np.ndarray]

    @classmethod
    def from_dict(cls, data: Dict[str, Any], decode_tensor_fn) -> "ReproduceResult":
        """Create from dictionary."""
        inputs = {}
        for name, tensor_data in data.get("inputs", {}).items():
            inputs[name] = decode_tensor_fn(tensor_data)

        return cls(
            result=ValidationResult.from_dict(data.get("result", {})),
            inputs=inputs,
        )


@dataclass
class MinimizeResult:
    """Result of minimizing a failure."""

    original_seed: int
    minimized_seed: int
    minimized_shape: List[int]
    result: ValidationResult

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MinimizeResult":
        """Create from dictionary."""
        return cls(
            original_seed=data.get("original_seed", 0),
            minimized_seed=data.get("minimized_seed", 0),
            minimized_shape=data.get("minimized_shape", []),
            result=ValidationResult.from_dict(data.get("result", {})),
        )


class ClientError(Exception):
    """Error from the gpuemu client."""

    pass


class Client:
    """Client for communicating with the gpuemu daemon.

    Example:
        >>> client = Client()
        >>> client.ping()
        {'version': '0.1.0', 'uptime_secs': 123}
    """

    def __init__(
        self,
        socket_path: Optional[str] = None,
        timeout_ms: int = 30000,
    ):
        """Initialize the client.

        Args:
            socket_path: Path to the daemon socket. Defaults to ~/.gpuemu/gpuemu.sock
            timeout_ms: Timeout for requests in milliseconds.
        """
        if socket_path is None:
            socket_path = os.path.expanduser("~/.gpuemu/gpuemu.sock")

        self.socket_path = socket_path
        self.timeout_ms = timeout_ms
        self._socket = None

    def _ensure_connected(self):
        """Ensure we have a connection to the daemon."""
        if not HAS_PYNNG:
            raise ImportError(
                "pynng is required for gpuemu-py. Install with: pip install pynng"
            )

        if self._socket is None:
            self._socket = pynng.Req0()
            self._socket.recv_timeout = self.timeout_ms
            self._socket.send_timeout = self.timeout_ms

            socket_url = f"ipc://{self.socket_path}"
            try:
                self._socket.dial(socket_url)
            except pynng.exceptions.ConnectionRefused:
                raise ClientError(
                    f"Cannot connect to daemon at {self.socket_path}. "
                    "Is the daemon running? Start it with: gpuemu daemon start"
                )

            self._check_protocol_version()

        return self._socket

    def _check_protocol_version(self):
        """Verify daemon protocol version is compatible (called once on connect)."""
        try:
            saved = self._socket
            self._socket = None
            ping_resp = self._send_request({"type": "Ping"})
            self._socket = saved
            daemon_pv = ping_resp.get("protocol_version", 0)
            if daemon_pv != PROTOCOL_VERSION:
                raise ClientError(
                    f"Protocol version mismatch: client={PROTOCOL_VERSION}, "
                    f"daemon={daemon_pv}. Please upgrade the "
                    f"{'client' if daemon_pv > PROTOCOL_VERSION else 'daemon'}."
                )
        except ClientError:
            raise
        except Exception:
            pass

    def close(self):
        """Close the connection."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request and return the response."""
        socket = self._ensure_connected()

        # Serialize request as JSON (simple protocol for MVP)
        request_bytes = json.dumps(request).encode("utf-8")

        try:
            socket.send(request_bytes)
            response_bytes = socket.recv()
            return json.loads(response_bytes.decode("utf-8"))
        except pynng.exceptions.Timeout:
            raise ClientError("Request timed out")
        except Exception as e:
            raise ClientError(f"Request failed: {e}")

    def ping(self) -> Dict[str, Any]:
        """Ping the daemon to check if it's alive.

        Returns:
            Dict with 'version', 'protocol_version', and 'uptime_secs'.

        Raises:
            ClientError: If the daemon has an incompatible protocol version.
        """
        response = self._send_request({"type": "Ping"})

        if response.get("type") == "Pong":
            daemon_pv = response.get("protocol_version", 0)
            if daemon_pv != PROTOCOL_VERSION:
                raise ClientError(
                    f"Protocol version mismatch: client={PROTOCOL_VERSION}, "
                    f"daemon={daemon_pv}. Please upgrade the "
                    f"{'client' if daemon_pv > PROTOCOL_VERSION else 'daemon'}."
                )
            return {
                "version": response.get("version", "unknown"),
                "protocol_version": daemon_pv,
                "uptime_secs": response.get("uptime_secs", 0),
            }
        elif response.get("type") == "Error":
            raise ClientError(response.get("message", "Unknown error"))
        else:
            raise ClientError(f"Unexpected response: {response}")

    def validate_op(
        self,
        op_name: str,
        inputs: Dict[str, np.ndarray],
        output: np.ndarray,
        **kwargs,
    ) -> ValidationResult:
        """Validate an op output against its reference implementation.

        Args:
            op_name: Name of the op (must be registered in gpuemu.toml).
            inputs: Input tensors as numpy arrays.
            output: Output tensor to validate.
            **kwargs: Additional kwargs to pass to the reference script.

        Returns:
            ValidationResult with pass/fail status and details.
        """
        # Encode tensors for transmission
        encoded_inputs = {
            name: self._encode_tensor(arr) for name, arr in inputs.items()
        }
        encoded_output = self._encode_tensor(output)

        request = {
            "type": "ValidateOp",
            "op_name": op_name,
            "inputs": encoded_inputs,
            "output": encoded_output,
            "kwargs": {k: str(v) for k, v in kwargs.items()},
        }

        response = self._send_request(request)

        if response.get("type") == "ValidationResult":
            return ValidationResult.from_dict(response.get("result", {}))
        elif response.get("type") == "Error":
            raise ClientError(response.get("message", "Unknown error"))
        else:
            raise ClientError(f"Unexpected response: {response}")

    def get_result(self, seed: int) -> Optional[ValidationResult]:
        """Get a stored validation result by seed.

        Args:
            seed: The seed of the validation run.

        Returns:
            ValidationResult if found, None otherwise.
        """
        request = {"type": "GetResult", "seed": seed}
        response = self._send_request(request)

        if response.get("type") == "ValidationResult":
            return ValidationResult.from_dict(response.get("result", {}))
        elif response.get("type") == "Error":
            if response.get("code") == "NotFound":
                return None
            raise ClientError(response.get("message", "Unknown error"))
        else:
            raise ClientError(f"Unexpected response: {response}")

    def list_results(self, limit: int = 100) -> List[ValidationResult]:
        """List recent validation results.

        Args:
            limit: Maximum number of results to return.

        Returns:
            List of ValidationResult objects.
        """
        request = {"type": "ListResults", "limit": limit}
        response = self._send_request(request)

        if response.get("type") == "Results":
            return [ValidationResult.from_dict(r) for r in response.get("results", [])]
        elif response.get("type") == "Error":
            raise ClientError(response.get("message", "Unknown error"))
        else:
            raise ClientError(f"Unexpected response: {response}")

    def store_baseline(self, tag: str) -> None:
        """Store current results as a baseline.

        Args:
            tag: Tag name for the baseline.
        """
        request = {"type": "StoreBaseline", "tag": tag}
        response = self._send_request(request)

        if response.get("type") == "Ok":
            return
        elif response.get("type") == "Error":
            raise ClientError(response.get("message", "Unknown error"))
        else:
            raise ClientError(f"Unexpected response: {response}")

    # =========================================================================
    # Phase 2: Fuzzing and Reproducibility
    # =========================================================================

    def fuzz_op(
        self,
        op_name: str,
        seed: Optional[int] = None,
        iterations: int = 100,
        fail_fast: bool = False,
        batch_sizes: Optional[List[int]] = None,
        seq_lengths: Optional[List[int]] = None,
        hidden_dims: Optional[List[int]] = None,
        dtypes: Optional[List[str]] = None,
        layouts: Optional[List[str]] = None,
    ) -> FuzzResults:
        """Fuzz test an op with random inputs.

        Args:
            op_name: Name of the op (must be registered in gpuemu.toml).
            seed: Master seed for reproducibility. If None, uses current timestamp.
            iterations: Number of test cases to generate.
            fail_fast: Stop on first failure.
            batch_sizes: List of batch sizes to use.
            seq_lengths: List of sequence lengths to use.
            hidden_dims: List of hidden dimensions to use.
            dtypes: List of dtype strings to use.
            layouts: List of layout types to use.

        Returns:
            FuzzResults with pass/fail counts and list of failures.

        Example:
            >>> results = client.fuzz_op("matmul", seed=12345, iterations=100)
            >>> print(f"Passed: {results.passed}/{results.total}")
            >>> for failure in results.failures:
            ...     print(f"  Seed {failure.seed}: {failure.failures[0]['message']}")
        """
        if seed is None:
            seed = int(time.time_ns()) & 0xFFFFFFFFFFFFFFFF

        # Build fuzz config
        fuzz_config = {
            "seed": seed,
            "shape_options": {
                "batch_sizes": batch_sizes or [1, 2, 4, 8, 16, 32],
                "seq_lengths": seq_lengths or [64, 128, 256, 512, 1024],
                "hidden_dims": hidden_dims or [256, 512, 768, 1024],
                "edge_cases": [[1], [1, 1], [1, 1, 1]],
            },
            "dtypes": dtypes or ["float32", "float16"],
            "layouts": layouts or ["Contiguous", "Strided", "Transposed"],
        }

        request = {
            "type": "FuzzOp",
            "op_name": op_name,
            "fuzz_config": fuzz_config,
            "iterations": iterations,
            "fail_fast": fail_fast,
        }

        response = self._send_request(request)

        if response.get("type") == "FuzzResults":
            return FuzzResults.from_dict(response)
        elif response.get("type") == "Error":
            raise ClientError(response.get("message", "Unknown error"))
        else:
            raise ClientError(f"Unexpected response: {response}")

    def reproduce(self, seed: int) -> ReproduceResult:
        """Reproduce a failing test case by seed.

        Retrieves the stored failure and regenerates the exact inputs
        that caused the failure.

        Args:
            seed: The seed of the failing test case.

        Returns:
            ReproduceResult with the original result and regenerated inputs.

        Example:
            >>> repro = client.reproduce(12345)
            >>> print(f"Op: {repro.result.op_name}")
            >>> print(f"Input shape: {repro.inputs['input'].shape}")
        """
        request = {"type": "Reproduce", "seed": seed}
        response = self._send_request(request)

        if response.get("type") == "ReproduceResult":
            return ReproduceResult.from_dict(response, self._decode_tensor)
        elif response.get("type") == "Error":
            raise ClientError(response.get("message", "Unknown error"))
        else:
            raise ClientError(f"Unexpected response: {response}")

    def minimize(
        self,
        seed: int,
        strategy: str = "binary-search-dims",
        max_iters: int = 100,
    ) -> MinimizeResult:
        """Minimize a failing test case.

        Attempts to find a smaller input that still triggers the failure.

        Args:
            seed: The seed of the failing test case.
            strategy: Minimization strategy. One of:
                - "binary-search-dims": Binary search to reduce dimensions.
                - "binary-search-values": Binary search to reduce values.
            max_iters: Maximum iterations for minimization.

        Returns:
            MinimizeResult with minimized seed, shape, and result.

        Example:
            >>> result = client.minimize(12345)
            >>> print(f"Minimized shape: {result.minimized_shape}")
        """
        # Convert strategy string to protocol enum
        strategy_map = {
            "binary-search-dims": "BinarySearchDims",
            "binary-search-values": "BinarySearchValues",
        }
        proto_strategy = strategy_map.get(strategy, "BinarySearchDims")

        request = {
            "type": "Minimize",
            "seed": seed,
            "strategy": proto_strategy,
            "max_iters": max_iters,
        }
        response = self._send_request(request)

        if response.get("type") == "MinimizeResult":
            return MinimizeResult.from_dict(response)
        elif response.get("type") == "Error":
            raise ClientError(response.get("message", "Unknown error"))
        else:
            raise ClientError(f"Unexpected response: {response}")

    def list_failures(self, limit: int = 20) -> List[ValidationResult]:
        """List stored failures.

        Args:
            limit: Maximum number of failures to return.

        Returns:
            List of ValidationResult objects for failed tests.

        Example:
            >>> failures = client.list_failures(limit=10)
            >>> for f in failures:
            ...     print(f"Seed {f.seed}: {f.op_name}")
        """
        request = {"type": "ListFailures", "limit": limit}
        response = self._send_request(request)

        if response.get("type") == "Results":
            return [ValidationResult.from_dict(r) for r in response.get("results", [])]
        elif response.get("type") == "Error":
            raise ClientError(response.get("message", "Unknown error"))
        else:
            raise ClientError(f"Unexpected response: {response}")

    @staticmethod
    def _encode_tensor(arr: np.ndarray) -> Dict[str, Any]:
        """Encode a numpy array for transmission."""
        return {
            "shape": list(arr.shape),
            "strides": list(arr.strides),
            "dtype": Client._numpy_dtype_to_protocol(arr.dtype),
            "data": base64.b64encode(arr.tobytes()).decode("utf-8"),
        }

    @staticmethod
    def _numpy_dtype_to_protocol(dtype: np.dtype) -> str:
        """Convert a numpy dtype to the protocol dtype string.

        Maps numpy dtypes to the Rust DType enum variant names
        (lowercase, matching serde serialization).
        """
        mapping = {
            "float16": "float16",
            "float32": "float32",
            "float64": "float64",
            "int8": "int8",
            "int16": "int16",
            "int32": "int32",
            "int64": "int64",
            "uint8": "uint8",
            "uint16": "uint16",
            "uint32": "uint32",
            "uint64": "uint64",
            "bool": "bool",
        }
        name = str(dtype)
        if name in mapping:
            return mapping[name]
        if "bfloat16" in name or "bf16" in name:
            return "bfloat16"
        return name

    @staticmethod
    def _protocol_dtype_to_numpy(dtype_str: str) -> np.dtype:
        """Convert a protocol dtype string back to a numpy dtype.

        Handles bfloat16 by falling back to float16 as proxy,
        since numpy has no native bfloat16.
        """
        mapping = {
            "float16": np.float16,
            "bfloat16": np.float16,
            "float32": np.float32,
            "float64": np.float64,
            "int8": np.int8,
            "int16": np.int16,
            "int32": np.int32,
            "int64": np.int64,
            "uint8": np.uint8,
            "uint16": np.uint16,
            "uint32": np.uint32,
            "uint64": np.uint64,
            "bool": np.bool_,
        }
        return np.dtype(mapping.get(dtype_str, np.float32))

    @staticmethod
    def _decode_tensor(data: Dict[str, Any]) -> np.ndarray:
        """Decode a numpy array from transmission format."""
        shape = tuple(data["shape"])
        dtype = Client._protocol_dtype_to_numpy(data.get("dtype", "float32"))
        raw = base64.b64decode(data["data"])
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()

    # =========================================================================
    # Execution Modes: Client-Side Fuzzing
    # =========================================================================

    def get_test_case(self, op_name: str, seed: Optional[int] = None) -> Dict[str, Any]:
        """Get a single test case from the daemon for client-side execution.

        The daemon generates random inputs. The client runs the actual op
        on GPU and submits the output for validation via submit_output().

        Args:
            op_name: Name of the op (must be registered in gpuemu.toml).
            seed: Master seed for reproducibility. Auto-generated if None.

        Returns:
            Dict with 'seed', 'inputs' (dict of name->ndarray), 'shape', 'dtype', 'layout'.
        """
        if seed is None:
            seed = int(time.time_ns()) & 0xFFFFFFFFFFFFFFFF

        fuzz_config = {
            "seed": seed,
            "shape_options": {
                "batch_sizes": [1, 2, 4, 8],
                "seq_lengths": [64, 128, 256],
                "hidden_dims": [256, 512],
                "edge_cases": [[1], [1, 1]],
            },
            "dtypes": ["float32", "float16"],
            "layouts": ["Contiguous", "Strided"],
        }

        request = {
            "type": "GetTestCase",
            "op_name": op_name,
            "fuzz_config": fuzz_config,
        }

        response = self._send_request(request)

        if response.get("type") == "TestCase":
            inputs = {
                name: self._decode_tensor(tensor)
                for name, tensor in response.get("inputs", {}).items()
            }
            return {
                "seed": response.get("seed", 0),
                "inputs": inputs,
                "shape": response.get("shape", []),
                "dtype": response.get("dtype", "float32"),
                "layout": response.get("layout", "contiguous"),
            }
        elif response.get("type") == "Error":
            raise ClientError(response.get("message", "Unknown error"))
        else:
            raise ClientError(f"Unexpected response: {response}")

    def get_test_batch(
        self, op_name: str, count: int = 10, seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get a batch of test cases from the daemon.

        Args:
            op_name: Name of the op.
            count: Number of test cases to generate.
            seed: Master seed. Auto-generated if None.

        Returns:
            List of test case dicts (same format as get_test_case).
        """
        if seed is None:
            seed = int(time.time_ns()) & 0xFFFFFFFFFFFFFFFF

        fuzz_config = {
            "seed": seed,
            "shape_options": {
                "batch_sizes": [1, 2, 4, 8],
                "seq_lengths": [64, 128, 256],
                "hidden_dims": [256, 512],
                "edge_cases": [[1], [1, 1]],
            },
            "dtypes": ["float32", "float16"],
            "layouts": ["Contiguous", "Strided"],
        }

        request = {
            "type": "GetTestBatch",
            "op_name": op_name,
            "fuzz_config": fuzz_config,
            "count": count,
        }

        response = self._send_request(request)

        if response.get("type") == "TestBatch":
            cases = []
            for case_data in response.get("cases", []):
                inputs = {
                    name: self._decode_tensor(tensor)
                    for name, tensor in case_data.get("inputs", {}).items()
                }
                cases.append(
                    {
                        "seed": case_data.get("seed", 0),
                        "inputs": inputs,
                        "shape": case_data.get("shape", []),
                        "dtype": case_data.get("dtype", "float32"),
                        "layout": case_data.get("layout", "contiguous"),
                    }
                )
            return cases
        elif response.get("type") == "Error":
            raise ClientError(response.get("message", "Unknown error"))
        else:
            raise ClientError(f"Unexpected response: {response}")

    def submit_output(
        self,
        op_name: str,
        inputs: Dict[str, np.ndarray],
        output: np.ndarray,
        seed: int,
        **kwargs,
    ) -> ValidationResult:
        """Submit an op output for validation against the reference.

        This is the core method for client-side and daemon-orchestrated
        execution modes. The client runs the actual GPU op and submits
        the result here for comparison.

        Args:
            op_name: Name of the op (must be registered in gpuemu.toml).
            inputs: Input tensors as numpy arrays.
            output: Output tensor from the op under test.
            seed: Seed of the test case (from get_test_case or get_test_batch).
            **kwargs: Additional kwargs for the reference script.

        Returns:
            ValidationResult with pass/fail status and details.
        """
        encoded_inputs = {
            name: self._encode_tensor(arr) for name, arr in inputs.items()
        }
        encoded_output = self._encode_tensor(output)

        request = {
            "type": "SubmitOutput",
            "op_name": op_name,
            "inputs": encoded_inputs,
            "output": encoded_output,
            "seed": seed,
            "kwargs": {k: str(v) for k, v in kwargs.items()},
        }

        response = self._send_request(request)

        if response.get("type") == "SubmitResult":
            return ValidationResult.from_dict(response.get("result", {}))
        elif response.get("type") == "Error":
            raise ClientError(response.get("message", "Unknown error"))
        else:
            raise ClientError(f"Unexpected response: {response}")

    def fuzz_op_client_side(
        self,
        op_name: str,
        run_op: "Callable[[Dict[str, np.ndarray]], np.ndarray]",
        iterations: int = 100,
        seed: Optional[int] = None,
        fail_fast: bool = False,
    ) -> FuzzResults:
        """Fuzz an op using client-side execution (THE RECOMMENDED DROP-IN PATH).

        This method generates random inputs via the daemon, runs the provided
        ``run_op`` callable on the client (which has GPU access), and validates
        the output against the reference script. This is how GPU developers
        should use gpuemu for fuzzing.

        Args:
            op_name: Name of the op (must be registered in gpuemu.toml).
            run_op: A callable that takes a dict of input tensors and returns
                     the output tensor. This is where you call your GPU kernel.
            iterations: Number of test cases to try.
            seed: Master seed. Auto-generated if None.
            fail_fast: Stop on first failure.

        Returns:
            FuzzResults with pass/fail counts and list of failures.

        Example:
            >>> client = Client()
            >>> results = client.fuzz_op_client_side(
            ...     "my_flash_attention",
            ...     run_op=lambda inputs: my_flash_attn(inputs["q"], inputs["k"], inputs["v"]),
            ...     iterations=50,
            ... )
            >>> print(f"Passed: {results.passed}/{results.total}")
        """
        if seed is None:
            seed = int(time.time_ns()) & 0xFFFFFFFFFFFFFFFF

        fuzz_config = {
            "seed": seed,
            "shape_options": {
                "batch_sizes": [1, 2, 4, 8, 16, 32],
                "seq_lengths": [64, 128, 256, 512, 1024],
                "hidden_dims": [256, 512, 768, 1024],
                "edge_cases": [[1], [1, 1], [1, 1, 1]],
            },
            "dtypes": ["float32", "float16"],
            "layouts": ["Contiguous", "Strided", "Transposed"],
        }

        cases = self.get_test_batch(op_name, count=iterations, seed=seed)
        total = 0
        passed = 0
        failed = 0
        failures = []

        for case in cases:
            total += 1
            try:
                output = run_op(case["inputs"])
                result = self.submit_output(
                    op_name, case["inputs"], output, case["seed"]
                )
                if result.passed:
                    passed += 1
                else:
                    failed += 1
                    failures.append(result)
                    if fail_fast:
                        break
            except Exception as e:
                failed += 1
                failures.append(
                    ValidationResult(
                        passed=False,
                        seed=case["seed"],
                        op_name=op_name,
                        max_diff=float("inf"),
                        max_rel_diff=float("inf"),
                        failures=[{"kind": "ExecutionError", "message": str(e)}],
                        timestamp=int(time.time()),
                        duration_ms=0,
                    )
                )
                if fail_fast:
                    break

        return FuzzResults(
            seed=seed,
            total=total,
            passed=passed,
            failed=failed,
            failures=failures,
        )
