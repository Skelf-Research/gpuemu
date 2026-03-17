"""Client for communicating with the gpuemu daemon."""

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import pynng
except ImportError:
    pynng = None


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationResult":
        """Create from dictionary."""
        return cls(
            passed=data.get("passed", False),
            seed=data.get("seed", 0),
            op_name=data.get("op_name", ""),
            max_diff=data.get("max_diff", 0.0),
            max_rel_diff=data.get("max_rel_diff", 0.0),
            failures=data.get("failures", []),
            timestamp=data.get("timestamp", 0),
            duration_ms=data.get("duration_ms", 0),
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
        if pynng is None:
            raise ImportError(
                "pynng is required for gpuemu-py. Install with: pip install pynng"
            )

        if socket_path is None:
            socket_path = os.path.expanduser("~/.gpuemu/gpuemu.sock")

        self.socket_path = socket_path
        self.timeout_ms = timeout_ms
        self._socket: Optional[pynng.Req0] = None

    def _ensure_connected(self) -> pynng.Req0:
        """Ensure we have a connection to the daemon."""
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

        return self._socket

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
            Dict with 'version' and 'uptime_secs'.
        """
        response = self._send_request({"type": "Ping"})

        if response.get("type") == "Pong":
            return {
                "version": response.get("version", "unknown"),
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
            return [
                ValidationResult.from_dict(r) for r in response.get("results", [])
            ]
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

    @staticmethod
    def _encode_tensor(arr: np.ndarray) -> Dict[str, Any]:
        """Encode a numpy array for transmission."""
        return {
            "shape": list(arr.shape),
            "strides": list(arr.strides),
            "dtype": str(arr.dtype),
            "data": base64.b64encode(arr.tobytes()).decode("utf-8"),
        }

    @staticmethod
    def _decode_tensor(data: Dict[str, Any]) -> np.ndarray:
        """Decode a numpy array from transmission format."""
        shape = tuple(data["shape"])
        dtype = np.dtype(data["dtype"])
        raw = base64.b64decode(data["data"])
        return np.frombuffer(raw, dtype=dtype).reshape(shape)
