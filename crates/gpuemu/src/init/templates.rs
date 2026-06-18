//! Embedded templates for project scaffolding.

/// Template for gpuemu.toml configuration file.
pub const GPUEMU_TOML: &str = r#"# gpuemu configuration file
# See https://gpuemu.dev/docs/configuration for full reference

[project]
name = "{{project_name}}"
framework = "{{framework}}"
version = "0.1.0"

[validation]
dtypes = ["float32", "float16"]
check_nan = true
check_inf = true
# seed = 12345  # Uncomment for deterministic testing

[validation.tolerances]
float32 = 1e-5
float16 = 1e-3
bfloat16 = 1e-3

 # Example op configuration - uncomment and modify for your ops
  # [[ops]]
  # name = "my_custom_op"
  # module = "my_module.custom_op"
  # reference = "scripts/ref_my_custom_op.py"
  # input_names = ["x"]
 # execution_mode = "client_side"   # "client_side" | "daemon_orchestrated" | "script_based"
  # op_script = "scripts/run_my_custom_op.py"   # only needed for script_based mode
  #
  # [ops.tolerances]
  # float32 = 1e-5
  # float16 = 1e-3
  #
  # [ops.invariants]
  # non_negative = false
  # shape_preserved = true
  # no_nan = true
  # no_inf = true

# Example kernel configuration - uncomment and modify for your kernels
# [[kernels]]
# name = "my_kernel"
# source = "kernels/my_kernel.cu"
# reference = "scripts/ref_my_kernel.py"
#
# [kernels.tolerances]
# float32 = 1e-5
#
# [kernels.artifact_checks]
# max_registers = 64
# max_spills = 0
# max_local_memory = 0
# required_patterns = []
# forbidden_patterns = []

[ci]
quick_dtypes = ["float32"]
thorough_timeout = 3600
parallel_jobs = 4

[policies]
fail_on_regression = true
warn_threshold = 0.1
"#;

/// Template for PyTorch reference script.
pub const PYTORCH_REFERENCE: &str = r#"#!/usr/bin/env python3
"""Reference implementation for {{op_name}} validation.

This script is called by the gpuemu daemon to compute expected outputs.
Inputs are received via JSON+base64 on stdin, outputs are written via JSON+base64 on stdout.
"""
import sys
import json
import base64
import numpy as np
import torch


def decode_tensor(tensor_dict):
    """Decode a tensor from the gpuemu protocol."""
    shape = tensor_dict["shape"]
    dtype = np.dtype(tensor_dict["dtype"])
    data = base64.b64decode(tensor_dict["data"])
    return torch.from_numpy(np.frombuffer(data, dtype=dtype).reshape(shape).copy())


def encode_tensor(t):
    """Encode a tensor for gpuemu output."""
    arr = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data": base64.b64encode(arr.tobytes()).decode("utf-8"),
    }


def reference(**inputs: dict) -> torch.Tensor:
    """Compute the reference output for {{op_name}}.

    Args:
        **inputs: Dictionary of input tensors.

    Returns:
        Expected output tensor.
    """
    # TODO: Replace with your op's expected behavior
    x = inputs.get("x") or inputs.get("input")
    if x is None:
        raise ValueError("Expected 'x' or 'input' in inputs")

    # Example: simple ReLU operation
    return torch.relu(x)


if __name__ == "__main__":
    # Read input from stdin
    input_json = json.load(sys.stdin)

    # Decode input tensors
    inputs = {
        name: decode_tensor(tensor)
        for name, tensor in input_json["inputs"].items()
    }

    # Get kwargs
    kwargs = input_json.get("kwargs", {})

    # Compute reference output
    result = reference(**inputs, **kwargs)

    # Encode and output
    output = encode_tensor(result)
    json.dump(output, sys.stdout)
"#;

/// Template for JAX reference script.
pub const JAX_REFERENCE: &str = r#"#!/usr/bin/env python3
"""Reference implementation for {{op_name}} validation.

This script is called by the gpuemu daemon to compute expected outputs.
Inputs are received via JSON+base64 on stdin, outputs are written via JSON+base64 on stdout.
"""
import sys
import json
import base64
import numpy as np
import jax.numpy as jnp


def decode_tensor(tensor_dict):
    """Decode a tensor from the gpuemu protocol."""
    shape = tensor_dict["shape"]
    dtype = np.dtype(tensor_dict["dtype"])
    data = base64.b64decode(tensor_dict["data"])
    return jnp.array(np.frombuffer(data, dtype=dtype).reshape(shape).copy())


def encode_tensor(arr):
    """Encode an array for gpuemu output."""
    np_arr = np.asarray(arr)
    return {
        "shape": list(np_arr.shape),
        "dtype": str(np_arr.dtype),
        "data": base64.b64encode(np_arr.tobytes()).decode("utf-8"),
    }


def reference(**inputs: dict) -> jnp.ndarray:
    """Compute the reference output for {{op_name}}.

    Args:
        **inputs: Dictionary of input arrays.

    Returns:
        Expected output array.
    """
    # TODO: Replace with your op's expected behavior
    x = inputs.get("x") or inputs.get("input")
    if x is None:
        raise ValueError("Expected 'x' or 'input' in inputs")

    # Example: simple ReLU operation
    return jnp.maximum(x, 0)


if __name__ == "__main__":
    # Read input from stdin
    input_json = json.load(sys.stdin)

    # Decode input tensors
    inputs = {
        name: decode_tensor(tensor)
        for name, tensor in input_json["inputs"].items()
    }

    # Get kwargs
    kwargs = input_json.get("kwargs", {})

    # Compute reference output
    result = reference(**inputs, **kwargs)

    # Encode and output
    output = encode_tensor(result)
    json.dump(output, sys.stdout)
"#;

/// Template for TensorFlow reference script.
pub const TENSORFLOW_REFERENCE: &str = r#"#!/usr/bin/env python3
"""Reference implementation for {{op_name}} validation.

This script is called by the gpuemu daemon to compute expected outputs.
Inputs are received via JSON+base64 on stdin, outputs are written via JSON+base64 on stdout.
"""
import sys
import json
import base64
import numpy as np
import tensorflow as tf


def decode_tensor(tensor_dict):
    """Decode a tensor from the gpuemu protocol."""
    shape = tensor_dict["shape"]
    dtype = np.dtype(tensor_dict["dtype"])
    data = base64.b64decode(tensor_dict["data"])
    return tf.constant(np.frombuffer(data, dtype=dtype).reshape(shape).copy())


def encode_tensor(t):
    """Encode a tensor for gpuemu output."""
    arr = t.numpy() if isinstance(t, tf.Tensor) else np.asarray(t)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data": base64.b64encode(arr.tobytes()).decode("utf-8"),
    }


def reference(**inputs: dict) -> tf.Tensor:
    """Compute the reference output for {{op_name}}.

    Args:
        **inputs: Dictionary of input tensors.

    Returns:
        Expected output tensor.
    """
    # TODO: Replace with your op's expected behavior
    x = inputs.get("x") or inputs.get("input")
    if x is None:
        raise ValueError("Expected 'x' or 'input' in inputs")

    # Example: simple ReLU operation
    return tf.nn.relu(x)


if __name__ == "__main__":
    # Read input from stdin
    input_json = json.load(sys.stdin)

    # Decode input tensors
    inputs = {
        name: decode_tensor(tensor)
        for name, tensor in input_json["inputs"].items()
    }

    # Get kwargs
    kwargs = input_json.get("kwargs", {})

    # Compute reference output
    result = reference(**inputs, **kwargs)

    # Encode and output
    output = encode_tensor(result)
    json.dump(output, sys.stdout)
"#;

/// Template for Python __init__.py in scripts directory.
pub const SCRIPTS_INIT: &str = r#"# gpuemu reference scripts
# This module contains reference implementations for ops and kernels.
"#;

/// Template for .gitignore.
pub const GITIGNORE: &str = r#"# gpuemu local state
.gpuemu/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
.eggs/
*.egg
.pytest_cache/
.mypy_cache/

# Build artifacts
*.ptx
*.cubin
*.fatbin
*.o
*.a
target/

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
"#;

/// Template for example test file.
pub const EXAMPLE_TEST: &str = r#"#!/usr/bin/env python3
"""Example tests for gpuemu validation.

Run with: pytest tests/test_ops.py
"""
import pytest
from gpuemu_py import Client, validate_op


@pytest.fixture
def client():
    """Create a gpuemu client."""
    return Client()


def test_example_op(client):
    """Test example op validation."""
    import numpy as np

    # Create test input
    x = np.random.randn(32, 128).astype(np.float32)

    # Compute expected output (reference)
    expected = np.maximum(x, 0)  # ReLU

    # Validate
    with validate_op(client, "example_op", {"x": x}, expected):
        pass  # Validation happens in context manager
"#;

/// Get reference script template based on framework.
pub fn get_reference_template(framework: &str) -> &'static str {
    match framework.to_lowercase().as_str() {
        "pytorch" | "torch" => PYTORCH_REFERENCE,
        "jax" => JAX_REFERENCE,
        "tensorflow" | "tf" => TENSORFLOW_REFERENCE,
        _ => PYTORCH_REFERENCE, // Default to PyTorch
    }
}

/// Render template with substitutions.
/// Uses a specific pattern to avoid partial matches by replacing
/// from longest key to shortest.
pub fn render_template(template: &str, substitutions: &[(&str, &str)]) -> String {
    let mut result = template.to_string();
    // Sort by key length descending to avoid partial matches
    let mut sorted_subs: Vec<_> = substitutions.to_vec();
    sorted_subs.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    for (key, value) in sorted_subs {
        result = result.replace(&format!("{{{{{}}}}}", key), value);
    }
    result
}
