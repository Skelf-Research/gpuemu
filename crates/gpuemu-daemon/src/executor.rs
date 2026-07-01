//! Reference script executor for running Python reference implementations.

use anyhow::{Context, Result};
use gpuemu_common::types::{DType, TensorData};
use std::collections::HashMap;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;
use tracing::{debug, error, info};

/// Configuration for the executor.
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Path to Python interpreter.
    pub python_path: String,
    /// Timeout for reference scripts.
    pub timeout: Duration,
    /// Working directory for scripts.
    pub working_dir: Option<std::path::PathBuf>,
    /// Promote float inputs to fp64 before running the *reference* script, so
    /// the oracle is a high-precision ground truth rather than computing in the
    /// kernel's (possibly low) input dtype. Does not affect op-script
    /// (kernel-emulation) runs, which must see their real dtype.
    pub oracle_fp64: bool,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            python_path: "python3".to_string(),
            timeout: Duration::from_secs(60),
            working_dir: None,
            oracle_fp64: true,
        }
    }
}

/// Promote a float tensor's raw little-endian bytes to fp64 bytes.
///
/// Returns `None` for non-float dtypes (left untouched). f16/bf16 are decoded
/// via the same conversions the validator uses, so representation is
/// consistent across the pipeline.
fn promote_float_bytes_to_f64(dtype: DType, data: &[u8]) -> Option<Vec<u8>> {
    let vals: Vec<f64> = match dtype {
        DType::Float64 => return Some(data.to_vec()),
        DType::Float32 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect(),
        DType::Float16 => data
            .chunks_exact(2)
            .map(|c| crate::validator::f16_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64)
            .collect(),
        DType::BFloat16 => data
            .chunks_exact(2)
            .map(|c| crate::validator::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64)
            .collect(),
        _ => return None,
    };
    let mut out = Vec::with_capacity(vals.len() * 8);
    for v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Some(out)
}

/// Executor for running reference implementation scripts.
pub struct Executor {
    config: ExecutorConfig,
}

impl Executor {
    /// Create a new executor with the given configuration.
    pub fn new(config: ExecutorConfig) -> Self {
        Self { config }
    }

    /// Run a reference script with the given inputs.
    ///
    /// The script is expected to read pickled inputs from stdin and write
    /// pickled output to stdout.
    pub async fn run_reference(
        &self,
        script_path: &Path,
        inputs: &HashMap<String, TensorData>,
        kwargs: &HashMap<String, String>,
        is_reference: bool,
    ) -> Result<TensorData> {
        info!("Running reference script: {:?}", script_path);
        debug!("Inputs: {} tensors", inputs.len());

        // Promote to fp64 only for the reference oracle, never for op-script
        // (kernel-emulation) runs which must observe their real input dtype.
        let promote = is_reference && self.config.oracle_fp64;
        let input_data = self.serialize_inputs(inputs, kwargs, promote)?;

        let mut cmd = tokio::process::Command::new(&self.config.python_path);
        cmd.arg(script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Some(ref wd) = self.config.working_dir {
            cmd.current_dir(wd);
        }

        let mut child = cmd.spawn().with_context(|| {
            format!(
                "Failed to spawn Python process: {}",
                self.config.python_path
            )
        })?;

        // Write input to stdin asynchronously
        if let Some(ref mut stdin) = child.stdin {
            use tokio::io::AsyncWriteExt;
            stdin
                .write_all(&input_data)
                .await
                .context("Failed to write to script stdin")?;
        }
        drop(child.stdin.take());

        // Wait for completion with timeout, killing the child if it exceeds the deadline
        let timeout = self.config.timeout;
        let result = tokio::time::timeout(timeout, async {
            let out = child.wait_with_output().await;
            out
        })
        .await;

        let output = match result {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => {
                anyhow::bail!("Failed to wait for process: {}", e);
            }
            Err(_) => {
                anyhow::bail!(
                    "Reference script {:?} timed out after {:?}",
                    script_path,
                    timeout
                );
            }
        };

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            error!("Reference script {:?} failed: {}", script_path, stderr);
            anyhow::bail!(
                "Reference script {:?} failed: {}",
                script_path,
                stderr.trim()
            );
        }

        // Parse output
        self.deserialize_output(&output.stdout)
    }

    /// Serialize inputs for the Python script.
    fn serialize_inputs(
        &self,
        inputs: &HashMap<String, TensorData>,
        kwargs: &HashMap<String, String>,
        promote: bool,
    ) -> Result<Vec<u8>> {
        // For MVP, we use a simple JSON protocol:
        // {
        //   "inputs": { "name": { "shape": [...], "dtype": "float32", "data": [base64] } },
        //   "kwargs": { "key": "value" }
        // }
        use base64::Engine;
        use serde_json::json;

        let mut input_map = serde_json::Map::new();
        for (name, tensor) in inputs {
            // Optionally promote float inputs to fp64 for the oracle.
            let promoted = if promote {
                promote_float_bytes_to_f64(tensor.dtype, &tensor.data)
            } else {
                None
            };
            let (dtype_str, data_ref): (&str, &[u8]) = match &promoted {
                Some(bytes) => ("float64", bytes.as_slice()),
                None => {
                    let s = match tensor.dtype {
                        DType::Float32 => "float32",
                        DType::Float16 => "float16",
                        DType::BFloat16 => "bfloat16",
                        DType::Float64 => "float64",
                        DType::Int32 => "int32",
                        DType::Int64 => "int64",
                        _ => "float32",
                    };
                    (s, tensor.data.as_slice())
                }
            };

            input_map.insert(
                name.clone(),
                json!({
                    "shape": tensor.shape,
                    "strides": tensor.strides,
                    "dtype": dtype_str,
                    "data": base64::engine::general_purpose::STANDARD.encode(data_ref)
                }),
            );
        }

        let payload = json!({
            "inputs": input_map,
            "kwargs": kwargs
        });

        Ok(serde_json::to_vec(&payload)?)
    }

    /// Deserialize output from the Python script.
    fn deserialize_output(&self, stdout: &[u8]) -> Result<TensorData> {
        use base64::Engine;

        let output: serde_json::Value =
            serde_json::from_slice(stdout).context("Failed to parse script output as JSON")?;

        let shape: Vec<usize> = output["shape"]
            .as_array()
            .context("Missing shape in output")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();

        let dtype_str = output["dtype"].as_str().unwrap_or("float32");
        let dtype = match dtype_str {
            "float16" => DType::Float16,
            "bfloat16" => DType::BFloat16,
            "float32" => DType::Float32,
            "float64" => DType::Float64,
            "int32" => DType::Int32,
            "int64" => DType::Int64,
            _ => DType::Float32,
        };

        let data_b64 = output["data"].as_str().context("Missing data in output")?;
        let data = base64::engine::general_purpose::STANDARD
            .decode(data_b64)
            .context("Failed to decode base64 data")?;

        Ok(TensorData::new(shape, dtype, data))
    }
}

/// Generate a reference script template.
#[allow(dead_code)]
pub fn generate_reference_template() -> String {
    r#"#!/usr/bin/env python3
"""Reference implementation template for gpuemu validation."""

import sys
import json
import base64
import numpy as np


def decode_tensor(tensor_dict):
    """Decode a tensor from the gpuemu protocol."""
    shape = tensor_dict["shape"]
    dtype = np.dtype(tensor_dict["dtype"])
    data = base64.b64decode(tensor_dict["data"])
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def encode_tensor(arr):
    """Encode a numpy array for gpuemu."""
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data": base64.b64encode(arr.tobytes()).decode("utf-8")
    }


def reference(**inputs):
    """
    Reference implementation.

    Args:
        **inputs: Named input tensors as numpy arrays.

    Returns:
        Output tensor as numpy array.
    """
    # TODO: Implement your reference logic here
    # Example:
    # return inputs["a"] + inputs["b"]
    raise NotImplementedError("Implement your reference logic")


def main():
    # Read input from stdin
    input_json = json.load(sys.stdin)

    # Decode input tensors
    inputs = {
        name: decode_tensor(tensor)
        for name, tensor in input_json["inputs"].items()
    }

    # Get kwargs
    kwargs = input_json.get("kwargs", {})

    # Run reference
    result = reference(**inputs, **kwargs)

    # Encode and output
    output = encode_tensor(result)
    json.dump(output, sys.stdout)


if __name__ == "__main__":
    main()
"#
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_inputs() {
        let executor = Executor::new(ExecutorConfig::default());
        let mut inputs = HashMap::new();
        inputs.insert(
            "a".to_string(),
            TensorData::new(vec![2, 3], DType::Float32, vec![0; 24]),
        );

        let serialized = executor
            .serialize_inputs(&inputs, &HashMap::new(), false)
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&serialized).unwrap();

        assert!(json["inputs"]["a"]["shape"].is_array());
        // No promotion: dtype is preserved.
        assert_eq!(json["inputs"]["a"]["dtype"], "float32");
    }

    #[test]
    fn test_serialize_inputs_promotes_to_f64() {
        let executor = Executor::new(ExecutorConfig::default());
        let mut inputs = HashMap::new();
        // One f32 element = 1.5 (0x3FC00000 little-endian).
        let bytes = 1.5f32.to_le_bytes().to_vec();
        inputs.insert(
            "a".to_string(),
            TensorData::new(vec![1], DType::Float32, bytes),
        );

        let serialized = executor
            .serialize_inputs(&inputs, &HashMap::new(), true)
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&serialized).unwrap();

        // Promoted: dtype becomes float64 and the value round-trips losslessly.
        assert_eq!(json["inputs"]["a"]["dtype"], "float64");
        use base64::Engine;
        let data = base64::engine::general_purpose::STANDARD
            .decode(json["inputs"]["a"]["data"].as_str().unwrap())
            .unwrap();
        assert_eq!(data.len(), 8);
        let v = f64::from_le_bytes(data.try_into().unwrap());
        assert_eq!(v, 1.5);
    }

    #[test]
    fn test_promote_float_bytes_f16() {
        // f16 1.0 = 0x3C00.
        let bytes = 0x3C00u16.to_le_bytes().to_vec();
        let out = promote_float_bytes_to_f64(DType::Float16, &bytes).unwrap();
        assert_eq!(out.len(), 8);
        assert_eq!(f64::from_le_bytes(out.try_into().unwrap()), 1.0);
    }
}
