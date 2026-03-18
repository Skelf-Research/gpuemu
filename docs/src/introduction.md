# Introduction

**gpuemu** is a GPU-less validation framework for deep learning kernels and custom ops. It enables comprehensive testing of GPU operations without requiring actual GPU hardware.

## Why gpuemu?

Testing deep learning operations is challenging:

- **GPU hardware is expensive** - Not everyone has access to GPUs for CI
- **Non-deterministic behavior** - GPU operations may produce subtly different results
- **Hidden bugs** - Numerical issues often only appear with specific inputs
- **Custom ops are risky** - Incorrectly implemented ops can silently corrupt models

gpuemu solves these problems by:

1. **Running validation on CPU** - No GPU required for correctness testing
2. **Systematic fuzzing** - Generate diverse test inputs automatically
3. **Reproducible failures** - Every failure can be reproduced from its seed
4. **Artifact checking** - Validate compiled kernel properties (register usage, etc.)

## Key Features

### GPU-less Validation
Validate PyTorch, JAX, and TensorFlow operations entirely on CPU with precise numerical comparisons.

### Intelligent Fuzzing
Automatically generate test cases with:
- Random shapes (edge cases and typical sizes)
- Multiple data types (float32, float16, bfloat16)
- Various memory layouts (contiguous, strided, transposed)

### Reproducer Generation
When a failure is found, gpuemu stores all information needed to reproduce it:
- Exact seed for deterministic regeneration
- Input data snapshots
- Automatic minimization to simplest failing case

### Artifact Inspection
Analyze compiled PTX/SASS to detect:
- Excessive register usage
- Memory spills
- Forbidden instruction patterns
- Regressions between versions

### CI Integration
Drop-in CI support for:
- GitHub Actions
- GitLab CI
- JUnit XML reports

## Quick Example

```bash
# Initialize a new project
gpuemu init --name my-project --framework pytorch

# Start the validation daemon
gpuemu daemon start --background

# Run fuzz testing
gpuemu fuzz --iterations 100

# Check for failures
gpuemu failures
```

## Next Steps

- [Installation Guide](./getting-started/installation.md)
- [Quick Start Tutorial](./getting-started/quickstart.md)
- [Configuration Reference](./getting-started/configuration.md)
