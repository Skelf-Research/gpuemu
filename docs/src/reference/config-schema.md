# Configuration Schema

Complete reference for gpuemu.toml fields.

See [Configuration Guide](../getting-started/configuration.md) for examples and usage.

## Schema

```
gpuemu.toml
├── [project]
│   ├── name: string
│   ├── framework: "pytorch" | "jax" | "tensorflow"
│   └── version: string
├── [validation]
│   ├── dtypes: string[]
│   ├── check_nan: bool
│   ├── check_inf: bool
│   ├── seed?: number
│   └── [tolerances]
│       ├── float32: number
│       ├── float16: number
│       └── bfloat16: number
├── [[ops]]
│   ├── name: string
│   ├── module: string
│   ├── reference: string
│   ├── [tolerances]
│   │   └── <dtype>: number
│   └── [invariants]
│       ├── no_nan: bool
│       ├── no_inf: bool
│       ├── non_negative: bool
│       └── shape_preserved: bool
├── [[kernels]]
│   ├── name: string
│   ├── source: string
│   ├── reference: string
│   ├── [tolerances]
│   └── [artifact_checks]
│       ├── max_registers: number
│       ├── max_spills: number
│       ├── max_local_memory: number
│       ├── required_patterns: string[]
│       └── forbidden_patterns: string[]
├── [ci]
│   ├── quick_dtypes: string[]
│   ├── thorough_timeout: number
│   └── parallel_jobs: number
└── [policies]
    ├── fail_on_regression: bool
    └── warn_threshold: number
```
