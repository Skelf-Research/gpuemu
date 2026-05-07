# CI Integration Tutorial

Setting up gpuemu in your CI pipeline.

## GitHub Actions

```yaml
name: GPU Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Install gpuemu
      run: curl -fsSL https://gpuemu.dev/install.sh | sh
    
    - name: Run validation
      run: |
        gpuemu daemon start --background
        gpuemu ci --quick --format junit --output results.xml
    
    - name: Upload results
      uses: actions/upload-artifact@v4
      with:
        name: test-results
        path: results.xml
```

## GitLab CI

```yaml
validate:
  image: python:3.11
  script:
    - curl -fsSL https://gpuemu.dev/install.sh | sh
    - gpuemu daemon start --background
    - gpuemu ci --format junit --output results.xml
  artifacts:
    reports:
      junit: results.xml
```

## Artifact Regression Testing

```bash
# Store baseline on main branch
gpuemu baseline main

# On PR, compare against baseline
gpuemu diff --baseline main --fail-on-regression
```
