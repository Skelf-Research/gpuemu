# CLI Reference

## Global Options

```
gpuemu [OPTIONS] <COMMAND>

Options:
  -v, --verbose  Enable verbose output
  -h, --help     Print help
  -V, --version  Print version
```

## Commands

### daemon

Manage the validation daemon.

```bash
# Start daemon in background
gpuemu daemon start --background

# Start daemon in foreground
gpuemu daemon start

# Stop daemon
gpuemu daemon stop

# Check daemon status
gpuemu daemon status

# View daemon logs
gpuemu daemon logs --lines 100
```

### init

Initialize a new gpuemu project.

```bash
gpuemu init [OPTIONS]

Options:
  -n, --name <NAME>          Project name [default: my-project]
  -f, --framework <FRAMEWORK> Target framework (pytorch, jax, tensorflow)
      --with-examples        Include example ops and tests
      --ci <PLATFORM>        Set up CI (github, gitlab)
  -t, --target-dir <DIR>     Target directory [default: .]
```

### test

Run validation tests.

```bash
gpuemu test [OPTIONS]

Options:
      --quick     Run quick validation (subset of shapes/dtypes)
      --thorough  Run thorough validation
      --seed <N>  Use specific seed
```

### fuzz

Fuzz test ops with random inputs.

```bash
gpuemu fuzz [OPTIONS]

Options:
  -o, --op <NAME>           Op name (omit for all ops)
  -i, --iterations <N>      Number of iterations [default: 100]
      --seed <N>            Master seed for reproducibility
      --fail-fast           Stop on first failure
```

### reproduce

Reproduce a failing test case.

```bash
gpuemu reproduce <SEED> [OPTIONS]

Options:
  -v, --verbose  Show detailed output including input values
```

### minimize

Minimize a failing test case to smallest reproducer.

```bash
gpuemu minimize <SEED> [OPTIONS]

Options:
      --strategy <STRATEGY>  binary-search-dims, binary-search-values
      --max-iters <N>        Maximum iterations [default: 100]
```

### failures

List stored failures.

```bash
gpuemu failures [OPTIONS]

Options:
  -l, --limit <N>  Number of failures to show [default: 20]
```

### lint

Lint kernel artifacts against policy rules.

```bash
gpuemu lint [OPTIONS]

Options:
  -k, --kernel <NAME>   Kernel name
  -p, --ptx <FILE>      Path to PTX file
      --format <FMT>    Output format (text, json) [default: text]
```

### baseline

Store current artifacts as a baseline.

```bash
gpuemu baseline <TAG>
```

### diff

Compare current artifacts against a baseline.

```bash
gpuemu diff [OPTIONS]

Options:
      --baseline <TAG>       Baseline tag to compare against
      --fail-on-regression   Exit with code 1 on regression
      --format <FMT>         Output format (text, json)
```

### ci

Run CI validation suite.

```bash
gpuemu ci [OPTIONS]

Options:
      --quick            Run quick validation only
      --baseline <TAG>   Compare artifacts against baseline
      --parallel <N>     Number of parallel jobs [default: 4]
      --format <FMT>     Output format (text, json, junit)
  -o, --output <FILE>    Output file (stdout if not specified)
```

### debug

Interactive debugging mode.

```bash
gpuemu debug [OPTIONS]

Options:
      --seed <N>   Start with specific seed
      --repl       Use REPL mode (default)
      --op <NAME>  Filter by op name
```

#### Debug REPL Commands

| Command | Description |
|---------|-------------|
| `list [limit]` | List recent failures |
| `show <seed>` | Show failure details |
| `reproduce <seed>` | Re-run validation |
| `minimize <seed>` | Minimize failing case |
| `export <seed>` | Export reproducer script |
| `tensor <name>` | Inspect tensor values |
| `refresh` | Reload failures |
| `status` | Check daemon status |
| `clear` | Clear screen |
| `help` | Show help |
| `quit` | Exit |

### status

Check daemon status.

```bash
gpuemu status
```

### version

Show version information.

```bash
gpuemu version
```
