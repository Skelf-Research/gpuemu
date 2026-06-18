# gpuemu

**Catch silently-wrong GPU kernels before they reach production.**

---

## The problem

The industry-standard correctness oracle for an LLM-generated GPU kernel is one line:

```python
torch.allclose(my_kernel(x), reference(x), atol=1e-5, rtol=1e-2)
```

One shape. One dtype. One seed. Every modern LLM-kernel benchmark — KernelBench,
TritonBench, GEAK, KernelBand, STARK — uses the same oracle. The kernels that pass it
are the kernels that ship to production.

That oracle is blind to bug classes that LLM-generated CUDA / Triton code routinely
contains:

- **Tail-mask leaks** in fused reductions (softmax `other=0.0` instead of `-inf`).
- **Accumulator scale bugs** in matmul (`acc=` instead of `acc+=`).
- **Missing normalisation** in attention (forgotten `1/√D`).
- **Online-softmax rescale bugs** in flash-attention (forgotten `acc *= α`).

In our measured 26-op corpus the standard oracle accepts **9 / 9** of these LLM-style
buggy kernels as correct (P1).

[**→ Read the full problem walkthrough**](why-gpuemu/the-problem.md)

---

## Why it matters for the industry

Every modern LLM training and inference stack now ships LLM-generated CUDA / Triton
kernels. A silently-wrong kernel runs at scale: a miscompiled matmul propagates through
every forward pass; a broken flash-attention degrades long-context quality without
crashing; an unmasked reduction taints metrics no one looks at. The cost is **GPU-hours
wasted on silently-broken work** and **slow, untraceable quality regressions** that
survive months of CI green builds.

Every published LLM-kernel benchmark shares the same oracle gap. The kernels they bless
are the kernels that ship.

[**→ Read the industry impact in detail**](why-gpuemu/the-industry-impact.md)

---

## What gpuemu does

gpuemu replaces "allclose on one shape" with an operator-domain-aware correctness regime
backed by four measured studies (P1–P4).

<div class="grid cards" markdown>

-   :material-target:{ .lg .middle } **Op-schema-aware fuzzing**

    ---

    Per-operator shape generators with boundary, regular, and adversarial value
    distributions.

    *Measured: 99 % bug recall under adversarial sampling (P3).*

-   :material-scale-balance:{ .lg .middle } **fp64 reference + calibrated tolerances**

    ---

    High-precision CPU reference; per-op `atol/rtol` derived from p95-of-controls.

    *Measured: 100 % illusion catch with 0 false positives across 5 GPU classes (P1);
    +23 pp recall over fixed `atol/rtol` (P2).*

-   :material-chip:{ .lg .middle } **Static PTX gating**

    ---

    Register pressure, spills, instruction count from the compiled artifact.

    *Measured: structural Δregs predicts Δperf% consistently across H100 / A100 / L40S /
    A10 / 3060 (P4).*

-   :material-replay:{ .lg .middle } **Reproducible failures**

    ---

    Bit-identical xorshift128+ in Rust and Python; exact input snapshots; byte-for-byte
    replay.

</div>

[**→ See the evidence (P1–P4)**](why-gpuemu/the-evidence.md)

---

## How gpuemu compares — and who already uses it

<div class="grid cards" markdown>

-   :material-compare:{ .lg .middle } **Compared to alternatives**

    ---

    Where gpuemu fits next to `torch.testing.assert_close`, KernelBench, NVIDIA
    Compute Sanitizer, Triton's own testing, HF Kernel Hub, and the OSS DL fuzzers.

    [:octicons-arrow-right-24: Compared to](why-gpuemu/compared-to.md)

-   :material-account-group:{ .lg .middle } **Who uses gpuemu**

    ---

    Three customer workflows — frontier-lab kernel teams, OSS-inference
    maintainers, and inference-as-a-service vendors — each with a real cited issue
    that the workflow prevents.

    [:octicons-arrow-right-24: Who uses gpuemu](who-uses-gpuemu/frontier-lab-kernel-team.md)

</div>

---

## Where to go next

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Get started in 5 minutes**

    ---

    Install the CLI, the Python client, and run your first validation.

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-bookshelf:{ .lg .middle } **Read a guide**

    ---

    For kernel authors, model developers, or custom-op integrators.

    [:octicons-arrow-right-24: Kernel Author](guides/kernel-author.md)

-   :material-flask:{ .lg .middle } **Run a tutorial**

    ---

    PyTorch, JAX, TensorFlow validation in 10 minutes each.

    [:octicons-arrow-right-24: PyTorch Tutorial](tutorials/pytorch-validation.md)

-   :material-book-open-variant:{ .lg .middle } **Architecture deep dive**

    ---

    Daemon, client, IPC protocol, sled storage, fuzzer internals.

    [:octicons-arrow-right-24: Architecture](concepts/architecture.md)

</div>

---

## Supported frameworks

=== "PyTorch"

    ```python
    from gpuemu.frameworks.pytorch import validate_pytorch

    with validate_pytorch(client, "my_op", {"x": x}) as ctx:
        ctx["output"] = my_custom_op(x)
    ```

=== "JAX"

    ```python
    from gpuemu.frameworks.jax import validate_jax

    with validate_jax(client, "my_op", {"x": x}) as ctx:
        ctx["output"] = my_custom_op(x)
    ```

=== "TensorFlow"

    ```python
    from gpuemu.frameworks.tensorflow import validate_tensorflow

    with validate_tensorflow(client, "my_op", {"x": x}) as ctx:
        ctx["output"] = my_custom_op(x)
    ```

---

## Platform support

| Platform | Status | Notes |
|---|---|---|
| **Linux** | Primary | Full workflow including artifact inspection |
| **macOS** | Core | CPU validation works fully; artifact inspection optional |
| **Windows** | Future | Not currently targeted |
