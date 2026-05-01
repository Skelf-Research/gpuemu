This tool catches a very *specific* class of bugs — the ones that are **expensive, silent, and usually found too late**.
Below is a practical taxonomy, with examples pulled directly from real CUDA / PyTorch / JAX failure modes.

I’ll frame each bug as:

> **What the dev intended → What actually happens → Why current tooling misses it → How your product catches it**

---

## 1. Indexing & layout bugs (the #1 category)

### Example

You write a CUDA kernel assuming **contiguous NCHW**, but PyTorch passes a **strided view**.

**Intent**

```cpp
int idx = n*C*H*W + c*H*W + h*W + w;
```

**Reality**

* Tensor is a view (`transpose`, `select`, `as_strided`)
* Strides ≠ dense
* Kernel reads *wrong memory* but doesn’t crash

**Why it slips through**

* GPU happily reads memory
* Output “looks reasonable”
* Tests pass on small inputs

**How your tool catches it**

* CPU mirror runs kernel logic against:

  * contiguous
  * transposed
  * sliced
  * weirdly-strided tensors
* Diff against **PyTorch CPU reference**
* Fails immediately with a minimal repro

> This alone saves weeks per year for kernel teams.

---

## 2. Boundary condition & tail bugs (off-by-one, partial tiles)

### Example

Your kernel assumes `T % BLOCK == 0`.

**Intent**

```cpp
for (int i = 0; i < BLOCK; i++)
  acc += x[base + i];
```

**Reality**

* Sequence length = 513
* Last block reads out-of-bounds or skips work

**Why it slips through**

* Most tests use “round” sizes (512, 1024)
* GPU OOB reads may not segfault
* Numerical error is subtle

**How your tool catches it**

* Shape fuzzing generates:

  * prime sizes
  * odd tails
  * minimal sizes (1, 2, BLOCK-1)
* CPU execution runs under ASan → **hard crash**
* You get a stack trace, not NaNs 3 weeks later

---

## 3. Accumulation precision bugs (fp16/fp32 mistakes)

### Example

You accidentally accumulate fp16 into fp16.

**Intent**

```cpp
float acc = 0.f;
acc += __half2float(x[i]);
```

**Reality**

```cpp
__half acc;
acc += x[i];   // silent precision loss
```

**Why it slips through**

* Results “look fine” for small inputs
* Error grows with sequence length
* Only shows up in training instability

**How your tool catches it**

* Contract says: `accumulate_fp32`
* CPU mirror:

  * runs high-precision reference
  * checks error bounds as shape grows
* Flags **numerical drift**, not just mismatches

---

## 4. Reduction associativity bugs (order-dependent math)

### Example

Parallel reduction changes summation order.

**Intent**

```text
sum(x) should be deterministic
```

**Reality**

* Floating point addition is non-associative
* Block-level reduction changes order
* Different shapes → different results

**Why it slips through**

* GPU results vary slightly run-to-run
* CI tolerances are too loose or too strict
* Hard to reason about

**How your tool catches it**

* CPU mirror runs:

  * multiple reduction orders
  * deterministic reference
* Enforces **error envelopes**, not bitwise equality
* Detects *unstable reductions*

This is *huge* for attention, softmax, layernorm.

---

## 5. Silent NaN / Inf propagation bugs

### Example

You forget to clamp or mask invalid values.

**Intent**

```cpp
y = exp(x - max);
```

**Reality**

* `x = -inf`
* `exp(-inf)` → 0
* Later divide by zero → NaN

**Why it slips through**

* GPU doesn’t crash
* NaNs propagate silently
* Loss explodes many steps later

**How your tool catches it**

* Invariants:

  * `no_nan`
  * `no_inf`
* CPU mirror:

  * injects adversarial inputs
  * checks propagation rules
* Fails *at kernel level*, not at model divergence

---

## 6. Autograd forward/backward mismatches

### Example

Forward and backward kernels disagree on layout or scaling.

**Intent**

```text
grad(x) matches d/dx forward(x)
```

**Reality**

* Forward uses fp16 accumulation
* Backward assumes fp32
* Gradients are subtly wrong

**Why it slips through**

* Training “sort of works”
* Converges slower
* No clear crash

**How your tool catches it**

* CPU mirror:

  * runs numerical gradient check
  * compares backward kernel vs finite differences
* Catches math inconsistencies *before training*

This is massive for custom ops.

---

## 7. Implicit assumptions about launch configuration

### Example

Kernel assumes:

* `blockDim.x == warp_size`
* `gridDim.x == batch`

**Reality**

* Launch config changes
* Kernel silently misbehaves

**Why it slips through**

* Kernel compiles
* GPU executes
* Assumption violated only in some models

**How your tool catches it**

* CPU mirror decouples:

  * logical indexing
  * physical launch config
* Tests multiple virtual launch shapes
* Flags hard-coded assumptions

---

## 8. Performance regressions *without running on GPU*

This is subtle but very real.

### Example

A refactor increases register pressure from 64 → 128.

**Reality**

* Occupancy halves
* Kernel is now slower
* CI with GPU not always available

**How your tool catches it**

* PTX/SASS lint:

  * register count
  * spills
  * shared memory usage
* Diff artifacts across commits
* “This PR increases register pressure by 2×”

No GPU required.

---

## 9. Triton / XLA “silent de-optimizations”

For **Triton** and **JAX** users:

* A small code change causes:

  * vectorization loss
  * scalar loads
  * worse tiling

**Why it slips through**

* Kernel still “works”
* Perf drops 30%
* Hard to spot in review

**How your tool catches it**

* Analyze generated IR
* Enforce expected patterns
* Fail or warn on de-opt

---

## The meta-point (this is important)

You are not catching:

* “CUDA crashes”
* “Compiler errors”

You are catching:

> **Silent semantic drift between intent and execution**

These are the bugs that:

* cost the most
* ship to production
* destroy confidence in custom kernels

---

## Why this category is perfect for a product

| Bug Type                     | GPU Needed? | Found Early Today? |
| ---------------------------- | ----------- | ------------------ |
| Indexing/layout              | ❌           | ❌                  |
| Precision drift              | ❌           | ❌                  |
| Reduction stability          | ❌           | ❌                  |
| NaN propagation              | ❌           | ❌                  |
| Autograd mismatch            | ❌           | ❌                  |
| Perf regression (structural) | ❌           | ❌                  |

Your tool is the **first systematic solution** to this layer.

---

If you want, next I can:

* Map these bugs to **real-world incident stories**
* Turn this into a **landing page narrative**
* Write the **Kernel Contract schema** that enables all of this

Just tell me where to go next.

