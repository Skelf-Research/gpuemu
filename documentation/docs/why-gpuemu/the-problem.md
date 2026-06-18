# The Problem

**Every LLM-kernel benchmark says your kernel is "correct" — and it is wrong.**

The industry-standard correctness oracle for a GPU kernel is one line:

```python
torch.allclose(my_kernel(x), reference(x), atol=1e-5, rtol=1e-2)
```

One shape. One dtype. One seed. This is the oracle inside **KernelBench**,
**TritonBench-revised**, **GEAK**, **KernelBand**, and **STARK** — every published LLM-kernel
benchmark. Kernels that pass it are reported as correct, and those numbers go on the
arXiv leaderboard.

That oracle has measurable, reproducible blind spots that LLM-generated CUDA and Triton
kernels routinely fall into. This page walks through one of them concretely; the same
pattern generalises across an entire bug taxonomy.

---

## A walkthrough: the tail-mask leak

Here is a softmax kernel an LLM generated. It looks fine:

```python
@triton.jit
def softmax_buggy(x_ptr, y_ptr, n_cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * stride + offs, mask=offs < n_cols, other=0.0)
    # BUG: 'other=0.0' on the masked load puts 0 into positions that should be -inf.
    # When max() runs over the row, it sees 0 in the tail and the softmax gets shifted
    # by a phantom non-negative element. The mask should set other=-float("inf").
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    tl.store(y_ptr + row * stride + offs, e / s, mask=offs < n_cols)
```

Run the field-standard oracle on the typical benchmark shape:

```python
x = torch.randn(8, 256, device="cuda")   # 256 is a multiple of BLOCK=64
assert torch.allclose(softmax_buggy(x), torch.softmax(x, dim=-1), atol=1e-5, rtol=1e-2)
# ✓ passes
```

It passes — every benchmark reports this kernel as correct. Now run the **same** oracle on
a shape where `n_cols` is *not* a multiple of `BLOCK`:

```python
x = torch.randn(8, 3, device="cuda")     # 3 is not a multiple of BLOCK=64
assert torch.allclose(softmax_buggy(x), torch.softmax(x, dim=-1), atol=1e-5, rtol=1e-2)
# ✗ AssertionError: max abs diff 0.31
```

The bug is exposed only by the shape. The benchmark never picks that shape, so the bug
ships.

In gpuemu's measured 26-op corpus this exact pattern (`softmax_llm_buggy`) is caught at
**100% recall** by `boundary` shape sampling and **0%** by `regular` shape sampling
([P3](the-evidence.md#p3-test-input-generation)).

---

## The general pattern

The tail-mask leak is one of four bug families that the one-shape oracle is blind to:

| Bug family | Surface when | Real example we caught |
|---|---|---|
| **Tail-mask leak** | `H` not a multiple of `BLOCK` | softmax `other=0.0` ↔ `-inf` |
| **Accumulator scale** | depends on K-dim length | matmul `acc = …` instead of `acc += …` |
| **Missing normalisation** | depends on `D` magnitude | attention forgets `1/√D` |
| **Online-softmax rescale** | `N > BLOCK_N` (multi-tile) | flash-attention forgets `acc *= α` after a max update |

Every benchmark in the LLM-kernel literature uses one fixed shape per operator. Every
benchmark misses every member of these four families when the chosen shape happens to be
benign — which is most of the time.

---

## What gpuemu does instead

gpuemu replaces the one-shape allclose oracle with:

1. **An op-schema-aware fuzzer** that generates boundary, regular, and adversarial
   shapes per operator (matmul `M×K×N`, attention `B,H,S,D`, …) over multiple dtypes and
   seeds.
2. **A high-precision (fp64) reference** that does not share precision with the kernel
   under test.
3. **Calibrated per-op tolerances** derived from p95-of-controls, not a single
   hand-picked `atol/rtol`.
4. **Reproducible failures**: every fuzz iteration is a seed; every failure includes a
   base64 input snapshot for byte-for-byte replay.

In the gpuemu 26-op corpus this regime catches **10/10** LLM-style bugs across 5 GPU
classes, with **0 false positives** on 16 correct controls
([P1](the-evidence.md#p1-the-correctness-illusion)).

---

## Next

- **[Industry Impact](the-industry-impact.md)** — what these missed bugs cost when LLM
  stacks ship them at scale.
- **[The Evidence](the-evidence.md)** — measured findings (P1–P4) backing every gpuemu
  default.
- **[Quick Start](../getting-started/quickstart.md)** — your first validation in 5 minutes.
