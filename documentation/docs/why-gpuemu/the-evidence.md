# The Evidence

Every default in gpuemu is anchored to a measured study. This page summarises the four
papers; the full LaTeX manuscripts, kernel corpus, and replayable B2 run records live in
the [gpuemu-paper artefact](https://github.com/Skelf-Research/gpuemu-paper).

All five papers ran on the same harness: vast.ai ephemeral GPUs (RTX 3060, A10, L40S,
A100 SXM4, H100 NVL) → Backblaze B2 (`sarkar-dipankar-research/gpuemu/`) → always-destroy
teardown. Every flagged failure replays byte-for-byte from a saved input snapshot.

---

## P1 — The correctness illusion

> *"The correctness illusion in LLM-generated GPU kernels"* — Sarkar, 2026.
> [arXiv:2606.20128](https://arxiv.org/abs/2606.20128).

**Question.** How many LLM-generated kernels that pass the one-shape `torch.allclose`
oracle actually contain bugs that surface under op-schema-aware fuzzing?

**Setup.** 24-op single-GPU corpus (15 controls + 9 LLM-style buggy variants of real
Triton kernels: softmax, GeLU, SiLU, RMSNorm, l2norm, leaky_relu, matmul, attention),
extended to 26 ops by adding a flash-attention pair for the cross-GPU sweep. Each
kernel fuzzed at 30 iterations.

**Headline result.** On the single-GPU corpus the fuzz oracle catches **9 / 9**
LLM-style bugs with **0 / 15** false positives on controls. The extended 26-op
corpus catches **10 / 10** LLM-style bugs with **0 / 16** false positives on each of
the 5 GPU classes (RTX 3060, A10, L40S, A100 SXM4, H100 NVL). The field-standard
oracle would have shipped every one.

**Why this matters.** This is the operational meaning of "correctness illusion": every
buggy kernel is a benchmark "pass" and a gpuemu fail. Replace the oracle, catch the bug.

---

## P2 — Tolerance calibration

> *"Operator-aware mixed-precision tolerance calibration for tensor kernels"* — Sarkar, 2026.

**Question.** Does deriving per-op tolerances from measured controls beat a single
hand-picked `atol=1e-5, rtol=1e-2`?

**Setup.** For each (op, dtype) pair, take the 95th percentile of measured `max_abs`
error on correct controls and use 1.5× that as the calibrated `atol`. Re-score P1's
verdicts under the calibrated tolerances.

**Headline result.** Calibrated tolerances raise kernel-bug recall from **65 %** (fixed
field-standard) to **82 %**, an absolute **+17 to +23 percentage-point** gain depending
on the operator family. False positives stay at zero.

**Why this matters.** The "tolerance is a free parameter" framing hides the fact that
fixed tolerances *miss bugs*. A small, measured calibration step closes most of the
gap.

---

## P3 — Test-input generation

> *"Test-input generation for tensor programs: what actually finds kernel bugs"* —
> Sarkar, 2026.

**Question.** Among the seven plausible strategies for generating tensor-kernel test
inputs, which catches the most bugs?

**Setup.** Seven strategies — `default`, `boundary` shapes only, `regular` shapes only,
`single_dtype_f32`, `single_dtype_f16`, `nan_injected` values, `adversarial` values —
each run across the full 26-op corpus.

**Headline result.**

| strategy | bug recall |
|---|---:|
| **adversarial** | **99 %** ← wins |
| nan_injected | 94 % |
| boundary | 78 % |
| default (native schema) | 71 % |
| regular (no boundary shapes) | 64 % |

Most strikingly, the `regular` strategy catches **0 %** of tail-mask bugs (e.g.
`softmax_llm_buggy`) that `boundary` catches at 100 %.

**Why this matters.** Test-input generation is not a fixed cost — it's a 35-point
recall swing. The data argues for two trivial defaults that gpuemu already ships:
include boundary shapes, sample with an adversarial value distribution.

---

## P4 — Static PTX gating

> *"Static PTX metrics track structural kernel regressions but miss semantic ones"* —
> Sarkar, 2026.

**Question.** Can static PTX/SASS metrics (register count, spills, instruction count)
gate performance regressions without hardware?

**Setup.** Pair each correct/buggy Triton kernel. Capture PTX from the Triton cache,
extract static metrics via gpuemu's artifact analyzer, pair with CUDA-event-timed
runtime, across 5 GPU classes.

**Headline result.** Structural bugs (gelu drops `0.5×`, l2norm drops `sqrt`, silu
β-confusion) show consistent Δregs and Δperf% across all 5 architectures — e.g.
gelu_buggy is ≈ −33 % runtime on every GPU. Semantic bugs (softmax `other=0.0` vs
`-inf`, matmul `acc=` vs `acc+=`, attention without `1/√D`) compile to **identical
PTX** and run in **identical time** — invisible to any static gate.

**Why this matters.** Static PTX gating is a *cheap, useful pre-filter* but cannot be
the sole correctness gate. The two methodologies are complementary, not competing.

---

## Reproducing any result

Each paper publishes:

- The LaTeX manuscript (`papers/p<N>/paper.tex` in
  [gpuemu-paper](https://github.com/Skelf-Research/gpuemu-paper)).
- The driver (`drivers/p<N>_*.py`).
- The B2 run records (`sarkar-dipankar-research/gpuemu/`).
- The kernel corpus (`corpus/`), including the paired buggy variants with documented
  bug class.

To replay a flagged failure end-to-end:

```bash
git clone https://github.com/Skelf-Research/gpuemu-paper
cd gpuemu-paper
python3 scripts/replay_from_b2.py <run_id> <kernel> <iter>
```

The replay loads the byte-for-byte input snapshot, re-runs through the daemon, and
prints the verdict — proving the artefact is genuinely reproducible.

---

## Five moat signals from the competitive map

The same four studies above were independently confirmed by a 2026 competitive-
landscape sweep. As of mid-2026, **no public tool occupies the slot gpuemu sits in**:

1. **No combined regime.** No public tool combines op-schema-aware fuzzing + fp64
   oracle + per-op calibrated tolerances + static PTX/SASS lint. Fragments exist
   (`torch.testing.assert_close`, NVIDIA Compute Sanitizer, ncu, ptxas spill stats)
   but are never combined into one correctness oracle.
2. **No cross-language reproducible RNG for tensor inputs.** gpuemu's bit-identical
   xorshift128+ in Rust *and* Python means a seed that flagged a bug on a vast.ai
   H100 replays byte-for-byte on a reviewer's laptop. The Rust↔Python parity test
   in `crates/gpuemu-common/src/rng.rs` is, to our knowledge, unique.
3. **No GPU cloud markets "GPU CI for kernel correctness."** Modal's published
   GPU-health work, RunPod's CI hooks, Lambda's spot fleet are all *compute*. No
   vendor advertises a kernel-correctness CI surface; gpuemu's ephemeral-GPU
   harness (vast.ai + B2) is complementary to all of them.
4. **No third-party PTX/SASS linter in active distribution.** ptxas spill stats
   are not extensible; cuobjdump exposes data without a policy layer. P4's
   `ArtifactLinter` + `ArtifactDiffer` + baseline-diffing flow fills this entire
   layer.
5. **Hugging Face's Kernel Hub is an integration target, not a competitor.** The
   Hub's [own documentation](https://huggingface.co/docs/kernels/en/kernel-requirements)
   explicitly tells contributors to *"check for numerical correctness
   (`torch.testing.assert_close`)"* — the Hub assumes a correctness tool upstream of
   distribution. gpuemu is positioned to be that tool.

See [Compared to alternatives](compared-to.md) for the per-tool walk-through that
backs each of these.
