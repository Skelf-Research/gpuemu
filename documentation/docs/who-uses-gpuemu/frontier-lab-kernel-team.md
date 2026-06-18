# Frontier-Lab Kernel Team

You are the 5–15 person GPU/TPU kernel team inside an AI lab that ships custom CUDA,
Triton, or Pallas kernels into pretraining and inference. Anthropic's Greenhouse board
in 2026 has [twelve simultaneously open roles][anth-board] touching this surface —
*Performance Engineer, GPU*, *TPU Kernel Engineer*, *Engineering Manager, Inference
Routing and Performance*, and more. None of them mention "correctness". OpenAI's
[Software Engineer, Inference — CUDA / Kernels][openai-cuda] cites "Triton, cuBLAS,
CUTLASS, mixed precision or tensor core optimization", again with no validation
language.

That gap is not because the work doesn't exist — it's because no industry-standard
tool names it. It hides inside generalist "performance engineer" roles and surfaces
as silent regressions weeks later. PyTorch's own measurement: [19.2 % of high-priority
`torch.compile` issues are silent wrong-output bugs][pt-silent].

[anth-board]: https://job-boards.greenhouse.io/anthropic
[openai-cuda]: https://openai.com/careers/software-engineer-inference-cuda-kernels/
[pt-silent]: https://arxiv.org/abs/2604.08720

---

## A pre-merge gate without a GPU

The workflow gpuemu enables for a frontier-lab kernel team:

1. A research engineer modifies a fused Triton matmul to add an FP8 mode.
2. PR triggers `gpuemu ci --parallel 8 --format sarif --output gpuemu.sarif` on the
   lab's runner (CPU-only is fine for the validation step; GPU is fine if the lab
   gates on real hardware).
3. The CI step runs an op-schema-aware seeded sweep: per-op shape generators
   (`MxKxN` for matmul, `B,H,S,D` for attention) crossed with `boundary | regular |
   adversarial` value distributions, each crossed with `float16 | bfloat16 | float32`.
   Every input is FP64-oracled, every output is checked against per-op calibrated
   tolerances (P2: +23 pp recall over the field-standard fixed `atol/rtol`).
4. Any flagged op posts a PR comment with a `gpuemu reproduce <seed>` line. The seed
   replays byte-for-byte on any machine — the lab's local box, the H100 cluster, the
   reviewer's laptop. P1 confirmed this across 5 GPU classes.
5. The CI run uploads SARIF to GitHub Code Scanning. The PR is *blocked* until the
   correctness check passes — same primitive as a perf-regression gate, except this
   one catches the bugs the perf gate is blind to.

---

## What this prevents

The four bug families the [problem walkthrough](../why-gpuemu/the-problem.md) enumerates
— tail-mask leaks, accumulator-scale bugs, missing normalisation, online-softmax
rescale bugs — all surface in this workflow because the sweep includes the *boundary*
shapes the field-standard one-shape oracle never tries. P3 measured this: the `regular`
strategy (the field default) catches **0 %** of tail-mask bugs; the `adversarial`
strategy catches **99 %** at zero precision cost on controls.

The lab gets:

- A **PR-blocking correctness gate** that distinguishes a real bug from a flake by
  shipping a replay seed.
- **Cross-architecture confidence**: P4 confirmed Δregs / Δinstrs are
  architecture-independent for the same PTX, so a gate calibrated on the lab's L40S
  runner transfers without re-calibration to the H100 production cluster.
- **A line of defence against quality regressions that don't crash** — the most
  expensive kind, because they ship and survive.

---

## Buying decision and budget shape

The buyer is typically an **Engineering Manager, GPU** or **Engineering Manager,
Inference** — the title literally appears in Anthropic's open roles. Budget sign-off
escalates to a Director of Infrastructure / VP Eng but not to the CTO; this is a
tooling line item, not a strategic procurement. The realistic price ceiling for a
20-engineer kernel team is **\$50–150 k / year per organisation**, priced as an
organisational site licence with a fair-use seat cap. Per-kernel metering would create
the same friction Monetizely warns about for CI tools and is incompatible with how
labs ship 100s of kernels per quarter.

If you'd like to pilot this workflow, see [Design partners](../why-gpuemu/design-partners.md).

## Where to go next

- [The problem](../why-gpuemu/the-problem.md) — the bug families this workflow
  catches.
- [The evidence](../why-gpuemu/the-evidence.md) — P1–P4 measured findings backing
  every default.
- [Compared to alternatives](../why-gpuemu/compared-to.md) — where gpuemu sits next to
  `torch.testing.assert_close`, KernelBench, Compute Sanitizer, and the rest.
