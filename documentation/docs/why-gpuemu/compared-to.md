# Compared to Alternatives

The first question a champion gets asked about gpuemu is "isn't this just
`torch.testing.assert_close`?" or "isn't this what KernelBench does?". The honest
answer to every variant is that gpuemu sits in a slot **none of the existing tools
occupies** — and below is the per-tool walk-through that shows why.

## The comparison table

| Tool | What it does well | The gap gpuemu fills |
|---|---|---|
| **`torch.testing.assert_close`** | Standard, simple, in-tree, runs anywhere | One shape, one dtype, one seed; no op-schema fuzz; no fp64 reference; no per-op calibrated tolerances. **P1** measured this catches 0/10 LLM-style buggy kernels in the 26-op gpuemu corpus across five GPU classes. |
| **KernelBench / TritonBench / GEAK / KernelBand / STARK** ([KernelBench][kb], [GEAK][geak], [STARK][stark]) | Leaderboards for LLM-generated kernels; useful research benchmarks | All four use the same one-shape `torch.allclose` oracle internally. Multiple 2025 retrospectives ([Youstra][jy], [Guo][sg]) document that KernelBench accepts kernels that compute only part of the output. Not user-facing; you submit *to* them, you cannot point them at your own kernels. |
| **NVIDIA Compute Sanitizer** ([docs][ncs]) | Memcheck, racecheck, initcheck, synccheck at runtime — catches memory and synchronisation bugs | **Detects no numerical bugs.** A tail-mask leak, accumulator-scale error, or missing normalisation produces correct-looking memory accesses and a silently-wrong result. Strictly orthogonal to gpuemu. |
| **Triton built-in testing** ([deepwiki][tt]) | pytest + `triton.testing.assert_close`, integrated with Triton's own CI | Same `assert_close` semantics as PyTorch (one shape, one dtype, one seed); no op-schema fuzz; no fp64 reference. The Triton harness is *very* good at finding compiler regressions; it is not designed to find LLM-generated kernel bugs. |
| **AMD ROCm Validation Suite** ([RVS][rvs]) | Hardware stress: temps, memory bandwidth, P2P throughput | Hardware-CI, not kernel-correctness-CI. Solves a different problem. |
| **HF Kernel Hub** ([kernels][hfk], [kernel-builder][hfkb]) | Distribution channel + ABI compatibility checks for sharable GPU kernels | The HF kernel-requirements doc explicitly says *"Don't forget to check for numerical correctness (`torch.testing.assert_close`)."* — **the Hub assumes a correctness tool upstream**. That's gpuemu's slot. **Integration target, not competitor.** |
| **ncu / cuobjdump / ptxas** ([cuda-binary-utilities][cbu]) | Surface PTX, SASS, register usage, spill stats | No lint policy, no baseline diffing, no built-in regression gate. ptxas spill stats aren't extensible. **No public third-party PTX linter** in active distribution. P4's static-PTX gate fills this entire layer. |
| **FreeFuzz / DocTer / DeepREL / NablaFuzz / FuzzGPT** ([ACL TOSEM 2025][acl]) | API-level Python fuzzers for DL frameworks | Target the *framework API* layer, not the kernel layer. The 2025 ACL TOSEM benchmarking study measured the seven SOTA fuzzers in this category collectively catch **6.5 %** (34/517) of real-world bugs. Adjacent, not competitive. |

[kb]: https://arxiv.org/abs/2502.10517
[geak]: https://arxiv.org/abs/2507.23194
[stark]: https://arxiv.org/abs/2510.16996
[jy]: https://www.jackyoustra.com/blog/kernelbench-agent
[sg]: https://simonguo.tech/blog/2025-10-automated-gpu-kernels.html
[ncs]: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/
[tt]: https://deepwiki.com/triton-lang/triton/7-testing-development-and-debugging
[rvs]: https://github.com/ROCm/ROCmValidationSuite
[hfk]: https://github.com/huggingface/kernels
[hfkb]: https://huggingface.co/docs/kernels/en/kernel-requirements
[cbu]: https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
[acl]: https://dl.acm.org/doi/10.1145/3729533

## Five moat signals

The competitive map above also surfaces five gaps that, as of mid-2026, **no public
tool occupies**:

1. **No public tool combines op-schema-aware fuzzing + fp64 oracle + per-op
   calibrated tolerances + static PTX/SASS lint.** Fragments exist (`assert_close`,
   Compute Sanitizer, ncu) but never combined into one correctness regime.
2. **No public tool offers cross-language reproducible RNG for tensor-kernel test
   inputs.** gpuemu's bit-identical xorshift128+ in Rust *and* Python (see [the
   evidence](the-evidence.md)) means a seed that flagged a bug on a vast.ai H100
   replays byte-for-byte on a reviewer's laptop.
3. **No GPU cloud markets "GPU CI for kernel correctness."** Modal's published GPU
   health work, RunPod's CI hooks, and Lambda's spot fleet are all *compute*; they
   are not validation infrastructure.
4. **No third-party PTX/SASS linter in active distribution.** ptxas spill stats are
   not extensible; cuobjdump exposes data without a policy layer. gpuemu's
   `ArtifactLinter` + `ArtifactDiffer` + baseline-diffing flow fills this entire
   layer (P4).
5. **Hugging Face's Kernel Hub is an integration target, not a competitor.** The
   Hub's own documentation explicitly assumes a correctness tool upstream of
   distribution. gpuemu is positioned to be that tool.

## Independent confirmation of the gap

The same gap shows up in third-party 2025–2026 writing:

- **STARK** ships an internal *"five-stage correctness harness covering smoke tests,
  shape sweeps across 10+ configurations, numerical stability under adversarial
  inputs, determinism verification, and edge-case coverage"* — language that mirrors
  gpuemu's product description. Agent teams now treat single-shape `allclose` as
  inadequate. gpuemu generalises that internal harness into a user-facing tool.
- **GEAK** introduces a "Benchmarking Agent" that *"runs correctness checks against
  a trusted reference"*. Same direction of travel.
- **PyTorch** itself: [arXiv 2604.08720][arxiv-silent] measures that **19.2 % of
  high-priority `torch.compile` issues are silent wrong-output bugs**, the
  second-largest category after outright crashes. The field knows the oracle is
  inadequate; nothing in the public tool inventory has replaced it.
- **Silent Data Corruption in LLM training** is now a recognized academic
  sub-field: [ACL 2025][sdc-acl] long paper, [LLM-PRISM][llm-prism] (arXiv 2604.10390),
  [TU Berlin coverage][sdc-tu]. The kernel-function-level fault sensitivity these
  papers identify is exactly the layer gpuemu addresses.

[arxiv-silent]: https://arxiv.org/abs/2604.08720
[sdc-acl]: https://aclanthology.org/2025.acl-long.996/
[llm-prism]: https://arxiv.org/html/2604.10390v1
[sdc-tu]: https://semiengineering.com/silent-data-corruption-a-major-reliability-challenge-in-large-scale-llm-training-tu-berlin/

---

## Where to go next

- [Who uses gpuemu](../who-uses-gpuemu/frontier-lab-kernel-team.md) — the three
  customer profiles this slot serves.
- [The evidence](the-evidence.md) — P1–P4 measured findings.
- [Design partners](design-partners.md) — pilot the enterprise tier.
