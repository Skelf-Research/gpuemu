# OSS LLM-Inference Maintainer

You maintain one of vLLM, SGLang, TensorRT-LLM, llama.cpp, MLC-LLM, TGI, exllama,
lmdeploy. Your issue tracker is full of the exact bug class that gpuemu targets:

- [**SGLang #21238**][sg21238] (Mar 2026) — degenerate output with Qwen2.5-Math-7B at
  `temperature=1.0`, FlashInfer sampling kernel, a *repeat* of a v0.1.2 bug that
  re-surfaced in v0.6.3. Classic regression-of-regression.
- [**SGLang #15996**][sg15996] — titled *"[CI Infrastructure] Unified output format
  for regression check"*. You are literally building this in-house, badly.
- [**SGLang #17839**][sg17839] — PCG CUDA-graph dropping `gpt-oss-120b` accuracy from
  0.80 to 0.745.
- [**vLLM #26378**][vllm26378] (Oct 2025) — "Wrong answer with `torch==2.9` and
  Inductor compilation".
- [**vLLM #20974**][vllm20974] — GPTQ-Int4 regression between 0.9.x releases.
- [**llama.cpp #20052**][llama20052] — multi-GPU layer-split producing garbage at
  context > 2048 on non-P2P PCIe.

[sg21238]: https://github.com/sgl-project/sglang/issues/21238
[sg15996]: https://github.com/sgl-project/sglang/issues/15996
[sg17839]: https://github.com/sgl-project/sglang/issues/17839
[vllm26378]: https://github.com/vllm-project/vllm/issues/26378
[vllm20974]: https://github.com/vllm-project/vllm/issues/20974
[llama20052]: https://github.com/ggml-org/llama.cpp/issues/20052

These all share the same diagnosis: the regression slipped past
`torch.testing.assert_close` because the failing shape, dtype, or value distribution
isn't in the test grid.

---

## A one-line CI gate

The workflow gpuemu enables for an OSS-inference maintainer:

```yaml
# .github/workflows/gpuemu.yml
name: gpuemu
on: [pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: gpuemu/validate@v1
        with:
          config: gpuemu.toml          # the project's op registrations
          parallel: 8
          comment-on-pr: true
```

That's it. The action pulls the gpuemu Docker image, runs
`gpuemu ci --format sarif`, uploads to GitHub Code Scanning, and posts a single PR
comment listing flagged ops with a `gpuemu reproduce <seed>` link per failure. The
comment is the artefact your contributors actually read; the SARIF lights up the
*Security* tab; the action's exit code blocks the merge.

This is the response to SGLang #15996.

---

## Free, sponsored, and badged

gpuemu's daemon and Python client are dual-licensed **MIT / Apache-2.0** and ship with
no telemetry. There is no paid tier for OSS projects today — the OSS tier *is* the
product for your use case.

For sponsored OSS arrangements (a corporate user of your project — Anyscale for vLLM,
NVIDIA for TensorRT-LLM, etc. — paying for branded "kernel correctness powered by
gpuemu" support and a project-specific rule pack), see
[Design partners](../why-gpuemu/design-partners.md).

If your project adopts the gpuemu Action, you can show a `Kernel correctness: gpuemu
✓` badge on your README, the same way you show test-pass badges today.

---

## What you get out of week one

- **The "validated against P1–P4 corpus" claim** on your release notes — you can
  state that the release passes gpuemu's adversarial-shape sweep across the 26-op
  paper corpus, with run records on B2.
- **A `gpuemu reproduce <seed>` link in every regression issue**, so the next time
  someone files "wrong answers with `torch==2.9`" your reviewers can replay the
  failure in seconds instead of bisecting for a week.
- **A pre-merge gate** that distinguishes a real correctness regression from
  unrelated CI flake — the gate either flags an op with a seed or it doesn't.

## Where to go next

- [The problem](../why-gpuemu/the-problem.md) — the bug families that hide behind
  `torch.testing.assert_close`.
- [Quick Start](../getting-started/quickstart.md) — your first validation in 5
  minutes.
- [Compared to alternatives](../why-gpuemu/compared-to.md) — where gpuemu fits next
  to the other tools you reach for.
