# Design Partners

gpuemu's daemon, CLI, Python client, and VS Code extension are dual-licensed
**MIT / Apache-2.0** with no telemetry. They are and will remain so. The
[research backing](the-evidence.md) (P1–P4) and the [kernel corpus](https://github.com/Skelf-Research/gpuemu-corpus)
are public.

We are working with **design partners on an enterprise tier** to extend the OSS core
with the surfaces a procurement department needs:

- **Private rule packs** — your kernel families, your invariants, your tolerance
  envelopes, versioned and signed.
- **Signed Kernel Correctness Reports** (HTML + PDF) suitable for SLA evidence
  packages, with ed25519 signatures customers can verify offline. See the
  [inference-vendor walkthrough](../who-uses-gpuemu/inference-vendor.md).
- **On-prem daemon** with SSO (SAML / OIDC), audit logs, and air-gapped operation.
- **Custom CI integrations** beyond the public GitHub Action — GitLab, Jenkins,
  Buildkite, Bazel, internal CI servers.
- **Support SLAs** and a private Slack / Teams channel with the maintainers.

We are deliberately not publishing tier names or prices in advance. **Pricing depends
on the shape of the first piloted use case**; the competitive research suggests
five-to-six-figure annual contracts for frontier-lab kernel teams and inference-as-a-
service vendors.

## Who we want to hear from

- **Inference-as-a-service vendors** that ship custom Triton / CUDA kernels and want
  to publish a signed correctness artefact with their SLA.
- **Frontier-lab kernel teams** that want a pre-merge correctness gate scaled to
  100s of ops with a fair-use site licence.
- **Maintainers of large OSS inference projects** (vLLM, SGLang, TensorRT-LLM,
  llama.cpp, …) who want a sponsored OSS tier — corporate users paying for the
  project-specific rule pack and the badge.

## Contact

**Email:** [me@dipankar.name](mailto:me@dipankar.name)

A one-line description of what your kernels look like (CUDA / Triton / Pallas, rough
op count, target GPU classes) is plenty for a first conversation. We do not collect
or store usage telemetry; the only information we have is what you tell us.
