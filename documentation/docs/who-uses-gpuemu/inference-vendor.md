# Inference-as-a-Service Vendor

You are Fireworks, Together, Anyscale, Modal, Replicate, Baseten, Predibase, or
Modular. You sell "we serve open-weight models with low latency" to enterprise
customers. Every kernel swap and every quantization variant carries SLA and
contractual quality risk.

A 2026 [Future AGI evaluation][futureagi] documents a Fireworks deployment of
Llama-3.3-70B-Instruct shipping on a quantized kernel variant with a non-default
`top_p` and no public disclosure. The result: JSON-mode adherence dropped 6 points
and 3-step tool-calling chains broke. Quote: *"the OpenAI-compatible API is a
contract about request and response shape, not about weights, precision, kernel, or
sampler defaults."*

[futureagi]: https://futureagi.com/blog/evaluating-fireworks-together-inference-2026/

No inference vendor publicly publishes a kernel-correctness methodology. Silence in a
competitive market means it is either secret IP or absent — and the Future AGI case
study suggests the latter.

---

## A signed Kernel Correctness Report

The workflow gpuemu enables for an inference-as-a-service vendor:

1. Before promoting a new kernel build to production, the deploy pipeline runs
   `gpuemu ci --parallel 16 --corpus your-prod-corpus.toml`.
2. After the run, the pipeline emits a signed, customer-facing artefact:

    ```bash
    gpuemu report --signed --format pdf \
        --output kernel-correctness-2026-06-11.pdf
    ```

    The PDF contains a per-op × per-dtype × per-shape pass matrix, the gpuemu corpus
    version, the run id, a SHA-256 of the result database, and an ed25519 signature
    over the report contents. Customers can verify the signature with one OpenSSL
    command.

3. The report is attached to the deploy ticket and surfaced on the vendor's status
   page. Enterprise customers receive it as part of their SLA evidence package.

This is the **trust artefact** the Future AGI case study was missing. It converts
gpuemu from an internal CI tool into something you sell *with* your inference SLA.

---

## Why this changes the contract

- **Public verifiability.** A customer who is on the fence about your service can
  read the signed report, verify the signature offline, and see — at the kernel
  level — what your service actually validated and how the validation went.
- **Pre-deploy correctness gating** with replay seeds. When something does go wrong
  in production, the on-call engineer has a corpus run id and a list of seeds, not
  a postmortem that says *"we believe it was the quantization step."*
- **Differentiation from vendors that don't publish.** When one vendor publishes a
  signed report and another publishes silence, procurement notices.

---

## Pricing shape

This is the highest-budget, lowest-volume customer profile gpuemu serves. Indicative
pricing (subject to a paid pilot):

| Component | Indicative |
|---|---|
| Annual platform licence | mid-six figures |
| Per-deployment certification fee | per signed report |
| Optional audit report | additional |

Realistic ceiling per vendor: **\$100–500 k / year**, structured as an annual licence
plus a per-certification fee. The comp is Datadog's enterprise APM tier and
SonarQube's enterprise model. The buyer is **VP of Engineering** or **CTO**, because
the pitch is risk-managed-against-customer-churn rather than developer productivity.

If you want to pilot, see [Design partners](../why-gpuemu/design-partners.md).

## Where to go next

- [The evidence](../why-gpuemu/the-evidence.md) — the four measured studies the
  signed report cites.
- [Compared to alternatives](../why-gpuemu/compared-to.md) — where gpuemu sits next
  to existing model-quality tooling.
- [Design partners](../why-gpuemu/design-partners.md) — the path to a piloted
  enterprise tier.
