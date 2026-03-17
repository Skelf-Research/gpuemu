# Validation Taxonomy and Policies

The validation system focuses on correctness and early structural regressions without relying on GPU execution. It groups checks into deterministic categories so failures are actionable.

## 1) Correctness checks

- **Reference equivalence**: Compare CPU mirror outputs to a canonical CPU implementation.
- **Layout robustness**: Validate behavior across contiguous and strided layouts.
- **Boundary safety**: Exercise tail and edge-case shapes to reveal OOB assumptions.

## 2) Numerical stability

- **Precision policies**: Enforce expected accumulation types (e.g., fp32 accumulation).
- **Tolerance envelopes**: Define acceptable error bounds per dtype and operator.
- **Reduction stability**: Detect order-sensitive reductions and unstable summations.

## 3) Invariants

- **No NaN/Inf**: Fail fast on invalid output unless explicitly allowed.
- **Monotonicity or symmetry**: Operator-specific properties where applicable.
- **Shape preservation**: Output shapes must match contract definitions.

## 4) Artifact-based checks

- **Register pressure**: Guardrails for occupancy regressions.
- **Spills and local memory**: Detect silent performance regressions.
- **Pattern expectations**: Presence or absence of expected instruction forms.

## 5) Fuzzing strategy

- **Shapes**: Include primes, odd sizes, and minimal edge cases.
- **Layouts**: Transposed, sliced, and strided views.
- **Dtypes**: Mix fp16/bf16/fp32 with explicit accumulation rules.

## Failure output

When a check fails, the system should report:

- The minimal failing input shape/layout
- The random seed used to generate the case
- The exact policy that failed
- A short diff summary of the outputs

