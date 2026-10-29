# Stage 0: FP32 dot-product bound for coefficient reconstruction

Status: approved by the user on 2026-10-29, after the first real fixed-context
run stopped.

The fixed-context instrument recomputes each rank-one coefficient `c = a·x` in
FP64 from the captured FP32 input and factor. It then requires the GPU FP32 value to
lie within `1e-6 + 1e-5·|a·x|`. The rule was inherited from the Pilot 3/4
instruments and kept by ADR 0010.

Run `stage0-geometry-v1-batch16-fp64-fixed-context-v1` sealed 15 units, then stopped
on `batch-0001-V1`. A bounded diagnostic in the same L4 session identified
layer 27 `mlp.down_proj` (d = 8960) on `gsm8k:test:65`. It measured
a·x = −2.018e-3, Σ|aᵢxᵢ| = 21.03 and error 1.025e-6. The allowance was 1.020e-6,
giving fraction 1.0048. The error is 0.82 FP32 units in the last place of Σ|aᵢxᵢ|. This
is below the a-priori worst-case bound of about d units. The other largest
cases were 0.2–0.8 such units, also with cancellation. The branch
reconstruction used at most 8e-4 of its allowance. Under cancellation, an absolute floor
of 1e-6 is finer than FP32 can resolve for these sums. A wrong captured tensor
errs on the scale of Σ|aᵢxᵢ| itself, about 10⁷ such units.

Decision:

- The coefficient allowance becomes `max(1e-6 + 1e-5·|a·x|, γ_d·Σ|aᵢxᵢ|)`, with
  `γ_d = d·u/(1 − d·u)` and `u = 2⁻²⁴`. This is the standard deterministic bound for
  an FP32 dot product of length d. It is set by d and the number format, not fitted
  to the observation.
- The legacy fraction is recorded per unit as `legacy_coefficient_max_fraction`, so
  every unit shows how often the old rule alone would have stopped.
- The branch reconstruction, KL, repeat, chunk, zero-control, mapping and integrity
  gates keep their thresholds.
- The failed v1 evidence is preserved. The replacement collection is
  `…-batch16-fp64-fixed-context-v2`.

Registration: the primary findings plan `…-batch16-primary-findings-v1` was frozen
before behavioral collection. It binds the v1 drift plan. A new plan
`…-batch16-primary-findings-v2` inherits that registration through
`--registered-from`, and only when these checks pass:

- the source plan is intact and was recorded `before-collection`;
- it binds the same behavioral collection;
- it has identical contrasts, statistics, hypotheses, limitations, margin and
  schedules.

The inheritance and its reason are stored in the plan. Otherwise the new plan is
labelled exploratory, as before.

This is a gate change made after a failure, recorded as an explicit user decision.
