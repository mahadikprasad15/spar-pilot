# Ticket 08: gap-aware rotation and chance-aware intruders

## Software scope

CPU public commands `stage0-intruders-{prepare,run,status,verify,recover-writer}`
consume the verified base spectral cache and the saved SFT64 and GRPO64 adapters.
Every adapted module/arm is measured with an exact FP64 reduced SVD of
`W0 + b a-transpose`; no small-update skip, GPU/model load or behavioral edit.

CPU notebook sections 23–25 prepare the branch, measure one missing unit by
default, then verify/report. They can be reached directly after sections 1–8;
primary GPU collection and findings are not dependencies. The complete real
inventory contains 392 module/arm units. Actual SVD time/storage are measured,
not inferred from earlier GPU timing estimates.

## Mathematical contract

- Principal angles compare entire old/new top-k left and right subspaces at
  base-defined k30/k50/k70. Stored bases must pass inherited orthogonality gates;
  QR re-orthonormalizes them for subspace arithmetic. Internal rotations/sign
  changes do not count as subspace rotation.
- Rank-one update operator norm is `||b|| ||a||`, also its Frobenius norm.
  Update/sigma_k and update/base-boundary-gap are separate diagnostics.
  Positive/zero is infinite, zero/zero undefined, no-next singular value undefined.
  Save explicit statuses and null numeric values; no ratio epsilon.
- Top-20 new left singular directions are detection candidates. For each energy
  level use m=max(k,40), keep overlap **norm**, and compare against all thresholds
  0.3, 0.4, 0.5, 0.6, 0.7. Primary threshold is strictly below 0.5.
- Chance is sqrt(m/d_out), retained per module/candidate. Threshold at or below
  chance is undecidable, with no eligible edit indices; raw overlaps/flags remain.
  No-intruder and undecidable modules are counted separately in each arm.
- Retain top-40 candidate singular values, gaps and overlaps for later V6c matching.
  Matching and merged-weight edits are Ticket 09, not this ticket.
- Validate all ranks/available columns before expensive decomposition; never clip.
- Full updated-matrix reconstruction below 1e-6 and orthogonality to 1e-6 are
  checked before storage truncation. All singular values FP64; retained U/V FP32.
- Save exact old/new gaps and descriptive numerical resolution indicator
  eps64*singular_dimension*sigma_max. Tied boundaries/vectors retain ambiguity
  warnings; this indicator adds no acceptance gate or intruder eligibility rule.
- Zero updates are measured and have exact zero subspace rotation, with boundary
  degeneracy still explicit. No “rotation under a degree” prediction or gate.

## Durability and report

Plans/configs, manifests/status/progress, source and implementation identities,
sealed module arrays/JSON, timing/bytes, candidates/geometry CSV, summary JSON,
plot-input JSON, SVG/PNG and HTML/Markdown are canonical under
`runs/stage0/<name>/intruders/`. Final publication rechecks source identities.
Sealed units resume, unsealed payloads recompute, corrupt or re-sealed changed
eligibility fails reconstruction. Stale writer release needs the exact token and
notes, and refuses a live local process. CPU status/verify import no model runtime.
Verification reconstructs angles/overlaps/ratios from saved singular arrays,
retaining hash-bound full-SVD reconstruction evidence without a fresh full SVD.

## TDD evidence

- Initial public test red: commands absent. Independent math tests added before
  implementation; public/math first green: 4 passed in 7.99s.
- Notebook handoff test red before new cells existed.
- Expanded tests exposed an unavailable-rank error occurring after SVD; validation
  moved before decomposition. No acceptance tolerance was changed.
- Focused suite then 11 passed in 30.45s.
- Startup affected baseline: 32 passed in 86.38s.
- Final affected regression and source pin: recorded at closeout below.

## Execution status

Software fixtures are synthetic and labelled. No real Qwen weight geometry,
GPU qualification, source upload or behavioral intervention was run. Real source
preparation and base spectral cache must be executed before this CPU branch.
The primary report remains independently runnable. Ticket 09 is the next
conditional implementation slice; absence of eligible intruders must be a
verified result, not inferred from missing execution.

## Closeout, 27 October 2026

- Final affected suite: **44 passed in 106.77s** (intruder geometry, base spectral
  cache, principal edits and shared Stage 0 workflow). This is not a new full-suite
  pass. Includes mutation of a source during the SVD blocking final publication.
- Notebook cells parse and have no outputs; source compiles and diff checks pass.
  Generated overlap figure inspected; raw candidate and angle tables retained.
- Synthetic validation snapshot and independent diagonal-oracle results are
  persisted under `artifacts/validation/stage0-ticket08/`.
- No real adapter geometry, GPU job, training, Drive upload or GitHub push.

- Notebook scientific implementation pin: `b27dbf94761a24dd292a432b4a7981b2ff6e4585`.
