# ADR 0008: Calibrate cross-batch agreement before authorizing execution

Status: approved by user, 2026-10-06. Real GPU acceptance remains pending.

## Evidence and decision

The v1 coordinate-wise `atol=rtol=1e-5` stress profile rejected batch 2.
Saved errors were small in aggregate relative L2 terms but larger near cancelling
coordinates. Similar errors occurred with and without padding. This supports
shape-dependent FP32 drift, but does not prove the absence of implementation bugs.
The old rule and its failed artifacts remain unchanged.

Use `pilot3-agreement-v2`, explicitly approved before collecting new data:

- Freeze 16 calibration examples and 16 separate validation examples: eight
  GSM8K and eight FineWeb in each cohort, sampled from four length strata with
  the prepared seed. Exclude the 16 examples used in the old stress profile.
- On all five checkpoint adapters, compare batch 1 against its repeat, deliberate
  right padding and batch sizes 2, 4, 8 and 16. Save all per-example sufficient
  statistics, source/runtime identities and completion hashes. Resume verified units.
- Fit each metric's envelope as its maximum calibration error. Freeze a 3x
  margin after operator review, before collecting validation. This is an
  engineering margin, not a statistical confidence guarantee.
- Reject candidate batches exceeding this frozen rule on independent validation.
  Repeat and padded controls must pass. Wrong sign, doubled scale and corrupted
  token-count mapping must be rejected. A wide rule that accepts these stops work.
- Profile eligible candidates on the separate longest-example stress workload.
  Stress disagreement still stops; no automatic threshold adjustment.

## Metrics and interpretation

Vector L2 difference is normalized by the maximum ordinary baseline magnitude
(sum/mean of token norms) of the two compared executions. Scalar norm differences
use this same physical scale. Dimensionless module and alternative block ratios
use `max(1, abs(reference), abs(candidate))`. Metrics include per-example values,
pooled token and equal-example weightings, every view/layer/module/checkpoint.

Direction discrepancy is `1 - cosine`. Vectors at or below the calibrated vector
resolution are counted as unresolved rather than used to calibrate direction
agreement. Reports flag block write vectors whose primary relative magnitude does
not exceed this resolution. This finite calibration does not guarantee numerical
accuracy on every unseen example, and statistical bootstrap intervals measure a
separate source of uncertainty.

Counts, defined coverage, exact zero writes, checkpoint hashes, hook identity,
rank-1 checks and nonfinite failures are not calibrated or relaxed. Negative
controls are necessary checks, not a comprehensive proof of correctness.

## Operational consequences and alternatives

Keep v1 preparation and failed profile. New calibration has its own named tree;
new diagnostic and profile paths bind runtime and rule identity. Use a new v2
execution/report name. No retraining or generated-answer evaluation is required.
A real GPU run remains necessary. The local tests use controlled fixtures and a
small randomly initialized Qwen model without downloads.

FP64 model reference was considered but is costly on the available GPU and is
not part of this protocol. Simply widening v1 tolerances would invalidate its
acceptance contract. Always using batch 1 is an available slower fallback.
