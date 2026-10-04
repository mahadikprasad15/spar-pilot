# Pilot 3 interview

This records agreed decisions during grilling. It is not a spec or permission
to begin implementation. Remaining decisions must be settled before a spec.

## Accepted: round 1

- Validate the activation-measuring instrument and describe the saved Pilot 2
  checkpoints at steps 0, 8, 16, 32, 64. No training or response generation repeats.
- Use FP32 throughout, matching Pilot 2.
- Measure each module's direct LoRA contribution on the adapted pass's fixed
  module input. Measure full adapted-minus-untuned change at block outputs.
  These are complementary quantities; the direct module contribution excludes
  upstream input changes.
- Batch forward passes. Profile candidate batches for memory, throughput and
  numerical agreement before selecting the run batch.
- Do not assume an accuracy collapse: corrected Pilot 2 accuracy was 68% to
  66%, while mean response length decreased from about 216 to 138 tokens.

## Accepted: round 2 (Q4–Q6)

- Use Hugging Face forward hooks and the same loaded model, with the adapter
  disabled and enabled in sequential passes on each batch.
- Feed identical held-out question plus gold-solution token sequences to both
  passes. No answer generation. Report question and solution token positions
  separately; exclude padding, system text and template/formatting markers from
  counted positions while retaining required context in the input sequence.
- Report token-weighted measurements as primary and equal-example-weighted
  measurements as a sensitivity view. Both use token-level measurements.

## Pinned research question: length and weighting

Could longer gold solutions dominate the aggregate effective-write profile,
and does the profile change when every problem receives equal weight? Length
is relevant because Pilot 2 changed generated-response length.

Fixed gold-solution lengths in Pilot 3 are distinct from generated lengths in
Pilot 2. A weighting discrepancy demonstrates sensitivity to the measurement
cohort's lengths; it does not establish why the adapter generated shorter
answers. Keep this question visible in the spec and final report. Further
length-stratified or generated-response experiments are not yet approved.

## Accepted primary summaries: round 3

- Normalize the magnitude of the mean block-output change by the average
  baseline activation magnitude, not by the magnitude of the average baseline
  vector, as the primary relative block summary.
- Use the ratio of average adapter-contribution magnitude to average ordinary
  module-output magnitude as the primary module summary.
- Retain mean change magnitude, the alternative block denominator and mean
  per-token module ratios as secondary diagnostics, separate from primary plots.

## Accepted: round 4 (Q10–Q12)

- Use FineWeb for the unrelated-text control; freeze the selected text and
  source identities at a pinned dataset revision.
- Exactly zero denominators yield undefined ratios (`null`), with explicit
  undefined counts and coverage. Small nonzero denominators remain defined;
  preserve denominator diagnostics. Do not silently add epsilon.
- A zero numerator with a nonzero denominator is a valid zero ratio.
- For ratios of average magnitudes, individual zero-denominator tokens remain
  in the aggregate; the aggregate is undefined only if its averaged denominator
  is zero. For mean per-token ratios, average only defined ratios and report
  the conditional coverage explicitly.

## Still open

Numerical acceptance tolerances;
checkpoint provenance and loader after the Drive rename; artifact details,
resume, plots and interpretive limits.

## Accepted: validation scope (Q16–Q18)

- Require exactly zero step-0 block/module changes, exact equality of recorded
  before/after base-weight hashes and verified loaded checkpoint file hashes.
- Check the rank-1 identity in all 196 modules at all five checkpoints on the
  same module input, with predeclared FP32 tolerance (still to be settled).
- CPU tests use all valid tokens of a tiny locally constructed model. Real
  validation samples up to 16 counted positions per module/checkpoint/input
  view, fixed reproducibly and spread across examples. Actual measurement
  aggregates still include every counted token.
- Include deliberately wrong cases in CPU tests to establish that the check
  catches errors. Expand real validation if results approach tolerances or
  show unexpected variation; do not silently relax acceptance thresholds.
- Stop on missing/mismatched inputs, incorrect hooks, nonfinite activations or
  failed validation. Persist diagnostics and failure status.

## Accepted: round 7 (Q19–Q21)

- Preserve complete GSM8K question-and-gold-solution sequences. Never silently
  truncate to match FineWeb length; stop if a hard sequence limit is exceeded.
  Relative normalization does not remove context-length differences. Keep
  question/solution views separate and cross-corpus comparisons descriptive.
- Construct and validate tokenizer-aware content masks. Exclude ambiguous
  boundary-crossing tokens and report their counts rather than guess ownership.
- Reuse one untuned block-activation reference per fixed batch across five
  checkpoints only after checking that switching checkpoints leaves disabled-
  adapter outputs unchanged. Require identical token IDs, masks, positions,
  frozen weights and numerical settings. Keep the temporary cache bounded
  and profile memory. Main forward work is one untuned plus five adapted passes
  per batch, excluding validation/preflight overhead.

## Accepted: round 8 (Q22–Q24)

- Validate both the pre-addition adapter contribution and the post-addition
  module output difference. Use predeclared FP32 tolerances, report maximum
  errors and contributions below subtraction resolution, and require negative
  controls. Exact tolerance constants remain to be confirmed before the spec.
- Explicit FP32 model computation; disable mixed precision and TF32; evaluation
  mode. Accumulate summary sums in FP64 and record numerical settings.
- Profile batches 1, 2, 4, 8, 16 including long GSM8K sequences and the bounded
  untuned-reference cache. Choose the fastest candidate passing memory and
  numerical-agreement checks. Benchmark OOM marks a candidate unsuitable;
  production OOM stops with saved progress. Agreement thresholds remain open.

## Accepted: progress and retained summaries (Q25–Q27)

- Treat numerical tolerances as predeclared engineering acceptance thresholds,
  not guaranteed floating-point error bounds. The direct-contribution proposal
  is atol=1e-6, rtol=1e-5; batch-summary agreement proposal is atol=1e-5,
  rtol=1e-5. The complete subtraction-rounding allowance is not yet settled.
- Save completed batch/checkpoint combinations with verified input/config hashes;
  freeze batch membership and numerical settings. Use completion markers and
  integrity checks to reject partial writes, and aggregate completed combinations
  exactly once on resume.
- Retain per-example vector sums, baseline and magnitude summaries and counts,
  alongside aggregate results. Do not retain full token-by-token activations by
  default. This supports weighting/length sensitivity and example-level uncertainty
  without repeating GPU inference; it cannot reconstruct arbitrary token-level
  analyses. NPZ arrays plus JSON/JSONL metadata are approved; exact schema remains
  to be specified. Use a baseline shard once per batch and hashed completion
  markers written last to validate each completed batch/checkpoint shard.

## Agreed workflow

- Complete grilling and confirm shared understanding before writing a spec.
- Apply to-spec: confirm the proposed external test boundaries, then synthesize
  the settled decisions as a ready-for-agent spec in the local issue tracker.
- Apply to-tickets: propose independently verifiable vertical slices with true
  blocking edges; obtain approval of granularity/dependencies before publishing.
- Implementation follows approved tickets with red/green CPU tests and real-model
  Colab preflight. No scientific inference, training or code implementation starts
  merely because the interview notes exist.

## Round 10: thresholds, inputs and report

- The proposed output-subtraction acceptance threshold per coordinate is
  1e-6 + 1e-5*abs(expected_write) + 4*epsilon_fp32*(abs(adapted_output)+
  abs(ordinary_output)), with epsilon_fp32=2**-23. Pre-addition branch validation
  uses only the first two terms. The user agreed but requested clarification of
  how this relates to the three instrument checks; explain before final design
  confirmation. These are engineering thresholds, not guaranteed error bounds.
- Use the verified Pilot 2 training-run manifest to obtain model/tokenizer pins
  and checkpoint identities. Resolve files under the new root and verify hashes;
  preserve historical configs rather than edit their absolute paths.
- Report block-write checkpoint/depth curves, layer-by-module ratio heatmaps,
  token versus equal-example weighting sensitivity, validation/denominator/
  resolution diagnostics, and 95% example-bootstrap intervals for primary scalar
  summaries (2,000 draws, seed 42). Resample GSM8K IDs jointly across checkpoints;
  resample FineWeb separately. Intervals do not represent training-seed variability.
- Final visual layout and test-boundary confirmation remain open before to-spec.

## Accepted: round 5 (Q13–Q15)

- Inspect the first 2,000 documents of a pinned FineWeb stream. Keep documents
  with at least 128 Qwen content tokens; select 150 with seed 42. Use their first
  128 content tokens. Freeze text, token IDs and source identities. This is a
  sample of that bounded pool, not a representative sample of the whole web.
- Present FineWeb as user text under the same Qwen chat wrapper. Primarily
  contrast it with GSM8K question-token measurements; solution-token results
  stay separate. Content and task differences remain, so this is a descriptive
  cross-input contrast rather than proof of task specificity.
- Exclude chat structure, system text, padding and task instructions from counted
  positions. Keep complete gold-solution content, including calculator annotations
  and the final answer marker, because those were in the training targets.
  This clarifies the earlier ambiguous phrase 'formatting markers'.
- Pilot 3 validates and describes activation changes. It does not establish a
  causal explanation for shorter generation. Later intervention experiments may
  use the measured vectors to test such explanations.
