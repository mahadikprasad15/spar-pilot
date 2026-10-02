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

FineWeb selection, lengths and presentation; numerical acceptance tolerances;
checkpoint provenance and loader after the Drive rename; artifact details,
resume, plots and interpretive limits.
