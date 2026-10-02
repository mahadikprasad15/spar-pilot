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

## Still open

Framework and adapter on/off procedure; fixed token sequences and counted
positions; token versus example weighting; unrelated corpus and selection;
normalizations; zero-denominator policy; numerical acceptance tolerances;
checkpoint provenance and loader after the Drive rename; artifact details,
resume, plots and interpretive limits.
