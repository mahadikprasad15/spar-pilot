# 03: Run matched FP32 baseline and checkpoint evaluations

Type: task
Status: resolved
Blocked by: 02 - Preflight, train and resume the rank-1 adapter

**What to build:** A public checkpoint-evaluation workflow that reuses the
existing harness for a fresh matched untuned FP32 baseline and all five adapter
checkpoints, without changing the frozen evaluation inputs.

- [x] Extend evaluation support to explicit FP32 while preserving existing
  Pilot 1 precision behavior and saved artifacts.
- [x] Derive configurations from frozen model/tokenizer/dataset revisions,
  actual prompt template and exact 150 item IDs; never resolve floating pins.
- [x] Evaluate untuned and step 0/8/16/32/64 variants in the same locked runtime
  using batch 1 and frozen greedy decoding, including the 1024-token cap.
- [x] Record adapter identities/hashes and save every configuration and response;
  reuse completed verified runs and process only missing items on resume.
- [x] Score saved responses with unchanged strict extraction and frozen
  flexible-v2 extraction; retain legacy outputs without changing old measurements.
- [x] Require checkpoint-zero behavior to match the untuned baseline before
  interpreting later changes; save evidence and block on an unexplained mismatch.
- [x] Reject incompatible runtime or comparison inputs and distinguish incomplete
  from completed cohorts. Do not run MMLU as part of Pilot 2.
- [x] CPU tests exercise the workflow through fake external model boundaries,
  including zero-checkpoint matching, preservation and evaluation resume.

## Comments

Approved as ticket 3. Existing baseline scores are contextual references,
not substitutes for the fresh matched FP32 measurement.


## Answer

Matched FP32 evaluation is derived from frozen inputs and checkpoint hashes.
Two CPU workflow tests pass, covering reuse and checkpoint-zero mismatch rejection.
Existing strict/flexible-v2 scoring is reused unchanged.
