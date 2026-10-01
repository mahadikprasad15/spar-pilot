# 04: Produce paired comparisons and checkpoint trajectory

Type: task
Status: resolved
Blocked by: 03 - Run matched FP32 baseline and checkpoint evaluations

**What to build:** A CPU-only comparison operation that turns complete matched
evaluation artifacts into an auditable exploratory report and trajectory plots.

- [x] Validate variant identities, common inputs, runtime, prompts, decoding,
  scorer versions and full item alignment before calculating paired changes.
- [x] Persist joined per-item evidence and source hashes for baseline and all
  checkpoints; do not silently discard invalid or capped items.
- [x] Report strict/flexible-v2 correct counts, accuracy and 95% intervals,
  invalid/capped rates and mean/median generated response lengths.
- [x] Report paired accuracy changes, mean-length changes and mean-length ratios
  against the fresh baseline using the recorded pre-run interval procedure.
- [x] Preserve item pairing in resampling and state that intervals do not
  estimate variation across training seeds.
- [x] Produce structured results, a readable report and standalone trajectory
  plots; include gold-target lengths and inspectable paired responses.
- [x] Keep historical and older Pilot 1 measurements separate and descriptive;
  do not introduce a binary collapse gate or causal claims from length similarity.
- [x] Refuse complete-cohort statistics for partial or incompatible evaluations;
  repeated report generation verifies source provenance and preserves old evidence.
- [x] CPU tests cover alignment, paired calculations, denominator policies,
  incomplete cohorts, source consistency and saved report behavior.

## Comments

Approved as ticket 4. No GPU or model download is needed for aggregation.


## Answer

Implemented provenance-checked CPU reporting, paired bootstrap intervals, joined
responses and standalone trajectory plots. Two CPU tests pass, including a
format-only strict-score change and corruption/incomplete-cohort rejection.
