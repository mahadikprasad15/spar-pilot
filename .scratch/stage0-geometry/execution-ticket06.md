# Ticket 06 execution record

## Software delivered

- Public saved workflow: `stage0-drift-prepare`, `run`, `status`, `verify`,
  `recover-writer`. Thin shared notebooks: GPU sections 16–19, CPU 15–16.
- Original fixed 150 GSM8K gold + 150 FineWeb token sequences; ten primary
  interventions; bounded untuned/SFT-zero/GRPO-zero controls on four documents.
- FP32 local-only one-model loader, exact payload checks for twelve adapters,
  same qualified session, TF32 off. Real inference singleton; eight-document
  durable save groups. Default 383 units; 3,000 main variant/doc combinations
  plus reference/restoration and bounded diagnostic forwards.
- Full-vocabulary directional KL in bounded chunks, FP64 reduction; explicit
  next-target mapping; signed roundoff retained under inherited arithmetic bound.
- Actual module branch/base output sums, unshifted views and positions;
  per-document/module sums and actual factor-norm products. Module RMS ratio,
  per-token mean ratios, equal-document means remain distinct.
- Strict switch/restoration, zero, repeat/chunk, coefficient/branch reconstruction
  and model-integrity checks. No scientific acceptance numbers changed.
- Sealed resumable groups, frozen source/runtime/tokenizer/settings, failed
  observations, attempt-time seals, cumulative budget ceiling and explicit
  writer recovery. GPU greedy allocation cannot fund drift.
- CPU reconstruction verifies all sealed scalar arrays and document aggregates.
  No historical values enter the new measurements. Provenance supports future
  compatible behavioral joins; this ticket makes no such claim.

## TDD and checks

- Initial public workflow test failed because drift commands did not exist.
- Factor-norm test failed before single-payload-pass norm reader existed.
- Notebook contract test failed before fixed-context stages were added.
- Independent full-vocabulary/directional KL oracle, target shift, general dense
  merged-delta write oracle, document weighting, and actual real-backend hook /
  output-head chunk path tested with an independent tiny NumPy model.
- 16 focused tests passed in 20.80s. Real model imports are blocked in the fresh
  CPU-verifier check. Unit fixtures neither download a model nor use a GPU.
- Earlier in-development run rejected changed source identities while files were
  being edited (3 failures, 7 passes). Fresh final focused run is authoritative.
  No failing acceptance threshold was relaxed.

## Execution still required

Real Stage 0 source/weight preparation, loader qualification, numeric review,
execution freeze and bounded GPU jobs have not been run locally. Ticket 06 is
software-complete, not experimentally complete. Inspect and reconcile any greedy
allocation before approving a separate drift bound. First collect the three
controls plus one SFT group, inspect actual time, then finish within an approved
remaining bound. Ticket 07 will compute paired intervals and human-review-aware
combined findings.

The notebook source pin will be recorded at release closeout. No GitHub push or
Drive upload is implied by this record.
