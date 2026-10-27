# Ticket14: base-input activation-energy decomposition

## Implemented scope

Public `stage0-energy-{prepare,predict,runtime,run,forecast,freeze,status,
verify-measurement,report,verify,recover-writer}` saved-workflow commands.
CPU37–39 / GPU38–43 extend the shared notebooks; setup and source selection are
restored from saved plans rather than in-memory objects from earlier cells.

- Original150 GSM8K/150 FineWeb token sequences and saved unshifted masks.
  Inputs are taken from the untuned base model, with its adapter disabled.
- SFT64, GRPO64, both saved step0 input factors, and the existing random64 control;
  k30/k50/k70 from the verified spectral cache. No new random adapter is generated.
- One base body forward per sequence supplies all selected factors/cutoffs.
  No output-head projection, backward, optimizer, tuned-arm forward or hidden
  repeat sweep. Singleton model scheduling preserves the existing qualified
  batching convention; eight sequences per durable save group by default.
- FP64 direct contractions for parallel, rest, twice-cross and independently
  computed total (a·x)². No full activations persisted; per-example sufficient
  sums retain views, modules, factor/cutoff identities and counted-token counts.
- User-corrected atol1e-5/rtol1e-5 checks: energy reconstruction, VᵀV identity,
  normalized parallel/rest orthogonality, normalized norm partition, and principal
  factor norm fraction against Ticket02 π_in using its projection implementation.
  Zero-factor denominators are undefined, not fabricated zeros.
- Failure preserves structured diagnostics and failed arrays before stopping.
  Frozen projected factors, original source hashes, qualification review, spectral
  cache, corpus and code identities are verified again on resume.
- Two bounded profile groups (one per corpus) precede the measured forecast and
  separate production numerical/budget review. Profile arrays are reused.
  Full cohort needs300 base forwards, not300×five factor conditions.
- First profile freezes its own recorded runtime; later energy groups cannot
  silently switch GPU/session. Qualification is a prerequisite, not a join of
  historical energy numbers. Cumulative Stage0 ceiling remains20 compute units.
- Raw NPZ, JSONL/config/projection receipts, progress/status/errors, attempts and
  integrity seals. Valid sealed groups are skipped after interruption; no
  optimizer or weight state changes.
- CPU pooled-token component ratios per module/corpus/view; bounded headline
  parallel/(parallel+rest), signed interference and undefined zero totals.
  Geometric π_in/π_out/chance context and descriptive equal-module layer/type
  summaries, CSV/JSON and HTML/SVG figures. No causal-damage claim.
- Hypotheses frozen before capture, with prior-exposure declaration and existing
  exposure receipts forcing exploratory interpretation. Scientific raw values
  remain distinct from implementation fixtures.

## Development evidence

- Session baseline: Ticket13 suite8 passed,3 Torch skips in58.39s.
- Independent cancellation/reinforcement, denominator and unshifted-mask tests:
  red before implementation, then2 passed in0.06s.
- Normalized projection test red before implementation, then3 math checks passed.
- Public workflow red before CLI wiring; then4 passed in30.53s, including full
  profile/freeze/interruption/resume/CPU report/payload rejection.
- Actual tiny-model hook seam red before extraction, then4 checks passed in2.69s
  in the working CPU Torch environment, including one disabled-adapter body
  forward per sequence and independent NumPy contractions.
- Expanded workflow failures:8 passed,1 Torch skip in45.94s, covering failed gate
  diagnostic retention, missing budget, changed session/projected factors,
  projection failure before model load and cumulative20-unit refusal.
- Notebook commands/AST checks passed. A stronger handoff-name test caught
  CPU/GPU default-name divergence and passed after correction.

## Related source-kind correction

A real spectral inventory is labelled `cpu-local-pinned-weights` because auditing
weights does not construct a model. Ticket13 previously compared that label
literally with R0's `real-cuda` execution kind. Added an explicit compatibility
mapping, preserving fixture/real separation. Regression test red then green.
This corrects a preparation rejection; no existing scientific result is changed.

## Execution boundary

No model download, CUDA execution, Colab spending, Drive upload or GitHub push.
Only CPU fixtures and an actual tiny CPU model ran. The default local Python has
an incomplete Torch binary; actual hook tests use the existing working CPU Torch
installation. Colab dependency pins and all numerical gates remain unchanged.

Final regression results, validation locations and release pin follow below.

## Final verification

- Ticket13/14 public/math/notebook suite: **18 passed,4 skipped in131.68s**.
  Skipped Torch checks all ran separately in the existing working CPU environment:
  energy checks **5 passed in2.71s** (including the actual hook test), and all
  three push/autograd checks **3 passed in3.31s**. All22 unique ticket tests pass.
- Existing spectral, drift, qualification/budget and workflow regression:
  **45 passed in129.16s**. Total67 unique affected tests across these recorded
  runs; no full legacy-suite claim.
- Passing snapshot retained at `artifacts/validation/stage0-ticket14/passing/`.
  Independent verification confirms300 synthetic base forwards, five factor
  conditions and complete CPU analysis. Original raw sums, projection checks,
  predictions, source artifacts, budget/attempt evidence, tables and SVG plots
  remain available. Explicit synthetic labels prevent research-result confusion.
- Corrected CPU/GPU default plan naming verified by a red/green handoff test.
  Both notebooks contain clean parseable cells and no saved outputs.
- Final preparation progress-only addition is verified below; it does not alter
  factors, reductions or gates. New validation snapshot selected for the final
  implementation identity rather than overwriting the preceding one.

- Final energy-only suite after preparation progress output: **9 passed,1 skipped
  in46.32s**. The skipped actual hook test passed separately; fresh verification
  of `artifacts/validation/stage0-ticket14/final-passing/` passed. No scientific
  result is implied by the synthetic snapshot.

- Scientific implementation/shared-notebook release pin: `3fb71062111e8c7c1dd29c15ac7650ae3ec1a6fe`.
