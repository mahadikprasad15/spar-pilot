# Ticket13: initial gradient pushes and training alignment

## Scope implemented

Public saved-workflow commands:
`stage0-pushes-{prepare,check,runtime,run,forecast,freeze,status,
verify-measurement,report,predict,reveal,verify,recover-writer}`.
CPU34–36 / GPU32–37 extend the existing shared notebooks.

- Initialization audit hashes exact A₀ payload bytes, cross-checking the original
  source audit. Three objectives per distinct A₀; no cross-A₀ comparisons.
- Matched eligible gold/own-correct-one questions and batches; GRPO keeps all512
  and dead groups. Original supervised/generated-token/end-turn contracts remain.
- No optimizer. Raw gradients and negative-gradient pushes saved per module and
  batch. Global masked CE/DAPO token denominator survives memory microbatches.
  Detached historical/group-normalized advantages; zero-advantage exact zeros.
- Lazy local FP32 model, source-payload checks, no TF32/autocast, checkpointed
  vocabulary head. Independent FP64 output-gradient×A₀ reconstruction requires
  relative error<1e-5; A-gradients exactly zero. Tiny CPU finite difference keeps
  epsilon1e-4 and relative error<1e-3. No acceptance number changed.
- One separately allocated batch per condition precedes a measured forecast and
  distinct production freeze. All costs share the cumulative20-unit ceiling.
- First gradient profile freezes its own session; completed R0 tokens remain
  immutable data and need not be regenerated after the sampling session ends.
  Gradient batches from different sessions are never silently joined.
- Resume skips valid sealed batches; changed sources, payloads, runtime and failed
  full integrity stop before collection. Graceful failure attempts a remaining-
  budget-bounded full integrity check; hard kills remain visibly unreconciled.
- Primary batch means with/without partial batch; full-batch-only consistency;
  token-weighted secondary means; principal output projection/chance; same-A₀
  direction-only comparisons. No cross-objective raw-norm interpretation.
- Separate prediction freeze after push summary and before trained endpoint
  reveal. Prior exposure forces exploratory interpretation. Paired gauge
  orientation, sign/ordinary push cosines, initialization A alignment, zero-B
  undefined alignment, and Gaussian controls. No causal/AdamW claim.
- Raw NPZ/JSONL/CSV, structured summaries, HTML, runtime/progress/errors, prediction
  and exposure receipts, and completion/verification graph.

## Development evidence

- Startup existing spectral/R0 tests:22 passed in78.57s.
- Initial math tests red before implementation. Two then passed in the working
  CPU PyTorch environment. Batch/mask test red before plan implementation.
- Actual collector tested against independent unchunked tiny autograd for CE and
  GRPO, plus dead sequences. Five initial tests passed in8.76s.
- Public workflow test red before CLI wiring; then exposed missing full integrity
  on graceful interruption. Added bounded failure readback, keeping gates fixed.
- Session-restoration test red on an overly strict sampling-session dependency;
  then passed with a separate frozen gradient runtime. Initial two public tests
  passed in46.82s.
- Missing-profile-approval test exposed a generic reservation KeyError; added an
  explicit actionable rejection before loading the model. No automatic approval.
- Notebook boundary test red before handoff cells, then two notebook checks
  passed in0.08s. Actual CPU Torch checks plus notebook test:6 passed in3.12s.

Final regression and retained validation evidence are recorded below at closeout.

## Execution boundary

No research-model download, CUDA execution, Colab spend, Drive upload or GitHub
push. Tiny CPU software checks do not qualify real GPU gradient collection.
The default local Python has an incomplete Torch library; actual Torch-backed
checks run separately in the existing working fastai environment (Torch2.7.1),
while source/workflow tests run in the regular CPU environment. Colab still uses
its mandatory pinned dependencies; no local or GPU acceptance gate is widened.

## Final validation, 27 October2026

- Final public/math/notebook suite: **7 passed, 3 skipped in54.49s** in the
  regular CPU environment. The three Torch-backed tests are separately executed
  in the working CPU environment: **6 passed, 4 deselected in3.02s**, including
  those three actual Torch tests. Together all **10 unique Ticket13 tests pass**.
- Shared source/workflow, spectral, R0 and qualification/budget regression:
  **42 passed in160.21s**. Total52 unique affected tests; no full-legacy-suite claim.
- Independent FP64 oracle: reconstruction relative error4.73e-17 and directional
  finite-difference relative error4.94e-10. Exact A-zero, dead-group and global
  microbatch-denominator checks passed; approved gates unchanged.
- Passing public fixture retained at `artifacts/validation/stage0-ticket13/passing/`:
  174 gradient batches, three conditions, raw NPZ/JSONL, means, CSV/HTML, prediction
  and reveal receipts. It is explicitly synthetic. Fresh CPU verification passed
  again; the public test also verifies with Torch/Transformers/PEFT imports blocked.
- Red/green logs and the actual tiny CPU oracle receipt are retained alongside
  the snapshot. Generated artifacts remain local canonical validation data.
- Both notebooks parse without outputs. Whitespace checks pass. No GPU run,
  model download, Colab spend, Drive upload or GitHub push.

## Fresh-kernel handoff correction

A final execution check (not just AST parsing) exposed an existing CPU section2
NameError: QUALIFICATION_NAME was first defined in a later section. Defined it
locally before source selection. The fresh-kernel test was red on that error;
then three notebook/handoff tests passed in0.11s. This adds one unique passing
test: **11 Ticket13/handoff tests, 53 unique affected tests** across the recorded
runs. No experiment code, source product, numerical gate or saved measurement
changed in this correction.
