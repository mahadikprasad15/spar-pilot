# Ticket 05 implementation evidence

Date: 2026-10-26. Software implemented and CPU tested; real-GPU collection pending.

## Delivered

- `stage0_evaluation`: public prepare/run/status/verify/recover-writer workflow
  consuming the exact Ticket 04 frozen execution and approved qualification.
  Source, adapter, cohort, scorer, scheduling, implementation and runtime
  identities are checked before expensive work. Completed sealed units are
  verified and reused; unsealed partial writes are recomputed.
- `stage0_evaluation_backend`: one lazy local-files-only FP32 CUDA model with
  all ten verified PEFT variants attached. Fresh SFT/GRPO references share the
  runtime, exact prompt IDs and ordered generation schedule with every edit.
  No optimizer, training replay or model download occurs.
- `stage0_evaluation_records`: independent CPU token/EOS/cap validation,
  strict/flexible-v3 scoring and descriptive summaries. Raw generated IDs
  include EOS; content lengths exclude it. An EOS in the last permitted slot
  remains an EOS stop; capped responses receive zero reward.
- Singleton real inference follows Ticket 04's qualified path. Eight sequential
  inference batches form a durable unit by default, giving 190 seals for 1,500
  responses. The final group has six examples. This storage choice does not
  change numerical batching or the per-input seed schedule. Larger CUDA batches
  require their own agreement evidence; larger CPU fixtures do not qualify them.
- Generation progress distinguishes outputs pending a seal from durable
  completed responses. Base SHA256 checks bracket clean attempts; parameter
  identity/version guards surround every sealed group. A known failed final
  hash invalidates all units from that attempt. Actual prompt IDs are checked
  against the saved qualification inputs, not merely prompt lengths.
- The initial run consumes the frozen production reservation. Attempts retain
  sealed timing/start evidence and deduct their elapsed time on resume. A killed
  attempt with unknown duration cannot silently reset its allowance. New bounded
  approval follows explicit cumulative usage reconciliation; the total 20-unit
  ceiling and same-runtime requirement remain in force.
- GPU notebook sections 12–15 expose prepare, collect/resume, usage
  reconciliation and CPU verification. CPU notebook sections 13–14 restore
  status and inspect the same raw results. Explicit optional pinned dependency
  setup and fresh-child model-class imports fill the GPU notebook's earlier
  setup gap; no installation or model import is added to CPU analysis cells.

## Red/green and verification

The first public workflow test failed because evaluation commands were absent.
The separate save-grouping test then failed on the missing public parameter,
and the notebook test failed before the handoff cells existed. Each behavior
was implemented before the corresponding green run.

The tests cover complete 1,500-record collection, partial and unsealed writes,
same-session missing-unit resume, source/response alteration, runtime mismatch,
strict/flexible extraction, cap/EOS limits, actual prompt-ID agreement,
exclusive ownership and explicit recovery, cumulative wall time, unknown
attempt duration, the 20-unit ceiling and failed final integrity. A stub model
exercises the real backend's adapter choice, greedy arguments, padding/masks,
termination and content/raw token handling without optional model imports.
A fresh Python process blocks Torch/Transformers/PEFT imports while verifying
the completed saved product.

Affected regression command:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_stage0_evaluation.py tests/test_stage0_qualification.py tests/test_stage0_edits.py tests/test_stage0_spectral.py tests/test_stage0_workflow.py tests/test_trace_session.py tests/test_notebook_session.py tests/test_behaviour_breakdown.py
```

Result: **100 passed in 189.30 seconds**. An additional prompt-ID identity test
passed separately (**1 passed in 0.10 seconds**). Limiting native CPU test threads
does not change the frozen GPU protocol or acceptance gates. Python compilation,
clean notebook code-cell AST checks and `git diff --check` pass.

## Durable validation and remaining execution

`artifacts/validation/stage0-ticket05/verification.json` records structured
software evidence. The same directory retains `paired-fixture-v1/`, an
explicitly synthetic complete 1,500-response result, copied and independently
verified under its canonical retained path. These adapters and outputs are CPU
fixtures, not trained-model results or real GPU qualification.

Before real execution, publish the tested notebook release, complete the actual
source/cache/edit workflows and the real Ticket 04 diagnostic, approve its
measured numerical gate and production budget, then collect with the same
recorded GPU session. Use the shared current release before qualification;
switching code/runtime after qualification is not silently accepted.

Ticket 05 emits raw automatic endpoint results only. Ticket 06's KL/effective
writes and Ticket 07's paired intervals, targeted human-review overlays and
non-inferiority interpretation remain separate implementation/execution work.
No real GPU work, model download, source-adapter mutation or Drive backup was
performed by this implementation task.
