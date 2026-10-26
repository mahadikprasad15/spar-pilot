# Ticket 04 implementation evidence

Date: 2026-10-26. Software and notebook workflow implemented; real Colab
qualification and the two evidence reviews remain pending.

## Delivered

- Public CLI stages prepare, record runtime/storage preflight, run, verify,
  approve the evidence-derived numerical allowance, and freeze a separately
  budgeted paired evaluation plan.
- Real CUDA loading is lazy and local-files-only. It requires the audited FP32
  model snapshot, pinned dependencies, one visible GPU, matching tokenizer and
  source identities, deterministic settings and a durable artifact-root
  readback before inference.
- The bounded run evaluates the first 16 items from the hash-pinned ordered
  Pilot 4 evaluation cohort for SFT V1 and GRPO V5. It captures teacher-forced
  logits over all answer-prediction positions, greedy completions, and a
  deliberate double-adapter negative control across 32 independently sealed
  units. Both real and negative worst-error coordinates and per-position
  maxima are saved.
- The hard 1e-3 error stop, >=1000x negative-control sensitivity, and strict
  common-prefix near-tie rule are implemented. The 3E allowance is proposed
  from observed evidence and needs its own review receipt.
- A hash-chained global reservation protects the 20-unit ceiling. Interrupted
  work remains reserved until a newer cumulative reading is recorded and
  reconciled. Resume requires the same recorded runtime; a new runtime needs a
  separately named qualification so paired evidence is not mixed.
- The GPU notebook exposes the workflow with explicit operator fields and a
  separate numerical approval and production-budget/freeze stage. The frozen
  production plan is not represented as a completed measurement. KL and write
  measurements need later reservations.

## Red/green and regression evidence

The qualification boundary tests first exposed missing public commands, then
covered the saved workflow. A follow-up test intentionally failed because
negative-control coordinates were absent; the math output now retains those
coordinates and per-position maxima.

Final affected regression command:

```text
pytest -q tests/test_stage0_qualification.py tests/test_stage0_edits.py tests/test_stage0_spectral.py tests/test_stage0_workflow.py tests/test_trace_session.py tests/test_notebook_session.py tests/test_behaviour_breakdown.py
```

Result: **90 passed in 145.10 seconds**. The suite uses CPU fixtures and does
not download a model or establish real GPU qualification. Python compilation,
notebook code-cell AST parsing, and `git diff --check` also pass.

## Remaining real-run prerequisites

- Restore the exact original Pilot 4 evaluation config at
  `artifacts/reports/pilot4-results-so-far-2026-10-13/inputs/evaluation-config.json`
  (the run plan checks its approved SHA256).
- Restore the verified Ticket 03 edit catalogue and audited model/tokenizer
  snapshot under the artifact root.
- In Colab, verify current cumulative Stage 0 usage and enter a bounded unit
  rate and wall-time reservation. No real GPU work has been started by this
  implementation task.
- Review the saved numerical evidence and approve its exact proposed 3E gate;
  then review the measured production forecast and reserve it under the same
  20-unit total ceiling.

The design and implementation do not imply that these runtime prerequisites,
qualification evidence, production freeze or scientific measurements already
exist.
