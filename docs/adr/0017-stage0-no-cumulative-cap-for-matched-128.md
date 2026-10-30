# Stage 0: no cumulative unit cap for the matched-128 GPU jobs

Status: decided by the user on 2026-10-30, before any E0.6a/E0.5 GPU unit was measured.

The spec set a cumulative Stage 0 ceiling of 20 Colab units and said to stop and ask
before exceeding it. On 2026-10-30 the operator recorded the real cumulative reading of
21 units, so every new reservation was refused. Asked, the user confirmed that 21 is
the true reading and removed the overall cap. The ceiling was costing GPU time rather
than protecting anything.

Decision:

- E0.6a distance and E0.5 push jobs reserve through `pilot_eval/stage0_job_budget.py`.
  It keeps every other check: a current cumulative reading, a reviewed approval with a
  session, a wall-time limit and a units-per-hour rate, one active GPU reservation at a
  time, and settlement against a newer reading. It does not compare the running total
  with a ceiling. Each reservation row records `budget_policy`.
- The usage ledger is unchanged. It stays append-only and cumulative readings cannot
  decrease, so it remains a true record of what Stage 0 cost.
- Their plans record `budget_ceiling_units: null` and `budget_policy`. They are new
  plans, because the code change alters their implementation fingerprints. The
  `-matched128-distance-v1` and `-matched128-pushes-v1` plans had measured nothing and
  are superseded by `-v2`.
- `stage0_budget.py`, `stage0_workflow.py` and `stage0_evaluation.py` are untouched.
  Completed products that fingerprint them (E0.4 behaviour, batching, drift, findings)
  keep verifying. Other capped jobs (R0, energy, trajectory) still refuse at 20 units.
  Running any of them now needs its own decision.

Real protection now comes from the per-job profile, forecast and frozen wall-time caps.
