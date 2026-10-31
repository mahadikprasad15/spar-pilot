# 0018: Explicit sampling GPU selection

Date: 2026-10-31

The user selected an A40 after repeated L40S/A100 placement failures and approved
removing the L40S-only validation restriction. Hardware selection is explicit;
this does not qualify another GPU by analogy. Keep the actual GPU identity in the
runtime hash and verify it in the worker. Keep FP32, TF32 off, pinned engine
versions, score agreement and exact replay requirements. Training still requires
L4 to match gold SFT. CPU fixtures never establish GPU qualification.

The user also withdrew the original budget cap as an execution blocker on this
date. The existing budget runner still enforces its original policy; changing
that workflow is a separate implementation step. Do not claim it is disabled or
reset historical spending records.

## Explicit per-request cap waiver

The saved job approval may set `waive_budget_limits: true`, alongside a nonempty
reviewer/reason and hashes of the current ledger and exact request. Reservation
validates those bindings and stores the approval in the hashed event chain. The
same reservation can launch despite the original cap/80% checkpoint or another
unreconciled cost. Costs and outstanding reservations remain visible; the waiver
never reports historical charges as zero or marks them settled. Ordinary jobs
retain the original policy. A bounded scientific child still needs its matching
launched job, source identity and worker identity. Runtime deadlines, export and
stop handling remain mandatory. Training/generation gates are not changed by this
bounded-smoke amendment. Old volume recovery is deferred, not declared complete.
