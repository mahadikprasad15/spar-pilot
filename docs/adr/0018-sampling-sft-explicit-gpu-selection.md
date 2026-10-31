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
