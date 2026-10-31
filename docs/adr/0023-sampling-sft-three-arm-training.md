# 0023: Explicitly selected three-arm training

Date: 2026-11-01

The user selected k0, k2 and k5 for separate seed42 training attempts. The
previous endpoint-only k0/k5 default is superseded for a new, explicit post-gate
selection; it does not retroactively change any old selection, completed run or
training result. The three arms are not merged and are not run concurrently on
one GPU. Existing stored response IDs, eligibility, omission and batch-order
contracts apply identically to k2.

This decision does not label ambiguous responses or waive the coverage/data gate.
Pending intended-answer labels need actual human review; more than five uncovered
problems in any arm blocks training. Full T15/T16 readiness, gold-matched L4 FP32
runtime, separate plan/job identity and measured report/selection remain required.
The user's explicit budget waiver may be recorded against the current ledger and
request; a waiver does not erase reservations or actual costs. Preserve all raw
sampling and training artifacts and independently verify exports before any Pod
stop. No claim of a completed model is made from CPU preparation or fixtures.
