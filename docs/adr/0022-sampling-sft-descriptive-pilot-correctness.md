# 0022: Pilot correctness is descriptive for generation

Date: 2026-11-01

The user explicitly retired the proposed 90% initial-attempt correctness alarm
as a prerequisite for full sampling and explicitly waived the budget caps for
this work. The measured pilot remains 19/32 correct under flexible-v3. This is a
policy decision; it does not change that measurement, the scorer, the saved pilot
report, or the correctness of any response. The old report and its historical
alarm remain available as originally written.

A new, explicit generation review bound to the exact generation plan and pilot
report may select `descriptive-correctness-v1`. The review records correctness
as descriptive and records any authorized budget waiver as `waived`, never as a
passed budget alarm. Actual spending and reservations remain in the original
ledger and each job still uses the existing request-bound authorization and
reservation lifecycle. A historical review without this policy retains the old
behavior. No unrecorded or implicit policy transition is permitted.

Exact real kill/restart replay, T3, selected runtime, source checks, the 2%
unfinished initial-attempt limit, per-attempt eligibility and later coverage/data
review are unchanged. The generation decision is not approval to train. If the
machine or runtime changes, apply the existing qualification requirements; a
machine-critical warning is not a finding about the source of numerical drift.
