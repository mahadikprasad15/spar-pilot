# 12: Production run

**Status:** ready to launch: record approved 2026-11-22; the Pod run is next
**Type:** task
**Blocked by:** 11

**This needs the user's approval** of the production launch record:

- a new run ID;
- tolerances (production-1 tolerances unchanged; new ones set from the measured floors);
- seeds;
- the hot-layer list and its provenance;
- the resolved open decisions.

- [ ] **Launch.** A fresh launch under the new run ID, with no handover from the rehearsal. Attended from the start through qualification sign-off; unattended afterwards, with the watchdog.
- [ ] **Scope.**
  - Attribution over 2 × 1,200 rows.
  - Aggregate, split, bootstrap and freeze.
  - **Greedy only, on cohort150:** 2 × 9 × 150 = 2,700 answers, + 150 untuned (proposed), + 300 intact repeat [user, 2026-11-19] = 3,150.
  - Report, verified copy and export, Pod stop.
- [ ] **Not in this launch:** sampled ablations, top-10 read ablations, held-out questions.

## Comments

- 2026-11-18 (round 1): the scope was updated from 15 × 150 to 9 conditions per arm, greedy only. The budget comes from the rehearsal (F5, decision 2).

## Answer

- 2026-11-22 (user): the production launch record `.scratch/c-attribution-pilot/launch-record-production.yaml` is approved.
  - Run ID `c-production-1`.
  - Cached-vs-forced tolerances 1e-3 per token and 3e-3 per row, provisional.
  - The untuned baseline is included: 3,150 answers.
  - Everything else is the rehearsal's.
  - Bundle `inputs-v2` (`ec3062d2`). Launch per `pod-runbook.md` section 10.
