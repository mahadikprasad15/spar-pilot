# 07: Capture, resume and durably verify raw evidence

**What to build:** One complete capture operation runs missing immutable batch/condition identities, resumes after a disconnect or partial write, and verifies required durable evidence before release.

**Blocked by:** 06: Profile and freeze the explicit production plan and predictions.

**Status:** claimed (software implemented and CPU-tested; real execution and prerequisite approvals pending)

- [ ] Public capture uses one model, includes the frozen random condition, shares only compatible base references, and saves all-site sums and declared retained differences.
- [ ] Interrupted and uninterrupted controlled runs yield the same accepted inventory/aggregates; only incomplete units repeat and valid source bytes remain unchanged.
- [ ] Temporary/staged writes cannot publish incomplete units. Resume checks unit hashes; corruption is rejected or repaired under an explicit policy without silent acceptance.
- [ ] Free-space preflight and actionable disk/write/sync failures preserve accepted units and suppress a premature release message.
- [ ] Durable synchronization reuses valid copied units; final verification checks required retained shards, identities and seals before announcing GPU release and waits for child exit.
- [ ] Show actual phase, condition, completed/sealed units, throughput and estimate; separate hash/sync work from forward progress.
- [ ] Reuse existing writer locks with one bounded public conflict check and a clear error; no separate broad contention framework is required.

## Approved Ticket 03 data dependency (A6–A7)

Carry per-document block-14 positions 0–4 through durable capture for whole-document bootstrap. For the optional activation-shaped null, capture base block-14 positions 1–127: FP64 Σh and Σhhᵀ plus count 25,400 (200 × 127), bound to source/runtime/site identities. At width 1,536 the second-moment matrix alone is 18 MiB; profile accumulation and transfer costs. Qualification owns numerical allowances. Missing moments mean this secondary null is unavailable; never fabricate covariance evidence or silently repair a failed matrix.

## Implementation handoff · 2026-10-16

See `TICKET_07_08_REPORT.md` and `docs/pilot5-capture-and-readouts.md`. Public saved-workflow commands and notebook stages are implemented with synthetic CPU execution evidence. No actual production or readout result is claimed. Ticket 06 qualification/scope/budget/predictions and Ticket 05 lens disposition remain explicit gates.
