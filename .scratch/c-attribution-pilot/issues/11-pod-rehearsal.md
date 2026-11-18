# 11: Attended Pod rehearsal

**Status:** needs-triage
**Type:** task
**Blocked by:** 10

**Needs the user's approval** of a rehearsal launch record (run ID `c-rehearsal-pod-1`).

- [ ] One Pod (A100-SXM4-80GB, 120 GB container disk, sl-0 attached, `HF_TOKEN` secret), one tmux window, one launch, no automatic restart. Follow a runbook derived from `.scratch/adapter-neuron-attribution/pod-runbook.md` with the lessons applied.
- [ ] Scope: 2 arms × 32 rows (covering EOS, capped, invalid and all-same groups; lesson r1: covering subsets), qualification checks (a)–(e), aggregation on the subset, a freeze, 2 conditions × 8 questions greedy and 1 condition × 4 questions × 8 sampled, one deliberate interruption with the local root deleted then resumed, export and clean restore, and a **real Pod stop** by the chain.
- [ ] Record facts, timings, peak memory and the qualified token budget for the production record. Every failure gets a lesson entry and a red/green fix before ticket 12.

## Comments

## Answer
