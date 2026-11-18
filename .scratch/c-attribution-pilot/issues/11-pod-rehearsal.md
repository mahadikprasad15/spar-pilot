# 11: Attended Pod rehearsal

**Status:** needs-triage
**Type:** task
**Blocked by:** 10

**This needs the user's approval** of a rehearsal launch record (run ID `c-rehearsal-pod-1`).

- [ ] **Setup.**
  - One Pod: A100-SXM4-80GB, 120 GB container disk, sl-0 attached, `HF_TOKEN` secret.
  - One tmux window, one launch, no automatic restart.
  - Follow a runbook derived from `.scratch/adapter-neuron-attribution/pod-runbook.md` with the lessons applied.
- [ ] **Scope.**
  - 2 arms × 32 rows, i.e. 4 whole questions × 8 draws per arm, chosen to cover short, long, EOS, capped and invalid rows and all-same groups.
  - Qualification checks (a)–(f).
  - Aggregation on the subset; a split; a freeze.
  - **Greedy only:** 3 conditions × 8 questions.
  - One deliberate interruption, with the local root deleted, then resumed.
  - Export and clean restore, and a **real Pod stop** by the chain.
- [ ] **Question IDs.** All rehearsal questions are drawn from cohort150, and their exact IDs are recorded and added to the cohort's `prior_uses` in the catalog.
- [ ] **Measurements for the production record:**
  - attribution throughput;
  - per-token and per-sequence cached-against-forced errors;
  - identity and batch-invariance floors, and finite-difference floors;
  - the gross/net ratio distribution;
  - bootstrap benchmark tiers: wall time and peak RSS, extrapolated to 150 questions;
  - greedy throughput at the fixed batch;
  - peak GPU memory;
  - export and readback time.
- [ ] **Failures.** Every failure gets a lesson entry and a red/green fix before ticket 12.
- [ ] **Measurements only.** The rehearsal proposes tolerance values and caps for the user's decision; it does not set them.

## Comments

- 2026-11-18 (round 1): greedy only. Rehearsal IDs are recorded. The rehearsal measures the budget and floors instead of assuming them (F4, F5, F7).

## Answer
