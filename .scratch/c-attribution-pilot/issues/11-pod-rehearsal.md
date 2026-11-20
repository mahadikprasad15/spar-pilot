# 11: Attended Pod rehearsal

**Status:** needs-triage
**Type:** task
**Blocked by:** 10

**This needs the user's approval** of a rehearsal launch record (run ID `c-rehearsal-pod-1`).

- [ ] **Setup.**
  - One Pod: A100-SXM4-80GB, 120 GB container disk, sl-0 attached, `HF_TOKEN` secret.
  - One tmux window, one launch, no automatic restart.
  - Follow `.scratch/c-attribution-pilot/pod-runbook.md` (written 2026-11-20 from the neuron-attribution runbook, lessons applied).
- [ ] **Scope.**
  - 2 arms × 32 rows, i.e. 4 whole questions × 8 draws per arm, chosen to cover short, long, EOS and invalid rows and all-same groups where available. Include the offline capped row. Online has no capped rows; record this rather than requiring one.
  - Qualification checks (a)–(f).
  - Aggregation on the subset; a split; a freeze.
  - **Greedy only:** 4 conditions × 8 questions (intact, top10-adapters, global64-reads, untuned).
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
- [ ] **Measurements only.** The rehearsal proposes tolerance values for the user's decision; it does not set them.

## Comments

- 2026-11-18 (round 1): greedy only. Rehearsal IDs are recorded. The rehearsal measures the budget and floors instead of assuming them (F4, F5, F7).

## Answer

- 2026-11-20 (user): the rehearsal record is approved.
  - Questions: attribution 817, 611, 285 and 186 for both arms; generation adds 13, 1116, 538 and 1300.
  - Matching seeds 20261121–24.
  - `noop_greedy_tokens` 16.
  - Conditions: intact, top10-adapters, global64-reads and untuned. Untuned was added so the adapter-off
    path runs on a GPU before production.
  - Cost is not tracked (`hourly_cost_usd: not-tracked`).

## Notes for the runbook (2026-11-20, from the 06–10 review fixes)

- **Install exactly the pinned packages.** `torch==2.9.0` from the cu128 index brings its own CUDA runtime,
  cuDNN and the other `nvidia-*` libraries, and those are part of the resume fingerprint. On a replacement Pod,
  install the same way. A resume refusal names any package that differs and the `pip install` that fixes it.
  The NVIDIA driver is recorded in `preflight.json` (`host.driver`) but never refuses a resume.
- **Replacement Pod.** Use this if the Pod is lost mid-run. The sealed work is on the volume, mirrored at every
  unit.
  1. Make sure the old Pod is really gone. A fresh heartbeat refuses the takeover, and the marker must be older
     than `marker_stale_seconds`.
  2. On the new Pod, set up as usual (same GPU type, same install, `HF_HOME` on the container disk).
  3. Reseal the record for this Pod:
     `python -m pilot_eval.c_attribution_chain rebind --record OLD.json --out NEW.json [--work-dir ...]`
     (it reads `RUNPOD_POD_ID`).
  4. Launch with `NEW.json`. Completed stages are re-verified and skipped. Preflight must match the qualified
     setup, or it refuses and names what to install or rent.
- **Stop withheld.** If the chain prints `STOP WITHHELD`, no backup holds all the local work and the Pod is left
  running (billing). Fix the volume or Hub problem, then relaunch with the same record on the same Pod; or stop
  it by hand. Details are in `chain/stop_withheld.json`.
- **Exercise in the rehearsal:** one rebind onto a second Pod after the deliberate interruption, if the budget
  allows; it costs one extra Pod setup (about 10 minutes).
