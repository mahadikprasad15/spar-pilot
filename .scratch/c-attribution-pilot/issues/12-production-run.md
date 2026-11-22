# 12: Production run

**Status:** done (2026-11-22, A100 Pod): complete, verified backups, Pod stopped by the chain
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

- 2026-11-22: `c-production-1` ran fresh on one A100-80GB from `fe8f1e7`, 08:43–10:21 UTC, wall 5,718 s, one attempt,
  no resume. Restored cleanly from HF into `artifacts/imports/c-attribution-production[-final]/`. Receipts are in
  `artifacts/imports/receipts/c-production-1.*.hf-upload.json`, and measurements in
  `artifacts/imports/measurements/c-production-1.json`.
  - **Qualification:** 14/14 checks pass; sign-off `accept` by the automatic rule. Cached vs forced: per token max
    1.21e-4 (online), 8.1e-5 (offline); per row max 2.82e-4 and 2.05e-4, all equal to the rehearsal. Batch
    invariance 2.2e-5 and 4.2e-5, against 1e-4.
  - **Stage seconds:**

    | Stage | Seconds |
    |---|---|
    | Preflight | 9 |
    | Qualification | 236 |
    | Rows | 3 |
    | Attribution | 933 |
    | Aggregate and freeze | 22 |
    | Generation | 4,500 |
    | Report | 2 |
    | HF export | 104 (1.35 GB, 943 files, one attempt) |

  - **Attribution:** 75 units per arm; 934 (online) and 859 (offline) tokens/s in steady units.
  - **Greedy:** 2,250 answers in 150 batches (median 28 s), not 3,150. For both arms, `match-global64`,
    `match-hot64` and `match-layer64` were skipped as `matched-control-unavailable`, as the spec prescribes
    (eligibility is never loosened). Some selected reads have no eligible candidate:

    | Group | Online deficient targets | Offline deficient targets |
    |---|---|---|
    | global64 | 11 of 64 | 21 of 64 |
    | hot64 | 4 of 256 | 6 of 256 |
    | layer64 | 22 of 1,792 | 40 of 1,792 |

    The read targets ran, with their paired Δ against intact but no target − control contrast. `match10-adapters`
    ran in both arms.
  - **Intact repeat:** 0 questions differ in either arm.
  - **Pod stop:** `runpodctl stop` returned 0. The closing records (`done`, `stop_record.json`,
    `last_attempt_end.json`) are on the volume only and have not been listed yet. The user's paste shows the stop
    record.
