# 14: Held-out follow-up (prepared, not launched)

**Status:** needs-triage
**Type:** task
**Blocked by:** 12, 13

- [ ] **Exclusion list.** Built from **every recorded prior use**:
  - cohort150;
  - the production-1 calibration and rehearsal items;
  - this pilot's rehearsal IDs;
  - every other test-split question listed in `catalog/cohorts.yaml` `prior_uses` or in run records.

  Each exclusion source is listed with its evidence. An unknown exposure is surfaced, not assumed absent.
- [ ] **Question set.** A sealed held-out set (GSM8K test minus the exclusions, seeded), at a size the user chooses.
- [ ] **Design frozen first.** The design (conditions, modes, sampled and/or greedy, analysis) is frozen and hashed after the exploratory pilot's report and **before** any follow-up outcome is seen. Sampled interventions (8 draws under the attribution policy) are the direct test of J; greedy is reported separately.
- [ ] **Confirmatory status.** The follow-up is not labelled confirmatory until prior exposure and the frozen plan have been checked.
- [ ] **Launch.** A workload estimate from ticket 12's measured throughput, a separate launch record, and separate approval. No automatic launch.

## Comments

- 2026-11-18 (round 1): the exclusion list now comes from every recorded prior use; the design is frozen first; there is no automatic launch (F7).

## Answer
