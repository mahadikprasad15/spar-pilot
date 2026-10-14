# 07: Capture, resume and durably verify raw evidence

**What to build:** One complete capture operation runs missing immutable batch/condition identities, resumes after a disconnect or partial write, and verifies required durable evidence before release.

**Blocked by:** 06: Profile and freeze the explicit production plan and predictions.

**Status:** ready-for-agent

- [ ] Public capture uses one model, includes the frozen random condition, shares only compatible base references, and saves all-site sums and declared retained differences.
- [ ] Interrupted and uninterrupted controlled runs yield the same accepted inventory/aggregates; only incomplete units repeat and valid source bytes remain unchanged.
- [ ] Temporary/staged writes cannot publish incomplete units. Resume checks unit hashes; corruption is rejected or repaired under an explicit policy without silent acceptance.
- [ ] Free-space preflight and actionable disk/write/sync failures preserve accepted units and suppress a premature release message.
- [ ] Durable synchronization reuses valid copied units; final verification checks required retained shards, identities and seals before announcing GPU release and waits for child exit.
- [ ] Show actual phase, condition, completed/sealed units, throughput and estimate; separate hash/sync work from forward progress.
- [ ] Reuse existing writer locks with one bounded public conflict check and a clear error; no separate broad contention framework is required.
