# 01: Restore a clean session, audit sources and smoke-test Colab

**What to build:** A fresh readable notebook restores Pilot 5 state, audits existing sources, declares stable named-site and report/optional-grade contracts, and demonstrates environment/storage readiness before larger slices depend on Colab assumptions.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Public restore/inventory operations on an offline fixture show actual missing/pending/completed states without starting a model or older pilot.
- [ ] The site contract distinguishes embedding output and block indices zero through 27, pre/post normalization and content/absolute positions.
- [ ] CPU restoration works without importing GPU libraries; actual GPU text-model imports and optional package conflicts are checked in a bounded fresh-process path.
- [ ] A recorded minimal fresh-Colab check mounts durable storage, writes/verifies a small artifact, disconnects/reconnects and reads it through the public restore path. Actual time is recorded, not promised.
- [ ] A changed runtime, owned active child and ambiguous source selection fail with actionable messages before inference.
- [ ] Limit the early grade contract to a schema version and not-requested/pending/validated status. Detailed grade fields belong to grading/report consumers, not this bootstrap ticket.
- [ ] Define supported analysis capabilities without selecting outcome rules: per-position cohort means at all sites; whole-example primary-window pooled evidence at all sites; per-position example evidence at the six retained blocks. Other data demands require a separately reviewed capture extension.
- [ ] Distinguish offline CPU completion from real-smoke evidence: this ticket cannot be reported as fully target-validated on CPU alone. The real GPU allocation and smoke duration are recorded.
