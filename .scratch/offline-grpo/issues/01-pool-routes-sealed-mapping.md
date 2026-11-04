# 01: Pool routes and sealed mapping

Status: resolved
Blocked by: None

## What to build

Inventory available historical pool evidence and support an evidence-selected existing-full-pool, reuse-and-extend, or fresh-512 route using the existing sampling workflow. Preserve historical evidence. Seal identified draws and map them to the saved optimizer windows; recompute frozen rewards and summarize each window without dropping dead groups.

## Acceptance criteria

- [ ] The inventory records found and missing evidence; fresh-512 can prepare when old raw groups are missing, while reuse rejects incomplete or invalid historical evidence.
- [ ] The sealed handoff identifies 512 questions and eight uniquely identified draws per question, preserving provenance, tokens, EOS/cap and masks; 64 windows match the saved question order and exclude held-out IDs.
- [ ] Rewards recompute from stored text, capped answers score zero, and per-window summaries retain all-correct, all-wrong and mixed groups.
- [ ] CPU fixtures cover both routes, identity and duplicate failures, ordering, held-out exclusion, reward and mask semantics. Fixture completion does not authorize a real pool run.

## Answer

Implemented public pool prepare, inventory, import and verification plus bounded fresh-R0 prepare/run/verify. Historical modules retain their fingerprints. Root regression passed 35 tests with 2 skips across offline pool/training and existing GRPO/R0 workflows. Latest focused pool suite passed 9 tests. The 512-group interruption/resume is synthetic CPU evidence. Real pool recovery, approved runtime/ledger and actual generation remain execution work.
