# 01: Source records

**Status:** needs-info
**Type:** research

Read the executed records the design could not reach from Git, and resolve conflicts. No GPU.

- [ ] Gold run: `sft.config.json`, `training.items.json` (hash), `lengths.json`, `inputs/order.json`, `meta/resolved_trainer.json`, `logs/steps.jsonl`, checkpoint seals. Check the recomputed first batch (`train:3104, 2902, 4885, 1746, 6609, 1148, 6833, 2308`) and every value in `design.md` §2.2 marked "code".
- [ ] Resolve the adapter-hash conflicts in `catalog/responses/gold-sft.greedy.test150.pilot2.yaml` (`26fdcb24…` vs `a20a344e…`; `71ee7a33…` vs `b26c524a…`), or record them unresolved.
- [ ] `sequences-v2` calibration, discovery and held-out question IDs (joint audit prep), for D11's exclusion list. List any other GSM8K train ID used by any run.
- [ ] `c-production-1`: locate `aggregate/<arm>/neurons.npz`, `freeze/<arm>.json`, their hashes and array conventions, read-only.
- [ ] Production-1: which `down_proj` layer was each arm's top adapter; joint-audit B/D list locations.

Needs the researcher's artifact root (local, Drive or HF). Read-only: nothing is moved or rewritten.

## Comments

## Answer
