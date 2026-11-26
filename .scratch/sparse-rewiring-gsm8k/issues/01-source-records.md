# 01: Source records and artifact locations

**Status:** needs-info
**Type:** research
**Blocked by:** —
**Group:** A

Read the executed records the design could not reach from Git. Read-only, no GPU. Needs the researcher's artifact root (local, Drive or HF).

- [ ] Gold run: `sft.config.json`, `training.items.json` (hash), `lengths.json`, `inputs/order.json`, `meta/resolved_trainer.json`, `logs/steps.jsonl`, checkpoint seals. Check the recomputed first batch (`train:3104, 2902, 4885, 1746, 6609, 1148, 6833, 2308`) and every value marked "code" in `design.md` §2.2.
- [ ] Resolve or record the adapter-hash conflicts in `catalog/responses/gold-sft.greedy.test150.pilot2.yaml`.
- [ ] `sequences-v2` calibration, discovery and held-out question IDs, plus any other GSM8K train ID used by a run (for D11).
- [ ] `c-production-1` `aggregate/<arm>/neurons.npz` and `freeze/<arm>.json`: locations, hashes, array conventions.
- [ ] Production-1: each GRPO arm's top `down_proj` layer and its trace/stop ranking files.

## Comments

## Answer
