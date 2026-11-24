# 01: Source records and artifact locations

**Status:** done, except production-1 per-neuron ranking files (on sl-0 only)
**Type:** research
**Blocked by:** —
**Group:** A

Read the executed records the design could not reach from Git. Read-only, no GPU. Needs the researcher's artifact root (local, Drive or HF).

- [x] Gold run: `sft.config.json`, `training.items.json` (hash), `lengths.json`, `inputs/order.json`, `meta/resolved_trainer.json`, `logs/steps.jsonl`, checkpoint seals. Check the recomputed first batch (`train:3104, 2902, 4885, 1746, 6609, 1148, 6833, 2308`) and every value marked "code" in `design.md` §2.2.
- [x] Resolve or record the adapter-hash conflicts in `catalog/responses/gold-sft.greedy.test150.pilot2.yaml`.
- [x] `sequences-v2` calibration, discovery and held-out question IDs, plus any other GSM8K train ID used by a run (for D11).
- [x] `c-production-1` `aggregate/<arm>/neurons.npz` and `freeze/<arm>.json`: locations, hashes, array conventions.
- [ ] Production-1 (layers found; ranking files need sl-0): each GRPO arm's top `down_proj` layer and its trace/stop ranking files.

## Comments

## Answer

2026-11-24: see `../source-records.md` (and `../source-records.json`). Read from the local copies on Prasad's Mac;
the HF mirror (`Prasadmahadik/spar-sampling-sft-artifacts`) was unreachable from here (proxy 403) and `sl-0` was not
needed except for one item.
- Gold: every file located and hashed; `load_sft` passes; the 512 items rebuild byte-identically; `order.json[0:8]`
  equals the recomputed first batch; `steps.jsonl` follows `order.json` for all 64 steps; `resolved_trainer.json`
  agrees with every "code" value in `design.md` §2.2. No `lengths.json` exists.
- Adapter-hash conflict: a hash-kind difference (directory digest vs safetensors file hash); full proof needs the
  Drive checkpoint directory.
- `sequences-v2`: 325 GSM8K train IDs, verified; no other run used train IDs outside gold 512 ∪ `sequences-v2`.
- `c-production-1`: both arms' `neurons.npz` and freezes located, seals verified, conventions recorded.
- Production-1: top `down_proj` layers online 13 (trace) / 14 (stop), offline 14 / 14. Per-neuron ranking files are
  only on `sl-0` (`$VOL/attribution/freeze`, `$VOL/attribution/store`); not yet read.
