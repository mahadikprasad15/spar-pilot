# 08: Pod input bundle

**Status:** needs-triage
**Type:** task
**Blocked by:** 02

**What to build:** a sealed, hash-listed input snapshot, so that the Pod restores **every** input the run binds (lesson 06a r1: a prep snapshot is not all of a manifest's inputs).

- [ ] **Contents**, each with its sha256:
  - the two behaviour `responses.jsonl` files (or their step-64 sampled subsets, recording the source sha256);
  - the two **source run configs**;
  - the online and offline step-64 adapters, and the untuned zero adapter;
  - the tokenizer files, and the pinned `generation_config.json`;
  - the cohort ID list;
  - the **hot-layer source file** named in the launch record;
  - the GSM8K test parquet;
  - the **catalog revision**: the commit hash of `catalog/`, plus the hashes of its files.

  The base model comes from HF at revision `989aa798…`, hash-checked.
- [ ] **Pod check.** On the Pod, the code revision contains `catalog/`, and `pilot_eval.catalog verify` on the restored bundle reports OK for the entries the run binds.
- [ ] **Export and restore.** Export to the private HF dataset under its own label. Restore on a clean root, and verify against the **seal**, not by rebuilding from the originals (lesson 06a r1).
- [ ] **Downloads.** Every download is hash-checked, with `HF_HOME` on the container disk (lesson pod-1).
- [ ] **Tests.** Export → clean restore → verify, on CPU with the fake Hub. A missing file fails with "missing", not "differs". A missing catalog revision is refused.

## Comments

- 2026-11-18 (round 1): the bundle now includes the source configs, the generation config, the hot-layer source and the catalog revision (F4, F7). The catalog commits were merged into this branch in `a43dca0`; `main` is not pushed.

## Answer
