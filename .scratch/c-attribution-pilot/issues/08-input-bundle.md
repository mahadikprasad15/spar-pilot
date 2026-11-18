# 08: Pod input bundle

**Status:** needs-triage
**Type:** task
**Blocked by:** 02

**What to build:** a sealed, hash-listed input snapshot so the Pod restores **every** input the run binds (lesson 06a r1: a prep snapshot is not all of a manifest's inputs).

- [ ] Contents: the two behaviour `responses.jsonl` files (or their step-64 sampled + greedy subsets, with the source sha256), the online and offline step-64 adapters, the untuned zero adapter, tokenizer files, the cohort ID list, and the GSM8K test parquet (for heldout sets), each with sha256. The base model comes from HF at revision `989aa798…`, hash-checked.
- [ ] Export to the private HF dataset under its own label; restore on a clean root and verify against the **seal**, not by rebuilding from originals (lesson 06a r1).
- [ ] Every download is hash-checked; `HF_HOME` on the container disk (lesson pod-1).
- [ ] Tests: export → clean restore → verify on CPU with the fake Hub; a missing file fails with "missing", not "differs".

## Comments

## Answer
