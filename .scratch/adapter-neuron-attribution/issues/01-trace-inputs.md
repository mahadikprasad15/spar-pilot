# 01: Freeze attribution inputs and base-correct traces

**Status:** needs-triage — draft for adversarial review
**Type:** task

**What to build:** Bind this study to the joint audit's sealed `inventory-v2` and `sequences-v2`; generate and freeze the base-correct trace set; build M_len position masks; freeze the ADR 0001 generation cohort reference.

**Blocked by:** none (needs the sealed joint-audit inventory and sequences on disk).

- [ ] Refuse to start unless `inventory-v2` and `sequences-v2` verify by hash; record their hashes in this study's manifest.
- [ ] Generate untuned greedy responses (pilot protocol, `max_new_tokens` 1024, FP32) for every discovery and held-out math question; save config and all responses as JSONL.
- [ ] Score with `score_gsm8k_flexible_v3`; keep correct, uncapped rows; tokenize prompt + response once with the frozen template; seal `base_correct` rows with masks, offsets and hashes.
- [ ] Stop (do not top up) if base-correct counts fall below the plan minimums.
- [ ] Build M_len masks for both trace sets: `response_content` minus final answer line minus last N content tokens; record EOS id and the `####` first-token id; save decoded masks for three examples per set.
- [ ] Record the ADR 0001 GSM8K cohort IDs and the untuned length distribution used for the early-stop threshold.
- [ ] CPU tests: mask construction on fixtures (final line, short responses, missing `####`), refusal on hash mismatch, deterministic row order.

## Comments
