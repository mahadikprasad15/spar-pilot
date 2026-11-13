# 08: Stage D generation check

**Status:** needs-triage — draft for adversarial review
**Type:** task
**Blocked by:** 06 (can run alongside 07)

**What to build:** 8 generation runs on the ADR 0001 cohort: `gold-sft` and `online-grpo` × {full, top-10 M_acc removed, top-10 M_len removed, matched 10 removed}.

- [ ] Edited checkpoints verified against the hook; pilot protocol decoding; config and all responses as JSONL.
- [ ] v3 accuracy, mean tokens, capped rate, early-stop rate (threshold from 01).
- [ ] Generation batch size from 05b, the same for every condition; generated tokens/s and utilization summary per run.
- [ ] Untuned run only if 01 flagged a protocol or runtime mismatch.
- [ ] Full-arm results compared with the existing pilot numbers as a consistency check.

## Comments
