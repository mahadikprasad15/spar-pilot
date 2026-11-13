# 08: Stage D generation check

**Status:** needs-triage — draft for adversarial review
**Type:** task

**What to build:** Generate on the ADR 0001 GSM8K cohort under the frozen Stage D conditions and score with flexible v3.

**Blocked by:** 06 (conditions). Can run in parallel with 07.

- [ ] Edited PEFT checkpoints for removal conditions, verified against the in-memory hook; hooked HF generation for constant c̄.
- [ ] Pilot protocol decoding; config and all responses saved as JSONL per condition.
- [ ] Accuracy (v3), mean tokens, capped rate, early-stop rate using the threshold frozen in 01.
- [ ] Untuned and full-arm rows regenerated under the same runtime, and compared with the existing pilot results as a consistency check.

## Comments
