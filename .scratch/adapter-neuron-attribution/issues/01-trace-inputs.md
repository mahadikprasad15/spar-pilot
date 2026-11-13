# 01: Inputs and base-correct traces

**Status:** needs-triage — draft for adversarial review
**Type:** task
**Blocked by:** none (needs the sealed joint-audit inventory and sequences on disk)

**What to build:** Bind this study to `inventory-v2` and `sequences-v2`; generate and freeze base-correct traces for the discovery and held-out math questions; build M_len masks; record the generation cohort and the untuned early-stop threshold.

- [ ] Refuse to start unless `inventory-v2` and `sequences-v2` verify by hash; record both hashes.
- [ ] Untuned greedy responses (pilot protocol, `max_new_tokens` 1024, FP32) for the 300 math prompts; config and all responses saved as JSONL.
- [ ] Score with `score_gsm8k_flexible_v3`; keep correct, uncapped rows; tokenize once with the frozen template; seal rows, masks, offsets and hashes.
- [ ] Stop if counts fall below the plan minimums; do not top up.
- [ ] M_len mask: `response_content` minus final answer line minus last N content tokens; record the `<|im_end|>` id and `####` first-token id; decoded masks for three rows.
- [ ] Record the ADR 0001 cohort IDs; compute the untuned early-stop threshold from existing untuned responses if protocol and runtime match, else flag for 08.
- [ ] CPU tests: masks on fixtures (final line, short response, no `####`), hash-mismatch refusal, deterministic order.

## Comments
