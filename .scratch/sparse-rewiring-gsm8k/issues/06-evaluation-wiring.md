# 06: Evaluation wiring

**Status:** ready-for-agent
**Type:** task
**Blocked by:** 03

Validation CE/greedy and cohort150 greedy for rewired checkpoints through the existing evaluation harness (config + JSONL per run, flexible-v3, explicit decoding arguments), plus catalog entry builders.

- [ ] Red/green on fixtures: loading a rewire checkpoint, deterministic masks at eval, checkpoint-0 identity with untuned outputs on a tiny model.
- [ ] Cohort150 greedy at step 64 and the final step only (D12, decided: evaluation only).

## Comments

## Answer
