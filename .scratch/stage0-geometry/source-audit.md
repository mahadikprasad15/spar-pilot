# Initial local source audit

Date: 2026-10-26. Read-only inspection; no model download or GPU execution.
Remote artifact completion has not been verified.

## Historical training

SFT config: `artifacts/validation/pilot5-ticket03-preflight/inputs/sft.config.json`.
FP32, L4, rank 1, alpha 1, dropout 0, learning rate 1e-4, AdamW,
64 updates, eight examples per update, maximum total sequence length 554.
`pilot_eval/sft_backend.py` implements summed shifted masked cross-entropy divided
by supervised-token count across the accumulation window. The nested inherited
source_config dtype is not the actual top-level SFT training dtype.

GRPO config: `artifacts/validation/pilot5-ticket03-preflight/inputs/grpo.training.json`.
FP32, A100-SXM4-80GB, rank 1, alpha 1, dropout 0, learning rate 1e-4,
64 updates, eight prompts times eight completions. Historical reward is
flexible-v3; loss is DAPO, beta zero, sample group SD (ddof=1) plus 1e-4.
The denominator is the entire window's completion-token count. Capped outputs
receive zero reward but their tokens remain in loss masks. Generation is HF,
temperature 1, top-k 0, top-p 1, cap 1024, with a continuously consumed RNG.
Current algorithm, train-engine and scorer source hashes match those recorded
in this saved config. Relevant code: `pilot_eval/grpo_algorithm.py`,
`pilot_eval/grpo_train_engine.py`, `pilot_eval/grpo_baseline.py`.

## Untuned sampling baseline

`artifacts/reports/pilot4-results-so-far-2026-10-13/inputs/baseline-config.json`
records 128 ordered questions, HF FP32, temperature 1, top-k 0, top-p 1,
cap 1024, eight draws, one question per generation batch. Seeds are 42 plus
ordered batch index (42 through 169), not 42 plus source question index.

Source run: `runs/pilot-4/Qwen--Qwen2.5-1.5B-Instruct/sampling-baseline/pilot4-untuned-sampling-v1`.
The local aggregate snapshot reports flexible accuracy 59.8633%, zero capped
outputs, eight all-wrong groups and twelve all-correct groups. This is sampled
training-cohort evidence, not held-out greedy accuracy. GPU identity needs the
source runtime metadata; the config alone does not establish it.

## Unresolved source facts

- No checkpoint-zero adapter tensor payload was found locally. Compare actual
  SFT/GRPO A tensors from durable storage; equal seeds are insufficient.
- Completed coefficient/KL supplemental production is not confirmed locally.
  See `docs/experiments/status.md`, `docs/research/pilot-program-handoff.md` and
  `docs/adr/0010-pilot4-coefficient-batch-sensitivity.md`. Fixed-write reports
  do not establish completion of the separate per-token supplement.
- Reconcile the proposal's correction count against the confirmed behavioral
  follow-up files before defining the E0.1 accuracy overlay.
- Audit the actual 128 IDs against the common 512 training IDs before R0.
