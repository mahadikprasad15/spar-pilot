# Stage 0: batch-agreement gate is the pre-registered 1e-3 hard stop

Status: approved by the user on 2026-10-29, after the first real batch16 agreement
attempt stopped.

The batch16 agreement check (`stage0-batch-run`) compared batched FP32
answer-position logits with singleton logits and required the error to stay
within the merged-loader allowance 3E. That allowance measures a different
rounding source: merging LoRA factors into FP32 base weights. Batching changes
matrix shapes and padding, so cuBLAS selects different kernels and summation
orders. Nothing makes that rounding smaller than the merge rounding.

The real attempt, `stage0-geometry-v1-behavior-b16-v1`, recorded a batched error of
5.083e-4 on `numeric-b16-V0a` (worst at `gsm8k:test:96`, position 134), against
3E = 4.520e-4 (E ≈ 1.5e-4). V0a fails, so the adapters are not the cause.
The magnitude is FP32 kernel rounding. A padding, mask, position-ID or KV-cache
fault produces order-one or larger logit errors.

Decision:

- Judge batched-vs-singleton logits, including cached decode, against the spec's
  pre-registered 1e-3 hard stop (`spec.md`, loader qualification). This number is
  not fitted to the failed observation.
- Use the same 1e-3 bound for the greedy near-tie rule. A divergence passes only
  if the reference top-two margin at the shared prefix is below the measured
  common-prefix error, which itself must be at most 1e-3. Every other divergence
  and every stop-reason difference still stops.
- The loader allowance 3E, the loader qualification and every spec tolerance stay
  unchanged. 3E is still recorded and bound in reviews and the frozen execution.
- Batching is a fixed, recorded condition shared by all variants and both
  references (`spec.md`: "Fix and record batching and generation settings").
  Batched outputs are not claimed to be token-identical to singleton outputs.
- The failed v1 evidence is preserved. The replacement agreement uses a new name,
  `stage0-geometry-v1-behavior-b16-v2`. Changed code changes the implementation
  identity, so v1 cannot be relabelled.

This is a gate change made after a failure, recorded as an explicit user decision.
Batch size 1 was the alternative inside the original gate and was rejected as too
slow for 1,500 behavioral generations.
