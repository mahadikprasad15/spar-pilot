# Pilot 4 algorithm and control status

## Verified so far

Actual pinned TRL 0.26.2 `GRPOTrainer.compute_loss` is used. Its DAPO branch
sums masked policy-loss contributions and divides each microbatch by the same
full-rollout completion-token count. Direct backward accumulates these terms;
there must be no second division by accumulation steps. BNPO is different.

The candidate configuration explicitly sets 64 completions, eight draws per
prompt, one iteration, beta zero, group scaling, sample standard deviation and
1e-4 stabilizer. It exposes resolved `GRPOConfig` arguments. These candidate
API settings do not fill or authorize the later scientific protocol freeze.
Eight prompts/64 draws are validated at the training-window boundary; capped
rewards become zero and padding tokens are excluded from the denominator.

Real tiny Transformers/PEFT comparisons use an independent per-sequence
calculation and unequal completion lengths, with a nonzero gradient witness.
Whole-window versus group-preserving microbatch gradients agree. These checks
use 1e-6 absolute and 1e-4 relative agreement for FP32 gradients; they do not
change any scientific acceptance threshold. Dead groups give exactly zero
policy gradient. A separate Adam test shows previous momentum can still move
parameters. Gradient consistency distinguishes first/zero gradients from a
numeric cosine and uses sorted trainable parameter names.

Primary sources:
- https://github.com/huggingface/trl/blob/v0.26.2/trl/trainer/grpo_trainer.py
- https://github.com/huggingface/trl/blob/v0.26.2/trl/trainer/grpo_config.py

## Learning gate failed; no full-run authorization

The initial binary-action control fixture does not pass the required 0.2
reward change at either sign/seed. All failed configs, adapters, samples and
logs are retained under the canonical diagnostics tree. The first construction
mistakenly used the prompt token as the model's zero padding embedding; that
setup bug was corrected, with the original evidence retained.

A further readout diagnosis found expected probability bounds of approximately
0.373–0.627 (seed 42) and 0.393–0.607 (seed 43). The frozen tiny output readout
has insufficient headroom for the intended learning demonstration. Sample
noise can exceed these bounds in observed reward, so this is an expected
policy-probability limitation, not a hard bound on sample means.

Requested correction, pending user approval: a separately versioned fixed
unit output direction orthogonal to the initial prompt hidden state. Opposite
output rows ±u/2 start with equal logits and permit a much wider probability
range. Keep the original binary task, seeds, 20 steps, learning rate and 0.2
criterion. Preserve the old failed fixture and freeze the new version before
any acceptance run. Do not tune it after viewing its outcome.

The learning gate remains required and must not be skipped or relabelled green.
The GRPO control command/notebook handoff remains pending this decision.
