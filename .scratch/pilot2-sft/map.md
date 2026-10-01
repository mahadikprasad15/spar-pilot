# Pilot 2: rank-1 SFT collapse

Status: needs-info

## Notes

The user invokes grill-with-docs, followed by to-spec, to-tickets and implement.
The seed brief requires shared understanding before any spec or training code.
This map records the inventory and current design frontier, not an approved spec.
Project tests passed at session start: 93 CPU tests.

### Existing instrument

- Pinned Qwen/GSM8K/MMLU evaluation plans, rendered prompts, frozen test IDs,
  greedy generation, item responses, summaries, uncertainty and resume checks.
- PEFT adapter loader, local adapter content hashes or pinned Hub revisions,
  and base-model compatibility checks.
- Strict/flexible-v1 GSM8K and post hoc CPU-only flexible-v2 rescoring.
- Four MMLU settings and explicit top-logit tie policy.
- Artifact governance and guided Colab evaluation notebook.

The saved local GSM8K baseline uses flexible-v2 accuracy 64/150 = 42.67% and
mean response length 220.02 tokens. Historical 64%/288 tokens are references.
The original rendered prompts contain Qwen's default system message despite
the written no-system-message protocol. Future comparisons must use the saved
rendered prompts deliberately rather than silently correcting old prompts.

### Required additions

- Freeze and save 512 GSM8K training examples at the baseline dataset revision;
  verify split provenance and held-out separation.
- Render training prompt/target and verify actual completion-only token labels,
  including the end-turn token; reject full-sequence overflow rather than truncate.
- Rank-1 adapter construction, actual target enumeration, zero-initial-write and
  frozen-base checks.
- Version-pinned TRL training configuration and optimizer/scheduler settings.
- 64 optimizer steps, checkpoint/log/status/resume artifacts and base-weight hashes.
- Derive checkpoint evaluation configs from the frozen baseline instead of
  resolving floating revisions in a fresh prepare operation.
- Checkpoint trajectory, paired per-item accuracy/length comparisons and training
  target-length diagnostics. The existing combined report requires identical
  adapter identity, so it is not an adapter-versus-baseline comparison report.
- Training Colab workflow and pre-registration before any real training run.

TRL and PEFT are not installed locally. Local metadata: torch 2.7.1,
transformers 4.57.3, accelerate 1.12.0. The recorded Colab runtime differs.
The repo has optional dependency ranges but no exact training requirements lock.
No original team training config was found in the local SPAR workspace.

## Decisions-so-far

The user's brief fixes the primary model family, GSM8K gold-solution SFT,
r=1, alpha=1, 512 training problems and 64 optimizer steps. It specifies adapter
checkpoints at steps 0, 8, 16, 32 and 64, single-GPU execution, CPU tests with
no model download, and reuse of the held-out evaluation instrument.

Recommendations in the pasted brief are not silently treated as confirmed
choices where the brief explicitly marks them open or asks for an interview.
No collapse tolerance has been approved.

### Interview round 1 resolved

The user has no original training repo, config or local path. Pilot 2 will
therefore select and document a new training protocol; it cannot claim exact
reproduction of the historical optimization recipe. The user approved comparing
the adapter with a matched untuned baseline on identical evaluation items and
scorer, reporting accuracy and response-length changes.

### Round 2 proposals (not yet confirmed)

- Learning rate 1e-4: explicit new adapter-training choice, not the historical LR
  or the TRL 0.26.2 SFTConfig default (2e-5).
- Effective batch 8 on one GPU, realized as microbatch 1 × accumulation 8.
  This is distinct from the evaluation batch size that caused OOM previously.
- Constant LR schedule, zero warmup, AdamW explicitly selected, max grad norm
  1.0, weight decay 0, LoRA dropout 0.
- Keep 512 selected gold solutions unchanged including calculator annotations
  and #### markers; seeded sample 42 from the pinned official train split.
- Same actual chat/system behavior as baseline, completion-only targets with
  end-turn included, packing off and fail on full rendered sequence overflow.
- Select a native-bf16 GPU platform; measure a fresh matched baseline in its
  locked training/evaluation environment if the original platform differs.

Official defaults were checked in TRL 0.26.2 SFTConfig, Transformers 4.57.6
TrainingArguments, and PEFT 0.18 LoraConfig. No library-default behavior will be
relied on silently; actual selected optimizer, LR/schedule and effective batch
will be recorded. Gradient-accumulation loss normalization still requires tests.

## Verified technical context

- The current Qwen config plus Transformers Qwen2 implementation imply
  28 decoder layers × 7 linear projections = 196 targets. Verify the pinned
  loaded model and actual target list, excluding lm_head, before training.
- PEFT all-linear excludes the output layer for a PreTrainedModel. Default
  LoRA initialization gives B=0; ordinary scaling alpha/r is 1 here.
- TRL completion_only_loss is distinct from assistant_only_loss. Actual masks
  and supervised Qwen end-turn token must be checked; dataset shape alone is
  insufficient evidence.
- TRL 0.26.2 is a candidate compatible by dependency metadata with the recorded
  Transformers 4.57.6. It is not yet a selected or runtime-tested lock.
- T4 lacks native BF16 acceleration; PyTorch can report emulated BF16 support.
  The brief's blanket statement that T4 cannot run BF16 is too strong.
- A resemblance between gold-target length and generated response length is
  suggestive evidence, not a causal demonstration of imitation.
- The historical 0.34–0.50 range is not an approved replication tolerance and
  does not establish an accuracy drop relative to our current baseline.

Primary references: https://huggingface.co/docs/trl/v0.26.2/en/sft_trainer,
https://huggingface.co/docs/peft/v0.18.0/package_reference/lora,
https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/cuda/__init__.py.

## Fog / current interview frontier

1. Confirm the proposed optimization settings or requested changes.
2. Confirm training selection and target formatting.
3. Choose the GPU platform for native-bf16 training and matched evaluation.

Checkpoint-evaluation scope, numerical collapse criteria, preregistered
diagnostic predictions and public training test seams still need agreement.

Moving to a different GPU/library stack requires a fresh matched untuned
measurement there before attributing differences to training. The existing
baseline remains preserved and descriptive.

After these answers, explain and resolve data/target formatting, optimization,
platform, checkpoint evaluation scope and preregistered predictions. Then use
the selected skills to agree testing seams, publish a spec and ticket breakdown,
implement with TDD, and review. No training has been run.
