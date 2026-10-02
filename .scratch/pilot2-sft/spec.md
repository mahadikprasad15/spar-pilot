# Pilot 2: exploratory rank-1 GSM8K SFT

Status: ready-for-agent

## Problem Statement

The researcher needs to measure how a small rank-1 SFT intervention changes
Qwen2.5-1.5B-Instruct's GSM8K accuracy and response length, using the evaluation
instrument established in Pilot 1. The earlier SPAR result reported accuracy
0.640 to 0.420 and mean response length 288 to 77 tokens, but its training
configuration and example indices are unavailable. Those aggregates cannot
identify the training recipe or serve as correctness gates for this run.

Pilot 1 also demonstrated that answer extraction can substantially change
measured accuracy. Its existing flexible-v2 score is 64/150 (42.67%), with mean
response length 220.02 tokens, under a different precision/runtime. Comparing
a newly trained score directly with those aggregates could confound training,
precision, formatting and scoring. Pilot 2 needs a fresh matched untuned
baseline, preserved item-level evidence and a fixed analysis plan.

## Solution

Build a parameterized, resumable training workflow and guided Colab notebook
for an exploratory, newly recorded SFT protocol. Train one rank-1 adapter on
512 frozen GSM8K gold solutions for 64 optimizer steps. Save adapter checkpoints
and resumable training state at steps 0, 8, 16, 32 and 64. Evaluate the matched
untuned baseline and every checkpoint on the same 150 held-out GSM8K problems.

Use unquantized FP32 on a Colab T4 with a real memory/stability preflight.
Report strict and frozen flexible-v2 accuracy, generated response lengths,
paired changes and uncertainty. Preserve every response, configuration,
training log and checkpoint. Register an exploratory analysis plan before
training; there is no directional prediction or binary collapse threshold.

## User Stories

1. As a researcher, I want a frozen training cohort so I can identify exactly
   which examples the adapter learned from.
2. As a researcher, I want training and evaluation split provenance so I can
   confirm that held-out problems were not used for training.
3. As a researcher, I want the original gold solutions preserved so I can
   attribute formatting changes to a recorded training target.
4. As a researcher, I want training prompts to match the actual evaluation
   prompts so an unnoticed template change does not confound the comparison.
5. As a researcher, I want verified completion-only loss so training optimizes
   the solution and its ending rather than the question.
6. As a researcher, I want complete examples without silent truncation so I
   know which target tokens received supervision.
7. As a researcher, I want the actual adapter target list saved so I can verify
   that the intervention covers every intended projection and excludes others.
8. As a researcher, I want a zero-write initialization check and frozen-base
   verification so checkpoint zero and later adapters have interpretable meaning.
9. As a researcher, I want explicit optimization and precision settings so the
   new protocol can be reproduced without unavailable historical configurations.
10. As a Colab user, I want a GPU preflight so memory or numerical failures are
    detected before the full training run.
11. As a Colab user, I want resumable training state so interruptions do not
    require restarting completed training.
12. As a researcher, I want loss, learning rate and gradient norm logs so I can
    inspect the training trajectory and numerical health.
13. As a researcher, I want a fresh matched untuned baseline so measured changes
    compare models in the same precision and runtime.
14. As a researcher, I want all checkpoint evaluations to reuse the same items,
    prompts, decoding and scorers so comparisons remain paired.
15. As a researcher, I want strict and flexible-v2 measurements from the same
    saved generations so output-format effects remain visible.
16. As a researcher, I want paired accuracy and response-length changes with
    uncertainty so a single aggregate does not hide item-level variation.
17. As a researcher, I want gold-target length summaries and checkpoint curves
    so I can explore how training relates to generated answer length.
18. As a researcher, I want an exploratory analysis plan recorded before training
    so later interpretations do not rewrite the protocol after observing results.
19. As a maintainer, I want CPU tests without model downloads so correctness
    checks run cheaply and independently of Colab availability.
20. As a Colab user, I want explanatory notebook cells and durable Drive artifacts
    so I understand each stage and can continue after a disconnected session.

## Implementation Decisions

### Protocol and data

- This is one exploratory training run under a new versioned protocol. Do not
  describe it as an identical reproduction of the historical training recipe.
- Use Qwen2.5-1.5B-Instruct, the model/tokenizer revisions and GSM8K revision
  frozen in the selected Pilot 1 source configuration. Derive variants from
  those pins; do not resolve floating Hub revisions independently per checkpoint.
- Select 512 distinct examples without replacement from official GSM8K `main`
  train using seed 42. Save ordered source indices, IDs, split, revision,
  selection algorithm, example content and hashes. A repeated preparation
  verifies and reuses the frozen cohort rather than selecting it again.
- Reuse the exact existing 150 evaluation IDs and order from Pilot 1's frozen
  test cohort. Verify split provenance and absence of identical training/eval
  questions; do not compare unqualified numeric indices across different splits.
- Preserve gold solutions, including calculator annotations and `####` markers.
- Training user prompts reproduce the actual saved GSM8K evaluation format,
  including its instruction and Qwen default system message. Record the known
  discrepancy with Pilot 1's written no-system-message protocol; do not silently
  fix the template within this comparison.
- Represent gold solutions as assistant completions. Supervise completion tokens
  and the Qwen end-turn token; mask prompt and padding tokens. Verify actual
  rendered token labels, including the prompt/completion boundary.
- Packing is disabled. Derive the sequence limit from the longest complete
  rendered training sequence, including prompt, solution and ending. Reject
  overflow or an incompatible model context limit; never silently truncate.

### Model, adapter and optimizer

- Use an unquantized FP32 frozen model and FP32 trainable adapters on one Colab
  T4. No FP16/BF16 autocast or quantization in this protocol.
- Use PEFT rank 1, alpha 1, dropout 0 and default zero-write initialization.
  Target q/k/v/o/gate/up/down projections in all 28 decoder blocks: exactly
  196 modules. Enumerate actual targets and save their names; exclude the
  output head, embeddings and other base parameters from training.
- Confirm the initial adapter has zero functional write and base parameters
  remain frozen. Hash base weights before and after training using stable
  parameter names and tensor contents; record and require equality.
- Train with TRL SFTTrainer for 64 successful optimizer updates, seed 42 and
  full determinism. Use microbatch 1 and gradient accumulation 8, for effective
  batch 8, learning rate 1e-4, constant scheduler, zero warmup, AdamW, zero
  weight decay and maximum gradient norm 1.0.
- Resolve and record the concrete AdamW implementation and all optimizer
  settings explicitly. Validate completion-token loss normalization across
  accumulation; the effective batch must not depend on unnoticed library defaults.
- Enable gradient checkpointing and disable the training KV cache. Record data
  shuffle order and actual example exposures. The intended 512 presentations
  over 64 effective batches must be verifiable rather than inferred from counters.
- Supply exact compatible dependency pins for GPU execution. TRL 0.26.2 is a
  investigated candidate, not evidence of an already validated installation.
  Record actual Python, PyTorch/CUDA, Transformers, TRL, PEFT, datasets and
  accelerate versions, hardware, repository commit and deterministic settings.

### Preflight, artifacts and resume

- Check existing artifacts before starting any operation. Keep a canonical
  configurable artifact root with training runs, evaluation runs and reports;
  the notebook may use the corresponding persistent Drive tree.
- Save immutable run configuration and input hashes, plus explicit status,
  timestamps, environment metadata and structured errors. Reject conflicting
  configurations under an existing run identity.
- Before full training, check actual device/dtype, targets, masks, longest-example
  memory requirements and finite loss/gradients through the intended training
  path. Persist results. Preflight must not advance the scientific training run
  or change its initial adapter/random state.
- Stop visibly on OOM, nonfinite values, mask/target/freeze failure or runtime
  mismatch. Do not silently lower precision, change batch settings, truncate
  examples or tune the learning rate to manufacture a historical-looking result.
- At steps 0, 8, 16, 32 and 64 save adapter weights plus optimizer, scheduler,
  random and sampler/data-position state sufficient for continuation. Record
  completeness and hashes so a partial checkpoint is not treated as resumable.
- Resume from the last complete verified checkpoint. Work after that checkpoint
  may need replay; an adapter file alone is not a complete training resume state.
  Preserve interruption evidence and avoid duplicate or misleading step logs.
- Save per-step loss, learning rate, gradient norm and numerical-health evidence
  as structured logs and CSV. Do not equate attempted/skipped updates with
  successful optimizer steps.

### Evaluation and analysis

- Pilot 2 evaluates GSM8K only. Run a fresh untuned FP32 baseline in the locked
  environment and evaluate adapter checkpoints 0, 8, 16, 32 and 64 there.
- Use evaluation batch 1 and the frozen greedy settings, prompt template,
  tokenizer, EOS behavior and response-length definition from the source
  evaluation protocol, including its 1024 generated-token cap.
- Derive checkpoint evaluation configurations from frozen inputs. Record adapter
  path and content hash, model/tokenizer pins, prompt IDs, seed, precision,
  decoding and scorer versions for every run. Save every response as JSONL.
- Reuse the existing evaluation/resume instrument and optional adapter loader.
  Verify completed outputs and process only missing item IDs after interruption.
- Score the same saved generations with unchanged strict extraction and frozen
  flexible-v2 extraction. Preserve any existing flexible-v1 outputs as legacy
  measurements. Flexible-v2 is fixed before this intervention; extraction never
  receives the gold answer, and invalid/capped outputs remain in the denominator.
- Report correct/total, accuracy and 95% intervals; mean/median response tokens,
  invalid and capped rates; paired per-item accuracy changes against the fresh
  baseline; mean-length changes and trained/baseline mean-length ratios.
- Paired uncertainty resamples matched item IDs together, not separate model
  cohorts. Record the interval procedure and its settings in the pre-run analysis
  plan. Identify intervals as conditional on this run and sampled cohort; they
  do not measure variability across training seeds.
- Verify checkpoint zero against the matched untuned outputs. An unexplained
  behavioral difference blocks interpreting the later checkpoint trajectory.
- Produce a checkpoint trajectory table/plot and preserve joined per-item
  comparisons with source hashes. Do not present incomplete evaluations as
  complete-cohort accuracy.
- Summarize gold-solution token lengths separately from full prompt/target/end-turn
  training lengths, stating each counting rule. Provide side-by-side responses
  for inspection. Similar gold/generated lengths are descriptive evidence and
  do not establish that target imitation caused the outcome.
- Save a dated local exploratory analysis plan before training, including the
  frozen protocol, measurements, interval procedures, diagnostics and evidence
  limits. No directional prediction or binary collapse threshold is registered.
- Present historical 0.640/0.420 and 288/77 values, and the earlier Pilot 1
  measurements, as separate contextual references. Report the actual outcome;
  neither agreement nor disagreement with those values is a software test gate.

### Interfaces and notebook

- Add parameterized public preparation, preflight/training and comparison
  operations, with replaceable external model/data boundaries for CPU tests.
  Reuse existing evaluation and offline rescoring operations.
- Keep training, checkpoint evaluation and aggregation separable so evaluations
  can resume without retraining and future adapter variants can reuse the instrument.
- Provide a guided Pilot 2 Colab notebook explaining setup, frozen data, the
  exploratory plan, preflight, baseline, training, resume, checkpoint evaluation
  and interpretation. Read persistent state rather than relying solely on live
  notebook globals. Preserve the existing Pilot 1 notebook and artifacts.

## Testing Decisions

- Test externally observable behavior through public preparation, training and
  comparison operations, using tiny models/fake external boundaries. Tests run
  on CPU with no downloaded model or dataset and no GPU requirement.
- Follow red/green TDD and existing configuration, workflow, backend, evaluation,
  recovery and offline-rescoring test conventions. Prefer assertions about saved
  artifacts and validated behavior over private implementation details.
- Cover deterministic cohort persistence, split separation, template identity,
  prompt/completion masks and supervised end-turn, no silent truncation, intended
  target coverage, zero initial write and frozen base weights.
- Cover successful update counting, accumulated loss behavior, checkpoint
  completeness, interruption/resume, incompatible configuration rejection and
  preservation of existing artifacts.
- Cover matched comparison input validation, item alignment, scorer consistency,
  invalid/capped denominator policy and no complete score from partial cohorts.
- Use actual GPU preflight checks for full-model placement, memory and numerical
  stability. CPU tests cannot establish that Qwen fits or trains stably on a T4.
- After GPU execution, require saved evidence for initial zero write, unchanged
  base weights, complete training state, successful updates and matched checkpoint
  zero behavior. These are instrument-integrity checks, not collapse criteria.
- Review the notebook's structure, instructional cells and command/config wiring.
  Report GPU execution as unverified until it actually occurs.

## Out of Scope

- MMLU evaluation for Pilot 2; the existing MMLU harness remains reusable.
- GRPO, additional adapters, steering, effective-write depth maps and causal
  tests of a chain-of-thought direction.
- Learning-rate sweeps, multiple training seeds and post-result recipe tuning.
- Quantization, sampled decoding and automatic precision changes.
- Changes to frozen GSM8K scoring rules or acceptance targets derived from the
  historical collapse or unblinded diagnostic review.
- Claims of exact historical replication, population-level certainty from one
  seed, or a causal explanation of shortening from behavioral results alone.
- Automatic remote training, external Research OS writes or guaranteed T4 fit.

## Further Notes

The user completed the grilling interview and confirmed the consolidated plan
before requesting this spec. Public test seams and both CPU/GPU checks were
explicitly approved. The source evaluation is preserved despite its recorded
system-message discrepancy.

Session-start verification: 93 CPU tests passed. No Pilot 2 training code or GPU
run has been executed. The next workflow stage is to propose implementation
tickets for approval before publishing them.

## Approved L4 hardware/throughput variant (2026-10-02)

See `docs/adr/0004-pilot2-l4-throughput.md`. A separately named L4 plan keeps
FP32 and all training/data/scorer settings, freezes evaluation batch 2, and runs
a fresh hardware preflight and matched L4 baseline/checkpoints. The old T4 plan
and artifacts remain immutable. A separate full-length eight-item benchmark
compares inference batches 1/2 before baseline, saving timings, memory and raw
outputs without adding diagnostic responses to scientific runs. Estimates are
conditional on those untuned examples and exclude training/setup. GPU execution
is verified in Colab, not inferred from CPU test success.
