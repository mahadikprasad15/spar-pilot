# Pilot 4: rank-1 GSM8K GRPO arm

Status: ready-for-agent

## Problem Statement

The researcher needs a one-seed GRPO arm alongside the completed rank-1 SFT
arm, with the same training problems, adapter budget, optimizer-step budget
and evaluation cohort. The objective is to describe reward-trained writes and
their behavioural effects, not to promise an accuracy gain. The comparison
must expose differences in sampled data, signal, precision and loss rather
than claim that all factors except the objective were matched when they were
not. Earlier scorer failures, numerical-gate failures and expensive integrity
checks make a verified, profiled and resumable workflow essential.

The supplied design is the scientific starting point. It contains explicitly
open settings and conflicts with the implemented SFT protocol. These are
recorded prerequisites, not permission to fill in defaults or alter old runs.

## Solution

Build a guided, parameterized workflow for one rank-1 GRPO intervention on
Qwen2.5-1.5B-Instruct: verify source evidence; sample an untuned training-cohort
baseline; resolve and freeze the remaining protocol and preregistration;
validate and profile the trainer; execute 64 optimizer steps; save checkpoints
at 0, 8, 16, 32 and 64; evaluate greedy behaviour at every checkpoint and eight
sampled answers per held-out problem at checkpoints 0 and 64; measure internal
writes on the existing fixed GSM8K and FineWeb sequences; and publish a paired
comparison with the SFT arm.

The full run consumes 512 distinct training prompts once, eight prompts per
optimizer step with eight completions per prompt. Every completion and its
reward/advantage is retained. Full scientific execution requires explicit
resolution of the prerequisites below; software work may implement and test
these gates without running an unresolved experiment.

## User Stories

1. As a researcher, I want the actual SFT source config verified so matching
   claims refer to an observed run rather than an assumed recipe.
2. As a researcher, I want the same 512 training problems and 150 held-out
   problems so cohort changes do not masquerade as objective effects.
3. As a researcher, I want model, tokenizer, prompt and dataset identities
   pinned so comparisons retain their meaning across sessions.
4. As a researcher, I want one imported, versioned correctness function for
   training and evaluation so reward does not silently use another scorer.
5. As a researcher, I want strict scoring logged separately so formatting
   changes remain visible without becoming the optimization target.
6. As a researcher, I want an untuned sampling pass on 128 training problems
   so initial reward, lengths and dead groups are measured before training.
7. As a researcher, I want capped answers identified so a length limit cannot
   silently manufacture a response-length effect.
8. As a researcher, I want completion length chosen from saved baseline
   evidence and frozen before training so the choice is reproducible.
9. As a researcher, I want exactly eight unique prompts and 64 completions
   per optimizer step so my data budget matches the intended design.
10. As a researcher, I want group-normalized advantages and the exact loss
    denominator recorded so the algorithm is independently checkable.
11. As a researcher, I want only the 196 intended rank-1 projections trained
    so the intervention excludes the language-model head and base weights.
12. As a researcher, I want zero initialization and an unchanged untuned
    model checked so the reference and checkpoint zero are meaningful.
13. As a researcher, I want tiny positive and negative learning controls so
    I can distinguish a working trainer from a plausible-looking log.
14. As a Colab user, I want a two-step target-GPU preflight with measured
    runtime and memory so I can review cost before the full experiment.
15. As a researcher, I want a dated preregistration after baseline sampling
    and before full training so outcomes do not determine the predictions.
16. As a Colab user, I want verified optimizer, scheduler, random-generator
    and data-order state so interruptions resume the same scientific run.
17. As a researcher, I want every rollout with token IDs, text and provenance
    so rewards, groups, caps and advantages can be audited offline.
18. As a researcher, I want reward, loss, lengths, gradient norms and gradient
    consistency per step so competing explanations of write size are visible.
19. As a researcher, I want all five adapter checkpoints saved so behavioural
    and activation trajectories can be compared rather than only endpoints.
20. As a researcher, I want greedy and sampled endpoint evaluation so the
    reported outcomes include both decoding policies relevant to GRPO.
21. As a researcher, I want per-item paired intervals and a preregistered
    non-inferiority margin so an inconclusive result is not called harmless.
22. As a researcher, I want identical fixed measurement sequences for both
    arms so different generated response lengths do not change the input.
23. As a researcher, I want block-output changes and direct module
    contributions separated so propagated effects are not called local writes.
24. As a researcher, I want mean write vectors and per-token rank-1
    coefficients retained so later direction and steering studies are possible.
25. As a researcher, I want forward KL to the untuned model on fixed contexts
    so behavioural, geometric and distributional changes can be read together.
26. As a researcher, I want precision-aware comparisons and missing SFT
    diagnostics reported honestly so unavailable observations are not invented.
27. As a Colab user, I want clear stages, subprocess exit detection and a
    verified GPU-release point so a dead process cannot waste my session.
28. As a maintainer, I want CPU tests without model downloads and immutable
    source artifacts so new work does not break or overwrite earlier pilots.

## Implementation Decisions

### Source contracts and unresolved prerequisites

- Reuse the completed SFT arm's verified cohort, model/tokenizer pins, actual
  rendered prompt contract, adapter target list and optimizer settings. Reuse
  Pilot 3's already-frozen token IDs, masks, documents and position rules;
  do not independently select another FineWeb slice or rebuild a supposedly
  identical cohort from current Hub data.
- The implemented SFT protocol uses FP32, not BF16. The supplied design uses
  BF16 without quantization for GRPO. Record this discrepancy in a matching
  audit and require an explicit precision decision before scientific execution.
  Do not silently change either arm or label a BF16/FP32 pair objective-only.
- The historical 0.640 accuracy is contextual. Checkpoint-zero correctness is
  token-identical equivalence to a freshly measured untuned baseline under the
  same current precision, prompts, batching and decoding, not achievement of
  an aggregate accuracy target. Old responses are reusable only when all
  relevant identities agree.
- The supplied design says "flexible" but does not specify a version. The
  approved corrected repository scorer is flexible-v3; record it as the
  proposed reward/evaluation version pending confirmation. Older flexible
  functions remain available for reproducing old reports, not as implicit
  alternatives. No new last-number heuristic is authorized.
- Known SFT optimization values are learning rate 1e-4, AdamW, betas 0.9/0.999,
  epsilon 1e-8, zero weight decay, constant schedule, zero warmup, clipping norm
  1.0, dropout zero and seed 42. Verify these against the selected actual source
  before declaring them matched; do not rely only on these documented values.
- The SFT loss sums supervised shifted-token cross-entropy and divides by the
  total supervised tokens across its accumulation window. The GRPO rule must
  use the corresponding whole-window completion-token normalization. In TRL
  0.26.2, DAPO uses the generation-window token count, whereas BNPO normalizes
  each microbatch before accumulation. Resolve the named rule from source and
  independent gradient evidence, not from interchangeable labels.
- The current direct dependency pins include TRL 0.26.2. A compatibility task
  verifies the exact trainer behavior and freezes every relevant config field.
  Any required version change creates a separate resolved environment; it does
  not rewrite earlier pilots' dependency or runtime records.
- Remaining run prerequisites include the precision/scorer decisions, exact
  loss configuration, baseline-subset selection, pilot sampling length limit,
  the rule for deriving the final limit above the observed 99th percentile,
  resolved generation filters, monitor actions and statistical method, and the
  dated preregistered margin/predictions. Existing input identities must also
  be available and verified. Missing values block freeze/full execution.

### Reward, sampling and GRPO training

- Optimize a binary reward from the existing flexible scoring function.
  Extract the numeric prediction without gold access, then compare with gold.
  Return zero for invalid/capped responses. Record strict extraction and
  correctness on every completion independently of its optimization reward.
  Enforce strict-correct implies flexible-correct for the chosen version.
- Use group size eight, temperature 1.0, reward standardization within prompt
  groups, zero KL penalty, one iteration per fresh rollout, and HF generation
  for training. No silent top-k/top-p filtering or other generation defaults:
  resolve and save all generation settings and the effective distribution.
- Use rank one, alpha one, zero dropout and the seven intended projections
  across all 28 decoder blocks. Exclude other trainable parameters. Freeze the
  untuned weights; no quantization. Record the effective base and adapter
  dtypes separately, train/eval modes and numerical settings.
- Eight distinct prompts produce 64 completions in each optimizer window.
  Microbatching and accumulation are memory settings that must preserve whole
  groups, one fresh generation window per optimizer update, and the loss
  denominator. The design's 16-by-four example is not a promise that it fits.
  Verify all 512 unique training IDs are consumed once and save their order.
- Save the exact standard-deviation convention, stabilizer, clipping and
  token-mask semantics. Confirm on-policy ratios/gradients through a direct
  calculation. Do not infer equivalence merely from num_iterations being one.
- All-equal reward groups have zero advantages and zero policy-loss gradient
  with beta zero. This does not imply zero parameter movement after earlier
  updates, because Adam momentum can persist. Log both gradient and actual
  parameter change rather than asserting that no-signal steps cannot move.
- Preflight and controls use disposable/restored state, separately named
  artifacts and their own seeds. They cannot become unrecorded full-run
  optimizer steps or consume the full-run data order/RNG state.

### Baseline, freeze and execution stages

- Before full training, sample eight answers for each of 128 frozen training
  problems. Preserve all 1,024 completions, strict/flexible scores, token IDs,
  ending/cap flags, group correct counts, length distribution and dead-group
  fraction. This is not a held-out behavioural evaluation or an X1 dataset
  covering the full training cohort.
- Record censored lengths rather than treating the pilot cap as true length.
  If the 99th percentile hits that cap, the sampling evidence cannot justify a
  final cap; collect an explicitly versioned extension before freeze. One final
  completion limit applies to full training and all Pilot 4 evaluations.
- Stage ordering is cheap CPU correctness, tiny-model smoke/learning controls,
  disposable target-GPU determinism and memory/timing checks, baseline sampling,
  resolved config plus preregistration freeze, full training, evaluation,
  measurement, completion verification and CPU comparison. A provisional cap
  needed by preflight is distinguished from the final frozen cap; capacity is
  rechecked for the final workload when it differs.
- Persist every one of 4,096 training completions, with optimizer step, prompt
  ID/index, group/draw ID, generation-policy identity, text, token IDs, counted
  length, stop/cap status, both extraction results, reward and advantage.
- Log loss, learning rate, strict/flexible mean reward, dead-group fraction,
  mean/90th-percentile completion lengths, cap fraction, pre-clip gradient
  norm, adapter norm and the chosen gradient-consistency statistic each step.
  Zero-gradient cosines are undefined with coverage, not arbitrary zeros.
- Nonfinite reward/loss/gradient stops execution. The design proposes cap-rate
  above about 2%, dead groups above about 80%, and length changes above 50% in
  eight steps as monitors. Save these as proposals until their exact policy is
  frozen; distinguish validity flags, inspection pauses and numerical errors.

### Durable artifacts and performance

- Use one configurable canonical artifact root with separate Pilot 4 plans,
  runs and reports. Earlier cohorts, checkpoints and reports are read-only.
  Save model, adapter, revisions, tokenizer/chat template, prompt indices,
  seed, precision, generation/scorer settings, hook and position rules where
  applicable, code identity, libraries and all resolved trainer arguments.
- Publish adapter checkpoints at 0, 8, 16, 32 and 64 with required optimizer,
  scheduler, RNG, trainer and data-order state. Verify manifests before resume.
  Resume must reproduce group membership, rollout position and diagnostic
  logging without duplicating committed completions. Any regenerated work
  after an interruption is explicitly separated from committed records.
- Lock runs; write payloads atomically and completion markers last; reject
  incompatible invocations without modifying existing successful evidence.
  Progress means verified work, not just process existence. Detect process
  exit, including zombies, and report the actual error instead of indefinitely
  polling a stale status. Use mutually exclusive stages on one GPU.
- Batch generation and measurement when supported. Freeze group membership,
  batch order, draw IDs and seeds. Preserve these on resume. Hardware identity
  belongs in run provenance; avoid new hardcoded L4-only dispatch behavior.
- Measure end-to-end cost including generation, backward, hooks, hashing,
  transfers, loading and saving. Report training and generation timings
  separately, peak memory and measured projected total cost. No compute-price
  or one-to-two-hour claim is a guaranteed acceptance criterion.
- Retain exact frozen-weight hashes at verified boundaries and inexpensive
  in-process guards between them, following the existing integrity policy.
  Publishing pending results requires the final boundary hash to pass. Do not
  repeat a whole-model CPU hash in every projection/token operation.

### Behavioural evaluation and statistics

- Greedy evaluation covers all five checkpoints on the same 150 held-out
  items: 750 responses. Sampled evaluation covers checkpoints zero and 64 with
  eight draws per item at temperature 1.0: 2,400 responses. Save all 3,150
  responses with stable IDs, draw IDs, seeds, token IDs/lengths and scorer
  decisions. Use the same frozen cap and generation settings per mode.
- A sampled accuracy is the mean binary reward across the eight draws per
  problem, then across problems, not pass-at-eight. Pair by problem and retain
  within-problem samples. A seed shared across checkpoints does not guarantee
  identical samples for changed distributions; document that distinction.
- Greedy decoding makes local next-token argmax choices; it does not search
  for the globally most probable full answer. Historical explanatory examples
  in the source document do not redefine this policy.
- Define greedy drop as checkpoint-zero accuracy minus step-64 accuracy. The
  non-inferiority gate passes when the upper end of its preregistered paired
  95% interval is below the preregistered margin. Ten percentage points is
  proposed, not approved. Failure to pass may be inconclusive; it is not alone
  proof of harm. Report the interval and point estimate regardless of label.
- Specify the paired interval method and decoding-uncertainty treatment before
  execution. Resample complete item records with their draws as appropriate;
  never count correlated draws as independent problems. Preserve exact seeds,
  draw indices and defined coverage. One training seed limits generalization.

### Internal measurements and SFT comparison

- Reuse Pilot 3 fixed sequences and masks, including complete held-out GSM8K
  gold text and the existing bounded FineWeb sample. Preserve question,
  solution and FineWeb views plus token/equal-example weighting. Do not measure
  each arm on its independently generated responses for this comparison.
- Retain Pilot 3 definitions: block-output adapted-minus-untuned change,
  norm of mean change divided by mean baseline norm, and direct module norm
  divided by ordinary module norm. Save sufficient per-example summaries and
  mean vectors with denominator and undefined/direction-resolution coverage.
- The existing activation preparation accepts an SFT FP32 source specifically.
  Add a verified GRPO source adapter behind the same workflow responsibilities;
  do not bypass validation or recast GRPO as an SFT config. Numerical
  calibration remains bound to model, precision, runtime and batch identity;
  a new BF16 workflow cannot inherit FP32 agreement evidence unchanged.
- Extend measurement to save signed per-token rank-1 input coefficients and
  forward KL(tuned || untuned) over the full vocabulary at matched fixed
  prediction contexts. Save token positions, masks, LoRA scaling/factor hashes
  and accumulation precision. Tokenization, shift alignment and normalized
  coefficient interpretation must be explicit, never inferred from filenames.
- KL uses paired distributions in a common measurement precision. Use bounded
  chunks and reductions rather than retaining all vocabulary logits. Define
  counted context positions and reductions before freeze. Identical models
  give zero KL; tiny independent calculations and wrong-direction controls
  verify the convention. Source inference/calibration gates still apply.
- Raw coefficients depend on the LoRA factorization: rescaling A and inversely
  rescaling B preserves the function. Save factor provenance and interpret
  coefficient variation with its output factor rather than as a stand-alone
  function-invariant effect magnitude.
- Additive raw outputs for coefficients/KL require a new versioned measurement
  product. Existing Pilot 3 summary-only artifacts cannot produce missing
  token coefficients or KL offline. New SFT forward passes for those additional
  measurements are separately named and reuse old adapters/sequences; no SFT
  retraining or repeated behavioural evaluation is implied.
- Compare paired write profiles, lengths/rewards, and signed SFT-versus-GRPO
  mean-vector cosines where precision and identity permit. Flag unresolved or
  zero directions. The proposed 0.05 isotropic cosine reference is contextual,
  not a validated empirical noise floor for anisotropic model writes.
- Existing SFT logs may lack the requested gradient-consistency statistic.
  Report it unavailable; do not reconstruct it from norms. Different consistency
  is descriptive and cannot by itself establish an Adam-based causal explanation.
- Save reward/write/KL trajectories for later matched-progress analyses; do not
  claim equal steps, equal norm or equal KL are interchangeable. An early SFT
  checkpoint is not a replacement for a lower-learning-rate SFT run.

## Testing Decisions

- The user confirmed the public workflow as the primary test seam: prepare,
  baseline sampling, train/save/resume, evaluate and measure/report with
  controlled model/data dependencies. Prefer these existing seams over a
  proliferation of internal mocks. Tests assert user-visible results and
  durable artifacts rather than private helper organization.
- Reuse prior training tests for sealed-checkpoint recovery, run exclusion,
  immutable inputs, corrupt manifest rejection and preservation of old runs.
  Reuse activation tests for known-zero, frozen weights, hook coverage,
  rank-1 identities, fixed masks, calibrated agreement and verified completion.
- CPU reward cases use the selected existing scorer's contract, including
  strict-correct implies flexible-correct, numeric normalization, capped and
  ambiguous answers, and gold-independent extraction. The source document's
  "72 apples, 5 left" last-number example conflicts with the existing ambiguity
  policy; no copied heuristic or gold-aware extraction is permitted.
- Tiny offline real-model tests verify prompt/group counts, zero advantages
  for dead groups, the direct chosen policy-loss gradient, accumulation-window
  token weighting and padding invariance at justified numerical scales. Positive
  toy learning and flipped-advantage negative controls exercise the trainer.
  Define their criterion before testing; "about 20 steps" is not an exact bound.
- Test resume across interruption with rollouts and random state, including
  no duplicate IDs, identical future groups, diagnostics continuity and
  completed-run reuse. Test failure before final integrity sealing leaves no
  apparently complete scientific unit. CPU orchestration tests need no download.
- Real target-GPU checks are separately saved evidence: two-step runs twice,
  frozen base, initial zero equivalence, rewards/rollouts reproducibility,
  finite gradients, final-limit capacity and complete-path performance.
  Local tests cannot claim these checks passed on Colab.
- Measurement extensions get independent known-answer coefficient/KL tests,
  mapping/sign/direction negative controls, position alignment, chunk-size
  agreement and bounded-memory tests. Report generation is CPU-only and rejects
  missing or incompatible scientific artifacts rather than inventing summaries.
- A full controlled workflow test traverses all stages, interrupts/resumes and
  produces the paired report. A guided notebook contract test verifies stage
  ordering, pinned code availability, subprocess termination handling, progress
  meanings and the completion check before GPU release.

## Out of Scope

- Accuracy gains or exact reproduction of unavailable historical settings.
- Three-seed headline claims, the SFT learning-rate sweep and objective-versus-
  data-source attribution; these are B0/X1 work.
- Sampled evaluation of SFT checkpoints or intermediate GRPO checkpoints.
- The full 512-problem untuned sampling baseline, no-reward-scaling ablation,
  nonzero-beta arms and a vLLM training implementation.
- Training on top of frozen SFT, causal patching/steering, placement sweeps or
  claiming the current one-seed geometry establishes a mechanism.
- Overwriting old scorer versions, configs, reports, calibration evidence or
  dependency pins; broad package restructuring.
- Sending the Adam discussion or any message to the mentor automatically.

## Further Notes

- Source: the user's "Pilot 4 — Rank-1 GRPO Arm: Design Doc", dated October 7,
  2026. Its simulation numbers, expected times/prices and historical aggregate
  scores are motivating statements, not verified results of this repository.
  Source SHA256: 0b59c6dc05363de557a4e74a2fd6fd6c0968e7ecc42c4ce0e0dde7e4ec6f049b.
- Specification synthesis uses the implemented SFT completion loss and FP32
  config, approved flexible-v3 scorer contract, Pilot 3 fixed-input/integrity
  decisions, and the TRL 0.26.2 trainer/config source. Preserve a source-to-spec
  reconciliation record so documentary conflicts remain visible.
- Intended inference workload is 1,024 baseline + 256 two-step reproducibility
  rollouts + 4,096 training + 750 greedy + 2,400 sampled = 8,526 generations,
  before any additional fresh zero-baseline checks or censored-length extensions.
  Activation passes, KL and supplemental SFT measurements are additional work.
- Decisions awaiting evidence/preregistration are legitimate stage gates. The
  spec is ready for implementing those gates, not authorization to silently
  choose their values or launch a full experiment.

## Comments

- October 7, 2026: user requested to-spec followed by to-tickets, without a
  grilling interview, using the supplied design. User confirmed public-workflow
  tests with controlled dependencies and tiny real-model gradient checks.
- October 7, 2026: user approved all nine proposed ticket titles, deliveries
  and blocking edges; published as separate ready-for-agent local issues.
- Baseline verification for this planning change: CPU/offline pytest completed
  with 194 passed, 19 skipped and 4 warnings in 405.60 seconds. Model-dependent
  coverage was incomplete: this shell's Torch installation could not import
  because its libtorch_cpu library was missing. No real GPU result is inferred.
