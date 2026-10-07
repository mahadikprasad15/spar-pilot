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
29. As a researcher, I want a weight-norm-matched random adapter measured on
    the same inputs so learned directions have an empirical control reference.

## Implementation Decisions

### Source contracts and unresolved prerequisites

- Reuse the completed SFT arm's verified cohort, model/tokenizer pins, actual
  rendered prompt contract, adapter target list and optimizer settings. Reuse
  Pilot 3's already-frozen token IDs, masks, documents and position rules;
  do not independently select another FineWeb slice or rebuild a supposedly
  identical cohort from current Hub data.
- FP32 without quantization is approved for both arms, including base and
  adapter weights. Record that the initial supplied design's BF16 text was
  superseded by the user's October 7 amendment; the precision gate is closed.
  Verify actual source and runtime dtypes rather than assuming the written
  choice guarantees a match. Any different precision is a new protocol variant.
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
- Remaining run prerequisites include scorer confirmation, exact
  loss configuration, baseline-subset selection, pilot sampling length limit,
  the rule for deriving the final limit above the observed 99th percentile,
  resolved generation filters, monitor actions, and the
  dated preregistered margin/predictions. Existing input identities must also
  be available and verified. Precision, behavioural bootstrap method,
  gradient-consistency definition and learning-control criteria are resolved
  by the amendment below; do not present them as unanswered decisions.
  Missing remaining values block freeze/full execution.

### Reward, sampling and GRPO training

- Optimize a binary reward from the existing flexible scoring function.
  Extract the numeric prediction without gold access, then compare with gold.
  Return zero for invalid responses. Capped responses explicitly receive
  reward zero, even when a numeric answer appears before truncation: the
  completion is unfinished and may lack its final answer. This conservative
  rule can push toward shorter answers. Choosing the final cap above the
  observed baseline 99th percentile reduces initial censoring, but does not
  guarantee low censoring as training changes the policy. Log cap fractions
  throughout training/evaluation and apply the frozen validity-monitor policy.
  Record strict extraction and
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
  norm, adapter norm and gradient consistency each step. Gradient consistency
  is the cosine between consecutive optimizer steps' flattened pre-clip
  adapter gradients across all trainable parameters, in a fixed saved parameter
  order, after the full accumulation window and before clipping. The first
  step and pairs with either gradient zero are undefined, with coverage and
  reasons. Capture the preceding gradient with sealed checkpoint state so
  this diagnostic remains continuous after checkpoint-boundary recovery.
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
  Training resumes only from the latest sealed checkpoint. Restore its state
  and redo later optimizer steps; mid-window continuation is out of scope.
  Rollouts/logs after that boundary belong to an incomplete attempt: preserve
  them as diagnostics and exclude them from the final scientific record.
  Checkpoint sealing binds rollout/log progress so the accepted history has
  one record per intended step/prompt/draw; exact partial-rollout replay is
  unnecessary. Recovery must reproduce future work from restored state.
  The largest checkpoint gap is 32 steps, so the possible redo cost is explicit.
- Reuse existing run locks, atomic payload writers and completion-marker
  helpers; do not build new persistence infrastructure for this pilot. Reject
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
- The approved behavioural interval method is a paired nonparametric bootstrap
  over the 150 held-out items, with 10,000 resamples and bootstrap seed 42.
  Each sampled ID carries both checkpoints' complete records, one draw per
  checkpoint for greedy and all eight for sampled evaluation. Recompute the
  drop, checkpoint zero minus 64, and take its 2.5th and 97.5th percentiles.
  Use the same method for greedy and sampled; preserve resampling indices.
  Checkpoint trajectory comparisons use the same item-bootstrap procedure
  against zero. This does not change Pilot 3's existing activation bootstrap.
  Do not count correlated draws as independent problems or infer training-seed
  uncertainty from item resampling. Report that the procedure uses observed
  draw sets and does not isolate repeated-decoding uncertainty on the exact
  fixed cohort. One training seed limits generalization.

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
  matching FP32 alone does not authorize incompatible calibration reuse.
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
- Add one random rank-1 adapter as a random-intervention reference, not an
  estimated null distribution, significance threshold or numerical noise floor.
  In each of the 196 modules draw independent random unit directions for the
  input/output factors using a separate fixed control RNG with seed 42; save
  construction method, order, factors and seed before measurements. Match each
  module's effective weight-update Frobenius norm to the verified GRPO step-64
  adapter, including its alpha/r scaling: the norm is scaling times the product
  of the two factor norms. A zero target norm gives a zero random update.
- Measure this random adapter with the same frozen sequences, views, numerical
  settings and validated instrument. Report its activation-write magnitudes
  and per-layer signed mean-write cosines against SFT and GRPO, under both
  weightings, with undefined/direction-resolution coverage. Weight-norm
  matching does not imply activation-norm matching. One realization cannot
  estimate a 95th-percentile null; existing numerical calibration retains its
  separate role. Additional random realizations are deferred.
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
  Run 20 optimizer steps for each control at two saved seeds, 42 and 43.
  In both seeds the positive control must increase mean rollout reward over
  steps 16-20 versus the untouched pre-training expected reward by at least 0.2; the negative control must
  decrease it by at least 0.2. Each seed's controls start from the same toy
  model/policy and use the same reward definition/settings apart from advantage
  sign. Construct a bounded learnable toy task whose initial expected reward
  is near the middle of its range, leaving room for both changes. Freeze its
  initialization, reward, generation and optimization settings before the
  acceptance runs. Do not tune thresholds/settings after observing a failure;
  diagnose it. These controls check implementation, not a statistical guarantee.
- Test interruption after a sealed checkpoint and during a subsequent window:
  recover from the sealed boundary, exclude/preserve the incomplete attempt,
  and verify accepted rollout/log IDs, future groups, RNG and gradient-cosine
  continuity plus completed-run reuse. No mid-window resume requirement.
  Test failure before final integrity sealing leaves no
  apparently complete scientific unit. CPU orchestration tests need no download.
- Real target-GPU checks are separately saved evidence: two-step runs twice,
  frozen base, initial zero equivalence, rewards/rollouts reproducibility,
  finite gradients, final-limit capacity and complete-path performance.
  Local tests cannot claim these checks passed on Colab.
- Measurement extensions get independent known-answer coefficient/KL tests,
  mapping/sign/direction negative controls, position alignment, chunk-size
  agreement and bounded-memory tests. Report generation is CPU-only and rejects
  missing or incompatible scientific artifacts rather than inventing summaries.
- Random-control tests verify 196-module mapping, unit directions, per-module
  effective norm matching, zero-target handling, seed reproducibility, unchanged
  trained source files and the same instrument gates. Reporting must expose
  realized activation magnitudes, unresolved directions and its one-realization
  limitation rather than turn a control cosine into a significance threshold.
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
- Mid-window scientific training resume, new locking/persistence frameworks,
  or an empirical direction-null distribution estimated from a single adapter.
- Sending the Adam discussion or any message to the mentor automatically.

## Further Notes

- Source: the user's "Pilot 4 — Rank-1 GRPO Arm: Design Doc", dated October 7,
  2026. Its simulation numbers, expected times/prices and historical aggregate
  scores are motivating statements, not verified results of this repository.
  Source SHA256: 0b59c6dc05363de557a4e74a2fd6fd6c0968e7ecc42c4ce0e0dde7e4ec6f049b.
  The retained source still says BF16; the user's approved October 7 amendment
  supersedes that text with FP32. Preserve both provenance and amendment.
- Specification synthesis uses the implemented SFT completion loss and FP32
  config, approved flexible-v3 scorer contract, Pilot 3 fixed-input/integrity
  decisions, and the TRL 0.26.2 trainer/config source. Preserve a source-to-spec
  reconciliation record so documentary conflicts remain visible.
- Intended inference workload is 1,024 baseline + 256 two-step reproducibility
  rollouts + 4,096 training + 750 greedy + 2,400 sampled = 8,526 generations,
  before any additional fresh zero-baseline checks or censored-length extensions.
  Activation passes, KL, supplemental SFT measurements and the one random
  adapter's forward-only measurement are additional work. Checkpoint recovery
  can redo incomplete work and therefore increase actual generation counts.
- Decisions awaiting evidence/preregistration are legitimate stage gates. The
  spec is ready for implementing those gates, not authorization to silently
  choose their values or launch a full experiment.

## Comments

- October 7, 2026: user requested to-spec followed by to-tickets, without a
  grilling interview, using the supplied design. User confirmed public-workflow
  tests with controlled dependencies and tiny real-model gradient checks.
- October 7, 2026: user approved all nine proposed ticket titles, deliveries
  and blocking edges; published as separate ready-for-agent local issues.
- October 7, 2026 amendment: user approved the reviewed critique's simplifications
  and clarifications: FP32 for both arms; checkpoint-boundary-only training
  recovery using existing infrastructure; 10,000 paired item bootstrap draws;
  consecutive pre-clip gradient cosine; two-seed positive/negative criteria
  with an attainable predeclared toy task; explicit capped-response zero reward;
  and one weight-norm-matched random-adapter control rather than a noise floor.
  Seed 42 follows the existing analysis convention; the toy pair is 42/43 and
  control randomness uses its own isolated generator. Existing pilot evidence
  and old acceptance thresholds remain unchanged.
- Baseline verification for this planning change: CPU/offline pytest completed
  with 194 passed, 19 skipped and 4 warnings in 405.60 seconds. Model-dependent
  coverage was incomplete: this shell's Torch installation could not import
  because its libtorch_cpu library was missing. No real GPU result is inferred.


### October 7 learning-control baseline amendment

User approved version 3: compare late-window rollout reward (steps 16-20)
against exact expected reward of the untouched binary policy, evaluated before
any optimizer update. In this two-token task expected reward equals probability
of token 1, so no sampled baseline estimate is needed. The 0.2 threshold,
seeds 42/43, 20 steps, optimizer and v2 readout stay fixed. Version 2 failed
its early-window rule because substantial learning occurred within steps 1-5;
its saved outcomes remain failed. This change is a disclosed post-diagnosis
engineering test revision, not a scientific experiment or retrospective pass.
