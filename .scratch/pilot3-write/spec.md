# Pilot 3: validated activation-write measurements

Status: ready-for-agent

## Problem Statement

The researcher has saved rank-1 SFT adapter checkpoints from Pilot 2 and wants
to understand how much they change activations, where those changes occur along
depth, and how measurements differ between task and unrelated text. Before
interpreting these measurements, the researcher needs evidence that the
instrument measures the intended quantities correctly.

Pilot 1 and Pilot 2 exposed scoring, prompting, memory, versioning, monitoring
and artifact-location failures. In particular, flexible-v2 extraction created
an apparent accuracy collapse that disappeared under flexible v3. Corrected
Pilot 2 baseline/final accuracy was approximately 0.680/0.660, while mean
generated length fell from 215.53 to 137.78 tokens. Pilot 3 must not assume an
accuracy collapse or turn descriptive activation measurements into causal claims.

## Solution

Build a parameterized, resumable, forward-only measurement workflow and guided
Colab notebook. Reuse the verified Pilot 2 checkpoints at optimizer steps
0, 8, 16, 32 and 64. On fixed, identical token sequences, measure both full
block-output changes and each adapted linear module's direct contribution.

Validate zero initialization, frozen base weights, checkpoint identity and the
rank-1 contribution identity. Freeze input masks, numerical settings and batch
membership after profiling. Save per-example sufficient summaries and verified
shards, then reconstruct plots and uncertainty on CPU without repeating model
inference. Preserve the interview and decision explanations in the repository.

## User Stories

1. As a researcher, I want to reuse saved adapters so I do not repeat training.
2. As a researcher, I want verified model and tokenizer revisions so checkpoints
   are measured with their associated untuned model.
3. As a researcher, I want checkpoint and frozen-base identity checks so an
   unintended weight change cannot masquerade as an adapter effect.
4. As a researcher, I want a true zero-adapter control so I can validate the
   measurement before interpreting a learned write.
5. As a researcher, I want a same-input rank-1 identity check so I know the local
   contribution is measured correctly.
6. As a researcher, I want negative controls so the validator demonstrably
   rejects incorrect scaling, signs, token alignment and module mapping.
7. As a researcher, I want decoder block outputs measured directly so final
   normalization conventions cannot silently change the hook location.
8. As a researcher, I want module contributions separated from propagated
   block changes so I do not confuse two different quantities.
9. As a researcher, I want all five checkpoints measured on identical tokens
   so generated-response differences do not confound activation subtraction.
10. As a researcher, I want complete held-out gold solutions preserved so
    truncation does not change the task being measured.
11. As a researcher, I want question and solution views separated so their
    different contexts remain visible.
12. As a researcher, I want unrelated FineWeb inputs frozen so cross-input
    comparisons have inspectable provenance.
13. As a researcher, I want actual token masks audited so instructions, padding
    and chat structure do not enter content summaries accidentally.
14. As a researcher, I want token-weighted and equal-example-weighted summaries
    so I can detect whether longer examples dominate the profile.
15. As a researcher, I want mean vectors and magnitude summaries retained so
    cancellation is distinguishable from small individual changes.
16. As a researcher, I want relative measurements with explicit denominators
    so raw activation scale does not silently determine the depth profile.
17. As a researcher, I want undefined ratios represented honestly so zero
    denominators cannot become apparent zero effects.
18. As a researcher, I want numerical-resolution diagnostics so small writes
    are not overinterpreted from rounded output subtraction.
19. As a Colab user, I want measured batch selection so the GPU is used
    efficiently without a brittle hardware-name restriction.
20. As a Colab user, I want the longest inputs included in preflight so a short
    audit is not mistaken for evidence that production fits in memory.
21. As a Colab user, I want one loaded model and a validated shared untuned
    reference so I avoid loading two full models or redundant baseline passes.
22. As a Colab user, I want completed shards reused after interruption so
    reconnecting does not discard expensive work.
23. As a maintainer, I want corrupt or incompatible artifacts rejected so
    resume cannot silently mix runs.
24. As a Colab user, I want monitoring failures distinguished from child-process
    failures so I do not accidentally start a second writer.
25. As a researcher, I want CPU-only report reconstruction so I can disconnect
    the GPU after measurements are safely saved.
26. As a researcher, I want checkpoint/depth curves, module heatmaps and
    coverage diagnostics so I can inspect the trajectory rather than one scalar.
27. As a researcher, I want paired example-bootstrap intervals so uncertainty
    preserves the matched checkpoint comparisons.
28. As a maintainer, I want offline CPU workflow and real-library tests so
    regressions are caught without downloading a model.
29. As a new user, I want guided notebook stages with inputs, outputs and
    rationale so I understand what each stage does and how to resume it.
30. As a researcher, I want the questions, alternatives and prior-pilot lessons
    preserved so later readers understand the choices behind the protocol.

## Implementation Decisions

### Workflow and interfaces

- Extend the existing evaluation package with decoupled preparation, validation,
  profiling, measurement and reporting responsibilities. Prefer existing public
  command and artifact contracts; keep notebook cells thin wrappers around them.
- Public operations accept a named plan, source training manifest and output
  root as appropriate. Model identity, corpus choice, checkpoint list, batch
  size and save policy belong in recorded configuration, not scattered constants.
- The initial scientific protocol is fixed below. Future variants require a
  separately named configuration, not silent changes during an existing run.
- One shared artifact tree holds both pilots' source evidence and Pilot 3
  outputs. Colab uses the approved shared Drive workspace and shared checkout;
  repository notebooks use the approved notebook folder. Historical run names
  and immutable configs remain unchanged.
- Record code revision, dependency versions, Python, GPU, CUDA, attention
  implementation, determinism controls, precision and matrix-multiplication
  settings. Verify resume compatibility before computation; an unverified
  environment change requires a new plan rather than appending to an old run.
- Workflow status distinguishes prepared, running, failed and completed work.
  Validation results and their identities accompany measured artifacts; a final
  scientific report requires all mandated gates and combinations to pass.

### Source identity and inputs

- Primary model: Qwen2.5-1.5B-Instruct, unquantized FP32. Obtain model/tokenizer
  revisions and checkpoint identities from the verified Pilot 2 training run.
  Never independently resolve floating revisions per checkpoint.
- Verify checkpoint file hashes and training metadata. Recorded before/after
  frozen-base hashes must match; verify the loaded untuned parameters against
  the source identity and hash them before/after this measurement workflow.
- Resolve relocated artifacts from the current root and manifest identities.
  Do not edit historical absolute paths or infer source identity from directory
  names alone. Missing source evidence stops preparation with useful diagnostics.
- Use all 150 frozen held-out GSM8K items from Pilot 2, with identical questions,
  gold solutions, ordering and source provenance. Preserve calculator annotations
  and final-answer markers in the gold solution.
- Render the source prompt and assistant gold completion under the source
  tokenizer/chat template, preserving actual system context. Save the rendered
  input and token IDs. This is teacher forcing, not generation.
- Keep complete GSM8K sequences. Validate the model's sequence limit and stop
  on an oversized example; never silently truncate or drop it.
- FineWeb control: use the user-approved `sample-10BT` configuration; inspect
  the first 2,000 documents of a pinned, explicitly
  recorded stream configuration. Keep documents with at least 128 Qwen content
  tokens, then sample 150 without replacement using seed 42. If fewer than 150
  qualify, stop rather than silently expanding the pool or changing eligibility.
  Freeze selected texts, token IDs, source identities and content hashes.
- Use the first 128 content tokens from each selected FineWeb document. Present
  the passage as user text under the same chat wrapper; freeze the final wrapped
  tokenization. Record the truncation rule and actual counted-token lengths.
- FineWeb is a sample of a bounded stream prefix, not a representative web
  population. Prefer comparison with the GSM8K question view; solution results
  remain separate. Context length and content/task differences remain confounds.
- Tokenizer-aware masks identify three views: GSM8K question, GSM8K solution,
  and FineWeb user content. Exclude system text, task instructions, chat/control
  tokens, padding and ambiguous boundary-crossing tokens. Report exclusion
  counts and empty-view coverage; keep excluded context in the input itself.

### Measurement definitions

- Hook the outputs of all 28 decoder blocks directly, before any final model
  normalization outside the last block. Verify hook identities and tensor shapes.
- Measure all 196 adapted projections: q, k, v, o, gate, up and down in each
  block. Verify rank 1, alpha 1, dropout 0 and exclusion of the language-model
  head against source checkpoint configuration and actual attached modules.
- For each counted token at each block, define change as adapted output minus
  untuned output on identical fixed model-input sequences. This includes
  propagated upstream changes; it is not an isolated module intervention.
- In an adapted pass, capture each linear module's actual input x. Measure its
  direct contribution as the scaled LoRA product on that same x, and its ordinary
  output as the frozen linear transformation on that x. The ordinary module
  output denominator is not borrowed from the untuned pass's different input.
- Use the actual source scaling alpha/r in the identity even though it equals
  one for this protocol. Ordinary module output includes any frozen bias;
  the direct contribution does not.
- Model computation uses FP32, evaluation mode, no autocast and TF32 disabled.
  Accumulate sufficient-summary sums in FP64. Record effective settings.
- For a selected view, primary weighting gives every counted token equal
  weight. Sensitivity weighting first averages each nonempty example's tokens,
  then gives every such example equal weight. Counts and empty examples remain
  explicit for both views.
- Save mean block change vector and mean untuned vector. The primary block
  scalar is magnitude of mean change divided by mean baseline magnitude.
  Also retain mean change magnitude and the alternative denominator, magnitude
  of mean baseline vector. Mean magnitude and magnitude of a mean differ when
  directions cancel; neither is a substitute for the other.
- The primary module scalar is mean direct-contribution magnitude divided by
  mean ordinary-output magnitude. Retain mean per-token ratios as a secondary
  diagnostic. Apply both weighting schemes consistently to numerator and
  denominator. Module ratios are not weight-norm ratios, additive shares of a
  block change or measures of causal importance.
- Exactly zero denominators produce null values and explicit coverage. Small
  nonzero denominators remain defined with denominator diagnostics; add no
  epsilon. Zero numerator with nonzero denominator produces zero.
- A ratio of means includes tokens with individual zero denominators and is
  undefined only if its mean denominator is zero. Mean per-token ratios use
  defined tokens only, report their conditional coverage, and under equal-example
  weighting average within each example's defined tokens before averaging
  eligible examples. An example with no defined ratios is excluded with a count.

### Instrument gates and numerical thresholds

- Gate A: step 0 must give exactly zero block changes and direct module
  contributions on all counted measurement positions. Approximate tolerances
  cannot excuse a nonzero initialization result.
- Gate B: frozen-base hashes must match exactly, and every loaded checkpoint
  must match its verified source files. Missing modules, incorrect hooks,
  misaligned inputs/masks, nonfinite values or identity failures stop the run.
- Gate C: validate the rank-1 identity in all 196 modules at all five checkpoints,
  including both the pre-addition branch and the adapted-minus-ordinary output
  subtraction on the same module input. CPU tiny-model tests cover all valid
  fixture tokens. Real validation uses up to 16 fixed counted positions per
  module/checkpoint/view, spread across examples; save their identities.
- Preserve full multiplication input shapes before selecting validation
  positions, so a sliced matrix multiplication does not change the calculation
  being compared. Scientific aggregates include every counted token.
- Per coordinate, the pre-addition acceptance threshold is
  1e-6 + 1e-5 times the absolute expected contribution. Post-addition subtraction
  adds 4 times epsilon32 times the sum of absolute adapted and ordinary outputs,
  with epsilon32 equal to 2 to the power minus 23. These are fixed engineering
  acceptance thresholds, not guaranteed floating-point error bounds.
- Save maximum errors, threshold comparisons and the fraction of sampled
  contributions below the subtraction-rounding allowance. This coverage refers
  to validation positions, not every measured token.
- Negative CPU controls include wrong signs, scaling, module/token mapping
  and nonzero writes above the floor. A synthetic alpha/r unequal to one must
  catch omitted scaling. Failures cannot be fixed by loosening thresholds.
- Borderline or unexpectedly variable real validation requires expanded
  diagnostics and inspection before proceeding; it must not silently pass by
  altering the acceptance policy.

### Execution, profiling and reuse

- Load one model. Run disabled-adapter and enabled-adapter forward passes
  sequentially on identical batches, with examples parallel within each batch.
  Do not load two full models or use generation for this pilot.
- Before reference reuse, verify that checkpoint switching leaves disabled-
  adapter outputs identical under fixed batch membership and numerical settings.
  If invariance fails, stop; do not reuse a mismatched reference.
- Reuse one untuned block-output reference per fixed batch across all five
  checkpoints. Keep the temporary cache bounded and release it after use.
  Main work is one untuned plus five adapted passes per batch, with validation
  and profiling overhead measured separately.
- Profile candidate batch sizes 1, 2, 4, 8 and 16 on fixed inputs including the
  longest GSM8K sequences and the real reference cache. Record synchronized
  timings, throughput, memory and summary agreement with batch 1.
- Summary agreement uses atol 1e-5 and rtol 1e-5; compare defined numerical
  summaries and require matching identity, count and undefined-value coverage.
  Choose the fastest candidate passing memory and agreement checks after review.
  Do not hardcode a T4/L4 or other GPU-family whitelist.
- Benchmark OOM marks that candidate unsuitable and continues safely to other
  candidates. Other validation errors stop. Production OOM stops and preserves
  completed shards; no automatic batch, precision or input-length fallback.
- Freeze ordered batch membership, token positions, masks and numerical settings
  before production. Estimates come from forward-pass profiling, not earlier
  autoregressive-generation timings.

### Artifacts and resume

- Each run persists a full configuration: source model/tokenizer/dataset pins,
  adapter/checkpoint identities, prompt template, selected indices/IDs, seeds,
  masks/views, hook/module identities, numerical settings, batching, measurement
  definitions, validation policy and artifact schema version. Explicitly record
  forward-only execution with no decoding or answer scorer, rather than inventing
  generation settings for this run.
- Save frozen input evidence, ordered batch manifests, status, logs, validation
  diagnostics, per-example summaries, aggregate arrays and structured results.
  Store numeric arrays as NPZ and metadata/records as JSON or JSONL.
- Retain per-example block vector sums, untuned vector sums, magnitude sums and
  counts; retain corresponding module magnitude/ratio sufficient summaries and
  coverage. These support both weighting schemes and example bootstrap without
  another GPU run. Full token-by-token activations are not retained by default.
- Save a shared baseline shard once per batch and checkpoint-specific shards
  per completed batch/checkpoint combination. Use temporary outputs, verify
  hashes and shapes, then write the completion marker last. Do not assume Drive
  provides an atomic multi-file transaction.
- Completion markers bind payload hashes to input/config/batch/checkpoint
  identities. Resume skips only verified complete combinations. Incomplete
  unmarked payloads are not scientific evidence; corrupt marked payloads stop
  with diagnostics. Aggregation counts every completed combination exactly once.
- Prevent concurrent writers to the same named run. Monitoring tolerates
  transient empty/partial progress reads while tracking actual process status;
  final scientific payload integrity remains strict. Instructions explain how
  to inspect an active child before starting another run.
- Reports require all configured combinations and passing gates. A partial run
  exposes progress and diagnostics, not a misleading completed scientific report.

### Reports, uncertainty and guided Colab

- CPU reporting reconstructs results from verified saved summaries without
  importing model-training dependencies or downloading/loading model weights.
- Block plots use token-view rows and weighting columns, checkpoint lines and
  95% intervals for the primary scalar. Module heatmaps use 28 layer rows and
  seven projection-type columns, separated by checkpoint/view; supplementary
  views show weighting and denominator diagnostics.
- Bootstrap examples with 2,000 draws and seed 42 for primary scalar summaries.
  Resample GSM8K IDs jointly across checkpoints and question/solution views;
  resample FineWeb separately. Recompute nonlinear ratios from resampled
  sufficient summaries rather than averaging previously calculated ratios.
  Report undefined replicate coverage rather than converting nulls to zero.
- Save structured results, plot artifacts and all provenance. Report zero and
  undefined coverage, denominator statistics, validation errors and sampled
  numerical-resolution diagnostics beside the scientific figures.
- Keep the pinned question visible: do longer gold solutions dominate the
  token-weighted write profile, and does equal-example weighting change it?
  Fixed gold lengths are not generated response lengths. No causal mechanism
  follows from weighting sensitivity or similarity to training-target lengths.
- Intervals condition on one training seed and the frozen cohorts, not variation
  over training seeds or a representative web population. Cross-input differences
  and depth profiles are descriptive, not layer-placement recommendations.
- Guided Colab stages explain dependencies, input/output locations, rationale,
  expected outputs, failures and resume: setup/Drive, source verification, input
  audit, instrument checks, batch profiling/review, measurement, CPU reporting.
  Include a clear point at which the GPU can be disconnected after verification.
- Verify notebook checkout revision and required entry points before expensive
  execution. Explain kernel/module staleness and fresh-process recovery after
  code changes. Preserve interview questions and alternatives in repo documents.

## Testing Decisions

- User confirmed the layered approach before this spec. Prefer the public
  workflow/artifact boundary as the main behavioral test seam. Do not couple
  tests to private helper names, call order or classes solely to mirror the code.
- Focused CPU tests use independent hand-calculated expected values for vector
  sums, cancellation, norm normalization, unequal example lengths, empty views,
  zero denominators, conditional ratios and nonlinear bootstrap reconstruction.
- Offline workflow tests exercise preparation through saved reports using
  controlled input/model fixtures, including interruption, resume without double
  counting, corrupt completed shards, incompatible configs, duplicate writers,
  transient monitoring reads and validation failure that blocks completion.
- Locally constructed tiny Qwen + PEFT integration tests exercise real hooks,
  adapter switching, padded batches, zero initialization, rank-1 identities and
  frozen weights. No CPU test downloads a model or requires CUDA.
- Real GPU preflight verifies actual saved checkpoints, longest-input capacity,
  strict FP32 settings, disabled-adapter invariance and batch-summary agreement.
  CPU integration success cannot be presented as GPU validation.
- Prior art includes the repo's matched-checkpoint evaluation tests, tiny-model
  SFT backend tests, immutable offline rescoring, paired reporting tests and
  notebook transient-status recovery tests. Extend their observable contracts
  rather than copying their implementation assumptions.
- Include planted negative cases and reports with undefined measurements.
  Passing numerical helpers alone is insufficient: inspect the connected
  report output and provenance, analogous to the earlier scoring failure.
- Implementation follows red/green TDD and commits after green steps. Run the
  existing offline CPU suite as a regression gate. Never change accepted
  numbers or tolerances merely to make tests pass.

## Out of Scope

- Retraining, new adapter objectives, additional model families, MMLU inference
  and repeated GSM8K answer generation or rescoring.
- Steering, projection/removal experiments, causal attribution of behavioral
  changes, SAE encoding, Jacobian lenses and layer-selection recommendations.
- Quantization, mixed-precision variants, merging adapters into base weights,
  arbitrary full-token activation retention or multi-GPU execution.
- Additional training seeds, representative web sampling, unapproved length
  stratification and generated-response activation cohorts.
- Automatic repair of historical configs, silent tolerance relaxation or
  automatic production fallback after OOM.

## Further Notes

- This spec supersedes intermediate open-question notes in the interview; the
  interview remains chronological evidence of how decisions were reached.
- The user confirmed the testing boundaries and authorized specification on
  2026-10-04. Specification does not authorize immediate implementation: next
  apply to-tickets, review the vertical slices and dependencies with the user,
  and publish tickets after approval.
- Earlier pilot safeguards are preserved in the companion learning document.
  Scientific results are still pending real source verification and GPU execution.

## Comments

- 2026-10-04: User confirmed the layered verification approach and requested
  creation of the spec after completion of the questions.
