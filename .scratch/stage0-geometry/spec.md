# Stage 0: controlled interventions on existing rank-1 adapters

Status: ready-for-agent
Approval: public saved-workflow test boundary and 15-ticket dependency graph approved by the user on 2026-10-26.
Date: 2026-10-26. Scientific design decisions are approved. No new GPU run or
implementation is authorized merely by publication of this specification.

## Problem Statement

Existing SFT and GRPO adapter runs differ in behavioral outcomes, response length
and internal measurements. The researcher needs to test whether specified
components of these rank-1 updates contribute to those differences. Descriptive
geometry alone cannot establish that removing a component preserves task accuracy
while changing length or distributional drift.

Earlier pilots also imposed substantial operational costs: hidden notebook state,
environment drift, repeated profiling, GPU time spent on CPU analysis, unclear
completion evidence, extraction errors and resume failures. Stage 0 must reuse
verified existing adapters, preserve raw evidence and isolate expensive model work
from repeatable analysis, within a cumulative ceiling of 20 Colab compute units.

## Solution

Provide two readable notebooks using the same parameterized saved-workflow
commands. The CPU notebook audits sources, computes and caches singular bases,
constructs interventions, performs paired statistics and produces reports. The GPU
notebook qualifies the instrument, measures costs, freezes approved execution
plans, and collects greedy responses, fixed-context KL and effective writes.
Both restore durable state independently; neither relies on the other's variables.

The main deliverable is E0.4: paired comparisons of unedited references, principal
component removal, size controls and random-subspace controls. Conditional intruder
interventions V6/V6c extend that comparison when eligible intruders exist. Dataset
surprisal, initial gradients, activation-energy decompositions and trajectories
are supporting work, scheduled only within remaining approved budget. Their
absence does not prevent a primary intervention report.

## User Stories

1. As a researcher, I want the original proposal and approved amendments retained,
   so the executed experiment can be traced to its actual decisions.
2. As a researcher, I want an evidence-backed source inventory, so planned products
   are not mistaken for completed measurements.
3. As a researcher, I want actual adapter tensors and scales verified, so the
   experiment uses rank one, alpha one and the intended adapted linear modules.
4. As a researcher, I want checkpoint-zero factors compared directly, so equal
   random seeds are not mistaken for equal initializations.
5. As a researcher, I want exact cohorts, prompts, masks, tokenization and source
   revisions bound to each run, so comparisons use the same intended inputs.
6. As a researcher, I want source scoring history and human labels retained, so
   extraction corrections do not silently rewrite original behavioral results.
7. As a Colab user, I want a CPU notebook that can restore state without CUDA,
   so preparation and analysis do not consume GPU credits.
8. As a Colab user, I want explicit environment setup and a release pin, so a
   reconnect cannot silently change the code or dependencies.
9. As a Colab user, I want runtime mismatches explained before expensive work,
   so environment failures do not waste long model runs.
10. As a researcher, I want each long job to publish sealed resumable units, so
    a disconnect repeats only unfinished work.
11. As a researcher, I want each product to have one writer, so two notebooks
    cannot corrupt or overwrite the same run.
12. As a researcher, I want failed products preserved with errors and diagnostics,
    so changed policies require explicit new execution identities.
13. As a researcher, I want progress counters tied to actual completed units,
    so process heartbeats are not confused with scientific completion.
14. As a researcher, I want measured timing, storage and credit forecasts, so
    decisions use actual costs rather than optimistic estimates.
15. As a researcher, I want one cumulative credit ledger across sessions, so
    diagnostics and optional work cannot bypass the 20-unit ceiling.
16. As a researcher, I want reusable weight decompositions, so multiple analyses
    do not repeatedly compute the same singular bases.
17. As a researcher, I want factor and subspace measurements per adapted linear
    module, so layer/type summaries remain traceable to their underlying values.
18. As a researcher, I want principal edits and their matched size controls saved
    as distinct adapters, so direction effects can be compared with magnitude effects.
19. As a researcher, I want each random control draw retained individually, so
    three draws are not disguised as a calibrated null distribution.
20. As a researcher, I want a mixed merged-weight loader with an intentional
    double-application control, so V6 does not accidentally apply LoRA twice.
21. As a researcher, I want numerical gates approved from bounded diagnostics,
    so tolerances are not silently selected after production failures.
22. As a researcher, I want new references evaluated with all edits in one runtime,
    so old hardware measurements do not confound the new paired comparison.
23. As a researcher, I want every generated answer and cap flag saved, so aggregate
    accuracy and shortening can be reconstructed and audited.
24. As a researcher, I want targeted intended-answer review, so extraction failures
    can be examined without another exhaustive labeling task.
25. As a reviewer, I want previous labels reused only for identical text, so a
    changed output cannot inherit an old human judgment.
26. As a researcher, I want KL and effective write collected together, so drift
    is not joined to incompatible historical write measurements.
27. As a researcher, I want formal non-inferiority with a fixed five-point margin,
    so favorable point estimates do not become unsupported preservation claims.
28. As a researcher, I want inconclusive and pending-review findings reported,
    so incomplete evidence is useful without being called a completed success.
29. As a researcher, I want rotation measured on whole subspaces, so harmless
    basis mixing does not become apparent mechanistic change.
30. As a researcher, I want chance-aware intruder eligibility, so undecidable
    modules are not counted as modules with no intruders.
31. As a researcher, I want V6 matched to nearby non-intruder singular values,
    so its effects are compared with a meaningful spectral control.
32. As a researcher, I want full and matched V6 products distinguished, so
    unavailable controls do not silently alter the main intervention.
33. As a researcher, I want rollout completion to recover the historical sampling
    contract, so missing questions are not generated under an invented procedure.
34. As a researcher, I want NLL and excess surprise named correctly, so model
    entropy is not mistaken for data entropy or KL.
35. As a researcher, I want raw gradients and negative-gradient pushes retained,
    so descent direction is not confused with the loss gradient or an Adam update.
36. As a researcher, I want factor orientation anchored to saved initialization,
    so equivalent sign changes do not reverse alignment claims.
37. As a researcher, I want principal, complementary and cross energies retained,
    so interference is visible rather than hidden inside a misleading fraction.
38. As a researcher, I want primary and optional stages independently runnable,
    so supporting analyses cannot hold the causal intervention path hostage.
39. As a collaborator, I want structured tables and raw files alongside figures,
    so I can inspect or reanalyze every reported result.
40. As a collaborator, I want one-seed and cohort limitations stated clearly,
    so descriptive differences are not generalized to all SFT or GRPO training.

## Implementation Decisions

### Scope, priority and public interfaces

- Reuse the existing pinned Qwen2.5-1.5B-Instruct model, Pilot 2 SFT and Pilot 4
  GRPO adapters. Expected adapter settings are rank one, alpha one, dropout zero,
  seven adapted linear module types across 28 decoder blocks. Verify these
  facts from selected sources; a similarly named folder is not evidence.
- Primary endpoint comparisons use checkpoint 64. Original intermediate states
  8, 16 and 32 and checkpoint zero remain available to optional trajectory and
  initialization analyses; they are not a mandatory production matrix.
- Extend the existing public workflow-command family with Stage 0 prepare/audit,
  basis, catalogue, qualification, freeze, run/resume, verify and report operations.
  Names are implementation choices; scientific identities and behavior are the
  contract. Prefer existing recovery, scoring, notebook orchestration and report
  mechanisms to parallel infrastructures or a broad central-CLI refactor.
- Use focused modules for source identity, spectral mathematics, intervention
  construction, model-backed collection, statistics and orchestration. Keep
  model-specific operations behind the existing replaceable backend pattern.
- Notebook cells expose readable settings, purpose, resource type, input product,
  output product and next prerequisite. Scientific logic lives in shared tested
  modules. Both notebooks evolve as slices land; there is no notebook per ticket.

### Source contract and status

- Every selected source has model/tokenizer revision, payload hashes, adapter
  factors/scaling, module map, cohort identities, prompts, mask/position rules,
  code identity and available runtime metadata. Record missing evidence explicitly.
- Audit actual checkpoint-zero A and B tensors. B must be zero for zero-update
  checks. If the two A0 states differ, optional initial-gradient comparisons
  compute both starts and label them; do not substitute one silently.
- Recover historical loss/reward contracts: SFT is supervised-token-normalized
  masked CE; GRPO is flexible-v3 reward, DAPO, sample group SD plus 1e-4, zero
  reward on capped continuations with their tokens retained in the loss mask.
  Saved source configurations and implementation hashes are authoritative.
- Identify the existing untuned eight-draw baseline and verify that all 128
  question IDs belong to the common 512 training cohort. Recover recorded seed
  ordering, generation engine, dtype, caps and prompt/template. Do not infer
  seed policy from source dataset IDs. Report whether required raw payloads exist.
- Distinguish fixed-write reports from the separate coefficient/KL supplement.
  Audit actual completion seals before claiming reuse. Historical human-answer
  corrections are bound to their confirmed source responses, not a fixed count
  copied from the original proposal.
- Use explicit states for prepared, incomplete, failed, awaiting-review,
  qualified, frozen, complete and not-applicable. Software implemented, CPU
  tests passed, real GPU qualified and scientific run complete are separate facts.

### Artifacts, coordination and resume

- Plans, runs, raw tensors/responses, diagnostics, metadata and reports use the
  canonical repository artifact root and a separately named Stage 0 namespace.
  Source code remains in the normal repository source tree, not generated runs.
- Frozen identities bind source hashes, model/tokenizer/scorer, cohort and masks,
  precision, batching/microbatching, intervention definition, runtime policy,
  code release, statistical settings, qualification receipt and approvals.
- Each job publishes durable work units at a scientifically meaningful boundary:
  module decompositions/edits, question sampling groups, gradient batches,
  variant evaluation batches and fixed-context measurement batches. Incomplete
  unit writes cannot be consumed or counted as completion.
- Publish verified unit payloads and content-bound receipts before downstream
  visibility. A final receipt reconciles expected units, hashes and counts;
  an empty DONE marker, a live PID or a JSON status alone is not completion.
- On resume verify sealed units, reuse them without recomputation, and repeat
  only unfinished units. Altered sources, cohorts, interventions, code/runtime
  contracts or gate policies require rejection or a separately named derived run.
  Preserve failed plans and diagnostics; never silently reseal altered evidence.
- One active writer owns each product. CPU/GPU notebooks write separate products
  and consume only verified products. Refuse an already owned product rather
  than assuming independent Drive clients provide a reliable distributed lock.
  Recover stale ownership only through an explicit recorded operation.
- Save every important product to durable storage before releasing a GPU. Local
  staging is permissible only with verified durable publication. Verify artifact
  roots and practical write/read behavior; do not claim that mounting Drive
  proves all subsequently written bytes are backed up.
- Fresh notebook kernels restore selections and prerequisites from saved plans.
  Missing optional jobs remain optional/pending. CPU setup/status/report paths
  must not import Torch, require CUDA or download a model unnecessarily. Actual
  CPU SVD may load explicitly selected cached/downloaded weight payloads.
- Real Colab GPU setup is pinned and checked before work. Record GPU name, driver,
  CUDA, Torch/dependency versions, numerical flags, device memory and code identity.
  Suitable available GPUs are allowed; FP32 weights/compute, no autocast and
  TF32 off remain mandatory. Preserve applicable deterministic and padding rules.
- New paired comparisons share one recorded session/GPU. Interrupted cross-session
  continuation may join only after applicable compatibility checks; otherwise
  it becomes a separately identified comparison. Historical seed equality or
  similar reward rates cannot substitute for these checks.

### Cost and approval contract

- The cumulative Stage 0 ceiling is 20 Colab compute units across all sessions,
  diagnostics, references, controls, repeated work and supporting jobs. Stop and
  ask before exceeding it. Approval of the overall ceiling is not an instruction
  to spend it all or approval of an unmeasured production job.
- Maintain a durable append-only usage/accounting ledger with timestamps, observed
  unit readings, session identity, per-job measured seconds, forecasts and budget
  approval references. Distinguish observed from estimated consumption; missing
  readings cannot be reported as zero. Reconcile session totals without double
  counting overlapping notebook jobs or resetting on resume.
- Measure a bounded representative workload and storage before proposing production
  cost. Display variant counts, expected model calls, completed/missing work,
  estimated units and storage. Reserve the approved projected cost before work;
  check budget at safe unit boundaries and stop early enough to respect the cap.
- The CPU notebook prepares work while the GPU is disconnected. GPU jobs load a
  model once where safe, switch verified adapters/merged modules with restoration
  checks, and avoid holding GPU time for CPU SVD, bootstraps or plotting.
- Prerequisite source audit approval, evidence-derived qualification approval,
  measured-budget approval and frozen production plan remain explicit stop points.
  Optional jobs run only after primary work is protected within the budget.

### Base spectral cache and geometric measurements

- Use each untuned adapted linear weight W0; the factor convention is update =
  (alpha/r) b a-transpose with a the transpose of the saved A row and b the saved
  B column. Never silently change the saved factor gauge.
- Compute exact reduced SVD in float64 on CPU, one module at a time. Retain all
  singular values and enough left/right vectors for k30/k50/k70, the floored
  intruder subspaces and construction/verification. Validate dimensions. Bind
  storage dtype and independent reconstruction/orthogonality evidence.
- k30/k50/k70 are the smallest ranks reaching 30/50/70 percent squared singular
  energy. Primary principal edits use k50. Cache verification applies to the
  full decomposition before retained-vector truncation; never test truncated
  reconstruction as though all directions were retained.
- Factor projection energy is ||basis-transpose v|| squared divided by ||v||
  squared, with zero-norm factors undefined. Report chance k/d, initialization,
  the existing GRPO-norm-matched random adapter and 100 Gaussian direction draws
  where proposed. Distinguish descriptive control bands from hard identities.
- Always execute rotation and intruder measures when E0.3 is run. Never skip them
  because the update/sigma_k ratio is small. Record that ratio and operator-update
  norm divided by sigma_k minus sigma_(k+1). A positive update over zero gap is
  infinite; zero over zero is undefined; no invented epsilon.
- Compare whole old/new top-k subspaces through principal angles, retaining the
  base-defined k. Report boundary singular gaps and degeneracy limitations.
  Individual vector identity inside degenerate clusters is not a stable mechanism.
- For each new top-20 left unit singular vector u, intruder overlap is
  ||U_old,m-transpose u||. Primary m=max(k50,40); k30/k70 sensitivity bases are
  similarly floored at 40. Keep the 0.3-0.7 sweep on norm, not squared energy;
  primary flag is strictly below 0.5.
- Report sqrt(m/d_out) beside overlaps as the RMS isotropic chance reference.
  A module is undecidable at any threshold at or below this reference. Log raw
  values but do not count undecidable modules as flag-negative or edit-eligible.
  Validate m against available singular dimensions rather than silently clipping.

### Principal intervention catalogue

- V0a and V0b are unedited SFT64 and GRPO64 references, freshly evaluated with
  interventions. Save their source adapter identities and fresh runtime outputs.
- V1 removes SFT b's projection onto the base top-k50 output subspace. V2 removes
  SFT a's projection onto the base top-k50 input subspace. Apply per adapted
  linear module, preserving alpha/r and all other factors.
- V3 removes a random k-dimensional output subspace from SFT b, using three
  separately saved Gaussian/QR draws and the declared seeds 1000,1001,1002.
  Record draw/module identity and bases or reproducible construction receipts.
- V4 scales the original SFT update per module to match V1's ||b|| ||a|| norm,
  preserving direction. Add the analogous size control matching V2. A zero target
  produces a zero update; unchanged zero source/target is an identity. Invalid
  scale requests must be reported, not replaced by a convenient epsilon.
- V5 applies the V1 output-side operation to GRPO. Other variants remain rank-one
  PEFT products. k=0 construction reproduces original factors byte for byte;
  finite-precision orthogonality uses a declared independently justified gate.
- Every variant has immutable edit specification, source hashes, per-module
  before/after norms, selected bases, seed, payload hashes and verification.
  No broad training or checkpoint mutation occurs.
- Preserve the unamended numerical acceptance values from the supplied design:
  full-SVD relative reconstruction below 1e-6; retained-basis orthogonality to
  1e-6; tiny-model gradient reconstruction relative error below 1e-5; directional
  finite difference at epsilon=1e-4 with relative agreement 1e-3; 100 Gaussian
  directions' mean overlap within 10 percent of k/d per module; and declared
  activation reconstruction checks to 1e-6. A finite-precision or stochastic
  failure stops the affected branch with diagnostic evidence; it is not permission
  to choose a more convenient tolerance. User-amended gates above take precedence
  only for the quantities they explicitly replace. Undefined relative denominators
  must be reported rather than hidden by an unapproved epsilon.

### Conditional intruder interventions and loader qualification

- V6 exists for either arm only when E0.3 finds primary-flagged directions in
  eligible modules, using m=max(k50,40), norm threshold 0.5 and threshold above
  the chance reference. Halve all flagged singular values, leaving singular
  vectors and other singular values unchanged. Log before/after values and every
  skipped module with a reason.
- Merge the original adapter into affected weights and save full edited matrices.
  In edited modules disable the adapter; in untouched modules retain it. Do not
  force this result into a rank-one adapter. Multiple singular-value changes
  need not admit a rank-two delta.
- V6c uses distinct eligible non-flagged directions from new top-40. Evaluate the
  overlap predicate for those candidates too; candidates outside detection top-20
  are not automatically non-flagged. Exclude undecidable modules. Minimize total
  absolute singular-value difference by assignment, with deterministic index
  tie-breaking, without control reuse.
- If a module lacks enough controls, exclude it from both edits of the matched
  comparison; retain full V6 separately. Save control/flagged singular-value ratios
  per pair. Repeat the matched comparison retaining only pairs with ratio in
  [0.8,1.25], without rematching. Undefined zero-denominator ratios are logged
  and excluded from that sensitivity. All comparison masks/variants are separate.
- Qualify the mixed loader using un-halved merged matrices against the original
  adapter model on 16 frozen held-out teacher-forced sequences. Record maximum
  absolute full-vocabulary logit error at every answer-prediction position,
  global maximum E and worst coordinates. Compare greedy sequences on the same
  16 prompts in the same runtime.
- An intentional negative loader applies the adapter twice in merged modules;
  its error must be at least 1000 times the real loader's. Retain raw errors and
  explicit zero-denominator status. A vacuous both-zero ratio does not demonstrate
  sensitivity and must not be reported as a successful negative control.
- Stop if E>1e-3. Otherwise propose the absolute allowance 3E for user approval.
  Keep calibration evidence and approved qualification separate; deriving a limit
  from the same observations is not independent validation. No automatic tolerance
  widening or promotion to qualified state occurs.
- A greedy mismatch qualifies as a descriptive near-tie exception only if the
  reference top-two margin is strictly below measured error at the same shared
  prefix. Save local prefix logits, candidates, margin and error, not comparisons
  of unrelated post-divergence contexts. A mismatch outside this rule blocks
  qualification. Near-tie exceptions never become claims of token identity.

### Paired behavioral collection and answer review

- Use the exact Pilot 4 held-out 150-question cohort, template/prompt, tokenizer,
  greedy procedure and 1024 new-token cap. Flexible-v3 is primary; strict is
  recorded alongside. Bind cap/end-token/invalid semantics to the existing scorer.
- Re-evaluate both references in the same runtime as all variants of the paired
  comparison. Fix and record batching and generation settings; all methods share
  the same input IDs and scheduling where applicable. GPU identity alone is not
  an equivalence claim across sessions or software versions.
- Save every response as JSONL with variant/question IDs, rendered prompt or
  source hash, generated text/token IDs, generated length, cap/end metadata,
  scorer version, extracted answers, valid flags, reward, runtime and unit ID.
  Statistics consume verified raw records rather than console aggregates.
- Review answer-changing cases and suspected extraction failures using the
  existing intended-answer review workflow. Reuse a confirmed human answer only
  when response text is identical, with links to the matching label/evidence.
  Do not copy old corrections onto new text or label an automated proposal human.
- Publish raw automatic results immediately when complete; separately publish
  targeted reviewed overlays and pending-review status. Partial review is not
  exact overall human-corrected accuracy. Strict-only formatting disagreement
  does not mandate exhaustive new review.

### Fixed-context drift and effective-write collection

- Use the fixed 150 GSM8K gold and 150 FineWeb measurement sequences selected by
  existing prepared products. Preserve token views, counted positions, masks,
  padding, position IDs and tokenizer identities; report each corpus/view separately.
- Compute full-vocabulary KL(p_variant || p_untuned) on identical teacher-forced
  contexts with float64 accumulation. Retain per-position KL and mapping. Do not
  store full-vocabulary log-probabilities as a production requirement.
- Collect effective-write sufficient statistics in the same runtime as any new
  KL: direct module update norm relative to base module output norm on declared
  inputs, plus sum of module factor-norm products where rank-one factors exist.
  For merged V6 products use the actual edited-minus-untuned matrix and its norm;
  never fabricate a rank-one factorization to fill that field. Report intervention
  representations and magnitude definitions separately.
- Reuse a disabled-adapter/untuned reference safely within frozen batches. Verify
  switching/restoration, source weight integrity and declared numerical identities.
  No-operator and zero-update controls need mapping and finiteness checks, not
  merely a small aggregate mean.
- New KL/write pairs do not use saved historical values as inputs. Historical
  same-run products may be analyzed together only with compatible verified
  provenance. Cross-session checks regenerate common inputs and recorded seeds;
  use independently established applicable envelopes, otherwise report separately.
  Old128/new384 reward-rate similarity is descriptive, never a hardware gate.
- Preserve the prior coefficient policy: batch sensitivity of raw coefficients
  is descriptive where authorized; strict repeat, chunk, KL, reconstruction,
  mapping and integrity rules are not globally relaxed. Stage 0 does not silently
  amend older protocol policies or acceptance numbers.

### Paired analysis, primary claims and preregistration

- Bootstrap paired evaluation items/documents using 10000 percentile resamples,
  seed 42 and 95% intervals. Resample all records of an item together. Tokens and
  adapted linear modules are not independent experimental subjects.
- For each predeclared edited-minus-reference accuracy contrast, claim preservation
  only if its lower 95% bound strictly exceeds -0.05. Note before execution that
  n=150 can leave this inconclusive. Report point estimates, intervals, gains,
  losses, invalid/cap rates and McNemar comparisons; do not infer preservation
  from a nonsignificant accuracy difference.
- Report median paired generated-token length difference, its paired interval,
  shorter-response share and cap rates. Median paired change is not the
  difference of marginal medians. Length is an observed response property, not
  by itself improved or harmed reasoning.
- Average fixed-context KL per document before bootstrap; distinguish equal-example
  aggregation from token weighting. Report principal edits against their references
  and magnitude/random controls, not only the most favorable variant.
- Predeclare the comparison family and hypotheses before relevant results. Present
  every specified contrast and interval; do not select an unadjusted winner and
  label the family a confirmed causal mechanism. Conclusions concern these
  interventions on these one-seed adapters and fixed cohorts.
- Freeze supporting hypotheses before their own measurements. An E0.5-informed
  prediction about trained E0.2 overlap must be timestamped after the gradient
  summary but before trained overlap reveal, and labelled a conditional prediction.
  It is not retroactive preregistration of already observed gradient behavior.
  Preparing a cache must not automatically reveal trained overlap summaries. If
  those results were already inspected while advancing the primary experiment,
  later gradient-to-overlap interpretation is explicitly exploratory; never
  manufacture an unobserved prediction or block the primary path to recover one.
- The main report can complete with optional analyses absent or V6 inapplicable.
  Conditional extension reports can add verified products without overwriting the
  primary report identity. Save JSON/JSONL, table exports, plots, configuration and
  an HTML narrative with explicit links between claims and source rows.

### Optional rollout completion and dataset measurements

- R0 completes 384 missing questions times eight draws, retaining the original
  128 groups and yielding 512 groups/4096 draws only after source compatibility
  and complete unique identities are verified. Preserve old/new origin per row.
  New seeds follow the audited ordered-batch scheme and are frozen explicitly;
  do not silently adopt the proposal's example source-index seed rule.
- Ticket 11 amendment (2026-10-27): exact replay of two recorded historical
  questions × eight draws, before production, may authorize the historical join
  only with matching token IDs, text, stopping metadata and verified source/settings
  identities/hashes. Use the original GPU type and software versions where
  identifiable. Any mismatch or unrecoverable historical seed/order requires a
  fresh uniform 512×8 cohort in the new session, not only a separate new 384.
  Preserve the original 128 as a separately labelled historical product; no
  reward-rate fallback or newly fitted numerical tolerance is allowed.
- Historical sampling is HF FP32, temperature one, top-k zero, top-p one, cap
  1024, one question per eight-draw group. Recover all recorded settings before
  generation; do not substitute a different engine to satisfy a cost estimate.
- Retain strict, flexible-v2 and flexible-v3 scores and cap information, without
  changing the historical training reward. Capped responses get zero reward.
  Derive gold, own_all, own_correct and one deterministically selected correct
  draw per eligible question, with saved selection seed/order and dropped IDs.
- Score each unique answer once under the untuned teacher-forced model, then use
  dataset memberships to aggregate; the overlapping correct subsets do not need
  redundant forwards. Mask exactly the SFT answer-token contract, including the
  audited end-token treatment. Save token surprise, entropy, rank and positions.
- NLL is mean negative log-probability per answer token and the primary measure.
  Excess surprise is NLL minus mean model entropy at those same positions,
  descriptive rather than KL. Summary primary ci_low/ci_high refer to NLL;
  excess-surprise intervals receive distinct names. Bootstrap whole problems.
- Preserve the user-approved own_all excess-surprise gate +/-0.02 nats/token.
  Its zero-expectation premise requires the sampled distribution, scored model,
  contexts, weighting and stopping treatment to match. A failed premise is a
  documented incompatibility, not evidence to widen the acceptance number.
- Report per-problem negative log(k/8) separately for eligible own_correct
  problems and an equal-problem summary. Do not compare this whole-answer quantity
  with per-token NLL or excess surprise. No inference of zero true correctness
  probability from zero successes in eight draws.
- Prediction P1 is own_all excess surprise near zero, own_correct slightly below
  zero, and higher gold NLL with positive excess surprise. The latter directional
  predictions are scientific hypotheses, not mandatory instrument identities.

### Optional initial pushes and activation/trajectory profiles

- At saved checkpoint-zero factors, compute masked SFT gold and own-correct-one
  CE gradients on the same eligible questions and frozen batches of eight. Preserve
  the total supervised-token denominator across microbatches. Record any final
  partial batch, rather than claiming 64 full batches after dropping questions.
- Compute GRPO pushes on the sampled groups using detached historical rewards
  and advantages, sample SD plus 1e-4, and the global DAPO completion-token
  denominator. Dead all-correct/all-wrong groups produce zero advantages. Do
  not normalize each answer independently or silently restrict GRPO to SFT's
  selected-correct problem subset without reporting that cohort distinction.
- Save raw gradients per module and batch. Push is their negative; all alignment
  cosines use push, including separately labelled sign-push diagnostics. No optimizer
  step is taken. Negative gradient is not a reconstruction of AdamW's historical
  update. Record factor gauge and A-versus-A0 alignment.
- Report push size, consistency, principal output overlap and alignment with
  endpoint b, initialization and Gaussian controls. Zero vectors give undefined
  cosine/ratios, not fabricated zero alignment.
- For input-energy profiles, use base-model module inputs and accumulate all
  arms' contractions in the same forward. Save parallel, complementary, twice
  cross and total sums in float64 plus counts. Verify their reconstruction.
  Headline principal fraction divides parallel by parallel plus complementary;
  normalized components against total include the signed cross term. Negative
  cross means cancellation; positive means reinforcement.
- Optional trajectory analysis uses original declared checkpoints and random
  control, without making intermediate capture a prerequisite for E0.4. If new
  KL is needed, collect corresponding effective writes in the same runtime.
  Size-matched observational curves do not establish a causal direction effect.

## Testing Decisions

- Approved primary boundary: invoke the same public saved-workflow
  commands called by notebooks, using tiny CPU-only artifacts and replaceable
  model dependencies. Assert observable outputs, identities, numerical outcomes,
  statuses and resume behavior; avoid tests that merely repeat private loops.
- Use existing public-command behavior tests for interrupted reports, tampered
  sources and label provenance; existing activation/trace tests for unit receipts,
  model switching, frozen batches and qualification; notebook tests for fresh-kernel
  restoration, CPU/GPU guards, release pins and operational progress.
- Add independent small-matrix/explicit-array oracles through that boundary for
  SVD reconstruction, energy ranks, projection edits, principal angles, gap ratios,
  intruder eligibility, matching, edit norm controls and V6 matrix reconstruction.
  Include rotated bases of the same subspace to demonstrate basis invariance.
- Tiny CPU model tests independently reconstruct gradient G*a0 and its sign,
  A-gradient zero at B=0, masked losses and microbatch normalization, detached
  advantages and dead groups. No Hub download or CUDA is required by unit tests.
- Loader fixtures must catch missing adapters, adapter double application,
  incorrect module masking and failed restoration after switching representations.
  Exercise exact 1000x, 1e-3 and near-tie rule boundaries with deterministic fake
  observations; real observed gates remain a separate approval artifact.
- Evaluation tests cover all 150 identities conceptually with tiny reduced fixtures,
  response persistence, tokenizer-length/cap semantics, raw-versus-human overlays,
  identical-text label reuse, gold-independent extraction, and all controls.
- Drift tests use an independent full-vocabulary reference distribution for KL,
  directional asymmetry, counted-position mapping, zero-update identities and
  numerical roundoff diagnostics. Reconstruct effective writes directly from
  small dense matrices, including merged interventions.
- Statistics tests establish paired-bootstrap behavior, median paired changes,
  exact non-inferiority boundary at -0.05, inconclusive intervals, capped outputs
  and separate random draws. Audit aggregate reconciliation against source rows.
- Failure tests cover incomplete writes, altered source/config/review, duplicate
  writers, corrupted seals, missing durable outputs, stale ownership, interrupted
  optional work and fresh notebook processes. No completed sealed unit is rerun
  merely because the notebook kernel restarted.
- Budget tests use an injected clock/meter: restored cumulative cost, forecasts,
  missing readings, overlapping session accounting, early stop and refusal to
  exceed total 20 units. Supporting jobs never bypass primary reservations.
- Optional-branch tests cover cohort mismatches, recorded seed restoration,
  duplicate/missing draws, zero-correct questions, subset membership, answer
  deduplication and distinct token/sequence units.
- Freeze missing real numerical criteria as pending approvals, not invented
  passing constants. Preserve already approved values and failed evidence.
- Required real-Colab evidence is separately recorded: actual environment/import
  and durable I/O checks, bounded timing, instrument/loader numerical evidence,
  approved gate, budget decision, interrupt/resume evidence and production
  verification. CPU tests do not claim target-GPU qualification.

## Out of Scope

- New SFT, sampled-SFT or GRPO training; additional training seeds; new model
  families, higher-rank adapters, NF4/BF16 experiments or reproducing a mentor's
  different recipe.
- Claims that KL is behavioral harm, that lens lists establish causal use, or that
  this one-seed intervention suite explains all historical SFT/GRPO differences.
- A new exhaustive human reasoning-quality audit or independent reviewer agreement.
- Automatically spending GPU credits, fitting thresholds after failures, exceeding
  the 20-unit total ceiling or silently expanding optional analyses.
- Treating unexecuted optional jobs as primary blockers, rebuilding every historical
  measurement, or requiring a universally available A100 40GB allocation.
- Broad repository refactors, moving source code into generated artifact trees,
  altering old reports/configs, or claiming remote backup without verification.

## Further Notes

- The original supplied proposal remains immutable. This specification synthesizes
  the subsequent approved scientific corrections, final six approvals and explicit
  exceptions; those decisions supersede conflicting original intent.
- The current local baseline is 485 passed, 35 skipped and 4 warnings. Some model
  tests were skipped because local Torch cannot load its CPU library. This is a
  stated environment limitation, not evidence that model-backed paths passed.
- Existing artifacts do not yet establish checkpoint-zero A equality or completed
  historical per-token KL production. Obtain evidence through the source audit.
- Keep a claim/evidence map for the primary report. Missing optional output,
  not-applicable V6, pending review and failed qualifications are explicit report
  states, not omissions to conceal.

### Ticket12 reporting conventions approved 27 October2026

The user approved target rank = 1 + number of vocabulary logits strictly greater
than the target, using the same full logit vector used for the NLL softmax.
Record output-head vocabulary size including padded rows, tokenizer vocabulary
size and padded-row count; do not trim the rank or softmax vocabulary.

Position summaries are disjoint first32/middle/last32; first32 takes precedence
for answers shorter than64 scored tokens. Report per dataset the number of
answers under64 and token counts in all three windows, including zero counts.
Gold tokens are exactly the saved Pilot2 loss-bearing tokens, preserving its
template/formatting and supervised end-of-turn. Samples use generated token IDs,
including the observed end-of-turn. Capped samples have no artificial token
appended and their missing end-of-turn count is explicit.

Primary means remain token-weighted. Add answer-weighted NLL, entropy and excess
surprise as sensitivity estimates with separately named intervals. Both primary
and sensitivity bootstrap whole problems, retaining all samples of each sampled
problem together. Saved memberships retain draw multiplicity after deduplication.
No numerical acceptance threshold changes.

### Ticket13 comparison conventions approved 27 October2026

Decide distinct initializations by canonical A0 tensor hashes from the verified
initialization audit, not seeds. Evaluate all three objectives at each distinct
A0; compare objectives only within the same A0. Gold and own-correct-one use
identical eligible IDs and batches. GRPO retains all512, with zero-advantage
groups contributing zero gradient. Cross-objective comparisons are direction-only
(cosines and principal overlap); do not compare raw push norms across objectives.
Primary push is the mean of saved batch pushes, reported with and without the
final partial batch. Exclude partial batches from consistency. Token-weighted
pooled push is secondary. Orient each endpoint pair by sign(a64 dot A0), flipping
a and b together when negative; zero dot is undefined. Report flipped-module
counts (expected none), rather than assuming orientation.

Implementation follows the approved within-session comparison policy: freeze
the first gradient profile runtime for all subsequent gradient objectives/resumes.
The completed R0 token cohort is source data with its sampling runtime retained;
using it does not join historical gradient values. A new gradient comparison may
start in a later suitable GPU session; changing session after its own profile
requires a separately named comparison, absent an approved compatibility check.

### Ticket14 numerical checks approved 27 October2026

The corrected approval supersedes the initially misstated absolute tolerance.
Use atol1e-5 and rtol1e-5 for FP64 energy reconstruction and normalized projection
checks. Compute total energy independently from a·x, never as the sum of the
components. Check V_kᵀV_k against identity; (a_parallel·a_rest)/||a||² against zero;
(||a_parallel||²+||a_rest||²)/||a||² against one; and
||a_parallel||²/||a||² against Ticket02's π_in for the same module, factor/checkpoint
and cutoff. Zero-factor normalized quantities remain undefined. Any failed check
stops and saves diagnostics; no gate widening or post-hoc threshold fitting.

Capture on the original150 GSM8K/150 FineWeb fixed inputs, preserving their
unshifted question/solution/user masks. All selected factors contract the same
untuned base inputs; one body forward supplies every factor and cutoff. Retain
FP64 sufficient sums and per-example counts, not full activation tensors.
