# Stage 0: geometry of existing rank-1 adapters

Status: design decisions, execution specification and ticket dependencies approved
on 2026-10-26. See `spec.md`, `map.md` and the per-ticket execution records for
current software and experimental execution status.

## Source and scope

The supplied proposal is preserved unchanged in
`sources/stage0-design-supplied-2026-10-26.md`. This working design records
review findings and approved amendments. `spec.md` is the execution contract;
the original proposal remains available for comparison. No new training is
proposed. No Stage 0 GPU job has launched.

The proposed research chain is:

1. Characterize the untuned model's sampled answers and gold training answers.
2. Compare their initial loss gradients through the saved rank-1 initialization.
3. Measure trained adapters relative to the untuned weight singular subspaces.
4. Measure behavioral changes and fixed-context distributional drift.
5. Edit adapter components and compare with size and random-subspace controls.

The first four steps provide descriptive evidence. The fifth tests effects of
specified interventions; it does not by itself identify all causes of the
historical SFT/GRPO difference.

### Approved priority

E0.4 is the main result. The gradient/data/geometry explanatory chain is
supporting evidence. Run supporting experiments when the GPU session has room;
their completion does not block the primary intervention result. Their costs
remain estimates until measured; do not assume they are cheap or extend a paid
session without the agreed budget.

Distinguish optional explanatory analyses from intervention prerequisites:

- Required for interpretable E0.4: verified source adapters and evaluation
  protocol, base singular bases needed to construct the selected edits, edit
  identity/loader validation, paired unedited references, selected controls,
  and the declared accuracy/length/KL outcomes.
- E0.3 primary intruder classification is required only to decide eligibility
  and direction selection for conditional V6/V6c. It does not block principal
  projection variants V1-V5. Always compute rotation and intruder measurements
  when executing E0.3; optional status does not reinstate the small-update skip.
- Full rollout completion R0, E0.6a, E0.5, trajectory/sensitivity expansions and
  the historical drift-versus-size analysis are supporting work. Their absence
  must be stated but does not invalidate completed intervention comparisons.
- New KL and accompanying effective-write measurements required by the approved
  within-session policy remain part of the primary outcome collection; the
  separate historical multi-checkpoint E0.6b explanatory analysis is optional.

The previous proposal's all-jobs-before-downstream stop order must be revised
in the execution specification so missing optional work cannot stall E0.4.
Predictions for supporting analyses must still be frozen before their results
are inspected; optional scheduling is not permission for retrospective prediction.

## Proposed execution structure

Two readable notebooks call shared repository modules: GPU computation and CPU
analysis. Code belongs in the repository's source tree, while plans, raw outputs,
status and reports belong under `artifacts/`. Each notebook restores saved paths
and decisions after restart. Analysis consumes verified saved products rather
than Python variables in another runtime. Work units must be resumable within a
long job; publication of a whole job alone would lose too much on disconnect.

Exact schemas, release pins, ownership and completion checks remain to be specified
after the scientific decisions. Existing workflow mechanisms should be reused.

## Findings requiring revision or explicit decisions

- E0.6a's NLL minus model entropy is not generally KL(data || model).
  Cross-entropy equals data entropy plus KL; model entropy cannot be substituted.
  Excess surprisal can be descriptive, with averaging and sampling conditions
  declared. Its proposed relationship to -log P(correct) mixes token and sequence
  quantities and is not an acceptance identity.
- E0.2, E0.3, E0.5 and V6 have explicit user corrections below, including the
  approved old top-m subspace and intruder overlap threshold.
- KL is distributional drift, not established behavioral damage. Comparing
  observational trajectories at similar update sizes does not isolate direction.
- Hardware and compatibility policy were amended by the user on 2026-10-26;
  the final six approvals below supersede earlier disputed proposals.
- New and old disjoint rollout cohorts need not have reward means within five
  points. Cohort diagnostics should not substitute for procedure verification.
- Several exact-zero, random-band and finite-difference acceptance rules need
  independent numerical justification; they must not be silently loosened later.
- Prediction timing conflicts: all predictions before GPU work versus some after
  the gradient summary. Separate prior hypotheses from conditional predictions.
- E0.4 must explicitly select its evaluation prompt, token cap, scorer and audit
  procedure. Pilot 1 and Pilot 4 references are not interchangeable.
- Timing and storage estimates are proposals, not verified resource budgets.

## Source audit in progress

Recover baseline rollout procedure and cohort membership, checkpoint availability,
training objectives/reductions, actual reward extraction, A0 equality, random
adapter construction and completed KL products. Record evidence paths and missing
facts. Human correction counts and their scope must come from the confirmed
behavioral follow-up, not the proposal's unverified count of 13.

## Decision tree

Design interview frontier is closed by the user's six final approvals below.
Source audits, evidence-derived qualification gates and measured budgets remain
execution prerequisites, not unanswered design-preference questions. Translate
approved requirements into explicit notebook stages, saved-workflow interfaces,
schemas and tests; preserve any evidence-based stop points.

The approved specification and 15 implementation slices are published in this
folder; `map.md` records the current implementation frontier.

## User amendments, 2026-10-26

### E0.6a

- Primary measure is NLL per answer token under the untuned model, with prompt
  masking matching SFT. This measures model-assigned surprise, not a general
  mathematical distance between distributions.
- Also report mean model entropy at those positions and excess_surprise = NLL
  minus that entropy. Excess surprise is descriptive, not KL. The zero-expectation
  interpretation requires samples from the distribution actually scored and
  compatible token/context weighting and stopping treatment.
- Retain own_all excess surprise within +/-0.02 nats per token as the user's
  requested acceptance threshold. Do not silently change it. Confirm its sampling
  premises before diagnosing a failure as an instrument bug.
- Record neg_log_p_correct per problem using k/8, for own_correct only, separately
  from per-token measures. Remove the old comparison or agreement test between
  this whole-answer quantity and excess surprise. How the per-problem quantity
  maps to a dataset-summary column remains to be specified.
- Summary columns: dataset, n_answers, n_tokens, nll, entropy, excess_surprise,
  ci_low, ci_high, neg_log_p_correct. The target of ci_low/ci_high must be explicit.
- Replace the former distance column with nll or excess_surprise as appropriate.
- P1: own_all excess surprise approximately zero; own_correct slightly negative;
  gold has much higher NLL and positive excess surprise. These are predictions,
  not all correctness identities. E0.6b KL remains unchanged.

### GPU policy

- Any suitable GPU; prefer A100 when available. FP32 and TF32-off remain mandatory.
- Each paired comparison runs in one session on one GPU. Record GPU, driver,
  CUDA and Torch versions. This includes gold versus own data, the activation
  pass, and all E0.4 variants with fresh V0a/V0b references.
- User proposes recomputing 16 saved values before cross-session or cross-GPU
  joins, using Pilot 3's three-times empirical envelope. Selection of values,
  reference repeat differences, per-metric aggregation, and near-zero treatment
  remain unresolved; this is not yet an executable gate.
- Compute effective write and sum of factor-norm products alongside any new KL.
  Historical values are checks, not substitute inputs. Saved KL/write values
  may be joined only if their provenance establishes matching A100 runs and
  measurement inputs.
- For R0, prefer the historical GPU type if identifiable. User proposes a
  five-point old128/new384 reward-rate fallback when it is unknown. This is
  recorded but scientifically disputed: disjoint question sets confound hardware
  with question difficulty. Agreement on common prompts is the proposed repair.
- User requests relative agreement 1e-4 on the same GPU type and the empirical
  envelope on different GPUs. Near-zero/reference-zero handling, session/runtime
  compatibility and the meaning of the envelope must be fixed before execution.

The original source remains unchanged. These amendments do not approve other
scope changes or settle the remaining mathematical and execution questions.

### Rotation and intruders (E0.3)

- Always compute rotation and intruder measurements. Remove the small-s-ratio
  skip rule and the prediction that rotation is under one degree.
- Retain update/sigma_k descriptively. Add update/gap_k, where
  gap_k = sigma_k - sigma_(k+1), using the operator norm of the effective update.
  For rank one this norm equals ||b|| ||a|| times the adapter scaling.
  This ratio is a sensitivity diagnostic, not by itself a rotation bound.
- Measure rotation using principal angles between entire old and new subspaces.
  Keep the fixed base-defined k when comparing the two. Mixing within either
  subspace does not constitute rotation of that subspace.
- Replace best overlap with any individual old vector by overlap with the whole
  old top-m left singular subspace. For a new unit vector u, retain both
  ||U_old,m^T u|| and its square. Primary m = max(k50, 40); sensitivity m values
  are max(k30, 40) and max(k70, 40). Each basis must exist within the matrix's
  available singular directions; validate dimensions rather than silently clip.
- Primary intruder threshold is projection norm < 0.5. Retain a 0.3-0.7 sweep
  on projection norm, not squared projection energy.
- Report sqrt(m/d_out) beside every overlap as the user-defined chance reference.
  Its square is the isotropic expected projected energy; sqrt(m/d_out) is the
  root-mean-square projection norm, not exactly its mean. This descriptive
  reference does not establish a calibrated null for new leading directions.
- A module is undecidable at any threshold <= sqrt(m/d_out). Preserve numerical
  overlaps but exclude that module from determinate flag/no-flag counts at that
  threshold. V6 eligibility requires 0.5 > sqrt(m/d_out) and a primary flag.
- Retain the proposed top-20 new candidate directions, subject to valid matrix
  dimensions. Degenerate clusters at the candidate boundary can still make
  individual candidate membership ambiguous; report boundary gaps.
- Zero-gap, final-index-with-no-next-singular-value and zero-update cases need
  explicit status fields. Never hide them behind an epsilon that changes the
  meaning of the ratio. Choice of those conventions is pending.
- The SVD artifact contract must support all requested old top-m bases, rather
  than assuming storage through k70 is always sufficient.

### Loss gradients and pushes (E0.5)

- Save the raw loss gradient g. Define push = -g, and use the push in every
  alignment cosine, including any separately labelled sign-push comparison.
- Push is the negative loss gradient, the local steepest-descent direction.
  Historical AdamW updates include preconditioning and optimizer state and need
  not be parallel to it. Do not label this a reconstruction of actual updates.
- Preserve the saved factorization: b a^T = (-b)(-a)^T. Anchor factor orientation
  to the saved a0, and record a-versus-a0 alignment rather than independently
  choosing signs for b. Near-initialization alignment is a hypothesis to verify
  from saved tensors. Do not silently reorient raw gradients or adapter files.
- Comparing the two arms still requires the actual A0 tensor comparison. The
  effective update is gauge invariant; factor-level cosines depend on this
  declared initialization convention.

### Activation-energy decomposition (E0.2)

For the same counted base-input tokens, save float64 raw sums and token counts:

- E_parallel = E[(a_parallel^T x)^2].
- E_rest = E[(a_perp^T x)^2].
- E_cross = 2 E[(a_parallel^T x)(a_perp^T x)].
- E_total = E[(a^T x)^2].

The reconstruction identity is E_parallel + E_rest + E_cross = E_total.
Report all three component values divided by E_total; these sum to one when
E_total is positive, but individual terms need not be bounded or positive.
Headline principal fraction = E_parallel / (E_parallel + E_rest), bounded between
zero and one when its denominator is positive. This excludes the cross term by
definition; it is not the fraction of total energy attributed to a component.
Negative cross terms indicate cancellation and positive terms reinforcement.
Keep raw sums so aggregation never averages incompatible ratios. All-zero or
zero-total cases require explicit undefined flags; no artificial denominator.

### V6: merged-weight intervention

- Run only if the approved E0.3 intruder rule finds intruders.
- Merge the original adapter into W, apply the SVD edit, and save full edited
  matrices for affected modules. Evaluate through a loader with explicit
  merged-weight support. In edited modules, use that matrix with the adapter
  disabled; keep the original adapter active in all other modules.
- For one singular-value edit, W_edit - W0 equals the original rank-one adapter
  plus one rank-one SVD correction, so its rank is at most two. Editing several
  singular values generally requires more than rank two. Full edited matrices
  avoid an artificial rank-two restriction.
- Halve all primary-flagged singular values in eligible modules. The primary
  rule is m = max(k50, 40), norm overlap < 0.5, and 0.5 above the module's chance
  reference. Apply this independently to SFT and GRPO if either has flags.
- Log every selected direction, its overlap, m, chance reference and singular
  value before and after editing. Log skipped modules with reasons, distinguishing
  undecidable from determinate modules with no flagged directions.
- Add V6c per edited arm: in the same modules, halve the same number of
  non-flagged directions from the new top-40. Detection remains on the top-20;
  evaluate the same overlap predicate on control candidates too, so directions
  21-40 are not treated as non-flagged merely because detection did not scan them.
  Exclude undecidable modules. Select distinct controls by minimizing total
  absolute singular-value difference, breaking assignment ties by index.
- Record sigma_control/sigma_flagged for every pair, alongside indices and
  absolute differences. In addition to the main matched comparison, repeat with
  only matched pairs whose ratios are in [0.8, 1.25], applying the same retained
  pair mask to V6 and V6c. Do not rematch after filtering. Report retained pair
  and module counts. Zero-denominator ratio handling remains an explicit edge
  case to settle before implementation.
- Modules short of eligible controls are excluded from both interventions in
  the matched comparison. Keep full V6 separately, including its full eligible
  flagged set. Full V6, matched V6/V6c and ratio-restricted sensitivity therefore
  need distinct variant identities and saved edit manifests.
- Before halving, validate the mixed loader using un-halved merged matrices
  against the original adapter model using the bounded diagnostic below.
- Do not merge the adapter again when loading already edited matrices. Preserve
  W0, original adapter, singular-value selection, correction factors and hashes
  separately. Other rank-one variants retain their PEFT representation.
- This is a different intervention representation, not an approximation forced
  back into rank one. No external-paper equivalence has been verified here.

### Merged-loader bounded diagnostic and gate approval

- Use 16 fixed held-out sequences in the same runtime for the adapter reference
  and un-halved mixed merged loader. Save cohort IDs and token/mask identities
  before execution. Measure max absolute full-vocabulary logit error at each
  answer-prediction position and the overall maximum E. Save worst coordinates.
- Also compare greedy generated token sequences on the same 16 prompts. The
  user allows a mismatch to be labelled a near-tie only when the top-two logit
  margin at the divergence position is strictly less than measured error.
  Evaluate this at a shared prefix, before divergent generations change the
  context. Store reference and loader logits, their top candidates, margins and
  error at divergence. A near-tie exception is descriptive, not exact greedy
  equivalence. Which error statistic and reference margin govern this exception
  must be pinned before execution; do not use later divergent contexts.
- Negative control: deliberately apply the original adapter twice in the
  merged modules, leaving other modules unchanged. Its measured error must be
  at least 1000 times the real loader's. Save both errors and use an explicit
  zero-error status rather than silently dividing by zero. The intended negative
  control failure must remain distinguishable from real-loader qualification.
- Hard stop if observed E > 1e-3. Otherwise propose absolute-logit allowance
  3E and present the evidence for user approval; do not automatically qualify
  the loader. This is a calibration-derived allowance, not independent validation
  on the same observations. Save the diagnostic and approved gate as separate
  products bound to the runtime, source hashes and loader implementation.
- A greedy mismatch not meeting the approved near-tie rule fails qualification.
  A failed hard stop or negative-control check cannot be repaired by silently
  increasing a tolerance.

## Final six approvals, 2026-10-26

This section supersedes conflicting provisional wording above and in the original
proposal. The user approved all six decisions explicitly.

### 1. Evaluation and targeted review

Use the Pilot 4 evaluation protocol: same 150 held-out items, recorded chat
template and prompt, greedy decoding, 1024-token cap, flexible-v3 primary and
strict alongside. Bind exact source configuration and tokenization in the audit.
Re-evaluate unedited SFT and GRPO in the same runtime as interventions. Reuse
human labels only for identical response text. Target new review at answer-changing
cases and suspected extraction failures; strict-only disagreement does not itself
require manual review. Keep raw scorer results and reviewed overlays distinct.

### 2. Accuracy-preservation claim

Declare accuracy preserved only if the lower 95% paired-bootstrap bound for
edited-minus-reference accuracy strictly exceeds -0.05. Report the full interval
and distinguish an inconclusive bound from evidence of loss. A statistically
detectable loss smaller than five points can still meet this non-inferiority
criterion; do not treat statistical difference and non-inferiority as opposites.
Use the existing declared 10000 resamples and seed 42. The preregistration must
explicitly warn that n=150 can yield an inconclusive non-inferiority result.

### 3. Magnitude controls

V4 scales the original SFT update to V1's effective-update norm separately in
each module, using ||b|| ||a|| (saved scaling is one), preserving direction.
Add the analogous per-module size control for V2. Report each of the three random
subspace draws separately; three draws do not constitute a calibrated null.
Every control receives its own immutable variant identity and edit provenance.

### 4. Historical compatibility

Remove the old128/new384 five-point reward-rate gate. Cohort reward rates are
descriptive only. Regenerate recorded historical prompts/seeds on common inputs
for compatibility checks. Use an existing independent numerical envelope only
where applicable to that metric/procedure. If none applies, report historical
and new results separately. Do not derive an envelope from the same old/new
differences being tested. New paired intervention comparisons need not wait for
historical numerical compatibility to be established.

### 5. Measured budget and hard ceiling

Measure required preparation and a bounded timing workload before presenting the
primary intervention budget. Optional supporting jobs use only remaining approved
budget. Stop and ask before exceeding 20 Colab compute units TOTAL for Stage 0,
across sessions, diagnostics, primary and supporting computation; not 20 per job
or notebook. Persist a cumulative usage ledger with observed unit readings and
forecast costs. An unknown remaining balance is not permission to exceed the cap.

Retain completed artifacts after interruption. Before joining an unfinished
comparison across sessions, restore the recorded environment and pass applicable
compatibility checks; otherwise use a separately named execution. Never silently
mix incompatible sessions to finish a paired comparison.

### 6. Numerical and reporting conventions

- Zero spectral gap with nonzero update: infinite sensitivity ratio. Zero over
  zero: undefined. Preserve reason/status; never replace with an arbitrary epsilon.
- Zero energy denominator: undefined.
- Zero flagged singular value: log the undefined control ratio and exclude that
  pair from ratio-restricted sensitivity comparisons.
- Near-tie decision uses the reference model's top-two margin and measured logit
  error at the same common prefix. Preserve both the diagnostic and exact greedy
  mismatch. No comparison of unrelated post-divergence contexts.
- E0.6a ci_low/ci_high refer to primary NLL. Add explicitly named fields for
  excess-surprise intervals rather than overloading the primary interval.
- Retain per-problem neg_log_p_correct and report an equal-problem average over
  eligible problems, separately from per-token measurements.

Implementation must preserve these approved numbers. Where source facts or
bounded measurements are still unavailable, record a pending state and the
required evidence rather than inventing a value or claiming qualification.
