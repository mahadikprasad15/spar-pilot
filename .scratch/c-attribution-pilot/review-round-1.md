# Correctness-attribution pilot: adversarial review, round 1

Reviewed commit: 1c572596f742677f2cebf4f15907464fa94ac5b9 on c-attribution-pilot. Scope: spec.md and all 14 tickets, plus the existing attribution instrument, generation and matching code. This is a design review, not evidence that the new code or GPU experiment has run.

## Review outcome

Revise the spec and affected tickets before implementation/launch. The leave-one-out score-function estimator and its sign convention are sound under the stated on-policy assumptions. The major issues are scientific interpretation and selection uncertainty, an independently reproduced matching failure, policy/precision verification, and a resource estimate that has not yet been established. The user wants a small, greedy-behaviour pilot, not a large new sampled-ablation suite.

## User decisions to carry into the revision

1. Attribution uses the existing final-checkpoint sampled responses for online and offline GRPO, each arm scoring its own responses on the same 150 question IDs. Keep all eight draws and all outcome types. Reuse is conditional on source/checkpoint and effective-policy verification.
2. Initial behavioural ablations are greedy only, on the full matched 150-question cohort, with the scorer, cap, checkpoint, prompt, batch policy and baselines bound and reported. These questions were used for selection and previous evaluations, so label results exploratory. Greedy outcomes are not a direct validation of the sampled-J target. A fresh held-out and/or sampled follow-up remains a separate later choice, not part of this launch.
3. Keep top-10 adapter removal per arm and a matched control. Do not include top-10 neuron-read ablations in the first pilot.
4. Add three distinct neuron-read removal groups per arm: (a) global top 64 across all 28 down_proj reads; (b) top 64 in each previously identified effective-write hot layer, removed together; (c) top 64 in every down_proj layer, removed together, up to 28 x 64 = 1,792 terms. Each group gets a size- and layer-distribution-matched control. These are removals of terms from LoRA reads only, not deletion of pretrained neurons, and are separate from adapter removals.
5. Use the same hot-layer list across arms, based on the recorded prior effective-write study; a union of the previously identified arm-specific hot layers is suitable if that is the documented list. Bind the exact layer IDs, source run, measurement and hashes before examining the new J rankings. If the historical hot-layer list is ambiguous, surface that decision; do not invent a list or select hot layers from the new J results.
6. Adapter controls: same module type and similar effective-write magnitude, proposed +/-20%; neuron controls: same layer and similar |a_j| * mean(|z_j|), proposed +/-20%. Do not scale target or control adapters/read terms. Specify the measurement cohort and weighting, exclude the selected targets, and report achieved matches and unavailable cases. User endorsed group-wise one-to-one matching rather than greedy sequential assignment.
7. Split the same 150 question IDs into two recorded, seeded groups of 75; keep all eight responses for each question together. Calculate and report adapter, global top-64 neuron, hot-layer and per-layer neuron ranking/sign stability. Stability is descriptive and never a gate. Use the full 150 questions for the pilot selection after the split analysis, and do not misdescribe later same-cohort ablation as held-out.
8. Bootstrap adapters by resampling whole question groups. Benchmark a chunked neuron bootstrap; prioritize the requested question split, selected-read stability and layer summaries, and undertake all-neuron intervals only if measured feasible. Neuron bootstrap estimates for candidates selected on the same cohort are exploratory, not post-selection-confirmatory intervals. Never introduce a stability threshold that blocks the pilot.
9. Trace the FP32/FP64 noise-floor, activation and accumulation/storage precision lessons across earlier pilots and the production-1 attribution; use what was actually verified. The new summed sequence-logprob has different scale and may exhibit signed cancellation. Do not silently loosen historical acceptance thresholds. Distinguish implemented/reproduced checks, measured floors, and unverified risks.
10. Report per condition: raw greedy accuracy, paired accuracy change in percentage points versus the same arm's intact model, 100 x ablated/intact accuracy (undefined if denominator is zero), target-minus-control difference, paired question-level uncertainty and gain/loss counts, invalid/cap and length. Include a compatible untuned baseline if available or regenerate it in the qualified runtime. Report group summed first-order sampled-J predicted removal effect alongside, explicitly not as a numerical prediction of greedy or a guarantee for full removal.
11. Helpful/harmful language is local and estimated: positive dJ/dlambda predicts improvement for infinitesimal strengthening near gate 1; a negative value does not prove full removal improves accuracy. Do not label a nonpositive selected score as helpful. If a requested exact-64 selection cannot be formed from positive scores, surface the shortfall and settle/document the handling before freeze rather than silently padding or changing group sizes.

## Substantive findings and required disposition

### F1: Different attribution and behavioural targets

Spec questions and report currently describe sampled-J ranking followed by greedy ablations. This scope is user-chosen and acceptable for a pilot. Do not treat positive or negative greedy results as proof or disproof of the sampled-J gradient. Full removal is also a finite, potentially interacting intervention; sum of local predicted effects is a first-order reference. Report both group predictions and measured outcomes under clear, distinct labels. Keep sample-based ablation and new cohorts deferred unless separately chosen. A direct later test of sampled J would require sampled intervention runs under the same policy.

Evidence: spec lines 11-20, 54-71, 119-141; issues/07-generation-ablation.md and issues/13-report-catalog.md.

### F2: Neuron ranking stability and 64-read scope

The proposed global search ranks 28 x 8,960 = 250,880 read terms per arm, while the original stability ticket only requires a split of the 196-adapter vector. User now also wants hot-layer and every-layer groups. Compute the requested neuron split summaries, keep them descriptive, preserve matched question IDs across arms, and report selected layer distributions and signs. Global 64, 64 per hot layer, and 64 per all layers are different budgets; use separate matching and do not interpret a larger-group drop without its size/layer-matched control. The signed-score concentration denominator must be unambiguous: a share of sum(abs(g)) is different from abs(sum(g)); the latter can approach zero through cancellation. Do not conflate pointwise selected-neuron intervals with simultaneous or held-out confirmation.

Evidence: spec lines 73-79, 103-110; issues/04-uncertainty-stability.md, issues/06-interventions-selection-freeze.md, issues/13-report-catalog.md.

### F3: Confirmed false-unavailable outcome in the existing matcher

The existing pilot_eval/attribution_ablation.py matched_controls function greedily picks a seeded candidate for each target in turn. I executed the actual function from the reviewed commit on a synthetic example (no model or production data involved):

- target indices 0,1 with writes 1.00 and 1.40; candidate indices 2,3 with writes 1.16 and 0.85; all same module type; exclude {0,1}; tolerance 0.20; seed 1.
- The function assigns candidate 2 to target 0, then returns target 1 unavailable and group=None.
- A valid full assignment exists: target 0 -> candidate 3; target 1 -> candidate 2.

This is a reproduced algorithmic failure mode, not evidence that any previous real study's controls were wrong or that this pilot's actual targets will encounter it. Use seeded, reproducible whole-group matching (such as maximum-cardinality bipartite matching), keeping the prespecified eligibility criteria. Declare unavailable only after checking group feasibility; do not loosen thresholds retrospectively. Add this counterexample for adapter and neuron-control matching. For the large all-layer control, feasibility can be checked layer by layer; failure in that condition must not silently block otherwise feasible global/hot comparisons.

Evidence: pilot_eval/attribution_ablation.py lines 161-177; issues/06-interventions-selection-freeze.md.

### F4: Original sampler provenance and numerical score fidelity

Token IDs, checkpoint hashes and recorded generation settings are valuable, but no generation-time per-token log probabilities were saved. Cached-vs-teacher-forced agreement on the current runtime does not by itself prove what policy generated historical draws. Bind the original base and adapter versions, tokenizer and prompt, original generation code and fully resolved generation configuration/logits processors, seeds/batch identities, EOS/cap semantics and runtime. Compare cached and full teacher-forced scoring on the saved prefixes, including representative short, long and capped cases, plus representative gate sensitivities where feasible. Record token errors and sequence-sum errors.

A max absolute per-token log-prob difference of 1e-3 permits a worst-case accumulated absolute bound of 1.024 over 1,024 tokens; this is a bound, not an observed discrepancy or proof of a gradient failure. Validate representative selected-scale gradients and numerical floors rather than relying on that scalar threshold alone. An exact seeded replay is useful but report-only if it diverges for understood hardware/batching reasons. A confirmed policy/checkpoint mismatch is material: repair, use suitable samples, or explicitly label the on-policy interpretation unestablished. Validate a no-op and matched generation settings/batch membership so numerical changes do not masquerade as small greedy effects.

Before freezing tolerances, compare past verified FP32/FP64 tests, Qwen full_fp64 handling, storage and cancellation floors; do not treat proposed tolerances as user-approved or loosen them to pass. Inputs and technical qualification remain hard safeguards; the user's no-gate decision applies to statistical stability, not source integrity or numerical correctness.

Evidence: spec lines 24-32, 42-46, 81-101; pilot_eval/grpo_gpu.py lines 62-87; docs/agents/experiment-lessons.md; issues/01, 02, 05 and 10-11.

### F5: Bootstrap, disk, CPU and total budget must be measured

A naive per-arm FP32 tensor of 1,200 responses x 28 x 8,960 neuron terms is about 1.2 GB before overhead. Materializing 10,000 FP64 bootstrap estimates for all 250,880 reads is about 20 GB per arm. These are conditional array-size calculations, not measurements of the planned implementation or proof it must allocate those arrays. Chunk or stream and preaggregate within question; benchmark wall time, peak CPU/GPU memory and export/readback on a representative workload before relying on the claim that bootstrap and freeze take under five minutes.

The revised scope has 9 greedy conditions per arm: intact, selected adapter and control, and three separate neuron selections with one matched control each. At 150 questions this is 2 x 9 x 150 = 2,700 greedy answers, plus 150 for an untuned run if regenerated, before qualification/rehearsal and any skipped conditions. Reusing compatible intact outputs may change the count. Measure end-to-end generation and qualification speed, storage/export overhead and bootstrap time; revise the workload and cap proposal rather than assuming the original 2-2.5 h / 3 h estimates cover the expanded scope. Do not start an automatic handover.

Evidence: spec lines 73-79, 143-172; issues/04, 08-12.

### F6: Comparative reporting and uncertainty

Report online/offline signed adapter scores, top-set overlap and layer/module distributions, global and per-layer neuron overlaps, and question-split stability. The arms' own sampled responses and difficulty-weighting differ; one seed per arm cannot establish an objective-specific or general mechanism. Pair bootstrap resampling by question ID across both arms and conditions, while keeping each arm's response groups distinct.

For each ablation, report paired uncertainty on ablated-minus-intact greedy accuracy, not just separate Wilson intervals and target-minus-control. Preserve the user's requested percentage remaining with the correct denominator. Report sum of first-order predicted removal-J for each target/control group, labelled as sampled local approximation, alongside actual greedy changes without claiming numerical agreement validates C. Report gains/losses, invalid/capped outcomes and lengths.

Evidence: spec lines 131-141; issues/07 and 13.

### F7: Catalog, cohort and rehearsal provenance

At review time catalog/ was absent from both this remote branch and remote main. The user reports the catalog commits exist on the Mac's local main. Therefore a clean checkout cannot get the catalog merely by merging origin/main. Identify the exact local catalog commits and incorporate the necessary content into the pilot branch or a sealed manifest, then verify that the resulting revision and all bound inputs are available on the Pod. Do not push main merely to solve this dependency.

Record all questions used for rehearsal and all other prior exposure. For a later held-out follow-up, exclude the original 150, the existing rehearsal 20, any newly used rehearsal questions, and other known prior uses. Freeze any follow-up analysis after the exploratory pilot and before seeing those later outcomes. Do not label it automatically confirmatory without checking prior exposure and the actual frozen plan.

Evidence: spec lines 31-34, 141, 157-159, 192-198; issues/08, 11 and 14.

## Ticket-by-ticket disposition

| Ticket | Review disposition |
|---|---|
| 01 | Preserve old metrics; implement summed generated-token metric and all relevant prompt-position contractions with CPU tests. |
| 02 | Preserve all draws and EOS/cap semantics; bind source model, tokenizer, effective policy and prompt provenance. |
| 03 | LOO and signs are correct under stated assumptions; test weights, group aggregation, positive/negative signs and signed neuron-sum identity. |
| 04 | Add requested 75/75 neuron and adapter split summaries; adapter bootstrap; benchmark chunked neuron bootstrap; no stability gate. |
| 05 | Review historical FP32/FP64 results; qualify cached/forced sequence scores, representative gradients and numerical floors; replay report-only unless source-policy mismatch is found. |
| 06 | Implement group-wise feasible matching; add reproduced failure fixture; freeze three new neuron scopes, controls and exact hot-layer provenance; do not rescale. |
| 07 | Update to the nine greedy conditions per arm and accurate cost; separate all interventions; report paired deltas, control contrasts and raw metrics. |
| 08 | Bind source data, exact catalog/manifest revision, checkpoints, tokenizer and cohort in a sealed, recoverable input bundle. |
| 09 | Re-estimate measured budget and enforce actual caps; preserve per-run state, no-op/weight-restore safety and stable qualified batching. |
| 10 | Test the real CPU boundary, ensure relevant dependencies/tests actually execute rather than being skipped, and preserve old metric behavior. |
| 11 | Attended Pod rehearsal with representative long/EOS/cap cases and exact rehearsal question IDs recorded; exercise recovery and stop. |
| 12 | Update the 2,700 (+ optional 150 untuned) greedy scope and measured launch budget; no automatic handover, no sampled ablations in the initial pilot. |
| 13 | Add group predicted sampled-J effects, greedy paired deltas and uncertainty, control feasibility, cross-arm overlaps/layer distributions, stability and exploratory labels. |
| 14 | Exclude every known previously exposed question and newly used rehearsal IDs; freeze follow-up design before seeing its outcomes; no automatic follow-up launch. |

## Verification and next action

Read all 14 tickets and the spec at the cited commit; inspected the relevant existing code and reproduced F3 on a synthetic input. The user's data, full repo checkout, GPU and the previously reported test environment were not available in this workspace. Attempted local python -m pytest -q: pytest is not installed here, so no project suite or GPU test was independently run. Claude's reported 804 passed / 42 skipped / 1 failed is not independently verified by this review.

Please reply in review-response-round-1.md point by point, distinguishing accepted decisions, fixes, residual risks and any genuinely unresolved user decision; then update the spec and tickets without changing historical study artifacts. Do not launch a GPU run as part of responding to this review.
