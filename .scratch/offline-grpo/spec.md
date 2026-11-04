# Offline GRPO Arm: Sampling, Training and Stage 0 Analysis

Status: ready-for-agent

## Problem Statement

Pilot 4 used fresh answers from the evolving learner for each GRPO window. We need a comparable fixed-data arm to study how replacing those answers with a frozen pool affects behavior, training data, adapter writes and internal measurements. The historical 128-question untuned baseline has a summary, but its raw groups have not been established as available; a synthetic full-pool fixture is not evidence that a real pool exists. Historical artifacts and their provenance must remain intact.

Online versus offline is the closest comparison, but it changes the answer source along with the realized answers, rewards, lengths, caps and dead groups. It does not isolate a pure effect of staleness. Comparisons with gold and sampling SFT also change answer provenance or selection and training procedure as well as objective. Geometry and correlations do not by themselves establish causal use or behavioral effects. One training seed per arm cannot estimate run-to-run variability.

## Solution

Build or recover a verified pool of eight answers for each of the 512 training questions, then train the exact checkpoint-0 adapter on the saved 64 ordered windows using the existing GRPO update without importance weights. Preserve the pool, scores, plans, responses, diagnostics, full recovery checkpoints and verification records in canonical artifact trees. Reuse compatible existing evaluation and Stage 0 workflows through a versioned source contract. A resumable driver carries the work from preflight through independent readback of the private backup.

One explicit prelaunch approval authorizes the named plan, spending and automatic continuation through the criteria-based gates below. The driver does not require further routine manual go signals. Any failed integrity, numerical or acceptance gate stops affected dependent work, preserves evidence and reports the state; independent work with verified dependencies may continue. Approval does not authorize an unspecified budget or a change to acceptance tolerances.

## User Stories

1. As a researcher, I want a fixed unweighted GRPO arm so I can compare it with the refreshed-data online arm under the same specified update.
2. As a researcher, I want an inventory of historical pool evidence so that I can distinguish verified raw draws from summaries or fixtures.
3. As a researcher, I want a verified existing full pool reused when available, so that valid work is not repeated.
4. As a researcher, I want the historical 128 groups reused and extended only when their raw provenance and compatibility are verified, including two-group exact replay, so that incompatible evidence is never silently mixed.
5. As a researcher, I want a fresh 512-question pool when the old raw groups are absent or replay fails, without falsely marking that historical audit complete.
6. As a researcher, I want each question, prompt, token sequence, draw, generation setting, scorer result, seed and origin recorded so that every pool result can be traced and reproduced within its stated runtime limits.
7. As a researcher, I want the pool mapped by question ID to the frozen optimizer windows and checked against held-out IDs so that training uses the intended cohort and order.
8. As a researcher, I want capped and dead groups retained with the online reward, mask, advantage and denominator rules so that the effective update is comparable.
9. As a researcher, I want the exact saved checkpoint-0 adapter loaded and hash checked, so that reseeding is not mistaken for the same starting state.
10. As a researcher, I want saved plans and full training recovery state so an interrupted run resumes at a verified boundary without repeating an update.
11. As a researcher, I want pre-update learner and generating-policy log probabilities and masks saved, so I can measure how the learner differs from the data source without changing the primary loss.
12. As a researcher, I want finite-sample negative KL estimates retained and sequence and token aggregations separately labeled, so estimates are not misrepresented.
13. As a researcher, I want training monitor thresholds to flag and continue as in the online plan, while genuine integrity and numerical failures stop affected work.
14. As a researcher, I want held-out greedy and compatible sampled evaluations with raw responses and paired item results, so behavioral comparisons are auditable.
15. As a researcher, I want checkpoint-0 greedy outputs checked against the online anchor in one runtime, so a harness discrepancy is caught before interpretation.
16. As a researcher, I want adapter direction, effective writes and spectral measurements compared under compatible definitions, so geometry is not confused with behavior.
17. As a researcher, I want the initial push computed on the fixed pool and the first training window checked separately, so the sign and batch identity of the comparison are explicit.
18. As a researcher, I want fixed-context drift and KL measured on the established probe cohorts, with direction and aggregation recorded, so they are distinct from training-data KL.
19. As a researcher, I want all eight established principal-cut conditions evaluated on paired questions, so cut effects can be compared with their controls.
20. As a researcher, I want dataset surprise, entropy and rank scored by the untuned model with compatible masks and both full-coverage and paired cohorts, so answer distributions can be compared honestly.
21. As a researcher, I want observed KL plotted against observed adapter norm as the core analysis, without implying a matched-norm intervention was performed.
22. As a researcher, I want a separately specified scaled-adapter extension if a matched-norm comparison is selected, with actual evaluation of the scaled adapters rather than algebraic rescaling of measured KL.
23. As a researcher, I want a preflight to verify inputs, runtime, storage, credentials, resource budget and release compatibility before expensive work.
24. As a researcher, I want progress, costs, partial failures and backup verification visible without interrupting the run, so I can safely monitor or resume it.
25. As a researcher, I want reports to distinguish measured results, descriptive associations and causal hypotheses, and to state single-seed and compatibility limitations.

## Implementation Decisions

- Use the verified Pilot 4 plan as the field-by-field reference for model and tokenizer revisions, numeric settings, prompt builder, cohort and window ordering, sampling parameters, reward, group advantage, token-level loss reduction, optimizer, checkpoints and monitor actions. Record all differences and use a new named plan. The primary method is fixed-pool, unweighted GRPO; the deferred TIS sensitivity arm is not implemented as an unused option.
- Inventory existing artifacts and their raw payloads before selecting a route. Reuse a full pool only if its provenance is verified; reuse and extend the historical 128 only after complete source/settings audit and the two-group exact replay; otherwise prepare a fresh 512 pool from independently verified inputs. Do not claim historical groups were audited when missing. Record pooling order and optimizer order separately. Equal answer text does not imply duplicate draw identity.
- Seal 512 distinct question IDs with eight uniquely identified draws each, and map them to 64 windows of eight distinct questions. Check held-out exclusion, prompt and token identity, EOS and cap semantics, scorer recomputation and all masks. Keep all-correct and all-wrong groups, zero their advantages when group variance is zero, and include their masked tokens in the global denominator. A cap gives zero reward while its generated tokens remain in the loss mask.
- Load and hash-check the exact checkpoint-0 adapter and configuration, verify target mappings and base identity, and save complete checkpoints at 0, 8, 16, 32 and 64 with optimizer, scheduler, RNG and cursor state. Save the plan, response JSONL, step logs, diagnostics and status. Use a versioned source contract with model, adapter, checkpoint, data and plan identities. New implementation identities must not require rewriting historical manifests or making historical products match new implementation hashes.
- Before each update, save aligned token log probabilities under the actual generating distribution μ and the current learner π, along with masks and grouping. Verify generation scores against teacher-forced scores on real draws including EOS and a long or capped answer. Report sequence sums of log μ minus log π, window means and length-weighted ratio-of-totals nats/token, with distributions and negative estimates preserved. The forward-KL interpretation requires draws sampled from μ, fixed in advance and not previously used in training; otherwise call it an empirical fixed-data log ratio. A separate fixed-subset checkpoint curve must label prior exposure. This diagnostic differs from fixed-context KL(adapter || untuned).
- Evaluate checkpoints on the frozen 150-question held-out cohort with greedy decoding, and perform the compatible eight-draw sampled evaluation at checkpoints 0 and 64. Save token IDs, answers and item scores; report paired transitions and pointwise paired item bootstrap intervals. Reuse old endpoints only after checking cohort, template, scorer, decoding, batch policy, checkpoint and runtime compatibility. Check checkpoint-0 token equality against the online anchor under one runtime.
- Cover the internal-measurement family of Stage 0 with adapter directions and effective writes, initial push and spectral geometry, and fixed-context drift and KL, including their established trajectories and applicable principal overlap and intruder analyses. Behavior and dataset analysis form the other two families. Compare product updates and effective writes, respecting rank-one factor gauge ambiguity. The initial push is minus the loss gradient, not an AdamW update. Compare the separately computed first-window push with the negative of the saved pre-clip first-window gradient using the same batch, parameter space and normalization.
- Evaluate all eight sealed principal-cut conditions: original, principal-output cut, principal-input cut, three random-direction controls, and output and input magnitude controls. Verify definitions, ranks and control construction against the sealed comparison plan; use paired behavioral evaluation, not cut construction alone, to assess effects.
- Score all pool answers with the untuned model for per-token surprise, entropy and rank, retaining zero-correct questions and reporting correct/incorrect and capped/uncapped subsets. Compare the full available gold and sampling-SFT cohorts and the common paired cohort separately, with token-weighted and equal-question or equal-answer summaries. NLL minus entropy is descriptive excess surprise, not KL.
- The core norm analysis reports observed fixed-context KL against observed update norm with definitions and checkpoints. A matched-norm claim requires a separately specified extension defining a common norm and target, construction of scaled adapters, and actual KL measurements of those adapters. Do not infer matched-norm KL by rescaling a previously measured KL.
- Treat layer similarity versus pre-update KL and time as descriptive, not as a causal slope: checkpoint and window difficulty also change, and only one seed is available. State that the online/offline contrast includes the effects of changed answers, reward, lengths and dead groups. Keep hypothesis-supporting patterns distinct from causal identification.
- Preflight checks the source and release pins, actual runtime and numeric configuration, model snapshot and cache, mounted durable storage and free space, private destination access without exposing credentials, transfer configuration, session persistence, budget and ledger constraints, lock state and estimated verification costs. An incompatible runtime requires a documented, explicitly approved compatible-runtime path, not a spoofed historical identity.
- The approved prelaunch plan freezes the route criteria, comparison protocol, predictions, spending bounds and automatic gate conditions. Following approval, pool verification, smoke, first-window gradient check and post-training integrity gates run automatically. Failed gates halt dependent work and preserve evidence. Analysis failures that do not compromise integrity are recorded while independent analyses continue. Never widen an acceptance tolerance to pass.
- Resume from sealed units and the latest verified checkpoint; replay an incomplete window only from its last sealed boundary. Reuse requires matching inputs, code and compatible runtime. Track child dependencies and status. Profile a real small unit, record uncertainty in time estimates, batch only within validated scientific contracts, cache verification in process where safe, and expose progress and log locations to another terminal. Export the pool, plans, raw outputs, logs, adapters and recovery state to the designated private repository and verify an independent manifest readback before calling backup complete or stopping the compute resource.

## Testing Decisions

The main public CPU fixture workflow exercises prepare, sealed pool mapping, the pool-backed training path, diagnostic recording, checkpoint interruption and resume. On the same synthetic mixed and dead groups with unequal token lengths, it compares the online and offline paths' rewards, advantages, global denominator, pre-clip gradients and one optimizer update. It verifies no duplicate draws, windows or updates, downstream source handoff, honest partial failures and recovery. CPU tests use mock models and require no downloads; a skipped import is not a pass.

Focused tests cover invalid historical reuse versus fresh fallback, pool identities and held-out exclusion, masks and scorer semantics, checkpoint hash, log-ratio sign and grouping, sequence versus token aggregation, identical policies, retained finite-sample negatives, and the first-window push sign. Test missing inputs or credentials, failed children and incomplete export. Numeric criteria come from the applicable existing specification or a separately reviewed criterion declared before execution; never adjust them to obtain a pass.

A later real-GPU smoke checks the actual base and checkpoint-0 adapter, runtime, generation-versus-teacher-forcing scores, finite loss and gradients, memory, timing and agreement with a same-input reference. Production gates verify the sealed pool, first-window gradient, base integrity and finite state, evaluation anchor, complete required products and independent backup readback. No real 64-step online replay, TIS arm or second seed is a launch prerequisite.

## Out of Scope

- Implementing or running TIS-2, fixed-pool SFT, independent diagnostic draws or additional training seeds in the primary arm.
- Claiming a pure causal effect of staleness or the objective from the proposed observational comparisons.
- Rewriting historical plans, evidence or acceptance thresholds to accept new code or runtime identities.
- Treating cut construction, geometric overlap or a descriptive slope as a behavioral causal result.
- Launching compute, pushing artifacts or implementing the work merely by approving this specification.

## Further Notes

The approved delivery is a specification followed by eight implementation tickets. The run still requires a concrete prelaunch approval covering its plan, predictions, destinations, resource and budget bounds. Historical data availability, compatible runtime and reusable analysis outputs must be verified rather than assumed. A completed process is not a completed experiment until all required products and the backup readback are verified.

Decision update, 2026-11-04: the overnight run does not require frozen predictions, a reporting protocol, a whole-run time estimate or the Stage 0 budget ledger. Approval binds the plan, preflight, runtimes and destination. A time limit is optional through `max_seconds`. Earlier mentions of predictions, spending bounds and budget in this spec are superseded for the overnight wrapper.
