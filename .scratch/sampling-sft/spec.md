# Sampling SFT: gold-guided data recipes and matched adapter training

Status: ready-for-agent
Date: 2026-10-28
Approval: scientific design and public saved-workflow test boundary confirmed by the user. Ticket breakdown approval is pending. Publishing this spec authorizes no paid job. First GPU spend requires explicit post-preflight approval; training requires separate post-gate approval.

## Problem Statement

The researcher wants to compare gold SFT with SFT trained on answers constructed by a gold-guided sampling procedure at three amounts of likelihood search. Existing gold-SFT and GRPO artifacts should be reused as reference products without rerunning gold training or confusing the proposed procedure with ordinary on-policy sampling.

Previous pilots suffered from environment drift, hidden notebook state, repeated expensive checks, ambiguous completion, score extraction errors and unreliable recovery. This experiment must be readable, affordable, auditable and resumable. Its data gate and adapter handoff must retain enough raw evidence for later evaluation and internal measurements without running those follow-ups now.

## Solution

Provide one parameterized saved-workflow interface used by a readable notebook and a cloud runbook. Prepare verified sources on CPU; qualify the sampling engine and measure costs on L40S; generate answers for the same 512 GSM8K training problems at k=0,2,5; grade and review them; publish descriptive gate statistics; and, after explicit approval, train k=0 and k=5 adapters on verified L4 using gold SFT's saved training protocol.

The experiment compares data recipes. Search amount k is a within-recipe contrast with gold conditioning and filtering fixed. It does not isolate a causal effect of on-policyness. Plain NLL, length, format and leakage are descriptive. The temperature-1 baseline and NLL offset are deferred, so sharpening relative to the model's own samples is not measured.

## User Stories

1. As a researcher, I want the original proposal and approved amendments preserved, so the experiment can be traced to its decisions.
2. As a researcher, I want source identities verified rather than inferred from filenames, so the intended model and training cohort are used.
3. As a researcher, I want actual training metadata distinguished from evaluation metadata, so historical hardware is not misidentified.
4. As a researcher, I want gold training reused without a rerun, so the new experiment adds only the approved sampling arms.
5. As a researcher, I want one saved plan describing prompts, tokenization, seeds and all decoding settings, so default changes cannot silently change the recipe.
6. As a researcher, I want generation and scoring prompts distinguished, so gold-guided proposals are not described as unconditioned policy samples.
7. As a researcher, I want scored token IDs retained through training, so decode/re-encode differences cannot change the supervised examples.
8. As a researcher, I want k=0,2,5 generated on all 512 problems, so search amount can be compared within the recipe.
9. As a researcher, I want data seed0 and training seed42 recorded separately, so their roles remain clear.
10. As a cloud user, I want actual hourly rates and conservative reservations checked before launch, so jobs cannot bypass budget ceilings.
11. As a cloud user, I want setup, idle, debugging, reruns and storage counted, so reported spend reflects the experiment's full cost.
12. As a cloud user, I want an 80% review checkpoint, so remaining headroom is not consumed without my approval.
13. As a cloud user, I want both durable export copies verified before automatic Pod stop, so stopping compute does not lose results.
14. As a cloud user, I want deletion and unexported termination to require my approval, so lifecycle automation cannot destroy the only copy.
15. As a notebook user, I want setup, CPU and GPU stages clearly labelled, so I can avoid paying for CPU work on an idle GPU.
16. As a notebook user, I want saved state restored in a fresh process, so execution does not depend on earlier cell variables.
17. As a researcher, I want a bounded throughput profile before production, so estimates come from measured work.
18. As a researcher, I want one frozen sampling runtime across arms and retries, so incompatible products cannot be mixed.
19. As a researcher, I want worker-level numeric settings verified, so parent-process flags are not mistaken for engine configuration.
20. As a researcher, I want a bounded score qualification with exact token alignment, so score errors cannot hide behind aggregate agreement.
21. As a researcher, I want the short pilot to show every alarm against its threshold, so revisions start from evidence.
22. As a researcher, I want exact interruption/resume replay qualified before full generation, so restart behavior is known in advance.
23. As a researcher, I want one writer and sealed block boundaries, so concurrent notebooks or interrupted writes cannot corrupt a run.
24. As a researcher, I want explicit finished, capped, exhausted and engine-error outcomes, so unfinished traces cannot enter training.
25. As a researcher, I want engine replays separated from model retries, so infrastructure failures do not strengthen answer selection.
26. As a researcher, I want every problem to have one final disposition, so missing data cannot disappear from denominators.
27. As a reviewer, I want automatic scores preserved alongside human labels, so intended-answer overrides remain auditable.
28. As a reviewer, I want engine completion and text immutable, so a human label cannot repair an unfinished or wrong trace.
29. As a researcher, I want coverage failures to block training, so the training cohort does not silently shrink.
30. As a researcher, I want smaller batches checked against independent loss arithmetic, so marked missing slots preserve the declared reduction.
31. As a researcher, I want gate comparisons on the three-arm common cohort, so k contrasts use the same problems.
32. As a researcher, I want each arm's full coverage and uncovered identities reported, so common-cohort analysis does not hide selection.
33. As a researcher, I want arm-specific covered training slots retained, so each new arm stays as close as possible to gold's exposure order.
34. As a researcher, I want plain NLL, lengths, formats and leakage with declared reductions, so summaries are interpretable.
35. As a researcher, I want whole-problem paired bootstrap intervals, so paired uncertainty does not treat tokens as independent samples.
36. As a researcher, I want missing baseline-dependent quantities marked not measured, so deferred work cannot be mistaken for zero effects.
37. As a researcher, I want no automatic equivalence or merging rule, so inconclusive evidence does not remove an arm.
38. As a researcher, I want an explicit gate approval before training k0/k5, so dataset and budget decisions precede optimizer work.
39. As a researcher, I want exact gold checkpoint0 identity and training metadata, so seed equality is not substituted for equal initialization.
40. As a researcher, I want adapters and gold-matching logs saved at the declared checkpoints, so later consumers have a complete handoff.
41. As a researcher, I want optional optimizer capture distinguished from valid recovery, so an adapter-only checkpoint is never treated as a resumable optimizer state.
42. As a researcher, I want honest implementation, qualification and execution statuses, so passing CPU fixtures is never reported as scientific completion.
43. As a collaborator, I want readable notebook code calling shared commands, so each stage has a clear purpose without embedded repair cells.
44. As a researcher, I want evaluation and internals deferred explicitly, so the adapter handoff does not expand into unapproved experiments.

## Implementation Decisions

### Public workflow and evidence model

- Extend the existing saved-workflow command seam and shared durable-execution conventions. The notebook and runbook call these same commands; scientific logic does not live in cells. Prefer reuse of existing source, scoring, training and artifact primitives over a parallel experimental framework.
- Deep interfaces separate source preparation, engine operations, sampler state, grading/review, gate reporting, training and cloud lifecycle. No broad prefactor is justified before implementation discovery.
- Canonical artifact roles are plans, runs and reports. Every product binds configuration, source hashes, model/tokenizer revisions, prompt identities, cohorts, code, runtime and parent products. Raw responses, token IDs, structured logs and statuses are durable.
- One writer owns a run. Verify existing sealed products before reuse. Reject changed inputs, identities, runtime or corrupted payloads. Partial products remain identifiable and never count as completion. CPU consumers restore without CUDA or notebook globals.
- Progress reports distinguish completed blocks/problems from elapsed heartbeats. Tests, real engine qualification, production collection, human review and training completion have separate statuses.

### Sources and reference training

- Use Qwen2.5-1.5B-Instruct and the exact saved 512 GSM8K training problems. Bind the existing 150 held-out evaluation identities for future handoff; do not evaluate them here.
- Verify gold's original training manifest, exact per-step order, optimizer configuration, tokenizer/masks, checkpoint steps, runtime and initial adapter. An L4 evaluation manifest alone is insufficient evidence of training hardware.
- Historical gold training is not rerun. The verified protocol is rank1, alpha1, dropout0 on seven projections across 28 decoder blocks; AdamW learning rate1e-4, betas0.9/0.999, epsilon1e-8, no weight decay, clipping1, constant schedule, microbatch1, accumulation8, 64 optimizer steps and seed42. This is one pass/512 exposures before omissions, not the separate mentor two-epoch recipe. Any contradictory original artifact blocks the affected handoff rather than being overwritten.
- Training is L4, FP32, with TF32 matching verified gold metadata. Exact optimizer/config and saved order are authoritative. Checkpoint0 must equal gold's saved A0/zero-B payload bit for bit, not merely share seed42.
- Gold targets retain the original solution and final marker; the same prompt mask and exactly one supervised end-turn convention carry through the existing trainer. Use stored response IDs for sampled targets; never silently retokenize or truncate them.

### Sampling recipe

- Data seed0; k0/k2/k5 generated for all512 problems. Use the supplied deterministic problem/block/step/retry seed derivation, fixed and versioned before qualification. Seeds are inputs, not a promise of reproducibility.
- Port the released greedy sampler, not the paper's exact MH algorithm or theoretical distribution guarantees. The proposer sees a cleaned gold solution and the decoded partial response in its user message, with that same prefix also appended as assistant prefill. This upstream duplication is intentional.
- Scoring uses the exact training/scorer prompt rather than the gold-guided proposer prompt. Score is mean response-token log probability including an actual stop token. Temperature0.6, repetition penalty1.1, top-p1 and all remaining decoding parameters are explicitly resolved; no engine defaults are inherited silently.
- Gold cleaning removes calculator markup and rewrites the final gold-number line to the agreed prose form. Preserve and hash raw gold, cleaned gold, template and tokenized prompts. Check context limits with the duplicated prefix, generation budget and scorer context accounted for separately.
- J64; response cap1024. Extend the current trace, perform k whole-trace uniform cuts and bounded regrowth proposals, score each full proposal, and accept only strict score improvement. Shorter/stop-only proposals are legal; grade only the final trace. k0 performs extensions without refinement.
- Preserve within-block refinement ordering; determine completion after the block's refinement. Continue blocks while no stop and length below cap, limited to twice the ceiling of cap/J. Do not increase cap, J or rewrite a prompt automatically.
- Store real stop/finish metadata, validate token positions and forbid retained tokens after the first stop. Do not invent a stopping token from decoded text. Same-boundary EOS/cap accounting and actual engine stop handling require a recorded preflight contract before qualification.

### Attempts, coverage and human review

- Each terminal trace attempt has exactly one state: finished, capped, block-budget-exhausted or engine-error. Operational interruptions are events, not trace outcomes. Persist pending state distinctly until a terminal record can be sealed.
- Empty output without an actual stop or missing/malformed scoring probabilities is an engine/protocol error. Stop-only is finished but incorrect and ineligible.
- Initial attempt plus at most3 model-outcome retries, with declared new seeds, stopping at the first eligible answer. Capped, exhausted and wrong final traces are ineligible. Engine failure replays the same logical request/seed/runtime once; a second consecutive failure stops the run. A logical request is bound to problem, request type, block, refinement step and model-retry index.
- Log both operational failures even if replay succeeds; do not count them toward the three model retries. If a run stops with engine error, it is blocked, not a completed dataset with a silently missing problem.
- Attempt state counts reconcile to terminal attempt records. A completed arm has512 final covered/uncovered rows. Every covered row references exactly one finished, correctly graded or confirmed-human-eligible attempt. Coverage includes retained and rejected attempts with reasons.
- Preserve flexible-v3 verdicts and strict diagnostics unchanged. Export flagged cases for human review. Separate confirmed intended-answer labels may correct eligibility when extraction failed; they cannot change tokens, supply gold, repair reasoning or complete an unfinished response. Agent suggestions remain separate from user confirmation.
- Coverage0 uncovered passes only the coverage gate; training still needs all other gates and explicit approval. With1–5 uncovered, mark missing slots and retain64 nonempty optimizer steps. More than5 blocks training. No gold, incorrect trace, duplicate fallback or unlimited retry fills a slot.
- T16 covers batch sizes3–7 under the existing total-supervised-token denominator: step completes, independent hand loss agrees, gradients are finite and scheduler advances exactly once. Failure stops for repair, without changing the gate or dataset policy.
- Train each arm on its own covered slots in gold's saved order. Report uncovered lists, union/intersection and pairwise symmetric differences; flag differences>5 before approval. This exposure difference remains a limitation of later trained-arm comparisons.

### Runtime freeze, qualification and replay

- Sampling is L40S/FP32/TF32off. Resolve and hash GPU type, driver, CUDA, Torch/vLLM versions, actual worker numeric flags, scheduling/batching limits, memory utilization, prefix caching and context limit. Record all additional package, template and algorithm identities separately. No silent GPU or precision fallback.
- Stage0.5 selects settings once using bounded measured throughput, memory and observed batch dependence. Freeze before the fixed32-problem k5 pilot. Profile candidates and qualification require budget reservations and paid-work approval.
- All arms/retries/resumes share one runtime hash. Changing it creates a new attempt and regenerates all k arms; do not mix runtime hashes in an arm or comparison. Reuse only compatible sealed evidence.
- Checkpoint completed block boundaries, including logical scheduler/order and RNG state. A partial block's scientific records are discarded and redone while its operational events remain. Submit dependent logical request groups in fixed recorded order by problem ID and request type.
- Clear prefix cache at every block boundary, in uninterrupted, resumed and production execution. Within-block caching stays on. Record additional prefill work.
- On the32-problem pilot compare uninterrupted execution with kill-after-block2/resume. Scientific traces, scores, decisions, IDs, steps and coverage must match exactly. Times, PIDs and event histories are separate and are not compared.
- A replay failure blocks Stage2. Requalify in this order: caching disabled entirely; scheduling/batch constraints; supported engine/framework deterministic modes. Every change gets a new runtime identity and qualification record, with its throughput cost. Unsupported controls or unsuccessful replay are reported, not patched by relaxing equality.
- T3 compares independent HF FP32 and vLLM scores on20 predeclared traces spanning short/long, stop/cap and empty-initial-prefix cases. Exact token/position alignment; each mean error<=0.001 nats/token and each maximum token error<=0.01 nats. Retain aligned IDs, scores, mean error, worst position and worst-token error. Missing categories block qualification; predeclared bounded capped fixtures may supply a category absent in the natural pilot.
- On T3 failure investigate prompt/template, slicing, then dtype/TF32. Block Stage2 without automatic tolerance changes. This is a score-agreement gate, not proof of identical near-tie decisions across engines.
- Pilot alarms: unfinished initial attempts>2%; initial correctness<90%; or Stage2's re-estimated reservation beyond remaining headroom. With32 problems, one unfinished exceeds2% and28-or-fewer correct falls below90%. Show values, denominators and thresholds. Engine failure separately blocks. Report median/p90 without automatically tuning J.
- Prompt/J revision needs approval, new plan identity and complete pilot rerun. Algorithm identity includes J. No cap increase or template revision follows automatically from an alarm.

### Data gate and training approval

- Report gold and each k arm on its eligible cohort, plus paired contrasts on the common intersection covered by all three sampling arms. Use corresponding gold problems for common-cohort comparisons.
- Plain NLL is HF FP32 negative mean response-token log probability, stop included, then equal-problem averaging. Keep per-token/answer values and identities sufficient to reproduce summaries.
- Report response-token length median/p90, format proportions, pass/unfinished/retry/coverage rates, and refinement acceptance by step and cut-position decile. Keep denominators explicit.
- Bootstrap whole paired problems10000 times with seed42 and percentile95% intervals. Length contrast is difference of arm medians recomputed in each resample; NLL contrast is difference of equal-problem means. Median paired difference is optional and separately named. Preserve cohort/seed/estimator identities.
- Leakage diagnostics are literal calculator-marker occurrence and the fraction of response word4-grams present in cleaned gold; under-four-word responses are undefined, not zero. Retain component counts and a predeclared deterministic word-normalization contract.
- Do not generate/reuse a temperature1 baseline or calculate offset columns. Mark typical-NLL offset and relative sharpening not measured. Later Stage0 definitions must agree or document differences.
- No automatic merging. A CI containing zero is inconclusive, not equivalence. Equivalence would require separately fixed margins and a whole interval inside them.
- After gate review and budget approval, train k0/k5 only, one seed42. k2 training requires separate approval after later endpoint evidence; no automatic meaningful-difference trigger is introduced.

### Training, handoff and lifecycle

- Required training captures: adapters at0/8/16/32/64; T15 exact gold checkpoint0 identity; gold-matching step logs; manifest with dataset/order, omissions, configuration, seeds, runtime and source hashes.
- Optimizer state and per-step gradients are optional captures. A claimed mid-training resume nevertheless requires verified complete optimizer/scheduler/progress/RNG and adapter recovery state. Without it, retain the interrupted attempt and restart from checkpoint0 under a new recorded attempt and valid budget reservation; never perform weights-only optimizer continuation.
- One experiment-wide ledger enforces USD25 total and sampling10/training6/storage-and-shared-overhead9 sub-caps. Defer baseline work without redistributing caps implicitly. Stage0 research follow-ups are separate from this sampler's Stage0 setup tests.
- Before a job reserve a conservative upper bound: hourlyGPUrate times estimatedhours times1.5 plus setup cost, avoiding double-counted setup. Launch only when spent plus every outstanding reservation including this job stays below both stage and total caps. Actual quotes replace planning prices before launch.
- Count setup, idle, debugging and reruns toward their served stage. Shared/storage charges count once. Initially reserve the expected month of network-volume storage and extend its reservation if retained. Pod stop does not end storage billing.
- At80% of any sub-cap or total, pause for approval before the next job even with headroom. Reconcile completed reservations to actual cost; interrupted or uncertain charges remain reserved until reconciled. Each arm summary includes a cumulative-ledger snapshot without duplicating shared charges.
- Verify both network-volume and downloaded copies by manifest/hash receipts before automatic own-Pod stop. Persist stop-requested before self-stop; an external client/provider check confirms stopped. Do not claim a process witnessed its own successful shutdown. Export failure preserves data and triggers explicit reporting rather than hidden idle spend or destructive fallback.
- Network-volume deletion and termination of an unexported Pod require explicit approval. No cloud rental, token handling, destructive action or paid launch is authorized by spec publication. The first sampling-GPU spend needs a post-preflight go-ahead; later training needs explicit post-gate approval.
- Notebook/runbook grows with the same workflow: readable setup and stage calls, CPU/GPU labels, expected artifacts, fresh-session restore, progress and resume instructions. Keep research exposition outside execution cells and avoid accumulating one-off repair cells.

## Testing Decisions

- Approved primary boundary: the same public saved-workflow commands used by the notebook/runbook, from source preflight through reservations/lifecycle, qualification, generation/review, gate, approved training and handoff. Test observable products, status transitions, errors and invariants rather than private helper structure.
- Use CPU-only, offline fixtures with tiny models and fake engine/provider adapters; no model downloads, paid endpoints or CUDA in unit tests. Independent arithmetic oracles cover token-score slicing, loss denominators, greedy acceptance, bootstrap estimands and cost accounting.
- Source tests reject changed model/tokenizer/cohort/order/A0/config hashes, missing historical evidence, incompatible runtimes and stale child products. They preserve historical gold outputs and distinguish source-audited from real-qualified status.
- Sampler tests cover k0, full-trace cuts, strict score acceptance, shorter/stop-only regrowth, exhaustion below cap, cap/stop accounting, malformed/empty engine outputs, engine-error replay and bounded model retries. Separate iteration-level monotonicity from potentially lower scores after extension.
- Workflow recovery tests kill before/after seals, restore fresh processes, reject concurrent writers, preserve sealed bytes, discard partial blocks, reproduce scientific records and reconcile attempts/dispositions without turning disconnects into model outcomes.
- Scoring tests use hand-built logits/masks, exact token IDs and shifts, real stop handling, missing prompt scores, and both T3 gates. Retain T1 prompt agreement, cleaning/grader identity, context-limit rejection, round-trip reporting, worker numeric settings and batch-dependence diagnostics from the design; CPU fixtures do not stand in for the real-engine checks.
- Review tests round-trip labels through saved exports, preserve automatic verdicts, reject altered source bindings and prevent labels from changing unfinished completion state. Coverage tests exercise0/1/5/6 uncovered, clustered omissions, symmetric-difference flags and exactly512 dispositions.
- Gate fixtures reproduce equal-problem means, stop-inclusive NLL, difference-of-medians bootstrap and undefined leakage; missing baseline is not rendered as zero. Completed reports must reconcile with raw rows and source identities.
- Training fixtures verify stored-ID masks, saved batches/checkpoint0, all T16 batch sizes, scheduler count, finite gradients,64 nonempty steps, required checkpoints/logs, and rejection of adapter-only recovery. Use established SFT tiny-model training prior art rather than mocking away the loss itself.
- Budget/lifecycle tests verify concurrent reservations, cumulative multi-stage limits,80% pauses, ongoing storage, setup/rerun allocation, uncertain-cost recovery, tampered ledgers, failed exports, stop-request versus stop-confirmed evidence, and deletion/unexported termination refusal.
- Notebook checks validate release pin, public-command wiring, fresh-session restoration, CPU/GPU stage labels and consistent run identity. Each implementation slice includes this user path; a final notebook-only ticket is not a substitute for incremental integration.
- Prior art includes the existing saved source-audit, SFT training, flexible-v3 rescoring, durable production recovery, Stage0 budget and notebook validation workflows. Reuse their evidence boundaries; do not import historical numerical tolerances into new gates without approval.
- Record real L40S score/replay/throughput qualification and verified L4 training evidence separately. Passing CPU tests means software behavior passed fixtures, not that the experiment was run or its runtime qualified. A failing gate stops and retains diagnostics; never edit an acceptance number to make it green.

## Out of Scope

- Gold-SFT retraining, GRPO training, held-out evaluation, internal activation/KL/geometry experiments or Stage0 research execution.
- Temperature1 512×8 baseline, historical sample reuse, typical-NLL offset or a relative-sharpening claim.
- Automatic k2 training, k10, k5+5 polish, self-correct SFT, random-correct/length-matched attribution controls and extra seeds.
- Causal claims about on-policyness or mediation; exact MH claims or theoretical guarantees of the released greedy heuristic.
- Automatic arm merging, unapproved equivalence margins, automatic cap/template/J changes, precision/GPU fallback, unlimited retries or duplicate/gold/incorrect slot filling.
- Paid execution, publication/push, persistent-volume deletion or unexported termination by virtue of this spec alone.

## Further Notes

The confirmed design and original supplied proposal are retained separately; this spec incorporates approved amendments and supersedes contradictory source-proposal defaults. Source audits are evidence-backed facts, while engine compatibility, original-training hardware and bounded runtime performance remain preflight/real-execution checks. Unsupported capabilities block dependent work rather than being silently assumed.

Round-trip score agreement and exact replay are different gates. Optional training optimizer capture and valid optimizer resume are different contracts. Generation completion, answer correctness, human review and software completion remain distinct states.

Metric normalization, p90 implementation, concrete fixture identities, engine version/defaults, actual stop metadata behavior and download/stop receipts must be made explicit in implementation preflight. A new scientific policy choice exposed there is escalated before execution. Do not introduce expensive profiling expansions under the name of verification.
