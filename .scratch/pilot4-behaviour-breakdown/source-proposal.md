## Pilot 4 follow-up: what is GRPO's 8-point gain?

Oct 22, 2026 · @Prasad Mahadik

### Purpose

In pilot 4, GRPO raised GSM8K accuracy by 8 points (68% → 76%) while SFT stayed flat (68% → 66%). This doc plans four analyses that decide what that gain is made of: better reasoning, answers that became readable to the scorer, corrected extraction errors, or sharpening of answers the model could already give.

The answer decides how we design the planned policy-distance experiment (gold SFT vs self-generated SFT vs Sampling SFT vs GRPO). That experiment is meant to explain a behavioural difference between training methods. If the difference is mostly answer format, plain accuracy would mislead us and every arm needs format-split outcome measures. If it is reasoning, the experiment is strongly motivated as planned.

All four analyses use saved outputs or short inference runs. No new training is needed. Extra training seeds are deferred (see Scope).

### What we already know

GRPO's gain arrives almost entirely at step 64, at the same checkpoint where answers switch to the canonical format; length does not change. Setup: Qwen2.5-1.5B-Instruct, rank-1 LoRA, 64 steps, 150 held-out GSM8K questions, one training seed per method. GRPO reward uses the flexible-v3 extractor.

| Measure (greedy, 150 questions) | GRPO step 0 | GRPO step 64 | SFT step 0 | SFT step 64 |
| --- | --- | --- | --- | --- |
| Flexible accuracy | 68.0% (102) | 76.0% (114) | ~68% | ~66% |
| Flexible unextractable | 8.7% (13) | 2.7% (4) | not yet pulled | not yet pulled |
| Strict accuracy | 5.3% (8) | 50.0% (75) | not yet pulled | not yet pulled |
| Strict-readable responses | 13 | 94 | not yet pulled | not yet pulled |
| Accuracy among flexible-readable answers | 74.5% (102/137) | 78.1% (114/146) | not yet pulled | not yet pulled |
| Mean response length | 215.5 tokens | 223.3 tokens | 215.5 tokens | 137.8 tokens |

GRPO flexible accuracy across steps 0/8/16/32/64: 68.0, 68.7, 71.3, 68.7, 76.0%. Strict accuracy: 5.3, 5.3, 10.0, 12.7, 50.0%.

GRPO step-64 change, paired bootstrap over questions: flexible +8.0 points (95% interval +2.0 to +14.7); length +7.7 tokens (−0.7 to +16.1). SFT change: −10.7 to +6.7 points.

GRPO sampled mode (8 draws per question, average reward): 54.4% → 67.5%; the greedy–sampled gap closes from 13.6 to 8.5 points; accuracy among readable draws 63.5% → 70.2%.

All intervals hold the training run fixed. They cover which questions were tested, not retraining.

Source: pilot4-grpo-fp32-v1 behaviour report in Drive. SFT numbers are from the reproducible SFT pilot; its strict and unextractable rates still need extracting (Analysis A).

### Competing readings of the GRPO gain

The current numbers fit several readings, and the accuracy totals alone cannot tell them apart. The strict-format jump and flat length lean towards format, but that lean is an inference.

| Reading | What it claims | Which analysis tests it |
| --- | --- | --- |
| R1 Reasoning | GRPO fixed answers that were wrong | A (wrong → correct), C (gain survives forced extraction), D (coverage grows) |
| R2 Readability | Answers flexible-v3 could not read became readable | A (unextractable → correct), C (base model near 76% under forced extraction) |
| R3 Extraction errors | flexible-v3 picked the wrong number from correct responses; a cleaner final format fixed that | B (hand audit of flexible "wrong" responses) |
| R4 Sharpening | Probability concentrated on answers the model could already give | D (coverage flat while average reward rises) |
| R5 Run luck | The step-64 jump would not recur in another training run | Not tested here (seeds deferred) |

Why the totals cannot separate R1 and R2: at step 0 GRPO had 102 correct, 35 wrong and 13 unextractable; at step 64, 114, 32 and 4. That fits "9 unreadable became correct, 3 wrong became correct" (mostly R2). It equally fits "9 unreadable became wrong, 12 wrong became correct" (all R1). Only per-question tracking separates them.

Why R3 matters: the reward reads answers with flexible-v3, not the strict scorer. So the strict-format jump (13 → 94 responses) was not rewarded directly. A plausible indirect route is that one clean final answer stops flexible-v3 grabbing a wrong number from a correct response, which earns reward. If so, some apparent wrong → correct flips are extraction fixes, not reasoning. This is a hypothesis to test, not a finding.

### Data inventory

Everything the analyses need already exists in Drive under SPAR, except the forced-extraction outputs (Analysis C). Folder names below are as saved; contents are inferred from names and sizes and must be confirmed on first load.

| Run | Drive folder | What it holds (to confirm) | Used by |
| --- | --- | --- | --- |
| GRPO, all checkpoints | pilot4-grpo-fp32-v1-behaviour-v1/results/responses.jsonl (15.2 MB) | Greedy responses at steps 0/8/16/32/64; 8 sampled draws per question at steps 0 and 64 | A, B, C, D |
| GRPO, behaviour summary | pilot4-grpo-fp32-v1/grpo-behaviour-report.console.log | Aggregate numbers in "What we already know" | Cross-check |
| GRPO, paired comparison | pilot4-grpo-fp32-v1-behaviour-paired-v1 | Earlier paired analysis; check before rebuilding | A |
| Untuned sampling | pilot4-untuned-sampling-v1/results/responses.jsonl (5.0 MB) | Base-model sampled draws | D (cross-check) |
| SFT, step 64 | pilot2-sft-fp32-l4-batch8-v1-step-64 and …-step-64-flexible-v2 | Greedy responses, scored | A, B, C |
| SFT, intermediate | pilot2-sft-fp32-l4-batch8-v1-step-32, …-trajectory | Responses and trajectory summary | Optional |
| SFT, scorer audit | pilot2-sft-fp32-l4-batch8-v1-flexible-v3-audit-v1 | flexible-v3 audit of SFT outputs | B |

Three consistency checks before any analysis:

Same 150 questions. Confirm the SFT evaluation used the same question IDs as GRPO (the GRPO report lists them). If not, restrict both to the overlap.

Same step-0 responses. Both runs report a step-0 mean length of 215.5 tokens, which suggests identical base-model responses. Confirm the prompt template and decoding settings match; if they do, use one shared step-0 file.

Same scorer version. Rescore every file with flexible-v3 and the strict scorer from one code version, rather than mixing stored scores (the SFT folders show flexible-v2 and v3 versions).

### Analysis A: per-question transitions

Question: which questions changed bin between step 0 and step 64, for each method? This separates R1 (wrong → correct) from R2 (unextractable → correct), which the totals cannot do.

Method.

Rescore every step-0 and step-64 greedy response with flexible-v3 and with the strict scorer. Assign each response one bin per scorer: correct, wrong, or unextractable.

For each method and scorer, build a 3 × 3 table: rows = step-0 bin, columns = step-64 bin, cells = number of questions. Four tables in total (GRPO and SFT, flexible and strict).

Report net gain = (questions moving into correct) − (questions moving out of correct), split by origin bin.

Test the GRPO-vs-SFT difference at step 64 directly: paired bootstrap over questions (10,000 resamples, same seed as the existing reports) on flexible accuracy, GRPO-64 minus SFT-64.

For SFT, also cross-tabulate flips against step-64 response length and the gold solution's number of steps (count of lines or <<…>> annotations in the gold answer).

Outputs. Four transition tables; the GRPO-vs-SFT paired interval; for SFT, the gain/loss counts and a table of losses by gold-solution step count.

What we read off.

GRPO: the share of its 12 net gains that come from unextractable vs wrong at step 0.

SFT: total churn (gains + losses), not only the net −2 points. A flat average with 20 gains and 23 losses means something different from 3 losses and nothing else.

SFT losses concentrated on many-step problems would link the shortening (215 → 138 tokens) to errors.

The strict-scorer tables show whether SFT also learned a canonical format (gold answers end in #### N).

Cost. Scoring only, no model calls. Under an hour once files are loaded.

### Analysis B: format-blind scoring and hand audit

Question: how much of each scorer's verdict reflects the answer's content, and how much reflects how it was written? This tests R3 (extraction errors) and checks that "unextractable" really means no answer.

Method.

Last-number scorer. For every greedy response in the four files (GRPO 0/64, SFT 0/64), take the last number in the text (strip commas, currency and units; compare numerically to gold). Record its verdict beside flexible-v3's and strict's.

Disagreement set. Collect responses where any two scorers disagree on correct vs not. Expected to be small: tens of responses per file.

Hand audit. Two people independently label each disagreement (or a random sample of 40 per file if larger). Labels: model's intended final answer; whether it is correct; and, if a scorer erred, why (picked an intermediate number, missed an unusual format, truncated response, other). Resolve conflicts by discussion; report agreement before resolution.

R3 check. Among GRPO step-0 responses that flexible-v3 marked wrong but that became correct at step 64, count how many already contained the correct final answer at step 0 per the hand label.

Outputs. For each file: accuracy under each scorer and under the hand label; a table of scorer error types; the R3 count.

What we read off.

If hand-labelled accuracy at GRPO step 0 is well above 68%, part of GRPO's "gain" was always there, hidden by extraction.

A large R3 count means some apparent reasoning gains are extraction fixes.

From here on, report accuracy among readable answers beside overall accuracy for every run.

Cost. Scoring is free; the audit is a few hours of human labelling.

### Analysis C: forced-extraction baseline

Question: if every model's answer is read off the same way, how much of GRPO's lead remains? This tests R2 directly, and whether the base model already had the ability.

What forced extraction is. It keeps each model's own saved reasoning unchanged and only standardises the final read-off:

Take the model's saved greedy response.

Append \nTherefore, the final answer is to it.

Let the same checkpoint continue greedily for at most 12 tokens.

Parse the first number in that continuation and compare to gold.

Checkpoints. Base (step 0), GRPO step 64, SFT step 64. Each model reads its own reasoning; we never swap reasoning between models.

Details to fix before running.

Keep the chat template exactly as in generation; append inside the assistant turn, after any end-of-turn token is removed.

Truncated responses (hitting the 1,024-token cap) stay in, flagged, and are reported separately.

Record, per response, whether the forced answer agrees with the original answer when the original was readable. Disagreement means the model re-decided at read-off; report the rate per checkpoint.

Outputs. Forced-extraction accuracy for the three checkpoints; paired bootstrap intervals for GRPO-64 minus base and SFT-64 minus base; re-decision rates.

What we read off.

Base under forced extraction near 76%: the base model already reached GRPO's answers; GRPO's gain was mostly presentation (R2).

GRPO-64 still clearly ahead of base under forced extraction: that lead is a gain format cannot explain (R1 or R4).

SFT-64 vs base under forced extraction: whether SFT's shortened reasoning lost or kept the answer content.

High re-decision rates weaken all three readings and must be reported.

Cost. About 450 short continuations (150 per checkpoint). Each is one forward pass over existing text plus 12 new tokens: minutes on an L4.

### Analysis D: coverage across sampled draws

Question: did GRPO enable the model to solve questions it could never solve before, or only make it give its existing answers more reliably? This separates R1 from R4 (sharpening).

Method.

From the saved GRPO sampled draws (8 per question, steps 0 and 64), compute coverage: the number of questions where at least one of the 8 draws is correct.

Compute it under flexible-v3 and under the last-number scorer from Analysis B, so a format change cannot create coverage.

Per question, record the number of correct draws out of 8 at each step. Plot the change per question: questions moving from 0/8 to ≥1/8 are new solves; questions moving from, say, 3/8 to 7/8 are sharpening.

Cross-check step-0 coverage against the untuned-sampling run, if it used the same temperature and questions.

Outputs. Coverage at step 0 and step 64 under both scorers; counts of new solves, lost solves and sharpened questions.

What we read off.

Coverage flat while average reward rises 54% → 68%: sharpening (R4).

Coverage rises, with new solves on questions that were 0/8 at step 0: genuine new ability (R1).

Eight draws is a small budget: a question with a 5% solve rate is missed most of the time. Treat small coverage changes (a few questions) as inconclusive.

Scope. GRPO only. SFT has no saved sampled draws, so SFT coverage would need new generation; deferred unless Analyses A–C make it necessary.

Cost. Scoring only.

### Decision rules

We fix these thresholds before looking at any results, so the outcome cannot shape the reading. They are judgement calls sized to 150 questions; the team can adjust them now, not after.

| Outcome | Evidence pattern | What it means | Consequence for the policy-distance experiment |
| --- | --- | --- | --- |
| Mostly format | ≥ 2/3 of GRPO's net gain comes from unextractable → correct (A) or from extraction fixes (B); base under forced extraction ≥ 73% (C) | GRPO mainly taught answer presentation | Every arm reports format and content accuracy separately (forced extraction + hand-checked scorer). Add a mechanistic test: do the late-layer GRPO adapter writes promote answer-format tokens, and does ablating them drop strict accuracy while forced-extraction accuracy holds? |
| Real reasoning gain | ≥ 1/2 of GRPO's net gain is wrong → correct after the B audit; GRPO-64 beats base under forced extraction with an interval above zero (C); coverage rises (D) | GRPO improved problem-solving | Run the experiment as planned. Prioritise self-generated-correct SFT: same data type as GRPO, plain SFT objective. Matching GRPO points to data source; falling short points to the objective. |
| Mixed | Neither pattern holds | Both format and content changed | Run the experiment, report both outcome measures for every arm, and do not headline plain accuracy. |
| SFT churns | SFT gains + losses ≥ 20 questions (A) | Off-policy gold data changes answers in both directions under a flat average | Add the style × content arms (format-stripped gold, model-rewritten gold) to attribute the losses. If losses sit on many-step problems, shortening is a candidate cause. |
| SFT stable | SFT gains + losses < 10 (A) | SFT shortened responses by a third without changing which questions it solves | Treat length compression as SFT's main behavioural effect and study it directly. |

The GRPO and SFT rows are independent; one GRPO outcome and one SFT outcome will both apply.

### Scope, limitations and open items

All conclusions from this plan describe one training run per method. Extra seeds are deferred for cost, so run-to-run variation (R5) stays untested.

Limitations.

One seed. GRPO's whole gain arrives in a single jump between steps 32 and 64, the kind of event most likely to differ between runs. Any conclusion here should be phrased as "in this run".

150 questions. One standard error on an accuracy near 75% is about 3.7 points. Differences of a few questions are not interpretable.

Forced extraction can re-decide. Analysis C reports how often it does; high rates weaken its conclusions.

Eight draws limit the coverage test to questions solved fairly often.

Open items.

☐ Confirm what each responses.jsonl holds (checkpoints, greedy vs sampled, prompt template).

☐ Confirm SFT and GRPO used the same 150 questions and identical step-0 responses.

☐ Write down the strict scorer's exact rule in this doc.

☐ Agree the decision-rule thresholds with the team before running.

☐ Choose two labellers for the Analysis B audit.

☐ Revisit seeds once A–D are in: if the conclusion hinges on GRPO's step-64 jump, one extra GRPO seed becomes worth its cost.

Order. A and B first (scoring only), then D (scoring only), then C (short inference). A, B and D can run in one session.

