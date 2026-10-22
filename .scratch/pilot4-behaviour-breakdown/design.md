# Pilot 4 follow-up: behavioural gain breakdown

Status: design interview; not an approved execution specification

## Objective

Explain the observed endpoint behavioural change in the existing Pilot 4 GRPO
run, alongside the saved SFT run. Separate scorer extraction, answer
presentation, answer correctness and finite-draw reliability before designing
the proposed policy-distance comparison. Whether to additionally audit the
validity of the reasoning is an open scope decision.

The supplied proposal is preserved unchanged in `source-proposal.docx`, with a
readable extraction in `source-proposal.md` and its SHA256 in
`source-record.json`. The comments below review the proposal; they are not
silent amendments to it. No follow-up analysis or inference has started.

## Verified starting evidence

The local snapshot at
`artifacts/reports/pilot4-results-so-far-2026-10-13/inputs/grpo-behaviour.json`
reports greedy flexible-v3 accuracy 102/150 to 114/150: **+8 percentage points,
net +12 correct responses**. This does not identify the gross gains or losses.
Its paired item-bootstrap change interval is +2.0 to +14.6667 points, using
10,000 resamples and seed 42, conditional on this training run.

Strict-correct responses increase from 8 to 75; flexible-invalid responses
decrease from 13 to 4; strict-invalid responses decrease from 137 to 56. Mean
greedy completion length changes from 215.53 to 223.27 tokens. These are
descriptive results, not an attribution of the gain to formatting or reasoning.

Sampled mean reward is a separate result: 54.4167% to 67.5%, from eight draws on
each of 150 questions at checkpoints 0 and 64. It is not pass-at-eight.

`inputs/evaluation-config.json` in that snapshot contains the 150 item IDs,
gold solutions, prompts, checkpoint identities and evaluation settings. Raw
held-out GRPO responses were not found in this local snapshot and must be
retrieved and verified from the canonical Drive artifact tree before analysis.

Canonical artifact-root-relative source:
`runs/pilot-4/Qwen--Qwen2.5-1.5B-Instruct/behaviour/pilot4-grpo-fp32-v1-behaviour-v1`.

## Proposed workstreams from the supplied document

| Stage | Work | Resources |
| --- | --- | --- |
| Source audit | Verify item identities, source records, prompts, decoding, scorer code, checkpoints and step-0 identity; retain method-specific step-0 files until equality is established | CPU / Drive |
| A | Rescore greedy endpoints; paired correct/wrong/scorer-invalid transitions; direct GRPO64 versus SFT64 comparison; SFT churn and exploratory length/difficulty breakdowns | CPU |
| B | Add a declared last-number diagnostic; independently audit intended answers and scorer failures using a reviewed audit sampling plan | CPU / reviewers |
| D | Compare mean sampled reward, observed eight-draw coverage and per-question correct-draw counts | CPU |
| C | Continue each checkpoint's own saved assistant response with a fixed answer cue; evaluate read-off changes and answer switching | New inference, if approved |

Timing estimates in the supplied document are proposals, not measured budgets.
The source audit and small workload measurements must inform execution estimates.

## Design issues to resolve

### 1. Answer correctness is not reasoning validity

Wrong-to-correct final-answer transitions can reflect different reasoning,
formatting, extraction or chance. If the target claim is improved reasoning,
the audit needs separate reasoning-validity labels and a rubric. Otherwise
report answer-level transitions and label causal explanations as hypotheses.

### 2. Eight-draw coverage is an observed sampling outcome

A 0/8 to at-least-1/8 transition does not establish a newly acquired ability.
For example, a model with a fixed 5% success probability has a 66.3% chance of
producing no correct answer in eight independent draws. More reliable sampling
can also increase observed coverage. Use observed coverage/observed new solves,
with explicit finite-draw uncertainty, rather than claims about zero support.

### 3. Forced extraction is an intervention, not a neutral reader

Appending a cue can elicit additional computation or a changed answer, even
when only 12 tokens are allowed. A surviving checkpoint difference does not
by itself prove reasoning improvement; a reduced difference does not prove
pure formatting. Define answer-switching denominators, cap treatment, exact
assistant-prefix token reconstruction, numeric parsing and interpretive limits.

### 4. A last-number scorer is not format-blind ground truth

It is a useful additional extraction heuristic, but can select an intermediate
or trailing unrelated number. The numeric grammar, units/percentage treatment,
cap handling and extraction/gold separation must be specified and tested.

### 5. Disagreement-only audit is selectively observed

Three scorers can agree and all be wrong about the intended answer. A random
sample of disagreements cannot be used as an unweighted overall hand-labelled
accuracy estimate. Decide between an exhaustive audit, a stratified audit with
known inclusion probabilities, or bounded targeted case analysis without an
overall accuracy claim. Ensure the R3 transition set is covered even when the
three scorers agree. Decide reviewer count, blinding and adjudication.

### 6. Net gain is not a count of gross improvements

Show every gain and loss flow. For attribution, specify whether each number is
a gross transition count, a net flow, or a reviewed response category. Dividing
gross gains by net +12 can exceed 100%, and audit-based explanations can overlap
unless response categories and priority rules are explicit.

### 7. Decision rules need a coherent estimand and complete outcomes

The proposed format/reasoning rules need not be mutually exclusive. The SFT
stable/churn thresholds leave 10–19 changes unclassified. Decide whether to
report continuous decompositions with compatible explanations, or use explicit
exhaustive classification rules. Preserve the original +8-point headline while
reporting any overlap-cohort comparison separately.

### 8. Observed evidence and future decisions must remain distinct

Aggregate outcomes and prior scorer problems are already known. The follow-up
can freeze a prospective analysis plan before the new breakdown, but cannot
claim blindness to all Pilot 4 outcomes. Flexible-v3 was developed post hoc in
the earlier SFT audit; see `docs/adr/0005-gsm8k-flexible-v3.md`. Preserve stored
scores and publish harmonized rescoring as a separately named product.

## First decision frontier

1. Scope: answer-level decomposition only, or add reasoning-validity annotation?
2. Audit resources: retain two independent human reviewers, or another declared
   reviewer/adjudication arrangement?
3. Execution order: CPU evidence first with a reviewed decision on inference,
   or commit now to the forced-read-off experiment regardless of CPU findings?

Answers will be recorded here and may open dependent decisions about sampling,
annotation rubrics, decision rules and continuation details. No ADR is created
until an actual consequential trade-off is settled. This document is not
`spec.md`, and no implementation tickets have been issued.

## Interview decisions — 2026-10-22

- **Q1 accepted:** answer-level decomposition first: answer correctness,
  extraction/presentation and observed sampled reliability. Do not label
  wrong-to-correct transitions as validated reasoning improvements. After
  reviewing the decomposition, decide whether a focused reasoning-validity
  audit of remaining unexplained cases is warranted; no exhaustive reasoning
  annotation is included in the initial scope.
- **Q2 accepted:** one human reviewer; no inter-reviewer agreement claim.
  Blinding, sampling and the annotation rubric remain to be settled.
- **Q3 accepted:** perform the CPU-only source audit and A/B/D first; decide
  whether to run C after reviewing those findings and its measured inference
  budget. This is the agreed execution sequence, not authorization to start
  implementation before the design/spec is settled.

## Second decision frontier

4. Human-audit coverage: explain the observed transitions through a targeted
   case audit with checks of unchanged cases, or obtain a complete human answer
   label for every unique greedy response? A targeted audit cannot be reported
   as exact overall hand-corrected accuracy; a probability-sampled audit would
   need its own weighted estimator and uncertainty.
5. Interpretation: report continuous transition/audit/coverage decompositions
   and compatible hypotheses, or retain revised, explicit categorical decision
   rules? The original numeric cutoffs are still unapproved.

### Round 2 decisions

- **Q4 accepted:** targeted human audit to explain gains/losses and possible
  extraction errors, plus a reproducible sample of unchanged cases. Do not
  report exact overall human-corrected accuracy from this targeted sample.
  The exact inclusion rules and unchanged-case sample remain to be approved.
- **Q5 accepted:** report counts and intervals, with potentially compatible
  explanations. The supplied numeric cutoffs for mostly-format, real-reasoning,
  SFT-stable and SFT-churn are not adopted as decision gates. Report churn
  continuously and describe evidential limits instead of forcing categories.

## Third decision frontier

6. Proposed audit inclusion: all flexible-v3 endpoint-bin changes
   (correct/wrong/invalid), every pair with an endpoint invalid under flexible-v3,
   and every pair with an endpoint disagreement between flexible-v3 and the
   declared last-number diagnostic on validity or extracted numeric value.
   Add a seeded random sample of 20 remaining stable pairs per method (or all
   if fewer), sampled from sorted item IDs with local seed 42. Do not inspect
   follow-up outcomes to choose the sample size. Strict-only format failures
   remain visible in the full automatic tables; they are not all automatically
   sent for human review. The same response can be shared by several pairs and
   receives one response-level label when its source identity is identical.
7. Proposed audit visibility: first show question and response under opaque
   identifiers, hiding method/checkpoint, gold and scorer outputs. Record the
   intended final answer or no-answer/ambiguous. After freezing those labels,
   compare with gold and reveal scorer decisions for error attribution. This
   is partial blinding: writing style can still suggest the method. No
   reasoning-validity judgment is requested in this first-stage rubric.

### Round 3 decisions

- **Q6 accepted:** the inclusion rules above and 20 remaining unchanged control
  pairs per method, using seed 42. Report the actual packet size before labelling.
- **Q7 accepted:** intended-answer identification first, followed by frozen labels
  and a separate gold/scorer reveal for error attribution. One reviewer; no
  inter-reviewer agreement claim.

The user requested the to-spec workflow followed by to-tickets. Scientific
interview decisions are now recorded; specification synthesis follows. The
public workflow test boundary must be checked with the user before publishing
the specification. Ticket granularity and blocking edges require a separate
review before individual tickets are published. New inference remains deferred.

## Specification handoff

The user approved the public saved-workflow boundary for tests: source
verification/import, rescoring, transitions/coverage, audit export, label
save/freeze, reveal and reporting. CPU fixtures cover fresh-runtime recovery
and changed-source/label rejection; real Drive acceptance remains separate.

`spec.md` synthesizes the agreed scope and contracts. `ticket-proposal.md`
contains five proposed vertical slices and their blocking edges, pending user
approval before individual issue publication. This handoff does not launch
implementation or new analyses.
