# Pilot 4 follow-up: behavioural gain breakdown

Status: ready-for-agent

## Problem Statement

The researcher observed greedy GSM8K flexible-v3 accuracy rising from 102/150
to 114/150 in the existing GRPO run, while SFT's aggregate accuracy stayed
approximately flat. Aggregate accuracy cannot identify answer gains and losses,
scorer extraction failures, format changes or changes in sampled reliability.
The next policy-distance experiment should not assume the eight-percentage-point
gain is a reasoning improvement before these alternatives have been examined.

The researcher needs a reproducible analysis of saved outputs, a manageable
single-reviewer audit and an inspectable report, without repeating training or
occupying a GPU for CPU work. The previous notebook's restart and environment
problems must not recur in this saved-output workflow.

## Solution

Build a CPU-only, source-bound workflow covering the supplied analyses A, B and
D: verify saved outputs; uniformly rescore greedy SFT/GRPO endpoints; describe
paired answer transitions and SFT churn; measure observed eight-draw GRPO
coverage; export a targeted blinded answer-level audit; freeze the reviewer's
intended-answer labels before revealing gold/scorers; and assemble the findings.

Use the existing public saved-workflow command boundary, also called by one
readable follow-up notebook. Each completed stage saves reusable results and
explicit status. Distinguish implemented software, automatic results, pending
human work and a completed real-data audit.

The first phase explains answer-level outcomes. It does not validate reasoning
chains, establish newly acquired ability, estimate training-seed variability,
or force the findings into the source proposal's categorical thresholds.
Analysis C, forced read-off, is deferred until the CPU findings and a measured
inference budget have been reviewed.

## User Stories

1. As a researcher, I want the supplied proposal preserved unchanged, so its
   original claims can be distinguished from agreed design amendments.
2. As a researcher, I want the exact source runs and score versions identified,
   so similarly named reports do not become interchangeable evidence.
3. As a researcher, I want the same question identities checked across methods,
   so cohort changes do not masquerade as method effects.
4. As a researcher, I want prompts, gold answers and decoding metadata checked,
   so response comparisons have an explicit compatibility boundary.
5. As a researcher, I want method-specific step-0 records retained until their
   equality is established, so similar aggregate lengths do not imply identity.
6. As a researcher, I want greedy and sampled outputs separated, so the
   eight-point greedy gain is not confused with sampled mean reward.
7. As a researcher, I want stored scores preserved beside harmonized rescoring,
   so scorer revisions do not erase prior evidence.
8. As a researcher, I want strict, flexible-v3 and a declared last-number
   diagnostic applied consistently, so extraction behaviour is inspectable.
9. As a researcher, I want extraction independent of gold comparison, so the
   scorer cannot choose an answer because it happens to be correct.
10. As a researcher, I want caps and invalid reasons retained, so a generation
    limit is not mistaken for the absence of an intended answer.
11. As a researcher, I want correct/wrong/invalid transition tables, so gross
    improvements and regressions remain visible under a small net change.
12. As a researcher, I want net flow accounting that reconciles to aggregate
    accuracy, so gains cannot be attributed using a misleading denominator.
13. As a researcher, I want a direct paired GRPO64-versus-SFT64 comparison, so
    separate within-arm intervals are not mistaken for an arm difference.
14. As a researcher, I want SFT gains, losses and churn reported, so a flat mean
    does not conceal different solved questions.
15. As a researcher, I want gold calculation counts and response lengths exposed
    alongside transitions, so potential shortening effects can be explored
    without claiming a causal explanation.
16. As a researcher, I want every required audit case selected deterministically,
    so inconvenient transitions cannot be omitted after inspection.
17. As a reviewer, I want the audit workload counted before labelling, so I can
    see the required cases and the 20 control pairs per method.
18. As a reviewer, I want question and response shown without gold or scorer
    results initially, so I identify the intended answer independently.
19. As a reviewer, I want absent and ambiguous answers representable, so I am
    not forced to invent a numeric answer.
20. As a reviewer, I want progress saved across sessions, so interrupted review
    does not lose labels or silently overwrite an earlier decision.
21. As a researcher, I want identical shared responses deduplicated with their
    source references retained, so one response does not receive conflicting
    labels merely because it appears in two method comparisons.
22. As a researcher, I want labels frozen before the second-stage reveal, so
    answer identification is distinguishable from hindsight about correctness.
23. As a researcher, I want scorer-error reasons recorded after reveal, so
    extraction fixes can be distinguished from unextractable responses.
24. As a researcher, I want the single-reviewer and targeted-selection limits
    visible, so the audit is not presented as independent agreement evidence
    or exact overall human-corrected accuracy.
25. As a researcher, I want all eight sampled draws checked per question and
    endpoint, so missing draws cannot inflate observed coverage.
26. As a researcher, I want correct-draw counts and observed coverage beside
    mean reward, so reliability changes and observed solve transitions are
    distinguished without claiming previously impossible answers.
27. As a researcher, I want paired question-level intervals, so the uncertainty
    matches the paired evaluation and conditions on the observed training runs.
28. As a Colab user, I want CPU-only entry points that do not import a model
    runtime, so scoring does not require CUDA or a compatible Torch install.
29. As a Colab user, I want fresh-runtime restoration from saved configuration,
    so later sections do not depend on hidden variables from earlier sessions.
30. As a researcher, I want changed sources or labels rejected under a frozen
    identity, so restarting cannot silently combine incompatible results.
31. As a researcher, I want tables, plots, per-item records and source identities
    saved together, so the aggregate claims can be audited.
32. As a researcher, I want a partial report while human review is pending, so
    completed automatic results remain useful without implying audit completion.
33. As a researcher, I want inference and reasoning-quality extensions deferred
    explicitly, so preparing this follow-up does not authorize extra GPU work.

## Implementation Decisions

### Scope and source contract

- Reuse existing output artifacts and checkpoint identities. No model calls,
  model downloads, activation measurements, adapter fitting or training occur
  in this phase. One training run per method remains the evidence boundary.
- Use parameterized public commands for preparation/source verification,
  rescoring, transitions, sampled coverage, audit export/label import/freeze,
  reveal and reporting. Reuse existing scoring, recovery and report conventions;
  add narrowly scoped source adapters rather than broad refactors.
- Require explicit source selection. Do not pick the most recent similarly
  named directory. Record source hashes, scorer implementation identity, model
  and tokenizer identities when available, prompt/decoding settings, question
  IDs, checkpoint/mode/draw identities, gold, raw text and cap metadata.
- Validate uniqueness and completeness separately for each checkpoint and mode.
  Greedy source rows are keyed by method/checkpoint/question; sampled rows add
  draw identity. Repeated baseline references in an exported paired trajectory
  are references, not extra evaluation responses.
- Compare methods on matching IDs, question/gold identities and compatible
  prompts/decoding. Report each original cohort separately. If IDs differ, a
  common-ID comparison must state its membership and denominator; never present
  an overlap-cohort estimate as the original 150-question headline. Material
  incompatibilities remain blockers to a matched comparison.
- Deduplicate step-0 responses only after verifying identical inputs, decoding
  settings and response content/token identity where retained. Otherwise retain
  separate baselines and state their differences. Never deduce equality from
  an identical mean length.
- Verify provenance only to the level supported by available artifacts. An
  exported paired-response file cannot independently establish checkpoint or
  runtime integrity. Missing receipts produce an explicit evidence limitation
  or block the claim requiring them, not a fabricated successful verification.

### Scoring and numeric semantics

- Import the existing strict and flexible-v3 scorers unchanged. Apply a single
  recorded implementation uniformly to every included response. Keep stored
  scores alongside new decisions; check historical consistency using the
  version applicable to the source. Report mismatches; do not silently repair
  source records or replace the old reports.
- Define each scorer's bin as correct, valid-but-wrong, or invalid. Invalid
  under a scorer does not establish that a human cannot identify an answer.
- The strict contract requires a final numeric `####` line, checks recognized
  conflicting markers/answer phrases and rejects capped responses. The spec
  adopts the existing implementation's exact behaviour, not a new simplified
  regex. Flexible-v3 first accepts its strict-final-line contract, otherwise
  uses the existing flexible fallback.
- The last-number diagnostic scans the response for the last complete supported
  numeric literal. Reuse existing exact numeric comparison semantics for signed
  integers, decimals, correctly grouped commas and simple rational fractions.
  Retain its span, original spelling, normalized value and invalid reason.
  Currency and adjacent unit text are not part of the numeric value; do not
  infer unit conversions or percentage scaling. Written-out numbers and
  unsupported notation remain unsupported. Reject malformed or undefined final
  numeric expressions rather than selecting an earlier gold-matching number.
  Tests must prevent partial matching inside unsupported numeric expressions.
- Extract without access to gold, then compare numerically in a separate step.
  Preserve capped responses and any extractable candidate for inspection, but
  keep their primary scored status invalid, matching the existing scorers.
  Report capped cases separately. This diagnostic is not a format-blind oracle.
- Preserve the proposal's distinction between overall accuracy, extractable
  fraction and accuracy conditional on scorer validity. Conditional accuracy
  has a changing denominator and is not itself a measure of reasoning quality.

### Paired transitions and uncertainty

- Produce four main endpoint transition tables: two methods by strict and
  flexible-v3. Include last-number tables as diagnostics. Save the contributing
  question IDs and response references for every cell.
- With bins C/W/U, report gross gains W→C and U→C, gross losses C→W and C→U,
  churn as gains plus losses, and net correct-count change as gains minus
  losses. Report net W-related flow W→C minus C→W and net U-related flow U→C
  minus C→U; their sum must equal the net change. Do not divide gross flows by
  net +12 and call the result an explained fraction of the gain.
- Preserve scorer-defined flows alongside the later human audit. The audit
  can identify extraction explanations for selected cases but does not rewrite
  the definition of the original +8-point result.
- Use 10,000 paired percentile bootstrap resamples of question IDs with local
  seed 42, using the existing report's interval convention. Within-arm endpoint
  differences and the direct GRPO64-minus-SFT64 comparison carry both records
  whenever an item is resampled. For sampled metrics retain all eight observed
  draws for the resampled item. Save resampling identity and denominators.
- Intervals describe question-level uncertainty conditional on the trained
  checkpoints and observed draws. They do not include training-seed variability
  or fully characterize uncertainty from new decoding draws. No new p-value,
  causal attribution or categorical significance gate is introduced.
- Save exact gold calculator-annotation counts and nonempty-line counts rather
  than selecting one after seeing the losses. Show SFT transitions and lengths
  by these descriptive measures with denominators; call neither a validated
  difficulty measure nor a causal explanation for errors.

### Targeted single-reviewer audit

- For each method select every question with a flexible-v3 endpoint-bin change,
  an endpoint invalid under flexible-v3, or an endpoint disagreement between
  flexible-v3 and the last-number diagnostic on validity or numeric value.
  Invalid endpoints are retained irrespective of whether both diagnostics fail.
- From the remaining stable pairs, sample 20 per method, or all if fewer,
  using sorted question IDs and a separate local seed-42 random generator per
  method. Record the candidate pool, selected IDs, inclusion reasons and seed.
  The random controls are not an independent population or a new test cohort.
- Strict-only format failures remain in the complete automatic results and
  may appear in random controls; they do not all automatically require review.
  Report the actual selected pair count and unique-response count before
  labelling. Never silently truncate the mandatory set to meet a guessed budget.
- Review both endpoint responses of each selected pair. Identical verified
  shared responses receive one response-level label with all parent references.
  The packet does not reveal pairs, checkpoint order or source filenames.
- Stage one shows only opaque response ID, question and saved response. Hide
  method/checkpoint, gold, scorer decisions and inclusion reasons. Style can
  still suggest the source; describe this as partial blinding.
- Stage-one labels contain a numeric intended final answer, absent, ambiguous,
  or pending, with optional evidence text and reviewer note. The reviewer may
  enter a numeric transcription of a written-out answer; retain the original
  evidence. Do not force ambiguous responses into a correct/wrong judgment.
- Save drafts incrementally. Explicitly freeze the complete selected packet's
  labels before gold/scorer reveal. Pending or unknown/duplicate response IDs,
  malformed labels and changed packet identities block a completed freeze.
- Stage two compares the frozen intended answer against gold and scorer output.
  Retain scorer agreement, missed/unextractable intended answers, wrong-number
  extraction and other discrepancies. The reviewer may assign explanation
  categories: intermediate number, missed format, conflicting/ambiguous answers,
  cap/truncation, other, or uncertain. Preserve explanatory notes and examples.
- No hidden edits to frozen labels. Corrections require a separately identified
  revision with reason and an explicit post-reveal status; do not present that
  revision as blind. One reviewer supplies no inter-reviewer agreement measure.
- Audit reports are limited to the selected set and known inclusion reasons.
  Do not estimate exact whole-cohort human-corrected accuracy or reason-validity
  rates. The automatic cohort totals and the targeted review are separate tables.

### Observed sampled reliability

- Use the saved GRPO eight-draw endpoint evaluations only. Verify eight distinct
  draws per question at both endpoints, common question identities and the
  declared temperature/decoding settings. Missing or duplicate draws block the
  fixed-eight comparison rather than reducing denominators silently.
- For flexible-v3 and last-number diagnostics report correct draws out of eight,
  mean draw reward, observed coverage (at least one correct draw), coverage
  gains/losses, and the full endpoint correct-draw-count transition table.
  Describe changes among already-observed solves separately from 0/8→positive.
- Call the latter observed coverage gains, not proof of newly acquired ability.
  Flat coverage with rising mean reward is compatible with sharpening but does
  not uniquely establish it. Last-number coverage is still extraction-dependent.
- A saved untuned-sampling run is a separate cross-check only if question cohort,
  evaluation role and decoding match. Training-cohort baseline samples cannot
  validate held-out coverage; do not pool additional draws opportunistically.

### Notebook, artifacts and recovery

- Provide one incremental, readable CPU follow-up notebook, with named inputs
  and short stage descriptions. Keep scientific logic in public tested workflow
  commands. Do not append repair cells to the historical Pilot 4 notebook.
- Every section restores its required saved paths from configuration. After
  runtime restart, setup plus the desired stage must work without rerunning
  expensive prior stages or relying on hidden notebook globals.
- Copy/verify source artifacts into a canonical analysis bundle and record
  availability and identity once. Subsequent stages operate on that verified
  saved bundle, checking the relevant parent receipts. Do not repeatedly scan
  unrelated Drive trees, model files or activation shards.
- Persist configuration, source records, normalized response JSONL, new scoring
  JSONL, per-question transitions/draw counts, audit selection and opaque packet,
  reviewer drafts, frozen labels, private mapping, reveal results, tables/plots,
  structured summaries, stage receipts, logs and explicit status.
- Separate reviewer-visible packet data from the private source/gold/scorer
  mapping. Do not promise access-control security when only two local files are
  provided; packet views and exports must actually omit the hidden fields.
- Resume completed verified stages; save partial review progress; publish stage
  outputs atomically. Changed sources, protocol or frozen labels require a new
  analysis/revision identity. Never merge incompatible evidence under one name.
- Report human-review pending separately from automatic-analysis completed.
  A child PID or an existing output directory is not completion evidence.
- The final report contains original/harmonized totals, full flow tables, direct
  paired comparisons, SFT churn/length descriptions, scorer disagreements,
  audited examples, sampled reliability, provenance and limitations. Partial
  reports prominently omit unfinished audit claims.
- Conclude with compatible explanations and a decision about whether to propose
  forced read-off or a focused reasoning audit. Neither launches automatically.

## Testing Decisions

- The user approved the existing public saved-workflow command seam used by the
  notebook: source verification → rescore → transitions/coverage → audit export
  → label save/freeze → reveal/report. Prefer this one integration boundary
  over exposing internal helpers solely for tests.
- Use tiny CPU-only fixtures, temporary artifact roots and no model/network
  downloads. Exercise the public commands and inspect persisted outcomes and
  errors, not internal calls or source-code shapes.
- Extend prior offline rescoring, SFT score-audit, paired reporting and notebook
  session/recovery patterns. Add independent hand-computable numeric, transition
  and coverage examples rather than deriving expected values from production
  functions under test.
- Verify immutable historical scores, strict/v3 parity, gold-independent
  extraction, fractions/negative/decimal/currency/commas, malformed expressions,
  ambiguity and cap handling. Do not change existing scorer acceptance rules.
- Verify net/gross flow identities, paired resampling, overlap denominators,
  differing step-0 retention, sampled draw completeness and observed-coverage
  versus mean-reward distinctions.
- Verify exact audit selection, deterministic controls, response deduplication,
  omission of hidden metadata from reviewer exports, draft resume, incomplete
  freeze rejection, immutable label receipts and versioned post-reveal changes.
- Exercise fresh-session command invocations without Torch/CUDA availability;
  stop/restart between stages; reject altered sources, wrong packet mappings,
  missing/duplicate labels and stale derived results. Idempotent replay must
  reproduce results without double counting.
- Validate report totals against the saved per-item records and partial-state
  visibility. Keep software tests separate from later real-artifact source
  acceptance and completed human review. No test passing substitutes for either.

## Out of Scope

- Training, new seeds, new sampled SFT responses or a fresh GSM8K evaluation.
- Forced read-off inference, activation ablation, KL/coefficient measurements,
  new policy-distance arms or a model/lens refit.
- Exhaustive reasoning-validity annotation and whole-cohort human ground truth.
- Inter-reviewer agreement, exact ability-support claims or causal attribution
  of response shortening to errors.
- Changing strict/flexible-v3 rules, overwriting historical artifacts, adopting
  the source proposal's numeric outcome categories or retuning after results.
- Publishing GitHub changes, launching experiments or implementation merely
  because this specification has been written.

## Further Notes

The supplied proposal and seven interview decisions are preserved alongside this
specification. Existing aggregate outcomes were already known; this is a
prospective follow-up analysis plan conditional on them, not an unseen-outcome
preregistration. The source proposal's forced-read-off claims and finite-draw
new-ability language are not adopted literally.

Implementation tickets are proposed separately and require the user's review
of granularity and blocking edges before publication. Source payload inventory
and raw-response availability must be verified during implementation acceptance;
the currently inspected local report snapshot alone is insufficient for A–D.
