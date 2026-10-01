# The pilot-1 interview, explained

Recovered from the original interview and checked against the final spec, ADRs
and actual saved runs. There were 49 numbered questions, a final design
confirmation, and later questions about test seams and exact logit ties.
Many questions repeated or refined previous decisions; they were not 49
independent scientific choices. A yes to a recommendation is not evidence that
its meaning was clear. This guide explains the implementation and consequences
without changing any experimental decisions.

## Decisions versus mismatches

- The original full-MMLU suggestion was superseded by a 1140-item balanced sample.
- No-system-message was agreed, but Qwen inserted its default system message.
- Some initial flexible-extraction wording suggested broader acceptance than
  flexible v1 implemented. Flexible v2 is a later explicit revision, not a rewrite
  of what the original score meant.
- Exact ties initially stopped the run; the user later approved versioned
  invalid/incorrect-and-continue scoring.
- A saved invalid answer record is compatible with a completed run. An unresolved
  execution error or missing item record is not.

## Every numbered question

### Q1: Reference provenance

**Why ask?** The old numbers are meaningful only alongside their prompts, item IDs and scorer.

**Operational consequence:** We lacked those artifacts. The implementation records new runs and labels the old numbers as references, not exact replications.

<details>
<summary>Original question wording</summary>

❓ Q1 — Baseline provenance: Where do 0.640 GSM8K, 288 tokens, and 0.570 MMLU come from? Please point to the paper, notebook, command, or prior run if one exists.

</details>

### Q2: Meaning of reproduction

**Why ask?** Matching a number and recreating a procedure are different objectives.

**Operational consequence:** We chose a repeatable new baseline; a score gap triggers investigation rather than automatic failure.

<details>
<summary>Original question wording</summary>

❓ Q2 — Meaning of “reproduce”: Is success matching those particular numbers under the source protocol, or producing a pinned, repeatable untuned baseline whose differences are explained?

</details>

### Q3: GSM8K item list

**Why ask?** Different problems can change accuracy even when the model is unchanged.

**Operational consequence:** 150 official test items were sampled and frozen. Saved IDs are reused for later paired comparisons.

<details>
<summary>Original question wording</summary>

❓ Q3 — GSM8K population: Which 150 held-out problems should be evaluated: a preselected list you have, a seeded sample from the official test split, or another split?

</details>

### Q4: MMLU population

**Why ask?** Full-test evaluation costs more; a subset has sampling variation.

**Operational consequence:** Full MMLU was initially proposed, then superseded by Q10: 20 items per subject, 1140 total.

<details>
<summary>Original question wording</summary>

❓ Q4 — MMLU population: Should each 0-shot and 5-shot scorer cover the full official test set, or a fixed pilot subset? Is 0.570 tied to a specific setting?

</details>

### Q5: Acceptance gates

**Why ask?** Scientific reference scores and implementation correctness need different treatment.

**Operational consequence:** Missing/corrupt records stop reporting. Historical 0.640/288/0.570 gaps do not automatically fail a run.

<details>
<summary>Original question wording</summary>

❓ Q5 — What is the acceptance gate when the original prompt and 150 indices are unavailable? Should 0.640, 288 tokens, and 0.570 be hard pass/fail thresholds, or comparison points with an investigation of differences?

</details>

### Q6: Primary metrics

**Why ask?** Choosing the headline result after seeing results invites selective reporting.

**Operational consequence:** All scorer cells are reported. Q35 finalized 5-shot constrained logits as primary MMLU; old 0.570 remains contextual.

<details>
<summary>Original question wording</summary>

❓ Q6 — Which outputs are primary? GSM8K has strict and flexible extraction; MMLU has generated-text string match and constrained A/B/C/D logits at both 0 and 5 shots.

</details>

### Q7: Old versus new prompts

**Why ask?** Unknown old settings limit comparability.

**Operational consequence:** New literal templates were versioned; recovered historical protocols would be separate variants.

<details>
<summary>Original question wording</summary>

❓ Q7 — Prompt lineage: Should we try to obtain the September scorer/prompt and pilot run configs from the team before freezing templates, or proceed with newly specified templates now?

</details>

### Q8: Seeded GSM8K selection

**Why ask?** A recorded selection makes repetition and comparison possible.

**Operational consequence:** Seed 42, no replacement, sorted source indices, pinned dataset revision. The saved list is more authoritative than the seed alone.

<details>
<summary>Original question wording</summary>

❓ Q8 — GSM8K sample selection: May we use a fixed seed (I recommend 42) to sample 150 distinct indices without replacement from the official test split, then save the sorted indices and dataset revision?

</details>

### Q9: Full MMLU cost

**Why ask?** Four settings over the full test set could consume substantial GPU time.

**Operational consequence:** The full-set proposal was superseded by the subject-balanced pilot subset; four settings were retained.

<details>
<summary>Original question wording</summary>

❓ Q9 — Compute budget for the full MMLU baseline: Are you willing to run all four scorer/shot combinations over the full test set even if generated-text scoring is much slower than constrained logits?

</details>

### Q10: Balanced MMLU subset

**Why ask?** Equal sampling prevents large subjects from dominating this pilot.

**Operational consequence:** 57 subjects × 20 = 1140. Overall accuracy equally weights subjects here; it is not full-test MMLU accuracy.

<details>
<summary>Original question wording</summary>

❓ Q10 — MMLU subset and summary: Is 20 seeded test items per subject (1,140 total) acceptable, with the same frozen items in every cell and both overall and per-subject accuracy reported?

</details>

### Q11: GSM8K prompt style

**Why ask?** Requesting reasoning and a marker can affect both correctness and response length.

**Operational consequence:** Zero-shot step-by-step user instruction with #### requested. The written no-system-message rule was not met: Qwen inserted its default system message.

<details>
<summary>Original question wording</summary>

❓ Q11 — GSM8K prompt: Should the untuned baseline use Qwen’s chat template, one user message asking for worked reasoning and a final `#### number` line, with no custom system message or few-shot examples?

</details>

### Q12: Strict versus flexible extraction

**Why ask?** Mathematical correctness and format compliance can disagree.

**Operational consequence:** One response gets two v1 scores. Flexible v1 was narrower than some interview wording implied; separately approved v2 accepts broader unambiguous final prose.

<details>
<summary>Original question wording</summary>

❓ Q12 — GSM8K extraction: What should count as strict versus flexible? For example, a response with correct arithmetic but no `####` line.

</details>

### Q13: MMLU text parsing

**Why ask?** Picking any letter that matches gold would inflate accuracy.

**Operational consequence:** An unambiguous initial choice can have an explanation, but additional standalone choice letters make the implemented response invalid.

<details>
<summary>Original question wording</summary>

❓ Q13 — MMLU generated-text scoring: Should a response count only when the first answer-bearing token is an unambiguous A/B/C/D choice, with explanations allowed after it?

</details>

### Q14: Logits versus sequence likelihood

**Why ask?** Scoring one token and scoring a multi-token string are different measurements.

**Operational consequence:** Q24/Q29 finalized single-token continuations after an assistant-side Answer: prefix; unsupported tokenization stops validation.

<details>
<summary>Original question wording</summary>

❓ Q14 — MMLU constrained logits: Should we score the next-token logits at a prompt ending in `Answer:` over tokenized choice continuations, or compare full conditional likelihoods of ` A`, ` B`, ` C`, ` D`?

</details>

### Q15: Generation caps

**Why ask?** Caps bound runtime but may interrupt an answer.

**Operational consequence:** GSM8K 1024 generated tokens; MMLU text 32. Capped items remain in the denominator and receive zero credit.

<details>
<summary>Original question wording</summary>

❓ Q15 — Generation limits: What caps should we use for GSM8K and generated-text MMLU, and do capped outputs count as wrong or get excluded?

</details>

### Q16: Vocabulary

**Why ask?** Calling an instruction-tuned model a base model or an unmatched result a reproduction is misleading.

**Operational consequence:** Reference result, untuned baseline and evaluation cohort were recorded in CONTEXT.md. This changes interpretation, not model output.

<details>
<summary>Original question wording</summary>

❓ Q16 — Vocabulary: Can we call the prior team’s numbers “reference results,” our no-adapter measurement an “untuned baseline,” and each frozen item list an “evaluation cohort”?

</details>

### Q17: Five-shot examples

**Why ask?** Example content/order and subject relevance can affect answers.

**Operational consequence:** Use the same subject’s first five official dev examples in published order, inside one user message; no training or test-answer exposure.

<details>
<summary>Original question wording</summary>

❓ Q17 — MMLU prompt construction: For 5-shot, should we use the five official `dev` examples of the same subject in their published order, followed by one test question, in one user message rendered with Qwen’s chat template? At 0-shot, same template with no examples?

</details>

### Q18: Version pinning

**Why ask?** Model weights, datasets and software may change between runs.

**Operational consequence:** Record exact model/tokenizer/dataset revisions and runtime versions; resolve them before freezing inputs.

<details>
<summary>Original question wording</summary>

❓ Q18 — Model and data versions: Should every run pin the exact Hugging Face model commit, tokenizer commit, dataset commits, Python/library versions, and prompt-template revision?

</details>

### Q19: Precision and platform

**Why ask?** Numerical settings can affect outputs and memory requirements.

**Operational consequence:** Unquantized bf16/fp16 and attention/hardware recorded. Your saved run used bf16/eager/Tesla T4. Other numerical settings require distinct variants.

<details>
<summary>Original question wording</summary>

❓ Q19 — Device and precision: Should a run declare dtype, attention implementation, quantization mode, and device, and forbid comparing scores across those settings without labeling them as separate variants?

</details>

### Q20: Counts and intervals

**Why ask?** Rounded accuracy hides denominator, extraction failures and question-sampling variability.

**Operational consequence:** Save item correctness, invalid/cap rates, counts and intervals. Future adapter comparisons should use paired question IDs.

<details>
<summary>Original question wording</summary>

❓ Q20 — Accuracy reporting: Should all requested cells report numerator/denominator, invalid and cap rates, per-item correctness, and a confidence interval, with paired item differences for future adapter comparisons?

</details>

### Q21: Resume contract

**Why ask?** Interruption should not erase progress or silently change the procedure.

**Operational consequence:** Immutable configs and durable item records; resume missing IDs. Final rules do not permit silently dropping failed questions.

<details>
<summary>Original question wording</summary>

❓ Q21 — Resume contract: Should each run have an immutable config, one JSONL record per item, explicit pending/running/completed/failed status, and a final summary derived only from complete item records? Re-running the same config should resume missing items.

</details>

### Q22: Exact numeric equality

**Why ask?** Formatting equivalence should not become tolerance for wrong arithmetic.

**Operational consequence:** Use exact rational comparison: 0.5 equals 1/2; near misses do not receive credit. No arbitrary expression evaluation.

<details>
<summary>Original question wording</summary>

❓ Q22 — Gold-answer comparison: For GSM8K, should strict and flexible extraction share the same exact numeric comparison to the dataset’s final answer, with no tolerance or rounding? A response saying `0.5` for gold `1/2` would be equal only if both parse to the same exact rational value.

</details>

### Q23: Ambiguous fallback

**Why ask?** The last number might be a unit/context quantity rather than the answer.

**Operational consequence:** V1 required a standalone final number. V2 rejects multiple distinct quantities in the final answer region.

<details>
<summary>Original question wording</summary>

❓ Q23 — Flexible GSM8K fallback: If a response contains no explicit final-answer phrase and ends with multiple numbers (for example, `3 boxes at $4 each`), should we choose the last number or mark it ambiguous?

</details>

### Q24: Choice-token validation

**Why ask?** A letter is not guaranteed to correspond to one token in every context.

**Operational consequence:** Validate A/B/C/D continuations as distinct single tokens in the rendered prefix. Do not silently replace logit argmax with sequence likelihood.

<details>
<summary>Original question wording</summary>

❓ Q24 — Constrained MMLU meaning: “Logit argmax” requires one token per A/B/C/D continuation. If the chosen prompt produces multi-token continuations, should the run fail validation and require a revised prompt, rather than silently switch to sequence likelihood?

</details>

### Q25: Context overflow

**Why ask?** Truncating examples or the question changes the test.

**Operational consequence:** Validate complete prompt length; stop and record the item if it does not fit. Do not silently reduce shots.

<details>
<summary>Original question wording</summary>

❓ Q25 — Prompt overflow: If an MMLU 5-shot prompt exceeds the model’s usable context after the fixed examples and question are rendered, should the run fail with the item and token count recorded, rather than silently drop examples or truncate text?

</details>

### Q26: Adapter identity

**Why ask?** Loading the wrong adapter/base pair would invalidate the comparison.

**Operational consequence:** Baseline adapter=null. Optional PEFT adapter identity and base compatibility are checked; stacks/merges are outside the current protocol.

<details>
<summary>Original question wording</summary>

❓ Q26 — Adapter identity: Should an optional PEFT adapter be a single explicitly named path/revision, validated against the expected base model, with baseline runs requiring `adapter=null`?

</details>

### Q27: Uncertainty and pairing

**Why ask?** A different sample can move the score; independent samples can obscure adapter differences.

**Operational consequence:** Wilson intervals for GSM8K, subject-level MMLU counts, and future paired resampling. This refined Q20.

<details>
<summary>Original question wording</summary>

❓ Q27 — Summary and uncertainty: Should GSM8K report a Wilson interval for accuracy and MMLU report both the item-weighted score and subject-level breakdown, with future adapter deltas using paired resampling on identical items?

</details>

### Q28: Package versus notebook logic

**Why ask?** Duplicated scoring logic can drift between environments.

**Operational consequence:** The Python package/CLI owns evaluation. Colab invokes it; CPU tests replace external model/dataset boundaries.

<details>
<summary>Original question wording</summary>

❓ Q28 — Execution surface: Should the evaluation logic live in a parameterized Python package/CLI, with Colab and RunPod notebooks or scripts only invoking it?

</details>

### Q29: Assistant answer prefix

**Why ask?** A user-side Answer: is separated from the scored position by Qwen’s assistant header.

**Operational consequence:** User message ends after options; assistant starts with Answer:. Both MMLU methods use that exact prefix.

<details>
<summary>Original question wording</summary>

❓ Q29 — MMLU answer prefix: May I revise the earlier prompt plan so the user message ends after the four options, and the assistant is prefilled with `Answer:`? The scored next token would then be the choice after that exact assistant prefix.

</details>

### Q30: Literal GSM8K instruction

**Why ask?** Small wording differences can change reasoning and formatting behavior.

**Operational consequence:** Freeze the exact step-by-step/#### template and save full rendered prompts. Actual system insertion remains a documented discrepancy.

<details>
<summary>Original question wording</summary>

❓ Q30 — Exact GSM8K instruction: Should the user prompt be: `Solve the following problem step by step. End your response with a final line in the form #### <number>.\n\nProblem: {question}` (with Qwen’s chat template and no system message)?

</details>

### Q31: Literal MMLU instruction

**Why ask?** Choice layout and instructions affect the answer distribution.

**Operational consequence:** Same Question/A./B./C./D. layout for demos and test question, no extra subject heading, assistant Answer: prefill.

<details>
<summary>Original question wording</summary>

❓ Q31 — Exact MMLU user prompt: Should it say `Choose the correct answer. Reply with only A, B, C, or D.`, then show same-subject 5-shot examples as `Question`, `A.`–`D.`, `Answer: X`, and finally the test question/options, with the assistant prefill from Q29?

</details>

### Q32: Length definition

**Why ask?** Counting prompts, padding or EOS would make token-length comparisons inconsistent.

**Operational consequence:** Count generated tokens before EOS, exclude EOS/padding, include invalid and capped items. Report mean, median and caps.

<details>
<summary>Original question wording</summary>

❓ Q32 — Response length: Should GSM8K mean length count generated tokens before EOS, excluding EOS and padding, include every scored item (including capped and invalid outputs), and report median and cap rate alongside the mean?

</details>

### Q33: Frozen cohorts and overlap

**Why ask?** Comparing different items or testing on training examples confounds results.

**Operational consequence:** Reuse the exact seed-42 IDs across variants; later GSM8K training must be checked for overlap with this held-out cohort.

<details>
<summary>Original question wording</summary>

❓ Q33 — Sampling seed and pairing: Should both benchmark cohorts use seed 42, remain fixed across every future adapter/decoding variant, and require a no-overlap check between GSM8K training items and its evaluation cohort?

</details>

### Q34: Determinism

**Why ask?** Greedy decoding still does not by itself guarantee identical computations across platforms.

**Operational consequence:** Enable deterministic settings and save hardware/software. Unsupported deterministic operations fail visibly; platform changes are recorded.

<details>
<summary>Original question wording</summary>

❓ Q34 — Determinism: The team note calls for `full_determinism=True`. Should baseline runs enable deterministic PyTorch settings, record the hardware/software environment, and fail visibly if a requested deterministic operation cannot run?

</details>

### Q35: Primary MMLU metric

**Why ask?** The closest score to the old number should not be selected after the fact.

**Operational consequence:** Primary metric is frozen-subset 5-shot constrained logits: your result 0.597. All four MMLU cells remain reported.

<details>
<summary>Original question wording</summary>

❓ Q35 — Primary MMLU metric: Since the 0.570 reference does not specify its scorer or shots, should this pilot name 5-shot constrained-logit accuracy on the frozen subset as its primary MMLU metric, while showing all four cells and treating 0.570 as contextual only?

</details>

### Q36: Reuse versus fresh inference

**Why ask?** Different generated responses could confound a scorer comparison.

**Operational consequence:** GSM8K strict/flexible share responses. MMLU text and logits require separate operations on the same prompts, at each shot count: five runs total.

<details>
<summary>Original question wording</summary>

❓ Q36 — Reuse of generations: Should one GSM8K generation be scored both strictly and flexibly, and one MMLU text generation be scored once by string match, while constrained logits are a separate inference on the identical prompt prefix?

</details>

### Q37: Logit response records

**Why ask?** A logit decision needs auditable evidence even without generated text.

**Operational consequence:** Save four logits, candidate IDs, selected/gold letters and score; generated_text=null is expected, not a parsing error.

<details>
<summary>Original question wording</summary>

❓ Q37 — What counts as a saved “response” for constrained logits: Should each item record the four candidate token IDs and logit scores, selected letter, gold letter, and correctness, with `generated_text=null` because no text was generated?

</details>

### Q38: Run identity and paths

**Why ask?** Overwriting results would hide setup changes and failed attempts.

**Operational consequence:** Final canonical hierarchy is experiment/model/dataset/cohort/variant/run-id. Configs stay immutable; changed settings use new runs and explicit report selections.

<details>
<summary>Original question wording</summary>

❓ Q38 — Run identity: Should `artifacts/runs/<run-id>/` contain immutable `config.json`, append-only `responses.jsonl`, `status.json`, and derived `summary.json`, with run IDs tied to the resolved config and attempts recorded separately?

</details>

### Q39: Completion gate

**Why ask?** Partial denominators can produce misleading headline accuracy.

**Operational consequence:** Require exactly one saved record per expected ID and no unresolved execution errors. A record may legitimately contain an invalid answer status.

<details>
<summary>Original question wording</summary>

❓ Q39 — Completion gate: Should a summary be publishable only when the expected item IDs have exactly one valid record each, with no unresolved item errors, duplicate IDs, or prompt/tokenization validation failures?

</details>

### Q40: Tests and real audit

**Why ask?** A script finishing successfully does not establish scoring correctness.

**Operational consequence:** CPU tests cover public scoring/protocol/run behavior; real-item audit checks prompts and outputs. Our five-item audit missed the prevalence of GSM8K formatting rejection.

<details>
<summary>Original question wording</summary>

❓ Q40 — Scorer validation: Before GPU runs, should CPU tests cover hand-worked GSM8K parse cases, ambiguous outputs, MMLU choice-token checks, mocked logits, prompt rendering, resume idempotency, and denominator rules; then manually audit a small fixed set of real item records before the full benchmark?

</details>

### Q41: Sampled decoding scope

**Why ask?** Sampling changes the question and requires an aggregation definition.

**Operational consequence:** Pilot 1 is greedy only. Temperature/top-p and multiple samples need a later protocol specifying average accuracy, voting, or pass-at-k.

<details>
<summary>Original question wording</summary>

❓ Q41 — Scope of sampled decoding: Should pilot 1 implement only greedy evaluation, while the config and evaluator interface leave room for later temperature, top-p, multiple samples, and per-sample seeds?

</details>

### Q42: Numeric grammar

**Why ask?** Accepted number forms are part of the measurement contract.

**Operational consequence:** Signed integers, decimals, grouped commas and fractions; no expressions. Strict/v1 remain narrow. V2 has separately documented prose/units rules.

<details>
<summary>Original question wording</summary>

❓ Q42 — GSM8K numeric grammar: Should strict `####` answers accept signed integers, decimals, comma-grouped integers, and exact fractions, with no units or percent sign on that line? Flexible extraction may accept a number next to a unit or in a final-answer phrase, but neither scorer accepts arithmetic expressions to evaluate.

</details>

### Q43: Contradictory MMLU text

**Why ask?** Choosing first, last or gold-matching letters changes credit.

**Operational consequence:** Contradictory or missing choices are invalid/incorrect and remain in the denominator; no gold-aware letter selection.

<details>
<summary>Original question wording</summary>

❓ Q43 — Invalid MMLU text: If generated text gives multiple choices, revises itself (`C ... actually B`), or contains no recognizable answer, should it be incorrect with a distinct parse status? The first unambiguous choice rule would accept `C` only when it is not contradicted later.

</details>

### Q44: Repeat versus resume

**Why ask?** A second complete execution is evidence about repeatability, not missing work.

**Operational consequence:** Resume incomplete runs; a deliberate complete replicate uses a new attempt ID. Never overwrite the first result.

<details>
<summary>Original question wording</summary>

❓ Q44 — Repeat runs: Should an exact config that is already complete remain immutable, while a deliberate replicate creates a new attempt ID and records its own outputs, even with the same seed?

</details>

### Q45: External harness comparison

**Why ask?** Different prompts/scorers can disagree without either implementation being broken.

**Operational consequence:** Use lm-evaluation-harness diagnostically when semantics/items can be matched. Pilot 1 has no automatic one-percentage-point agreement gate.

<details>
<summary>Original question wording</summary>

❓ Q45 — External cross-check: The project note mentions `lm-evaluation-harness` agreement within one point, but its default prompts may differ from ours. Should we use it only as a diagnostic when the exact same cohort and prompt/scoring semantics can be matched, with no one-point acceptance gate in pilot 1?

</details>

### Q46: Results contents

**Why ask?** A single number cannot reveal extraction failures or procedural differences.

**Operational consequence:** Report raw counts, per-subject counts, lengths, caps/invalids, intervals, source references and protocol limitations.

<details>
<summary>Original question wording</summary>

❓ Q46 — Results format: Should final summaries include both overall and per-subject MMLU counts, strict/flexible GSM8K counts, response-length distribution, invalid/cap rates, the prior reference values with provenance, and an explicit statement that our sampled protocols are not numerically equivalent to those prior runs?

</details>

### Q47: Exact flexible fallback

**Why ask?** The earlier Therefore, 12 example contradicted a number-only rule.

**Operational consequence:** The interview corrected that example. V1 settled on a number-only final-line fallback; v2 is a separate subsequent revision.

<details>
<summary>Original question wording</summary>

❓ Q47 — Flexible GSM8K fallback: When there is no `####` line or explicit final-answer phrase, should the fallback accept only a final nonempty line containing exactly one number (aside from surrounding punctuation), rather than searching backward for the last number anywhere in the reasoning?

</details>

### Q48: Final MMLU layout

**Why ask?** This restated and froze previous prompt decisions.

**Operational consequence:** Same-subject dev demos inside the user message, unanswered test question, assistant Answer: prefix. Actual default system insertion was an implementation mismatch.

<details>
<summary>Original question wording</summary>

❓ Q48 — Literal MMLU v1 layout: Should each user message begin `Choose the correct answer. Reply with only A, B, C, or D.`, followed by zero or five same-subject dev examples in published order, each as `Question: ...`, `A. ...` through `D. ...`, `Answer: X`, then the test question/options without its answer; the assistant prefix is `Answer:`?

</details>

### Q49: Within-subject uncertainty

**Why ask?** Resampling must respect the deliberately balanced subject design.

**Operational consequence:** 2000 bootstrap draws resample items within each subject; intervals cost CPU only. Future paired adapter deltas resample matching IDs together.

<details>
<summary>Original question wording</summary>

❓ Q49 — MMLU uncertainty: Since we deliberately sample 20 questions per subject, should each MMLU cell report a 95% interval from resampling items within each subject, and future adapter differences use the same paired item IDs during resampling?

</details>

## Later questions

**Public test seams:** confirm tests cover scoring, protocol construction and
run/resume behavior. This was a development-workflow question, not an accuracy
parameter. A user should not need to approve this kind of routine architecture
choice without a specific trade-off.

**Exact logit ties:** the original recommendation stopped a run on equal largest
logits. A real bf16 tie occurred. The revised explicit policy records no chosen
letter, invalid status and incorrect credit, retains the item in the denominator,
and continues. Ties are reported separately; no fractional credit is assigned.

## Current comparison context

Your GSM8K strict/flexible-v1/flexible-v2 accuracies are 7.3%/8.7%/42.7% on the
same 150 responses; 46% was a non-blinded diagnostic final-answer review.
Historical GSM8K accuracy was 64% under an unrecovered procedure. Mean length
is 220.02 tokens versus the historical 288.

MMLU text/logit 0-shot is 58.2%/57.8%; text/logit 5-shot is 60.0%/59.7% on the
same balanced sample. The historical 57% has unknown scorer/shot settings.
Do not select whichever metric happens to be closest as proof of reproduction.

## How to prioritize decisions

First decide the scientific question, comparison, evaluation population and
meaning of a correct answer. Then decide prompts, decoding, resource limits and
numerical settings. Finally document routine implementation defaults such as
paths, metadata and test seams. Save all choices, but concentrate discussion on
those that affect the scientific claim or consume substantial compute.
