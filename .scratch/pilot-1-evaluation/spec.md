# Pilot 1 evaluation harness

## Purpose

Establish a reproducible untuned baseline for `Qwen/Qwen2.5-1.5B-Instruct` on frozen GSM8K and MMLU evaluation cohorts, then reuse the same instrument for a single optional PEFT adapter. This is a new, versioned protocol. Earlier SPAR observations of GSM8K accuracy 0.640, mean response length 288 tokens, and MMLU accuracy 0.570 are reference results with unrecovered prompts, items, and scorers, not acceptance gates. Report differences with provenance and possible explanations; do not claim an identical reproduction.

## Scope

- Python package and command-line entry point using Hugging Face `transformers` and `datasets`; a notebook or cloud launcher calls that same entry point.
- Greedy evaluation in pilot 1. Reserve configuration fields for later sampled decoding; do not assign a sampling aggregation rule yet.
- Load a pinned model and tokenizer revision, with `adapter=null` for the untuned baseline or one explicitly named PEFT adapter path/revision. Validate adapter/base compatibility. Adapter stacks and merged adapters are separate future variants.
- Run on one GPU in Colab or RunPod. Record dtype, attention implementation, quantization, device, library versions, hardware, and deterministic settings. Prefer unquantized bf16 when supported, otherwise unquantized fp16; distinct precision or quantization settings are distinct variants.
- CPU unit tests use no model download and replace the external model boundary with a fake.

## Frozen evaluation cohorts

- Resolve and pin dataset revisions before selection. Select 150 distinct indices without replacement from the official GSM8K `main` test split using seed 42.
- Select 20 distinct test indices without replacement within each of MMLU's 57 subjects using seed 42, for 1,140 items. Keep subject and source index for every item. This is a subject-balanced pilot subset, not full-test MMLU.
- Persist the ordered item IDs, selection method, seed, source split and revision, and dataset hashes under `artifacts/cohorts/`. Reuse the exact IDs across scorer, shot, adapter, and decoding variants. Confirm GSM8K evaluation IDs do not overlap any later GSM8K training IDs.
- Use the five MMLU `dev` examples of the same subject in published order for 5-shot; use none for 0-shot. Never include a test answer in its prompt.

## Prompts and decoding

Version every literal prompt template. Render Qwen's chat template and save the complete rendered prompt per item. Use no system message.

GSM8K v1 user text, with `{question}` replaced by the dataset question:

```text
Solve the following problem step by step. End your response with a final line in the form #### <number>.

Problem: {question}
```

MMLU v1 user text starts with `Choose the correct answer. Reply with only A, B, C, or D.` It then shows zero or five examples, each as `Question: ...`, `A. ...` through `D. ...`, `Answer: X`, followed by the unanswered test question and four options in the same format. The assistant is prefilled with `Answer:`; both MMLU scorers use the same prefix. No extra subject heading is added.

- Greedy generation: `do_sample=false`, one sequence, GSM8K `max_new_tokens=1024`, MMLU generated text `max_new_tokens=32`. Save resolved generation settings, including EOS and pad token IDs and stop reason.
- A capped output remains in the accuracy denominator and counts as incorrect even if it contains an answer. Track the cap rate.
- GSM8K response length counts generated tokens before EOS, excluding EOS and padding; include all items, including invalid and capped outputs. Report mean, median, and cap rate.
- Validate that every 5-shot prompt fits the usable context. Record item and token count and fail rather than truncating examples or the question.

## GSM8K scoring

- Extract the gold final answer from the dataset's `####` marker. Use exact numeric equality after normalizing signed integers, decimals, comma-grouped integers, and rational fractions; no tolerance or rounding. Never evaluate arbitrary arithmetic expressions.
- **Strict:** accept exactly one parseable final line in the form `#### <number>`; the line may contain no units or percent sign. Missing, multiple, malformed, or non-final markers are incorrect.
- **Flexible:** first use a valid strict marker; otherwise recognize an explicit final-answer phrase such as `Final answer: <number>` or `The answer is <number>`; otherwise accept only a last nonempty line containing one number and optional surrounding punctuation. Do not search backward through reasoning for the last number. Contradictory or ambiguous answers are incorrect. A percent sign requires an explicit future rule and is currently rejected.
- Score each generated response both strictly and flexibly. Save raw response, extracted value, extraction status, gold value, and correctness for each scorer. Invalid outputs remain in the denominator.

## MMLU scoring

- Run both scorers at 0-shot and 5-shot on the same 1,140 items. Report all four cells separately; designate 5-shot constrained-logit accuracy as the primary pilot MMLU metric. The historical 0.570 is contextual only because its scorer and shot setting are unknown.
- **Generated-text string match:** greedily generate up to 32 tokens from the assistant-side `Answer:` prefix. Accept an unambiguous A/B/C/D answer; multiple choices, revisions such as `C ... actually B`, missing choices, and caps are invalid and incorrect. Save raw text, parsed choice, parse status, gold choice, and correctness.
- **Constrained logits:** compare next-token logits for candidate continuations ` A`, ` B`, ` C`, ` D` at the same assistant-side `Answer:` prefix. Before evaluation, verify that each candidate is exactly one token in context; fail validation if not, rather than switching to sequence likelihood. Save candidate token IDs, four scores, selected letter, gold letter, and correctness; set `generated_text=null` because this scorer does not generate text.
- A subject-balanced item-weighted accuracy and per-subject numerator/denominator are required. Report a 95% uncertainty interval by resampling items within each subject. For future adapter differences, resample paired item IDs. Report invalid and capped output rates.

## Run artifacts and resume

- Keep all outputs under the repo's `artifacts/` root. Cohort manifests live under `artifacts/cohorts/`. Runs use the canonical hierarchy `artifacts/runs/<experiment>/<model>/<dataset>/<cohort>/<variant>/<run-id>/` with `inputs/`, `checkpoints/`, `results/`, `logs/`, and `meta/` directories.
- Each run has immutable `config.json` and `meta/run_manifest.json` recording model, tokenizer, optional adapter, prompt template/version, exact prompt indices, dataset and model revisions, decoding settings, scorer, seed, software/hardware, output paths, and start time. Save a copy of the cohort manifest or a content-addressed reference in `inputs/`.
- Save one JSONL record per expected item in `results/responses.jsonl`, including the rendered prompt, raw generation or four-choice scores, gold answer, parsed answer, correctness, token counts, stop reason, and item provenance. Save recoverable errors separately.
- Maintain `meta/status.json` and `checkpoints/progress.json` during a run; log progress with completed/total items and batch size. Batching and checkpoint interval are configurable. On restart, read durable item records and status, reject a config mismatch, and process only missing IDs. A completed run is immutable; a deliberate replicate uses a distinct attempt/run ID.
- Produce `results/results.json` only when every expected item has exactly one valid record and there are no unresolved item errors, duplicate IDs, or prompt/tokenization failures. The aggregate is derived from saved item records. A failed or partial run remains visibly incomplete.

## Reporting and correctness gates

- GSM8K: strict/flexible correct and total, 95% Wilson intervals, mean/median generated-token length, invalid and capped rates.
- MMLU: correct and total for all four cells, 95% subject-stratified sampling intervals, per-subject counts, invalid and capped rates. State that this is a sampled protocol and cannot be compared numerically as if it were full MMLU or the earlier SPAR run.
- Before a GPU run, CPU tests must check independently worked GSM8K parsing cases, contradictory MMLU text, choice-token validation and mocked logits, prompt rendering, cohort selection, resume idempotency, duplicate/error handling, and denominators. A small fixed real-item audit checks rendered prompts and per-item outputs before the full cohort.
- Correctness and completeness checks are hard gates. Historical accuracies are not. An external `lm-evaluation-harness` comparison is diagnostic only when cohort, prompt, and scorer semantics are matched; pilot 1 has no one-point agreement gate.

## Confirmed public test seams

1. **Scoring API:** score saved GSM8K and MMLU item responses without a live model; expose extraction status and correctness.
2. **Protocol API:** build frozen cohorts and fully rendered prompts from pinned dataset records and a tokenizer-like chat-template boundary.
3. **Run API/CLI:** execute or resume a configured run against an injected model/dataset boundary and inspect its public status and result artifacts.

Tests will exercise these public behaviors and use a fake only at external model/dataset boundaries.
