# Paste into Claude with repo-folder and Drive access

You are helping assemble a source-backed research dossier for SPAR Team2's selective-fine-tuning pilots1–5. Produce a research narrative for colleagues, not a dump of execution logs or a promotional summary.

## Access and first reads

Local repository: `/Users/prasadmahadik/spar/pilot1`.
GitHub: `https://github.com/mahadikprasad15/spar-pilot`.
Google Drive logical root: `My Drive/SPAR/spar-pilot/artifacts`.
Colab spelling: `/content/drive/MyDrive/SPAR/spar-pilot/artifacts` (not automatically accessible on this computer).

Read these first:
1. `README.md` and `docs/experiments/README.md` — experiment map and navigation.
2. `docs/experiments/pilot-{1..5}.md` and `docs/experiments/status.md` — current questions, configurations and evidence status.
3. `docs/methods.md` and `docs/artifacts.md` — shared conventions and access requirements.
4. `docs/research/source-inventory.json` — local source hashes and starter report inventory.
5. `CONTEXT.md`, the corresponding `.scratch/<pilot>/spec.md`, relevant complete ADR filenames and actual frozen runtime/config/review artifacts.

`docs/research/pilot-program-handoff.md` supplies additional research context.
The methods primer and `output/learning/` are optional explanatory material, not
the report structure. Use ordinary technical research writing; define project-specific
terms without teaching general engineering concepts or adding a colleague FAQ.

The Git clone excludes `artifacts/`. Confirm Drive/local access before promising a full synthesis. If a source is inaccessible, list the exact missing artifact and access needed; continue using available evidence without inventing it. Do not download all large binary shards initially: inventory small configs, manifests, status/seals and summaries, then fetch only raw evidence needed for a calculation or audit. Do not interrupt any active Pilot5 GPU process.

## Task and ordering

### Phase A — establish what each experiment means

Build one matrix for pilots1–5:
- RQ, hypothesis/prediction and whether exploratory/preregistered/post hoc;
- model/tokenizer/dataset revisions; exact train/test/measurement cohorts and possible overlaps;
- all training choices (rank/alpha/dropout/targets, base/adapter precision, quantization, optimizer, losses and normalization, LR/schedule, microbatches/accumulation, steps, seeds, rollouts/cap);
- prompt rendering, masks/content positions and generation/evaluation scorers;
- measurement/retention/reduction precision, numerical flags and hardware;
- controls, estimands, weighting, statistics and acceptance gates;
- actual run status, source artifact paths/hashes, reasons/dated decisions and known deviations.

Separate intended settings from actual executed settings and from explanatory rationale inferred after the fact. Per-run configs outrank notebook defaults. An issue marked resolved can mean implemented software only. A completion seal still needs verification under the corresponding workflow; a successful child exit, mere status flag or HTML caption alone is insufficient evidence.

### Phase B — build a current evidence inventory

Write `source-inventory.csv` (or equivalent structured table) with pilot, scientific product, arm/checkpoint/cohort/variant, source path, SHA256, completion evidence, scorer/runtime/weighting, verification scope and whether raw inputs are accessible. Preserve history and select the current scientific run explicitly. Distinguish synthetic diagnostics/fixtures from actual model results.

Start with preserved local report inputs at `artifacts/reports/pilot4-results-so-far-2026-10-13/inputs/`, the A10040 review and existing SFT audits. The13Oct report is a snapshot; it does not establish latest completion. Cell27 was reported completed later. Pilot4 v2 KL/coefficients need a fresh status lookup. The supplied Pilot5 capture log records durable completion and the user reported cell24 complete; no completed learned-readout/automatic-analysis result has been supplied. Inspect current artifacts before extending those claims.

### Phase C — analyze and write the story

Organize around questions:
1. Which apparent behavior changes survive correct scoring and matched baseline checks?
2. How do behavior and response length develop under SFT and GRPO?
3. How do local adapter contributions differ from propagated block changes on math question/solution versus unrelated text?
4. What does the norm-matched random intervention establish, and what remains unknown?
5. Once available, how do fixed-context KL and coefficient summaries relate to behavior/write geometry?
6. Once available, are Pilot5 mean differences task-readable under the frozen reference/null procedure, and do the two arms differ directly?

Do not conflate format compliance, correctness, response length, internal magnitude, task readability and causality. Preserve the post-hoc flexible-v3 correction and audit evidence boundary. Matching problems/rank/steps is not an objective-only causal comparison. One seed is not cross-seed robustness; bootstrap intervals condition on their sampled population. KL is conditional and direction-specific. Raw coefficient magnitude is factorization-dependent. One random adapter is not a null distribution. Missing qualification is unavailable, not a negative finding. JLens qualification is scoped; no automatic extension to all layers/positions. Use primary results and direct contrasts separately from secondary searches; keep the four-test/two-test Holm families and C3 calibration caveat intact.

Do not redo training, generation, calibration or GPU measurements. Do not relax numerical limits. Read-only recomputation of derived CPU statistics/figures from already available data is allowed. If a proposed new analysis is outside the frozen procedure, mark it exploratory and propose it separately before treating it as confirmatory. Do not call external graders, incur paid compute, publish/share files externally, or message collaborators without authorization.

## Deliverables

Save a separately named research-report directory with:
- `report.docx` (or a native Google Doc if explicitly available/authorized), plus PDF and readable Markdown source.
- An abstract, pilot map, per-pilot RQs/methods/results, integrated discussion, limitations and next experiments. Begin with a clear result story, then progressively explain methods.
- Methods/config tables with provenance, data dictionary, completed/pending ledger and verification scope.
- `figures/`: editable/vector SVG or PDF plus PNG previews and the exact plotting code/data. Every caption states population, N, scorer/checkpoint, weighting/denominator, uncertainty method and source.
- Machine-readable source inventory and figure-to-data manifest; exact metrics tables as CSV/JSON. Record code/environment used for any recomputation.
- An explicit list of unavailable evidence, unresolved discrepancies and analyses not yet executable.

Suggested figures, only where verified data supports them:
1. Program dependency/RQ diagram.
2. Paired SFT/GRPO checkpoint accuracy and response-length trajectories, with matched baselines and scorer labels.
3. Strict versus flexible-v3 scoring audit to show formatting/scorer effects, distinctly post hoc.
4. Separate depth panels for GSM8K question, solution and FineWeb block writes; named numerator/denominator and weighting.
5. Direct projection contributions versus propagated block changes, with separate axes/definitions.
6. SFT/GRPO/random signed direction comparisons with uncertainty/coverage, if actual direction evidence is available.
7. KL trajectories and coefficient distributions with fixed-context/mask/factor conventions, if supplemental execution is complete.
8. Pilot5 task-reference overlap versus lens-specific nulls, direct paired arm contrasts, legacy/new robustness and N-curves, if completed and verified.

Use a consistent SFT/GRPO/random palette, direct labels, readable units and no smoothing that invents between-checkpoint values. Do not combine incomparable ratios on a single axis or manufacture error bars from aggregate means. Prefer6 useful figures over20 repetitive ones. Keep dense provenance in an appendix, with source links near each result.

Finish by stating what the evidence supports, what it does not establish and which minimum next measurements would resolve the main uncertainty. Keep a polished scientific voice without hiding failures or overstating mechanism/novelty. Before final delivery, check every reported number against its source, every panel against its plotting data and every completion claim against actual artifacts. Report verification limits honestly.
