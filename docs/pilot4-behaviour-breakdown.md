# Pilot 4 behavioural follow-up

The workflow imports saved SFT and GRPO answers, harmonizes scoring, measures
paired transitions and sampled reliability, supports intended-answer review, and
assembles a source-bound findings report. Every stage runs on CPU; none loads a
language model or regenerates answers.

## Run locally

Use `notebooks/pilot-4-behaviour-breakdown.ipynb` from the repository root, or the
commands below. The notebook contains one incremental workflow through combined findings
(section 13). Set the source selection path once; subsequent sessions can restore the
frozen analysis directly. The local notebook is not yet a published Colab release.
Its Colab setup requires a supplied checkout at its explicit harness commit; it
does not silently clone a moving branch or install model dependencies.

```sh
python -m pilot_eval behaviour-prepare --selection /absolute/path/selection.json \
  --name pilot4-behaviour-breakdown-v1 --output-root artifacts
python -m pilot_eval behaviour-score \
  --config artifacts/plans/pilot4-behaviour-breakdown-v1/behaviour.analysis.json \
  --output-root artifacts
python -m pilot_eval behaviour-verify \
  --config artifacts/plans/pilot4-behaviour-breakdown-v1/behaviour.analysis.json \
  --output-root artifacts
```

## Source selection

Paths are absolute or relative to the selection file. Native SFT evaluation
exports require an explicit checkpoint. GRPO exports select greedy endpoints and
exclude sampled draws. Each config must supply independent expected question IDs
through `items` or `cohort_manifest.item_ids`; an explicit `expected_ids` selection
is also supported and must agree with any configured cohort.

```json
{
  "schema_version": 1,
  "evidence_kind": "real-data",
  "sources": [
    {"method": "sft", "format": "evaluation", "step": 0,
     "responses": "sft-0/responses.jsonl", "config": "sft-0/config.json"},
    {"method": "sft", "format": "evaluation", "step": 64,
     "responses": "sft-64/responses.jsonl", "config": "sft-64/config.json"},
    {"method": "grpo", "format": "grpo-behaviour",
     "responses": "grpo/responses.jsonl", "config": "grpo/config.json"}
  ]
}
```

Each source can supply a `sha256` map for `responses` and `config` from a
historical receipt. Declared digests are checked before import and retained in
the frozen source inventory.

Other supported formats are SFT `paired-trajectory` (JSONL or JSON array) and
GRPO `grpo-paired` (one greedy draw per pair). Repeated SFT baseline references
are checked for equality and counted once per question. Keep provenance receipts
with the imported files. The new workflow hashes the exported bytes; that alone
does not prove historical checkpoint or training integrity.

## Outputs and recovery

The plan is saved under `artifacts/plans/<name>/behaviour.analysis.json`.
The run is under `artifacts/runs/pilot-4/behaviour-breakdown/<name>/`:

- `inputs/`: immutable source bytes and normalized records, retaining raw scores.
- `units/`: separately sealed method/checkpoint scoring units.
- `results/responses.jsonl`: all selected responses, extractions and exact comparisons.
- `results/results.json`: original counts, common IDs, compatibility and scoring summaries.
- `results/report.md`: readable tables, denominators and limitations.
- `meta/status.json`, `logs/run.log`, `complete.json`: progress and completion evidence.

Rerun the scoring command with the same plan to reuse verified units. The optional
`--max-new-units 1` provides a bounded stop for recovery checks. A changed original
file, snapshot, scored unit or implementation is rejected under the same identity.
If original exports are unavailable, the verified source snapshot remains usable.
Use a new analysis name for an intentional change. Earlier reports are not edited.

## Interpretation

Strict and flexible-v3 use existing scoring implementations unchanged. The
last-number diagnostic extracts without gold, then compares exact numeric values.
Unsupported numeric expressions and capped responses remain invalid. This
diagnostic can select an incidental final number: it is neither format-blind
ground truth nor a reasoning-quality judge.

Method-specific baselines remain saved. Shared-reference eligibility requires
matching exported content and declared input/decoding settings. Separate original
and common-ID denominators prevent overlap from silently replacing the headline.
Missing historical decisions and incompatible settings remain explicit.

## Paired transitions and sampled reliability

Notebook section 6 runs Ticket 02. Section 7 runs Ticket 03 independently of
section 6. After a restart, run setup and settings, then the desired section.
These stages need NumPy and the `report` extra (Matplotlib), and no model packages.

```sh
python -m pilot_eval behaviour-transitions \
  --config artifacts/plans/pilot4-behaviour-breakdown-v1/behaviour.analysis.json \
  --name pilot4-behaviour-transitions-v1 --output-root artifacts
python -m pilot_eval behaviour-sampling \
  --config artifacts/plans/pilot4-behaviour-breakdown-v1/behaviour.analysis.json \
  --sampled-source-index 2 --name pilot4-behaviour-sampling-v1 --output-root artifacts
```

The sampled source index is explicit: inspect `sources` in the scoring plan.
Index 2 applies to the example above, not every selection. It must identify a
frozen native GRPO export containing sampled endpoints. The command reads that
verified source snapshot; it does not fetch files or generate missing draws.

Reports live under `artifacts/reports/<name>/`, with configuration, parent hashes,
sealed `units/`, per-question/per-response evidence, `results/report.md`, SVG
figures, `meta/status.json` and a completion receipt. The `bootstrap/` directory
retains the actual NumPy index arrays and sorted item identities. The configured
10,000 resamples use a local seed-42 generator and 2.5/97.5 percentile bounds,
matching the existing interval convention. Within an item, paired endpoints and
all eight observed draws remain together.

Both commands support `--max-new-units 1` for a bounded stop. Repeat the same
command to verify completed outputs or reuse sealed units after interruption.
Changed parent sources, implementations, or output hashes are rejected. Use a
new report name for an intentional protocol change.

Transition reports contain all C/W/U cell IDs for strict, flexible-v3 and the
last-number diagnostic, gross gains/losses, churn, net W/U flows and the direct
paired GRPO64-minus-SFT64 difference. Each original cohort and the common-ID
restriction remain explicit. Changes in input/decoding contracts within a method
block a matched transition report. Gold calculator-annotation counts and
nonempty-line counts describe SFT transitions and response lengths; neither is
a validated difficulty or causal measure.

Sampling reports require exactly eight distinct draws for every held-out question
at both endpoints, matching prompts/gold and decoding contracts. They retain
correct-draw counts, a 9×9 count-transition table, mean draw reward, observed
coverage and its gains/losses. No additional untuned run is selected or pooled by
these commands. Training-cohort draws cannot substitute for held-out evaluation.

The intervals condition on these trained checkpoints and saved responses. They
do not capture training-seed variability or uncertainty from fresh decoding.
Observed 0/8-to-positive changes are coverage gains, not proof of new ability.
Automatic transitions are not validated reasoning gains. The combined findings
remain Ticket 05; the intended-answer audit is described below.

## Intended-answer audit

Ticket 04 adds notebook sections 8–12. Everything runs on CPU. The notebook calls
the same public commands used by the tests:

```sh
python -m pilot_eval behaviour-audit-prepare \
  --transitions artifacts/reports/pilot4-behaviour-transitions-v1 \
  --name pilot4-answer-audit-v1 --output-root artifacts
python -m pilot_eval behaviour-audit-status \
  --config artifacts/plans/pilot4-answer-audit-v1/behaviour.audit.json \
  --output-root artifacts
```

Preparation selects the approved flexible-v3 transitions, invalid endpoints and
numeric/validity disagreements with the last-number diagnostic. Each method gets
20 additional remaining stable pairs, or all if fewer, using a separate Python
`random.Random(42)` on its sorted question IDs. Pools, selections and reasons are
saved. Strict-only formatting failures are not all mandatory.

The status command prints mandatory/control pair counts and the deduplicated
response workload before review. Identical responses are merged only when the
question, gold, content, cap/token metadata and complete declared input/decoding
settings match; retained token IDs must agree where both sources provide them.
Every parent reference is retained privately. Opaque IDs and shuffled order are
created once and recovered from a saved construction checkpoint after interruption.

### Review and save

The run is under `artifacts/runs/pilot-4/behaviour-audit/<name>/`:

- `reviewer/review.html`: self-contained offline editor.
- `reviewer/packet.json`: opaque response IDs, questions and responses only.
- `reviewer/draft.json`: current draft export, rebuilt from sealed label history.
- `private/`: source mapping, selection reasons and construction evidence.
- `drafts/`: immutable, packet-bound incremental label snapshots and receipts.
- `results/`: workload only; this does not establish completed human review.

Share only the reviewer folder for the first stage. The page omits methods,
checkpoints, gold, scorer decisions, pair grouping and reasons. Style can still
suggest a source; this is partial blinding, not access-control security.

Identify the response's intended answer, rather than grading correctness or
reasoning. Choose a numeric transcription, absent, ambiguous or pending. Evidence
and notes are optional; preserve the supporting text when transcribing a written
number. The supported numeric grammar is the existing exact integer/decimal/
comma/fraction grammar; scientific notation and undefined fractions are rejected.

Use **Download draft** to save `answers.json`. Browser caching is temporary.
Import the file through notebook section 9, or:

```sh
python -m pilot_eval behaviour-audit-save \
  --config artifacts/plans/pilot4-answer-audit-v1/behaviour.audit.json \
  --labels /absolute/path/answers.json --output-root artifacts
```

Then import the regenerated `reviewer/draft.json` into the page before continuing.
The export's `base_revision` prevents an older page from overwriting newer labels.
Repeating the exact last import is idempotent. A stale import is rejected; keep
both downloads and reconcile against the current draft explicitly, rather than
discarding unsaved work. Different reviewers, packets, unknown/duplicate IDs and
malformed labels are rejected. The sealed history survives a missing draft cache.

### Freeze, then reveal

After all selected responses are labelled, explicitly enable notebook section 10
to freeze, then run section 11 to reveal. Pending labels block completion.

```sh
python -m pilot_eval behaviour-audit-freeze \
  --config artifacts/plans/pilot4-answer-audit-v1/behaviour.audit.json --output-root artifacts
python -m pilot_eval behaviour-audit-reveal \
  --config artifacts/plans/pilot4-answer-audit-v1/behaviour.audit.json --output-root artifacts
```

`labels/frozen.json` retains the complete pre-reveal labels and draft identity.
The separate `reveal/` report compares those labels with gold and all three
scorers. It distinguishes agreement, missed intended answers, wrong-number
extraction, and absent/ambiguous reviewer labels. It never labels an automatic
answer transition a validated reasoning gain.

The revealed `review.html` supports response/scorer explanation categories:
intermediate number, missed format, conflicting/ambiguous, cap/truncation, other
and uncertain. Download `explanations.json` and save it in notebook section 12:

```sh
python -m pilot_eval behaviour-audit-explain \
  --config artifacts/plans/pilot4-answer-audit-v1/behaviour.audit.json \
  --annotations /absolute/path/explanations.json --revision review-v1 --output-root artifacts
```

Annotations bind the packet, revealed-result hash and reviewer. Each immutable
snapshot gets a new revision name. Partial explanations remain explicitly partial;
they do not change the frozen answer labels. These are selected-case findings,
not exact whole-cohort human-corrected accuracy, reasoning-validity rates or
inter-reviewer agreement.

### Corrections after reveal

Ordinary draft saves stop after freezing. An answer correction requires a named,
reasoned revision and is always marked **post-reveal exposed**:

```sh
python -m pilot_eval behaviour-audit-revise \
  --config artifacts/plans/pilot4-answer-audit-v1/behaviour.audit.json \
  --labels /absolute/path/corrections.json --revision correction-v1 \
  --reason 'Explain the correction' --output-root artifacts
```

The correction file uses the same answer-label envelope, with `base_revision`
equal to the original frozen draft revision. It may contain only changed labels;
each correction revision starts from the original frozen answers. To retain
several corrections together, include all of them explicitly. Original frozen
labels and the original reveal remain unchanged. Corrected reports/pages are
under `revisions/<revision>/`. Explanations for one of those pages require the
matching `--source-revision <revision>`; no latest revision is chosen implicitly.

Software fixture round trips and the real reviewer's outstanding labels are
separate evidence. Preparing a real packet does not complete the human audit.


## Combined findings (Ticket 05)

Run notebook sections **1–2, then 13** to restore existing reports. Sections 3–7
create the automatic source products only when needed. Sections 9–12 are the
optional complete-transcription journal/freeze/reveal route; do not run them for
an incomplete assisted-review backup.

Section 13 accepts explicit completed transition and sampling directories. The
audit config is optional. For the approved assisted route, supply both that
config and the separately generated confirmed-correction report directory.
Selections are saved per analysis; an existing findings plan restores its exact
named parents after restart. No directory-wide latest-run discovery is used.

```sh
python -m pilot_eval behaviour-findings \
  --config artifacts/plans/<analysis>/behaviour.analysis.json \
  --transitions artifacts/reports/<transitions> \
  --sampling artifacts/reports/<sampling> \
  --audit-config artifacts/plans/<audit>/behaviour.audit.json \
  --assisted-report artifacts/reports/<confirmed-corrections> \
  --name <findings> --output-root artifacts
```

Omit the last two source options to generate an automatic-only partial report.
A verified complete frozen/revealed selected audit can instead be supplied with
`--audit-config` alone. Software completion, automatic results, pending review,
assisted targeted review and a completed frozen selected audit are distinct
states. Explanatory annotations remain separate from intended-answer labels.

The command verifies source/config/output receipts and reconciles endpoint totals,
transition cells, direct contrasts, sampled draw counts, coverage IDs and saved
paired bootstrap intervals against the per-item records. Assisted corrections
are recomputed from the bound proposals and explicit confirmations; original
backups and their source chains are checked. Unconfirmed proposals do not enter
the confirmed-case totals. It does not silently rewrite earlier scores or use
original confidence intervals for revised totals.

Outputs under `artifacts/reports/<findings>/`:

- `report.html`: analyst presentation with figures, tables, case links and limits.
- `results/report.md`, `results/results.json`, `results/summary.csv`: readable and
  structured findings. Recorded historical flexible decisions remain distinct
  from harmonized strict/flexible-v3/last-number decisions.
- `results/paired-items.jsonl`, `results/sampled-counts.jsonl`: inspectable source
  rows underlying the report.
- `plots/`: standalone SVG and PNG exports.
- `config.json`, `meta/run_manifest.json`, `meta/status.json`,
  `checkpoints/`, `units/`, `complete.json`: source identities, progress and
  completion receipts. The plan is `plans/<findings>/behaviour.findings.json`.

Three CPU units—synthesis, figures and presentation—seal independently. Use
`--max-new-units 1` to stop after a bounded unit and repeat the same command to
resume. Completed verified outputs are reused. Changed evidence, labels,
implementation or corrupt outputs stop reuse; choose a new report name for an
intentional revision. CPU reporting does not require repeating inference.

For transfer to another machine, retain the named scoring plan and frozen input
bundle, transition/sampling reports (including bootstrap arrays), and any audit,
assistance, backup and confirmation artifacts referenced by the selected
correction report. Keep the canonical artifact subpaths under the new root.
Adapter weights, model files and activation caches are not inputs to this CPU
report. The analyst report intentionally contains answers and gold comparisons;
share the separate `reviewer/review.html` packet for blinded review.

### Evidence from the 2026-10-22 assisted review

All 215 status classifications were recovered: 210 numeric, three ambiguous and
two absent. Six numeric transcriptions were already present; 16 flagged numbers
were subsequently confirmed. The other 188 proposals remain unconfirmed and do
not affect the correction analysis. The nine corrected shared untuned responses
were rechecked explicitly by the reviewer.

Only these confirmed-case changes give GRPO 111→117/150 (+4 pp) and SFT
111→100/150 (−7.33 pp). These are partial correction overlays, not exhaustive
human-scored accuracy. Original sampled reward/coverage analyses remain
unmodified and extraction-dependent. No conclusion about reasoning validity,
newly acquired ability or training-seed reproducibility follows from these counts.
