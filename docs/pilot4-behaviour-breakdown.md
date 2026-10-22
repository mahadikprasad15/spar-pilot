# Pilot 4 behavioural breakdown: saved-output scoring

Ticket 01 imports existing greedy SFT and GRPO responses at steps 0 and 64,
preserves their recorded decisions, and applies a common set of scorers. It runs
on CPU. It neither loads a language model nor regenerates answers.

## Run locally

Use `notebooks/pilot-4-behaviour-breakdown.ipynb` from the repository root, or the
commands below. The notebook contains setup, settings, import, scoring and report
sections. Set the source selection path once; subsequent sessions can restore the
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
Automatic transitions are not validated reasoning gains. Human audit and the
combined findings remain Tickets 04–05.
