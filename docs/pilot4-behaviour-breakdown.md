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

Tickets 02–05 add transitions, sampled reliability, the blinded human audit and
the combined findings. Those analyses are not implied by Ticket 01 completion.
