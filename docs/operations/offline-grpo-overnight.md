# Offline GRPO: launching an overnight run

Run everything on one GPU session, in `tmux`, with the repository on the Pod's own
disk and `--output-root` on the volume. The network volume does not allow `chmod`
or setting timestamps, so `git clone` must not target it and `rclone` needs
`--local-no-set-modtime`. `A` below is the artifact root; `NAME` is the run name.

Qualification records the git commit, so any later commit means requalifying.
Stay on one commit from step 3 onward.

## 1. Source inputs, before renting a GPU

```bash
python3 scripts/offline_grpo_source_check.py --root $A
```

All seven checks must say `OK`. A `MISSING` line or failed check names what to
restore. Earlier inputs live in Drive `artifacts/{plans,runs}`, and in the private
Hugging Face artifacts repo. Compare before overwriting anything that differs.

## 2. Software, model cache, parity

```bash
python3 -m pip install -r requirements-pilot2.txt pyarrow
hf download Qwen/Qwen2.5-1.5B-Instruct --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
HF_HUB_OFFLINE=1 python3 -m pytest -q -rs tests/test_offline_grpo_engine.py::test_pinned_trl_online_and_offline_update_parity
```

Parity must say `1 passed`; a skip does not count.

## 3. Stage 0 qualification

The spectral cache is generated, never copied or backed up. The overnight stage
reuses this one because the name matches `NAME-spectral`.

```bash
python3 -m pilot_eval stage0-spectral-prepare --output-root $A --audit plans/stage0-geometry-v1/stage0.plan.json --weights-dir inputs/models/<revision> --name NAME-spectral
python3 -m pilot_eval stage0-spectral-run --output-root $A --config plans/NAME-spectral/stage0.spectral.json
python3 -m pilot_eval stage0-edits-prepare --output-root $A --spectral plans/NAME-spectral/stage0.spectral.json --name NAME-edits
python3 -m pilot_eval stage0-edits-run --output-root $A --config plans/NAME-edits/stage0.edits.json
python3 -m pilot_eval stage0-qualification-prepare --output-root $A --edits plans/NAME-edits/stage0.edits.json --cohort reports/pilot4-results-so-far-2026-10-13/inputs/evaluation-config.json --name NAME-qualification
python3 -m pilot_eval stage0-qualification-runtime --output-root $A --config plans/NAME-qualification/stage0.qualification.json
```

Qualification reserves from the Stage 0 ledger, which needs a current usage reading
(`stage0-budget-record`) under its 20-unit ceiling. Write an approval with `reviewer`,
`notes`, `session_id`, `max_units_per_hour`, `max_seconds` and `source_reviewed`, then:

```bash
python3 -m pilot_eval stage0-qualification-run --output-root $A --config plans/NAME-qualification/stage0.qualification.json --approval <approval.json>
```

Read `results/diagnostic.json`. If you accept it, write a review with `reviewer`,
`notes`, `diagnostic_sha256` (of `complete.json`) and `absolute_gate` (the exact
`proposed_absolute_gate`), then run `stage0-qualification-approve --review <review.json>`.

## 4. Fidelity tolerance

```bash
python3 scripts/offline_grpo_measure_fidelity.py --root $A --frozen plans/pilot4-grpo-fp32-v1-frozen-v2/grpo.frozen.json --output $A/notebook/NAME/fidelity-measure.json
```

Choose the limit from this, above rounding and far below a real fault. On the A100,
the largest measured difference was 6.7e-5 nats, and `0.001` was chosen.

## 5. Plan and preflight

```bash
export HF_HUB_ENABLE_HF_TRANSFER=0
python3 scripts/offline_grpo_prepare_inputs.py --root $A --name NAME --fidelity-tolerance <nats> --hf-repo <owner/private-repo>
python3 -m pilot_eval offline-grpo-overnight-template --output-root $A --inputs notebook/NAME/inputs.json --name NAME
python3 -m pilot_eval offline-grpo-overnight-prepare --output-root $A --spec plans/NAME/offline.overnight.input.json
python3 -m pilot_eval offline-grpo-overnight-preflight --output-root $A --config plans/NAME/offline.overnight.json
```

## 6. Approve and launch

Write an approval with `protocol: offline-grpo-prelaunch-v1`, `approved: true`, the
`reviewer`, the plan and preflight hashes, both runtimes and `repo`. `max_seconds` is
optional. Then:

```bash
python3 -m pilot_eval offline-grpo-overnight-run --output-root $A --config plans/NAME/offline.overnight.json --authorization <approval.json>
python3 -m pilot_eval offline-grpo-overnight-status --output-root $A --config plans/NAME/offline.overnight.json
python3 -m pilot_eval offline-grpo-overnight-verify --output-root $A --config plans/NAME/offline.overnight.json
```

After losing local files, extract the backup at the same root, then run
`offline-grpo-overnight-restore --manifest <manifest.json>`. Resuming needs the same
session. A new session needs a new plan.
