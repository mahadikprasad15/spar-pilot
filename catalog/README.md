# Artifact catalog

One place that says what exists, what produced it, and where it is. The catalog points at artifacts
wherever they live (local artifact root, Drive, HF datasets, the RunPod sl-0 volume). It never moves
or rewrites them: sealed paths and hashes stay valid.

```text
catalog/
  locations.yaml   storage roots and their kind (local-artifacts, local-path, hf-dataset, runpod-volume, drive)
  arms.yaml        canonical arm names, aliases used by older runs, checkpoint hashes per step, training data
  cohorts.yaml     question sets: split, size, ID digest, source file, every prior use
  responses/       one YAML file per response set (generated or training texts)
```

## Commands

```sh
pip install -e ".[catalog]"                       # PyYAML
python -m pilot_eval.catalog verify               # check every local entry against bytes
python -m pilot_eval.catalog verify --root hf-sampling-sft-local=/path/to/hf-sampling-sft
python -m pilot_eval.catalog find --arm k2         # aliases resolve to canonical arms
python -m pilot_eval.catalog find --c-suitability on-policy --mode sampled
python -m pilot_eval.catalog find --arm online-grpo --include-replays
```

`verify` checks file sha256, total and selected row counts, draws per question, question IDs against
the cohort, per-row `adapter_sha256` against the arm checkpoint, checkpoint hashes and cohort ID
digests. Entries at non-local locations are reported `NOT-LOCAL`, never as passing.
`local-artifacts` defaults to `<catalog>/../artifacts`.

## Response-set entry

```yaml
id: online-grpo.s64.sampled-x8.test150   # <arm>.s<step>.<mode>[-x<draws>].<cohort-short>[.<qualifier>]
kind: evaluation          # evaluation | training-data | training-rollout | replay
arm: online-grpo          # canonical name from arms.yaml
checkpoint_step: 64       # null when not tied to one verified checkpoint
cohort: gsm8k-test-cohort150
cohort_coverage: full     # or subset (only questions in the cohort; some missing)
question_field: item_id   # row field holding the question ID
draws_per_question: 8     # null when the file mixes conditions
policy: {mode: sampled, temperature: 1.0, top_p: 1.0, top_k: 0, repetition_penalty: 1.0,
         max_new_tokens: 1024, eos_token_id: 151645}
scorer: {name: gsm8k-flexible-v3, code_sha256: ..., cap_reward: 0}
selection: {correctness_filtered: false, gold_access: false, all_groups_retained: true, best_of_n: false}
c_suitability: on-policy  # on-policy | off-policy | fixed-text | greedy-only | insufficient
replay_of: null           # id of the set this copies; replays are hidden from `find` by default
generator: ...            # optional: who produced the texts when it is not the arm checkpoint
files:
  - {location: local, path: <relative to the location root>, sha256: ..., rows_total: 3150,
     select: {step: 64, mode: sampled}, rows_selected: 1200}
evidence:
  status: verified        # verified | inferred | unknown
  checked: [what was checked against bytes]
  note: required unless verified — what is missing or assumed
notes: free text (group counts, caveats)
```

`c_suitability` refers to correctness-based attribution (estimating dJ/dλ): `on-policy` = sampled from
that exact checkpoint under the target policy, unselected; `off-policy` = sampled from another policy;
`fixed-text` = usable for fixed-text contrasts only; `greedy-only` = one greedy response per question.

## Convention: a run is not finished until it is catalogued

When a run produces checkpoints, responses or a question set:

1. Add or update the entries here in the same change as the run's records: new checkpoint steps in
   `arms.yaml`, new question sets (with every prior use) in `cohorts.yaml`, one `responses/*.yaml` per
   response set. Use canonical arm names; add any new run-specific name as an alias.
2. Copies, re-scorings and imports of existing draws are `kind: replay` with `replay_of`, never a new
   evaluation.
3. When a set is used for something new (an evaluation, a selection, a Stage D), add that use to the
   cohort's `prior_uses`, so later work knows what is still held out.
4. Run `python -m pilot_eval.catalog verify` and commit only when it reports OK. Record unverified
   facts as `inferred`/`unknown` with a note; do not mark them verified.
5. Files that exist only remotely (Drive, HF, sl-0) still get an entry, with the hash if known.
