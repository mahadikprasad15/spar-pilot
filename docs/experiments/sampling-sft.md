# Sampling SFT

Sampling SFT compares gold SFT with training data constructed by a gold-guided
proposer and greedy likelihood search. Generate k=0,2,5 on the original 512
training problems; retain k2 as a data/gate arm and seek separate approval to
train k0/k5, seed 42. This is a data-recipe comparison, with no causal claim
about on-policyness. Gold training is not rerun.

## Current implementation

Ticket01 provides CPU source preparation, audit, verification and restoration.
It does not load a decoder/tokenizer, sample, train, qualify a GPU or reserve
cloud resources. The notebook is `notebooks/sampling-sft.ipynb`; all scientific
work is delegated to the public commands below.

The audit checks gold's pinned cohort and saved masks, exact 512-problem order,
64 eight-problem batches, original L4 training metadata, completion/base-hash
records, step logs and all five rank1/alpha1/FP32 adapter payloads. It also
checks the selected GRPO prepared reference and matching declared model,
tokenizer and base identities. It reads historical sources without modifying
them. The base hash is verified against saved metadata; base weights are not
loaded or rehashed, and historical training is not replayed.

## Source selection

Choose one canonical artifact root containing the historical products. It may
be local or the mounted Drive artifact root. All selected paths must resolve
inside that root; resolving a symlink outside it is rejected.

Create a selection containing explicit `sft` and `grpo` paths. Gold accepts its
original training configuration or a prepared SFT activation configuration with
an explicit original-training reference. GRPO accepts its prepared activation
configuration. For example:

```json
{
  "sft": "plans/pilot2-sft-fp32-l4-batch2-v1/sft.config.json",
  "grpo": "plans/pilot4-grpo-fp32-v1-writes-v1/activation.prepared.json"
}
```

These are selectable historical names, not evidence that the files are present.
An evaluation config/report is not a substitute for the original training
manifest, runtime, order or checkpoint payloads.

## CPU workflow

From the checked-out repository, using an environment with the project's CPU
dependencies installed:

```sh
python -m pilot_eval sampling-sft-prepare --selection artifacts/plans/sampling-sft-v1/source-selection.json --name sampling-sft-v1 --output-root artifacts
python -m pilot_eval sampling-sft-audit --config artifacts/plans/sampling-sft-v1/sampling-sft.plan.json --output-root artifacts
python -m pilot_eval sampling-sft-status --config artifacts/plans/sampling-sft-v1/sampling-sft.plan.json --output-root artifacts
python -m pilot_eval sampling-sft-verify --config artifacts/plans/sampling-sft-v1/sampling-sft.plan.json --output-root artifacts
```

The notebook supplies the same selection and commands with short, readable
cells. On reconnect, run its setup/source-selection cells and the final restore
cell. No generation is relaunched.

## Outputs and status

The immutable source plan is saved under the selected artifact root's plans
role. Its run contains source selection, run/software manifests, structured
status/progress/logs, a source-audit result and a completion seal. The result
retains source hashes and missing-file observations, original configuration,
runtime, masks contract, exact batches and checkpoint inventory.

`source-audited` with `source_ready=true` means the selected source checks
passed. `blocked` with `source_ready=false` means the audit completed but found
missing/contradictory evidence. Audit/verify exit 0 can therefore describe a valid
blocked report; downstream launchers must check readiness, never exit status
alone. Corrupt or changed sealed inputs/products reject verification with exit 1.

All GPU-qualified/executed flags remain false. Runtime hash is null until a real
sampling configuration is qualified. The approved recipe is resolved, while
unqualified engine fields are listed explicitly and cannot inherit defaults.

If sources are added or corrected, preserve the earlier plan and use a new plan
name. Repeating an unchanged completed audit verifies and reuses its seal.
An interrupted audit with no final seal can rerun its cheap CPU checks. A stale
writer marker requires explicit inspected-owner recovery; it is never cleared
automatically. The recovery command requires the saved owner token and review
notes, refuses a live local owner, and retains an ownership-recovery event:

```sh
python -m pilot_eval sampling-sft-recover-writer --config artifacts/plans/sampling-sft-v1/sampling-sft.plan.json --output-root artifacts --owner-token RECORDED_TOKEN --review-notes 'Inspected the stopped writer before recovery'
```

## Next stages

Tickets02–06 implement budget/export controls, bounded engine checks, the exact-resume pilot, full-arm generation and separate eligibility/coverage review. Real collection and human review remain execution steps.
Gate statistics and approved training remain later implementation tickets. Sampling hardware is L40S; training requires verified
L4/FP32 gold metadata. GPU-version/scheduling/default settings and actual stop
semantics are resolved before engine qualification, not fabricated by this CPU
audit.

The first paid job requires explicit post-preflight approval. The total budget
is 25 USD with sampling 10 / training 6 / storage and shared overhead 9 sub-caps.
Evaluation, internals, Stage 0 research, the 512 × 8 baseline/offset and extra arms
remain deferred.

The confirmed design, specification and ticket graph are retained in the local
Sampling SFT issue-tracker effort. CPU fixture completion does not establish
real Drive source readiness or L40S/L4 qualification.

## Budget and cloud lifecycle

Ticket02 adds the CLI-first budget/export/stop controls and the
[RunPod handoff guide](../operations/sampling-sft-runpod.md).
The optional notebook can inspect the ledger and run a disabled-by-default
zero-cost smoke. No engine, live cloud qualification or paid approval is implied.

## Bounded engine and pilot workflows

[Engine checks](../operations/sampling-sft-engine.md) and the
[fixed32 pilot](../operations/sampling-sft-pilot.md) use public saved-workflow
commands behind the reserved-job entrypoint. The notebook exposes optional CPU
plan cells; CLI/RunPod remains primary. CPU fixture and subprocess-kill evidence
is not a real engine qualification or paid-job approval.

## Full generation

[Generation runbook](../operations/sampling-sft-generation.md): real qualification
and exact-plan review precede budgeted k0/k2/k5 jobs. Sealed blocks, model-outcome
retries and automatic512-row dispositions share one frozen runtime. Completed
collection remains separate from human coverage review and training approval.

## Eligibility and coverage

[Review workflow](../operations/sampling-sft-review.md): source-bound intended-answer
labels have explicit human provenance and saved revisions, separate from agent
suggestions and unchanged automatic scores. Reviewed coverage preserves original
training slots and reports omissions/common cohorts without authorizing training.
