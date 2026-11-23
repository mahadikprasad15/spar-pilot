# Sparse rewiring on GSM8K — implementation review

Reviewed 2026-11-23. **Hold A100 qualification on this snapshot.** Two interfaces fail on the default execution path. The chain can also accept the wrong Pod ID, and its backup inventory can omit evidence needed to analyze or resume the experiment.

The baseline is `7667739`, the C-attribution commit named in the design. The review covers [the design and tickets](https://github.com/mahadikprasad15/spar-pilot/pull/2) at `9ccd41b`, the integrated [Pod chain](https://github.com/mahadikprasad15/spar-pilot/pull/17) at `dbca6a3`, and the separate heads below. The local review merge `44a84f6` brings those heads together without changing their code. The [spec](spec.md), [design](design.md), [decision log](decision-log.md), [launch draft](launch-record-draft.yaml), [ADR](../../docs/adr/0024-sparse-rewiring-a100-training.md), and [experiment lessons](../../docs/agents/experiment-lessons.md) govern the review. The later launch draft records the D1–D4, D7–D10 and D15 decisions. D11 is still proposed.

| Reviewed heads | Tickets |
| --- | --- |
| `dbca6a3` | 02, 03, 05, 06, 07, 08, 09, 14 |
| `3ef8e43` | 04 validation sampler |
| `f3b5fc4`, included by `6afbb3a` | 10 attribution comparison |
| `6afbb3a` | 11 rank comparison |
| `c7d595b`, `baff8c2` | 12 standalone analysis and 13 dynamics figure |
| `8cab36b` | 20 A100 ADR |

## Standards and experiment lessons

### P1 — The runner accepts the launch record's Pod as the current Pod (ticket 14)

`build_rewire_chain` fills the observed `current_pod` with `record['pod_id']`. This bypasses the shared runner's `RUNPOD_POD_ID` lookup. With a different environment ID, the actual check accepted the record's Pod. The stop command would target that Pod. The fixture issued no stop. Pass through `None` so the shared check reads the real environment, and test both mismatching and missing IDs before the launcher takes ownership. The recovery instructions also point to the C-attribution CLI, whose validator rejects sparse records. Exercise the eventual sparse entry point and watchdog with the same guard. [Constructor](../../pilot_eval/rewire_chain.py#L399-L407) · [Pod check](../../pilot_eval/c_attribution_chain.py#L293-L312) · [Stop](../../pilot_eval/c_attribution_chain.py#L736-L745)

### P1 — Backups can claim complete coverage without the evidence needed to resume or report (tickets 06, 08, 14)

The stage expects checkpoint directories and writes an arm seal called `training-complete.json`. The shared copier discovers `complete.json` units. In a clean-directory CPU recovery probe, the copier returned `verified=True` and restored the checkpoint and survivor JSON. It omitted the arm seal, step/validation/settle logs, hardening record and gate snapshots. The resume check can skip an arm after verifying only its checkpoints. The plan, resolved input provenance and diagnostics need a shared-format seal and explicit coverage. Require `verify_arm` under the current plan, runtime and commit when resuming, and assert that a clean restore can build the dynamics and hardening reports. [Training output list](../../pilot_eval/rewire_chain.py#L245-L255) · [Arm seal](../../pilot_eval/rewire_checkpoint.py#L313-L364) · [Copy and coverage](../../pilot_eval/attribution_chain.py#L339-L362) · [Resume check](../../pilot_eval/c_attribution_chain.py#L495-L505)

### P1 — Restarting after a saved checkpoint leaves a broken seal in the backup tree (tickets 08, 14)

Discarding an interrupted attempt removes the tensor payloads but leaves `complete.json` under the recursively scanned run tree. In a CPU probe, the real copier copied the replacement checkpoint and then returned `verified=False` for the archived seal. This can leave a completed retry without verified backup coverage. Archive the prior checkpoint receipts and logs in an evidence unit that the copier can verify independently. Test a crash after the real checkpoint sink has run, then restore to an empty work directory. The current chain test crashes after creating an unsealed dummy file. [Discard logic](../../pilot_eval/rewire_checkpoint.py#L367-L399) · [Existing crash fixture](../../tests/test_rewire_chain.py#L302-L310)

### P1 — Qualification loses the runtime identity and never binds the rows the trainer consumes (tickets 05, 07, 14)

Sparse preflight writes `chain.identity` directly. The shared runner replaces it with the handler's missing `identity` return field and saves `qualified_identity=None`. The fixed-path preflight seal can independently refuse different receipt bytes, including a host-driver change on an otherwise matching replacement Pod. Separately, the trainer consumes arbitrary rows by ID. A CPU fixture changed completion tokens and labels while retaining IDs, and both runs used the same plan hash and produced different CE. Return a verified identity covering the code, runtime, bundle, plans, frozen gold rows/order, validation and cohort inputs. Use attempt-scoped preflight receipts and verify current inputs before any completed stage can be skipped. [Sparse preflight](../../pilot_eval/rewire_chain.py#L143-L177) · [Identity handoff](../../pilot_eval/c_attribution_chain.py#L524-L539) · [Row lookup](../../pilot_eval/rewire_training.py#L212-L217) · [Plan input hashes](../../pilot_eval/sparse_rewiring_plan.py#L374-L392)

The review also rechecked two prior lessons. Sparse preflight inherits IEEE setters and readback, and the shared chain writes closing records before stop. The previously documented failed-stop case remains relevant. Both stop forms can fail after `done` is written, leaving a billable Pod with a closed run. Carry the failed-stop check into the sparse launcher and rehearsal. Full-size export needs a separate capacity measurement. The uploader's workspace check requires roughly twice the uncompressed export size in free space after training, around 69 GB for the checkpoints alone. The draft's 60 GB pre-training minimum does not establish that capacity. [Closing and stop](../../pilot_eval/c_attribution_chain.py#L668-L693) · [Stop result](../../pilot_eval/c_attribution_chain.py#L736-L753) · [Export space check](../../pilot_eval/neuron_audit_export.py#L345-L350)

## Spec and ticket conformance

### P1 — The default gold-reference worker fails after the untuned evaluation (tickets 09, 14)

The spec requires untuned and gold step-64 validation references before training. The child adapts `generation`, then calls the inherited gold adapter loader, which requires `inputs.sources.gold-sft.adapter_role`. The sparse record schema excludes `inputs.sources`. The actual child process with a validated record and saved tiny Qwen2 sealed untuned results and then raised `KeyError: 'sources'`. Add a verified gold adapter source to the sparse record/bundle and exercise the actual worker with a saved tiny PEFT adapter. The current chain test substitutes the reference stage. [Spec §Training dynamics](spec.md#training-dynamics-figure-requested-by-the-researcher-2026-11-23) · [Worker](../../pilot_eval/rewire_evaluation.py#L378-L398) · [Schema](../../pilot_eval/sparse_rewiring_plan.py#L63-L70) · [Test stub](../../tests/test_rewire_chain.py#L268-L297)

### P1 — The canonical frozen plan cannot start the trainer (tickets 05, 07, 14)

The plan builder records the read LR at `optimizer.reads.lr`. The trainer reads `plan['read_lr']` when called by the chain. Passing a real `build_plan` result reproduced `KeyError: 'read_lr'` before step 0. The CPU chain test constructs the plan manually and inserts the top-level field. Have the trainer consume the ticket 05 plan and require the integrated test to build and read it through the real producer. [Plan producer](../../pilot_eval/sparse_rewiring_plan.py#L338-L363) · [Trainer](../../pilot_eval/rewire_training.py#L133-L149) · [Fixture](../../tests/test_rewire_chain.py#L165-L198)

### P1 — Cohort150 is optional in production and reachable in rehearsal (tickets 09, 14)

D12 requires cohort150 at steps 64 and 1,024 in production, and the qualification rehearsal uses validation only. The chain silently skips cohort150 whenever either raw input is absent, with no check for 150 unique pinned IDs. The eval guard allows the final step, and the real-path test explicitly runs cohort150 at a short rehearsal's final step. Require the frozen cohort and both outputs in production, and refuse cohort150 in any real qualification or rehearsal. Use a separate synthetic cohort for the CPU coverage test. [Spec §Validation and cohort150 roles](spec.md#validation-and-cohort150-roles) · [Evaluation branch](../../pilot_eval/rewire_chain.py#L303-L331) · [Rehearsal test](../../tests/test_rewire_chain.py#L347-L364)

### P2 — Approved settings are not consistently enforced by the record and trainer (tickets 05, 07)

D2–D4, D8 and D10 are signed, but the schema validates several values only by type or broad range. A record with D2 still signed accepted a different mask LR. The trainer takes read betas and epsilon for both groups, hardcodes weight decay and hard-concrete constants, and lets a caller disable gradient checkpointing independently of the record. It also schedules validation CE using the gate snapshot cadence, even when a valid rehearsal record gives them different values. Lock approved production values. Derive or verify the settings actually used from the record, and test differing rehearsal cadences. [Production rules](../../pilot_eval/sparse_rewiring_plan.py#L76-L110) · [Validation](../../pilot_eval/sparse_rewiring_plan.py#L208-L228) · [Optimizer wiring](../../pilot_eval/rewire_chain.py#L213-L235) · [Cadence](../../pilot_eval/rewire_training.py#L389-L391)

### P2 — The analysis can substitute incomplete evidence for final results (tickets 10, 12, 13)

The attribution CLI reads only the `survivors` array. It accepts a checkpoint-0 export with `survivors_omitted=true` as an empty set, so an early export can produce a plausible zero-overlap report. Require a sealed final binary export and record its digest, arm and step alongside the source manifest. The standalone analysis supplies fixture-tested helpers but still needs a reader for production artifacts. The figure accepts a contiguous truncated step log and treats evaluation series as optional. Its events panel omits expected-active change and switched-on/off counts. Bind figure inputs to the expected steps and evaluation cadence in the plan, and include these measures in the figure and CSV. [Attribution CLI](../../pilot_eval/rewire_attribution.py#L409-L421) · [Survivor reader](../../pilot_eval/rewire_attribution.py#L217-L221) · [Analysis output](../../pilot_eval/rewire_standalone.py#L437-L445) · [Figure input checks](../../pilot_eval/rewire_dynamics_figure.py#L38-L73) · [Events panel](../../pilot_eval/rewire_dynamics_figure.py#L221-L234)

## Delivery status

- Ticket 01 still owns source artifact identities, the verified gold order and real attribution manifests. Ticket 04 implements a sampler with unreadable-source checks, but its production caller must supply the complete run-record inventory. The proposed normalized-text and duplicate exclusions also need agreement before freezing D11.
- Ticket 15 still owns measured A100 qualification for both ranks. Ticket 16 still owns the approved sealed launch. Ticket 17 still owns production execution and activation-shift collection. Ticket 18 still owns the production report and catalog verification. The A100 ADR is already available in the separate ticket 20 branch.
- The deferral of per-neuron causal checks is consistent with D23. The change retains gold's completion-loss denominator, D14 mask-group replay facilities, checkpoint 960 before hardening, and factors-and-masks-only checkpoint files in the tested paths.

## Verification

- A local merge of the reviewed heads succeeded without conflicts. All **161** sparse-rewiring tests passed on CPU in Python 3.12 with torch 2.9.0, transformers 4.57.6 and PEFT 0.18.0. The original local PyTorch installation was missing its CPU library, so testing used a temporary environment with the pinned versions.
- The adjacent attribution and SFT suite reported **151 passed, 1 skipped, 2 failed**. Both failures came from watchdog tests that expected an `actions.log` file. The watchdog script and its tests are identical to the baseline. The full merged suite was interrupted after 73 passes and remains unverified.
- Independent CPU probes exercised the real plan producer, the child gold-reference path, mismatched Pod identities, clean-directory copy and restore, interruption after a sealed checkpoint, and changed completion tokens under the same pinned plan hash. GPU qualification, live Hub export and actual Pod shutdown were outside these checks.
