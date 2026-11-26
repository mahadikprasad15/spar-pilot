# Sampling SFT k = 2 re-review at `6831f74`

Reviewed `sparse-rewiring/sampling-k2` against [the original handoff](review-implementation.md), [the spec](spec.md), and [the approved design decisions](design.md). The implementation is now committed. **Hold the K2 production GPU launch.** The remaining launch blockers are input provenance, executable identity, and enforcement of the registered training and routing protocol. Post-hoc fixes can proceed independently.

## Disposition of the original review

| Finding | Re-review disposition |
| --- | --- |
| 1. Scientific inputs | Training export, adapter weights and config, target masks, EOS, and cohort accounting are checked. Held-out target provenance and full recipe enforcement remain open. |
| 2. Executable code | The guard detects edits to listed files. Shared dependencies outside that list still evade it. |
| 3. Conditional routing | Default Run A selects seed 42. Partial checkpoints, the three-versus-four boundary, and a wrong explicit scaled penalty are covered. Routing still accepts evidence from a different completed plan and can accept a different calibration. |
| 4. Sampling recipe | The four-attempt policy, retry seeds, and disabled top-k are fixed. The loader still permits drift in other settings and the stage drops evidence required by the spec. |
| 5. Reference controls | Both production references now include cohort150 accuracy and mean length on both cohorts. The local CPU suite used a reference-worker stand-in because PEFT is missing. |
| 6. Activation shifts | Paired forwards fix the false active pole. The completed-arm entry point crashes, and the checkpoint-only fallback accepts intermediate checkpoints as verified. |
| 7. Comparative report | A parameterized HTML comparison and a two-survivor test are in place. Lens coverage and rank-deficient principal angles remain open. |
| 8. Stale Gold analysis | The standalone unit now fingerprints the adapter and activation-shift seal, and rebuilds when either changes. The focused regression passed. |

## Standards

### P1 — Bind the shared code used at runtime

The file allowlist omits `sft_backend.py`, which provides completion loss, along with live dependencies such as `rewire_dynamics.py` and `c_attribution_chain.py`. Both the dirty check and digest are limited to the allowlist. In a temporary Git fixture I changed the completion-loss denominator in `sft_backend.py`. The strict check still returned the same full identity.

Cover the executable tree or enforce a clean source checkout with a tracked-source binding. Add a refusal test for a changed shared dependency, with unrelated local result caches permitted.

[Executable allowlist](../../pilot_eval/sparse_rewiring_k2_plan.py#L58) · [Scoped dirty check](../../pilot_eval/sparse_rewiring_k2_plan.py#L203) · [Training import](../../pilot_eval/rewire_training.py#L37)

### P1 — Verify the provenance and effective recipe of held-out targets

Stage 1 accepts arbitrary model and tokenizer directories and checks only the validation ID digest. The output records recipe settings but has no model, tokenizer, executable, or frozen-input content identity. An artifact from a different proposer model can therefore pass launch validation when the prompt IDs and labels still match. The loader also does not enforce the declared `max_blocks`, `stop_token_ids`, or accept rule. For example, targets from the supported `--block-limit 1` path can carry a different search budget with the same checked metadata.

Before generation, verify the independently pinned model and tokenizer files and the frozen validation input. Bind these identities, the engine/precision and the effective recipe to the output. Validate them again at launch. Add refusal tests for changed proposer identity, search budget, and stop IDs.

[Generation CLI](../../scripts/generate_val128_k2_targets.py#L339) · [Target payload](../../scripts/generate_val128_k2_targets.py#L308) · [Recipe validation](../../pilot_eval/rewire_k2_launch.py#L155) · [Existing model pins](../../pilot_eval/c_attribution_bundle.py#L31)

## Spec

### P1 — Require the actual registered Run A and its Stage 1 calibration for routing

The production routing paths do not supply `expected_plan`. They verify the arm against its own plan and accept the matched regime string, seed 42, expected final step, and shared input digests. A CPU fixture with real checkpoint and arm seals, matching input digests, and a step-1024 matched plan at `0.003` instead of `0.0006` authorized matched seed 43. The same plan-identity gap affects Run B. Run B also rereads the supplied headroom directory without comparing its seal digest to the one recorded in the gate receipt, so a different sealed calibration with the same six input digests can be substituted.

Derive the expected matched seed-42 plan from the sealed production record and verified inputs. Check the plan, code/model provenance, and final binary checkpoint before routing. Use the specific verified Stage 1 headroom seal selected by the receipt, and refuse a different calibration. Test another completed plan, a mismatched headroom seal, and an authentic volume restore.

[Optional plan comparison](../../pilot_eval/rewire_k2_launch.py#L623) · [Receipt recheck](../../pilot_eval/rewire_k2_launch.py#L748) · [Run B authorization](../../pilot_eval/rewire_k2_launch.py#L772) · [Matched seed-43 routing](../../pilot_eval/rewire_k2_launch.py#L831)

### P1 — Seal only the registered production hyperparameters and seeds

Production checks shape, step counts, some schedule values, and the matched penalty. It does not fix the optimizer and hard-concrete constants or the data shuffle and arm seeds. A CPU probe sealed and read back a production draft with mask LR `0.01`, data-order seed `999`, and replication seed `999`. The approved values are mask LR `0.1`, order seed `42`, and replication seed `43`. The production scheduler treats any selected arm other than seed 42 as the replication arm.

Bind all preregistered training constants and the exact production seed set. Exercise refusal through the production sealer and loader. Keep rehearsal overrides in rehearsal mode.

[Production constants](../../pilot_eval/sparse_rewiring_k2_plan.py#L159) · [Arm validation](../../pilot_eval/sparse_rewiring_k2_plan.py#L334) · [Preregistered constants](spec.md)

### P2 — Preserve and resume Stage 1 generation evidence

Generation accumulates attempts in memory and writes only at completion. An interruption loses all progress despite the sampler offering block callbacks and resume. The output drops decoded text, the attempt log, and terminal disposition per uncovered ID. Preserve these records and use the sampler's durable progress hooks to resume under the same manifest. Test an interruption and resume with a fixture and compare the final targets and evidence to an uninterrupted run.

[Sampling loop](../../scripts/generate_val128_k2_targets.py#L243) · [Output payload](../../scripts/generate_val128_k2_targets.py#L320) · [Final-only write](../../scripts/generate_val128_k2_targets.py#L390) · [Sampler callbacks](../../pilot_eval/sampling_sft_sampler.py#L88)

### P2 — Fix and enforce the completed-arm activation export path

The recommended `--arm-dir` path unpacks two values from `rewire_checkpoint.load_into`, which returns the seal dict. I reproduced `ValueError: too many values to unpack (expected 2)` on a real sealed CPU fixture. The fallback accepted its step-0 checkpoint and marked it verified because it only checks the checkpoint's own seal.

Use the single return value. Require completion and final-plan identity for production exports, including the model and validation input identities. Test the actual entry point with a real sealed completed arm and unchanged paired-forward results. Test rejection of an intermediate checkpoint.

[Broken call](../../scripts/extract_survivor_shifts.py#L64) · [Loader return contract](../../pilot_eval/rewire_checkpoint.py#L293) · [Checkpoint-only path](../../scripts/extract_survivor_shifts.py#L87)

### P2 — Reject missing lens matrices and use the actual subspace rank

The decoder silently reuses the raw Logit Lens vector when a fitted lens lacks a layer. Identity is expected for the final layer only. Require all fitted layers used by survivors, run the specified orientation check, and bind comparable caches to the same model and lens revisions. Separately, QR includes arbitrary basis directions for duplicate or linearly dependent rows. On CPU, duplicate copies of two orthogonal vectors yielded two reported angles of `0°` where the true row spaces have a single angle of `90°`. Use a rank-revealing decomposition and test missing nonfinal layers and duplicate rows.

[Decoder fallback](../../notebooks/decode_down_proj.py#L221) · [Principal angles](../../notebooks/build_down_proj_report.py#L45)

## Verification

- The focused K2 launch and K2/Gold plan suite passed 64 tests. `git diff --check c5ca4aa...6831f74` passed.
- The full suite stopped during collection on missing PEFT and PyArrow. The actual pinned training export and adapter config bytes were unavailable in the local unpacked artifacts, and the GPU path was not run.
- CPU probes independently reproduced the executable-identity gap, wrong-plan routing, off-spec production settings passing the seal, completed-arm export crash, intermediate export acceptance, and false principal angles.
