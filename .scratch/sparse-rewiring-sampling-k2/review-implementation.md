# Sampling SFT k = 2 sparse rewiring implementation review

Reviewed 2026-11-26 on `sparse-rewiring/sampling-k2` at `c5ca4aa`, including the local K2 files and the committed Gold rewiring code they reuse. See the [K2 spec](spec.md) and [approved decisions](design.md#7-decisions-log-approved--operational).

**Hold the K2 GPU launch.** Fix the production launch gate, input and code identity, and held-out sampling recipe before generating calibration targets or running the study. The post-hoc analysis issues can be fixed independently. Preserve the Gold v1 plans and historical seals.

## Standards

### 1. P1 — Verify the identity and contents of all scientific inputs

The K2 training loader accepts direct JSON/JSONL or an export directory with an absent or incomplete seal. The reference worker can load an adapter other than the pinned K2 checkpoint-64, and the adapter config is missing from the plan identity. The validation loader accepts a raw list and only requires one supervised token in the completion. A CPU probe confirmed that it accepts `input_ids=[11,12,13]`, `labels=[11,12,999]`, and `prompt_tokens=2`, scoring prompt tokens and a substituted label without EOS.

For production, require the verified `l4-training-data-v1` export, reconcile its covered and uncovered IDs and original step slots, and check the dense checkpoint's pinned SHA and config identity. Require the K2 target schema and hashes, verify exact cohort coverage accounting and recipe, and validate that prompt labels are masked and completion labels equal the stored response IDs with one final EOS. Add refusal tests for missing seal entries, malformed labels, wrong adapter weights or config, and changed coverage.

[Training loader](../../pilot_eval/sparse_rewiring_k2_plan.py#L474) · [Validation loader](../../pilot_eval/rewire_k2_launch.py#L71) · [Adapter resolution and hashing](../../pilot_eval/rewire_k2_launch.py#L128) · [Existing training export verifier](../../pilot_eval/sampling_sft_training_data.py#L154)

### 2. P1 — Bind the launch to the executable code

The seal and run CLI use a helper that reads Git HEAD. The current checkout can seal against `c5ca4aa` even though the K2 launcher and plan are untracked and the shared chain and qualification source have local modifications. Further source edits could run or resume under the same stated code identity.

Require clean, committed executable sources at seal and run, or bind and recheck the actual executable source contents. Test refusal after changes to a tracked dependency and when the launcher exists only as an untracked file.

[Seal and run CLI](../../pilot_eval/rewire_k2_launch.py#L534) · [Commit helper](../../pilot_eval/attribution_chain.py#L787) · [Experiment provenance lesson](../../docs/agents/experiment-lessons.md#L25)

## Spec

### 3. P1 — Enforce the conditional two-penalty rule on every production entry point

Default Run A schedules seeds 42 and 43. The separate `check-run-b` command accepts the latest checkpoint with a state file as the final result. A CPU probe with an unsealed step-64 state of zero active gates returned the recommendation to seal and run scaled Run B. The scaled draft accepts any positive `--l0-lambda` without verifying the gate or the calibration.

Run matched seed 42 first. Require a verified completed matched seed-42 arm at step 1024, under its expected plan, before routing. At four or more survivors, allow the matched seed 43. Below four, allow the two scaled seeds using the sealed Stage 1 penalty. Require the gate and calibration to refer to the same experiment inputs and refuse a missing or invalid scaled penalty. Cover incomplete or tampered checkpoints, the three-versus-four boundary, direct CLI invocations, and a scaled penalty that differs from the sealed calibration.

[Default scheduling](../../pilot_eval/rewire_k2_launch.py#L444) · [Survivor gate](../../pilot_eval/rewire_k2_launch.py#L362) · [Penalty validation](../../pilot_eval/sparse_rewiring_k2_plan.py#L194) · [Protocol](spec.md#3-stages-2--3--training-protocol--conditional-two-lambda-rule)

### 4. P1 — Match the qualified K2 sampling recipe

The held-out generator makes three total attempts with retry seeds `0, 10007, 20014`. The training recipe makes one initial attempt plus up to three retries with seeds `0, 1001, 1002, 1003`. The HF generator also inherits `top_k` from the model generation config, while the qualified sampler disables the filter. These differences can change both coverage and the headroom estimate.

Reuse the established retry seed function and bound, and ensure the chosen engine uses the qualified proposal settings, including no top-k restriction. Add a spy-engine case that succeeds only on the fourth attempt and a test of the resolved proposal settings.

[Held-out retry loop](../../scripts/generate_val128_k2_targets.py#L215) · [HF generation settings](../../scripts/generate_val128_k2_targets.py#L101) · [Established retry policy](../../pilot_eval/sampling_sft_generation.py#L37) · [Qualified engine settings](../../docs/operations/sampling-sft-engine.md#L65)

### 5. P2 — Write the full reference bar

The reference worker computes val128-k2 CE and val128 greedy accuracy. It does not evaluate cohort150 or include mean response length in its reference summary. Measure and persist the missing controls for both production references before training. Cover both reference arms and both greedy cohorts in the CPU test.

[Reference worker](../../pilot_eval/rewire_k2_launch.py#L227) · [Stage 1 spec](spec.md#2-stage-1--held-out-val128-k--2-target-generation--reference-bar)

### 6. P2 — Measure activation shifts against an actual base forward pass

The K2 exporter computes the base projection from the rewired model's hidden state, so upstream rewiring is present on both sides of the subtraction. A two-layer CPU fixture gave a reported downstream shift of `-0.3302865` while paired base and rewired forwards yielded `+1.9572531` on identical tokens. This flips the reported active pole.

Run the fully base and rewired model on the same token sequence, using the intended completion positions on both passes. Reuse the existing paired-forward method and require verified final checkpoint identity for the exporter. Add a two-layer regression where upstream rewiring changes the sign.

[K2 exporter](../../scripts/extract_survivor_shifts.py#L130) · [Gold paired-forward measurement](../../pilot_eval/rewire_activation_shift.py#L70) · [K2 checkpoint loader](../../scripts/extract_survivor_shifts.py#L29)

### 7. P2 — Connect K2 exports to the comparative HTML report

The HTML builder uses a hardcoded Gold cache and expects Gold-specific rank fields that K2 survivor exports do not supply. The small-survivor test exercises registry and probe helpers without building an HTML report.

Parameterize the report's inputs and output, use the supplied experiment and arm metadata, and render the requested Gold versus K2 overlap, R-Lens similarity and principal angles, comparison probe, and activation sign. Exercise the real HTML builder with a one-to-three-survivor K2 fixture and keep the existing Gold output under regression coverage.

[Report builder](../../notebooks/build_down_proj_report.py#L15) · [Small-survivor test](../../tests/test_sparse_rewiring_k2_plan.py#L317)

### 8. P2 — Rebuild or refuse stale standalone analysis units

The committed analysis driver's documented sequence builds standalone units first and then calls it with `--gold-adapter`. The second call retains the old sealed unit, so the cosine stays absent. A later activation-shift unit is similarly ignored. Bind the standalone unit to these inputs and rebuild on a change, or require `--redo standalone` with an actionable error. Test the documented command sequence.

[Documented commands](../../pilot_eval/rewire_analysis.py#L13) · [Sealed-unit reuse](../../pilot_eval/rewire_analysis.py#L269) · [Optional standalone inputs](../../pilot_eval/rewire_analysis.py#L284)

## Review verification

- The focused K2 launch, K2 plan, and Gold plan tests passed all 60 cases using the local environment with working PyTorch.
- The broader rewiring run had 142 passes, 59 failures, 6 skips, and 4 setup errors. Missing PEFT, unavailable PyTorch precision APIs, and fixture state leakage after a failed setup affected that run. The full repository suite stopped at collection on missing PEFT and PyArrow.
- The partial-checkpoint route, malformed validation target, and activation-shift sign reversal were reproduced on CPU. No GPU or paid Pod was used.
