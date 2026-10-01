# 05: Deliver the guided Colab workflow and integration review

Type: task
Status: ready-for-agent
Blocked by: 04 - Produce paired comparisons and checkpoint trajectory

**What to build:** A explained Pilot 2 Colab notebook connecting frozen
preparation, exploratory plan, GPU preflight, baseline, training, checkpoint
evaluation, resume and reporting through the public repository operations.

- [ ] Explain the research goal, new-protocol status, each stage's purpose,
  fixed settings, expected artifacts and interpretation limits in notebook cells.
- [ ] Use the pinned environment and repository version, detect the actual T4
  and FP32 settings and preserve outputs in a configurable persistent Drive tree.
- [ ] Guide input inspection and exploratory-plan recording before training,
  then memory/stability preflight and complete matched evaluation/reporting.
- [ ] Restore state from saved manifests after reconnecting; show how to resume
  training/evaluations without regenerating completed compatible work.
- [ ] Surface failure logs and explain that changing settings requires an explicit
  recorded variant; do not add silent automatic recovery changes.
- [ ] Preserve the Pilot 1 notebook and historical experiment artifacts.
- [ ] Verify notebook structure and command/config wiring without GPU downloads;
  run the full CPU suite and review implementation against the spec and repo rules.
- [ ] Report exactly which checks ran and clearly distinguish local verification
  from real Colab execution; do not claim GPU success before it is observed.

## Comments

Approved as ticket 5. Completes the implementable workflow; conducting the actual
remote GPU experiment is separate from local implementation verification.

