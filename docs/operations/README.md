# Execution guides

Use [experiment records](../experiments/README.md) for questions/configurations and
[the status ledger](../experiments/status.md) for completion evidence. These guides
describe stage commands, review fields and recovery. Notebook section numbers can
change between releases; match the stage name and pinned notebook revision.

| Workflow | Guides |
|---|---|
| Evaluation and scoring | [Recovery](../colab-recovery.md), [logit ties](../colab-logit-ties.md), [progress](../colab-progress-recovery.md), [flexible-v2](../colab-flexible-v2.md), [flexible-v3](../colab-flexible-v3.md) |
| SFT | [Pilot 2](../pilot2-sft.md) |
| SFT writes | [Pilot 3](../pilot3-instrument.md) |
| GRPO | [Training](../pilot4-grpo.md), [fixed writes](../pilot4-fixed-writes.md), [coefficients/KL](../pilot4-token-measurements.md), [final handoff](../pilot4-final-handoff.md), [resume](pilot4-resume.md) |
| Readable traces setup | [Bootstrap](../pilot5-bootstrap.md), [corpus](../pilot5-corpus.md) |
| Instrument qualification | [Capture diagnostic](../pilot5-capture-diagnostic.md), [lens/profile](../pilot5-lens-and-profile.md), [JLens](../pilot5-jlens-qualification.md), [capture qualification](../pilot5-capture-qualification.md) |
| Capture and analysis | [Capture/readouts](../pilot5-capture-and-readouts.md), [prereveal review](../pilot5-prereveal-review.md), [automatic analysis](../pilot5-automatic-analysis.md) |

[Workspace layout](../workspace-layout.md) documents persistent versus temporary
storage. [Artifact access](../artifacts.md) describes the evidence needed to
reproduce a result. Completed stages should be restored/verified according to
their workflow instead of rerunning the notebook from top to bottom.
