# Workspace layout

```text
spar-pilot/                   Git repository
  notebooks/pilot-{1..5}-colab.ipynb
  pilot_eval/                 shared measurement and training package
  tests/
  docs/experiments/           experiment records and status
  docs/operations/            execution guide index
  docs/adr/                   decisions
  .scratch/                   specifications and tickets

My Drive/SPAR/spar-pilot/     persistent workspace
  artifacts/
    cohorts/
    plans/
    runs/
    reports/
    notebook/
```

Colab mounts the persistent artifact root at
`/content/drive/MyDrive/SPAR/spar-pilot/artifacts`. Temporary harness checkouts,
model caches and staging directories are separate. Their locations are defined
by each notebook release; do not assume a shared checkout across all pilots.
Pilot 5 uses revision-specific harness directories. Restarting a runtime loses
temporary files and Python variables, not verified artifacts already saved to Drive.

The notebooks pin harness commits. Reopening a notebook from GitHub can update
the launcher while an existing frozen execution still requires its original
compatible code. Follow the named workflow's restoration and verification path.
See [operations](operations/README.md), [artifact access](artifacts.md), and
[current status](experiments/status.md).

The Drive folder `SPAR/pilot1` was renamed to `SPAR/spar-pilot` in place; children
and existing run names were retained. Historical absolute paths may therefore
require validated migration rather than string replacement.

## Historical paths and resume

Archived configs record the paths and runtime used at execution. Those strings
are not rewritten. In particular, Pilot 2 evaluation configs contain absolute
adapter paths under the old Drive root. Existing reports, response files and
checkpoints are retained, but do not rerun historical training/evaluation or
the old comparison command after this rename expecting automatic resume.
That requires a validated path migration and the recorded runtime. Use the
new CPU section to audit the completed run without invoking those commands.

Historical `git show <commit>:output/jupyter-notebook/...` instructions retain
their old paths because those files exist at that historical commit. Current
GitHub and Colab links use `notebooks/...`.
