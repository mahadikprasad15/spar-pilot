# Artifacts and reproduction

## Access and roots

The Git repository contains code, notebooks, specs and documentation. Model
checkpoints, saved responses and large measurement arrays are outside Git under
an ignored artifact root. A clone alone cannot reproduce an existing report.
No public artifact download or blanket Drive permission is configured by this
repository; request the relevant run inventory and access from the maintainer.

| Environment | Root |
|---|---|
| Local workspace | `/Users/prasadmahadik/spar/pilot1` |
| Local artifacts | `<repository>/artifacts` |
| Persistent Drive | `My Drive/SPAR/spar-pilot/artifacts` |
| Mounted in Colab | `/content/drive/MyDrive/SPAR/spar-pilot/artifacts` |

Local result snapshots may be smaller than Drive runs. Absolute paths in recorded
logs describe their original environment. Resolve artifact-relative paths against
the chosen root; do not edit frozen files to make paths appear compatible.

## Layout

```text
artifacts/
  cohorts/           frozen selections and item identities
  plans/<name>/      prepared, reviewed and frozen workflow configs
  runs/              responses, adapters, arrays, logs and completion evidence
  reports/<name>/    derived results, tables, figures and provenance
  notebook/          operation console logs and review material
```

Run layout varies by pilot. Typical files are `meta/status.json`, runtime/source
manifests, `logs/run.log`, `logs/errors.jsonl`, structured results, response JSONL,
checkpoint directories and hash-sealed units. The producing stage's schema is
authoritative. A status counter or live PID is not a completion inventory.

## What to retain or share

A useful result bundle includes:

1. Exact frozen config, protocol version, code revision and runtime manifest.
2. Cohort IDs, token/template/mask identity and dataset/tokenizer revisions.
3. Required adapter/checkpoint files and their source hashes.
4. Canonical results and responses or sufficient per-example arrays for the claimed analysis.
5. Completion/verifier evidence, failed attempts where relevant, and report provenance.

A config is useful for review but contains references, not necessarily the data
or weights it names. Generated HTML is a presentation output, not the only source.
Dataset/model terms still apply when redistributing their material.

## Verification and resume

Use the notebook's pinned harness and the workflow's saved plan. First inspect
existing state. Public verification commands consume saved evidence; they are
preferable to hand-editing status files or rerunning completed GPU stages.
For example, with actual paths substituted:

```sh
python -m pilot_eval activation-verify --config EXECUTION_CONFIG --output-root ARTIFACT_ROOT
python -m pilot_eval traces-production-verify --config PRODUCTION_CONFIG --output-root ARTIFACT_ROOT
```

Consult the corresponding [operations guide](operations/README.md) for the exact
stage and required parents. Sealed units can be verified/reused; an unsealed unit
may repeat. Changed source, runtime, scope or numerical policy can require a new
named execution. Never bypass a failed identity check by changing stored hashes.

## Evidence hierarchy

Frozen execution artifacts and verifier outputs establish what ran. Structured
report inputs establish what a report analyzed. User-supplied console logs provide
reported execution evidence. Specs/notebook defaults describe intended behavior;
ticket completion describes software availability. The [status ledger](experiments/status.md)
records which of these supports each current claim.

The local 13 October Pilot 4 snapshots and the supplied Pilot 5 completion summary
have different dates/scopes. This documentation update did not fetch or rehash the
complete live Drive tree.
