# 06: Deliver the guided Colab workflow

**What to build:** Provide an explained notebook that runs the approved workflow, makes artifacts easy to find and supports safe recovery and CPU-only reporting.

**Blocked by:** 05 — Reconstruct the scientific report on CPU.

**Status:** resolved

- [x] Stages call shared commands rather than duplicate measurement logic; explain purpose, expected outputs and rerun behavior.
- [x] Use shared checkout/Drive names; verify checkout entry points and dependency versions before expensive work.
- [x] Display actual sources/masks and reviewed profiling evidence before production; explain when the GPU may be disconnected.
- [x] Recovery instructions distinguish live child processes, stale imported code, incompatible plans and incomplete artifacts.
- [x] Link the decision guide, spec/design and earlier-pilot lessons; validate notebook syntax and workflow integration on CPU.
- [x] Clearly distinguish locally tested behavior from real GPU preflight still required in Colab.

## Answer

Delivered `notebooks/pilot-3-colab.ipynb`: 14 explained sections call shared
commands for source/input freezing, mask audit, real instrument validation,
profiling/review, execution freezing, resumable measurement, strict completion
verification and CPU report/figures. Shared checkout/Drive names and pinned
core revision `c547501` are checked before work. This revision contains all
measurement/report commands; the notebook lives at the newer delivery revision.

Input and profiling reviews guard execution freezing. The helpers persist
console/PID records, distinguish boot identities, refuse a second live child,
tolerate partial status via the shared monitor, check process exit and require
separate scientific verification. Setup also blocks switching code with an
active recorded child. Recovery instructions cover stale imports, partial
artifacts, corrupt marked evidence, incompatible runtime/settings and OOM.
Section 11 identifies the GPU-disconnect point; a CPU-only reconnect route is
explicit. The interview, spec/design, lessons and instrument guide are linked.

Verification: notebook code syntax and nbformat validation passed. Three CPU
tests cover the artifact/stage contract, connected saved workflow and immutable
reconnect/CPU report reuse, plus real subprocess success/failure, partial status
and live-child protection. Full offline suite: **178 passed, 13 existing
warnings, 353.90 seconds**. Colab Drive mounting, package installation and actual
source/model/CUDA execution are not locally verified and remain notebook gates.
