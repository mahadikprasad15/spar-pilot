# 06: Deliver the guided Colab workflow

**What to build:** Provide an explained notebook that runs the approved workflow, makes artifacts easy to find and supports safe recovery and CPU-only reporting.

**Blocked by:** 05 — Reconstruct the scientific report on CPU.

**Status:** ready-for-agent

- [ ] Stages call shared commands rather than duplicate measurement logic; explain purpose, expected outputs and rerun behavior.
- [ ] Use shared checkout/Drive names; verify checkout entry points and dependency versions before expensive work.
- [ ] Display actual sources/masks and reviewed profiling evidence before production; explain when the GPU may be disconnected.
- [ ] Recovery instructions distinguish live child processes, stale imported code, incompatible plans and incomplete artifacts.
- [ ] Link the decision guide, spec/design and earlier-pilot lessons; validate notebook syntax and workflow integration on CPU.
- [ ] Clearly distinguish locally tested behavior from real GPU preflight still required in Colab.
