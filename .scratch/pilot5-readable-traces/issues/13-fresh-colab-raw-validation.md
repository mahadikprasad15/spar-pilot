# 13: Validate the completed raw notebook in fresh Colab

**What to build:** Run final raw-path deployment regression after the early environment smoke, using a bounded real capture/readout, durable sync and reconnect/reuse demonstration.

**Blocked by:** 11: Publish the automatic research report and verified handoff.

**Status:** ready-for-agent

- [ ] Record the actual clean environment, imports, source inventory, bounded workload, durable verification and reconnect reuse through public stage calls.
- [ ] Reviewer check: code is readable with explicit imports/named settings, purpose/input/output notes and no required repair cells. Keep this judgment separate from automatic checks such as parsing cells and rejecting expression-valued string fields.
- [ ] Incompatible runtime and active-child failures are actionable before inference; interruption cleans up its owned child.
- [ ] The check neither runs nor waits for optional assistant work or human grading. Those remain independent status entries.
- [ ] Separate offline implementation tests from target-GPU validation and from the actual full scientific experiment; no blanket guarantee about every Colab allocation.
