# 07: Restore and verify the complete audit package

**Status:** needs-triage — draft for adversarial review

**What to build:** At any stage, including preparation or after a crash, snapshot all available sealed artifacts, input/config manifests, progress, status and failure evidence; export to the configured private HF dataset repository and verify an independent fresh-download restore. Produce a clearly labeled partial or complete receipt.

**Blocked by:** 03a, for the artifact and seal layout. It must be implemented and available before production ticket 04. It does not depend on a final report or successful collection.

- [ ] Default to `Prasadmahadik/spar-sampling-sft-artifacts`, configurable; validate private repository and actual upload/readback using an online Hub client separately from the offline worker cache. Include an inventory and hashes; document reconstructible omissions and regeneration dependencies.
- [ ] A clean-root restore verifies every included item and distinguishes partial from complete; account for export time in runtime/cost.
- [ ] Detect authentication, disk, upload and readback failures; preserve local work and never stop a Pod before verified external copy.
- [ ] CPU integration uses mocked I/O and interrupted/resumed cases; no live external write required for tests.
- [ ] Snapshot only stable sealed data, include partial/error status and budget timing, and support invocation before a report exists. Partial verification is never labeled full completion.

## Comments

### 2026-11-10 — first implementation (Claude)

- `pilot_eval/neuron_audit_export.py`, reusing `sampling_sft_hf` (private check, `require_space`, `retryable`, `transfer_deadline`, archive+index commit). `snapshot` archives only verified sealed runs/units plus status/log evidence. It lists unsealed/interrupted/corrupt items and missing expected units, and records the scope and the omitted reconstructible inputs. `upload` re-verifies files after archiving and reads back via a fresh pinned-revision download; 401/403 are fatal and 429/5xx are retried. `restore` requires an empty root, verifies every file and re-verifies block seals. `may_stop_pod` is true only for a verified restore of the same snapshot; it can additionally demand `complete`.
- Default repo `Prasadmahadik/spar-sampling-sft-artifacts` (configurable). Text files containing credential-like strings block the snapshot.
- Tests: 10 with a mocked Hub (no network): partial→complete after resume, round trip, resume without recommit, public repo refused, readback mismatch, auth vs transient errors, changed file, secret, corrupt unit, tampered archive.
- Local real snapshot (no upload yet): `exports/joint-neuron-adapter-audit/prep-v2/snapshot-8bde88e707dab1bd.json`, 25 files, complete for its preparation scope only.
- **Not done:** a live HF write/readback (needs your go-ahead); export timing in the overall budget is recorded per call but not yet wired into a runbook.
