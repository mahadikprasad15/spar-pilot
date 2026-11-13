# 01: Inventory and bind audit sources

**Status:** needs-triage — draft for adversarial review

**What to build:** A parameterized, read-only inventory that identifies every intended arm, step-64 and step-0 checkpoint, base model, tokenizer and available training/rollout/selection lineage. Produce a sealed machine-readable inventory with verified hashes, absent evidence, ranks and module coverage; reuse compatible existing verification without requiring inaccessible historical graphs unnecessarily.

**Blocked by:** None.

- [ ] Start from the local imported offline-GRPO step-64 and step-0 tensors and the heatmap's internals source; locate their authoritative adapter configuration and verify their hashes and provenance. Their directories do not themselves contain adapter configs. Missing arms or initialization evidence remain explicit.
- [ ] Trace the internals report's loader and recorded config hash first, then its training launcher and logs. The training plan reports rank 1, alpha 1 and seven projection types, and a locally available original zero-checkpoint config matches its original hash; the internals step-64 config has a different recorded hash. Do not substitute one for the other without proof.
- [ ] If scale or targets remain unknown, mark the actual adapted arm unresolved; do not claim an actual within-arm read ranking, since its incoming activations also depend on the adapter configuration.
- [ ] Identify unsupported rank, scales or module mappings; distinguish file hashes from parameter hashes.
- [ ] CPU fixtures exercise changed, absent and mismatched sources; no downloads.
- [ ] Report exactly what must be restored before a seven-arm comparison.
- [ ] Record the expected saved PEFT tensor key list and module/shape mapping for every checkpoint so later live-model checks can detect missing, unexpected or inactive adapters.

## Comments

### 2026-11-10 — first implementation (Claude)

- Code: `pilot_eval/neuron_audit_inventory.py` (`build` / `verify` CLI), tests `tests/test_neuron_audit_inventory.py` (14 CPU fixtures: changed, absent, mismatched, zero-B, unexpected/missing keys, unsupported options, target mismatch, out-of-root paths, sealed reuse, corrupt seal).
- Sources plan: `<artifact_root>/plans/joint-neuron-adapter-audit-v1/inventory.sources.json`; sealed output: `<artifact_root>/runs/joint-neuron-adapter-audit/inventory/inventory-v1/` with `artifact_root=/Users/prasadmahadik/spar/pilot1/artifacts` (main checkout).
- Result: 5 arms load-ready by payload/config/seal identity (Gold SFT, online GRPO, Sampling SFT k0/k2/k5: 28 layers, rank 1, 196 modules, scale 1.0, init/audited key lists and semantic configs match). Untuned baseline has no adapter. Offline GRPO is **unresolved**: step-0/64 weights match the internals record, but `adapter_config.json` and `complete.json` are absent for both steps (4 restore requirements, candidate source `Prasadmahadik/spar-offline-grpo-v1@6b3fafd9…`, unverified). The base snapshot is not local; expected hashes are recorded only.
- **Correction to the spec premise:** the internals plan records config `1360b89e…` for *every* offline checkpoint, step 0 included; `05e94f39…` is the Gold SFT step-0 config used as the initialization source. PEFT writes `target_modules` in set order, so identical configs get different file hashes (e.g. Sampling SFT k5 step-64 `54eb451e…`). The inventory therefore adds a separate semantic config hash. The missing file still has to be restored to verify `1360b89e…`. A step-64-specific config change is not indicated.
- Lineage gap: `plans/pilot4-grpo-fp32-v1-frozen-v2/grpo.frozen.json` (online GRPO) is not local.
- Live PEFT routing/activity is not checked here (03a/03b).

### 2026-11-10 — offline GRPO restored (Claude)

- Downloaded from private HF dataset `Prasadmahadik/spar-offline-grpo-v1@6b3fafd9f666268a287bea5f05f3eeb5088fc8ee` into the import dir: `checkpoint-{0,64}/adapter_config.json`, `checkpoint-{0,64}/complete.json` and training `complete.json`. All four checkpoint files match the internals-recorded hashes (config `1360b89e…`, seals `0f6f15b7…` / `8ddc7f2b…`). The config is r=1, alpha=1, dropout 0, the seven projections in a different order.
- New sealed inventory `inventory-v2`: 6 load-ready + untuned baseline; 0 restore requirements; `seven_arm_complete: true` by payload/config/seal identity. `inventory-v1` is kept as the earlier record.
- The repo also has checkpoint-8/16/32 configs/seals, which were not downloaded because they are not needed.
