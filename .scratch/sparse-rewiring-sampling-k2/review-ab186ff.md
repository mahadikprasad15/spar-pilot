# Sampling SFT k = 2 re-review at `ab186ff`

Reviewed exact commit `ab186ff71bf6214d069f0fdd86aeb22022a1e68c` on `sparse-rewiring/sampling-k2`, against [the previous review](review-6831f74.md) and [the K2 spec](spec.md). The working checkout was already at `3bb75fc`, so I ran the review and tests in a detached copy of `ab186ff`.

**Hold the K2 production GPU launch.** Stage 1 target generation cannot reach model loading via either normal CLI configuration. The production loader can also accept targets from an unqualified generator.

## Standards

### P1 — Fix both Stage 1 provenance paths

With `--bundle`, the CLI calls `collect_provenance()` without a pin override. The helper explicitly passes `pinned=None` into `check_base_model()`, bypassing its trusted default and raising `TypeError: 'NoneType' object is not subscriptable`. With no bundle, the helper reads `pinned['file_sha256']` from `pinned_identities()`, whose result is keyed by filename, so it raises `KeyError: 'file_sha256'`. That helper measures the supplied snapshot and is intended for diagnostics, so using its results as trust anchors would still allow the snapshot to certify itself.

I reproduced both failures on CPU with minimal local files and the real provenance helper. The current test supplies a custom non-null pin map and misses the CLI default. For production, use the independently pinned base-model identities, then record the flat SHA-256 map after verification. Require a verified bundle for production if the no-bundle entry point cannot verify the tokenizer and validation source. Exercise the CLI with its real default pin flow in the regression test.

[Provenance branches](../../scripts/generate_val128_k2_targets.py#L112) · [CLI call](../../scripts/generate_val128_k2_targets.py#L555) · [Trusted pins and diagnostic identity shape](../../pilot_eval/c_attribution_bundle.py#L28) · [Verifier](../../pilot_eval/c_attribution_bundle.py#L172) · [Test fixture](../../tests/test_rewire_k2_launch.py#L262)

### P1 — Enforce qualified generator identity and precision at launch

The target loader compares the model, tokenizer, and validation hashes and checks the recipe constants. It only requires nonempty engine and executable fields and does not check precision readback. A CPU probe used matching input hashes and an accepted recipe with `engine='unqualified-engine'`, `executable_sha256='arbitrary'`, and `matmul_precision='medium'`. The loader accepted it.

Require the generator identity authorized for Stage 1, its registered executable digest, and the required IEEE readback. Bind these expectations to the approved record so a deliberate generator change has an explicit provenance path. Add refusal cases for changed engine, code, and precision. Keep spy engines explicit in test fixtures.

[Target provenance validation](../../pilot_eval/rewire_k2_launch.py#L217) · [Production loader](../../pilot_eval/rewire_k2_launch.py#L315)

## Spec

### P2 — Bind activation shifts to the frozen base model

The prior unpacking crash is fixed and intermediate checkpoints are refused. The exporter still loads an arbitrary same-shaped model and tokenizer from `--model-dir`. It checks the arm and K2 validation file, then records the supplied config hash without checking the actual base weights and tokenizer. A different same-shaped snapshot can therefore produce shifts labeled as verified.

Verify all model and tokenizer files against the frozen experiment bundle. Include their verified identities in the export. Test refusal when the model weights change with the config held constant.

[Checkpoint and validation checks](../../scripts/extract_survivor_shifts.py#L64) · [Unverified model load and config hash](../../scripts/extract_survivor_shifts.py#L329)

### P2 — Compare actual lens assets and validate orientation

Missing fitted layers now fail and rank-revealing SVD fixes the principal-angle bug. Streaming still resolves mutable `main` and the local path accepts arbitrary weights and lenses. The decoder stamps revision labels without binding them to the loaded data, and the report accepts a comparison when one cache lacks revision metadata. The orientation check compares the cosine of `Jv` and `Jᵀv` against the same input `v`. Those dot-product numerators are identical, and the function always labels the orientation `J @ v` without testing a known target.

Fetch immutable revisions or verify local asset hashes, record the actual model, tokenizer and fitted-lens identities, and require compatible provenance in both comparative caches. Test orientation against a known asymmetric mapping and reject a transposed fixture.

[Streaming URLs](../../notebooks/decode_down_proj.py#L36) · [Orientation check](../../notebooks/decode_down_proj.py#L190) · [Cache labels](../../notebooks/decode_down_proj.py#L303) · [Comparison guard](../../notebooks/build_down_proj_report.py#L154)

## Prior findings and verification

- Shared Python-source binding and production hyperparameter and seed enforcement address the prior findings. Routing checks the expected matched plan and binds the selected Stage 1 headroom seal.
- Stage 1 now preserves attempt and disposition evidence. A separate CPU probe paused the real sampler immediately after its first completed block, restarted it from the persisted checkpoint, and reproduced identical targets and evidence with a deterministic fixture.
- The completed-arm exporter now uses the correct checkpoint-loader return value. Missing fitted lens layers and rank-deficient subspaces have regression coverage.
- The focused K2 launch and K2/Gold plan suite passed **66 tests**. `git diff --check 6831f74...ab186ff` passed. The full suite stopped at collection on missing PEFT and PyArrow. No GPU run or production artifact restore was performed.
