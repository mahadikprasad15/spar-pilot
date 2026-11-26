# Sampling SFT k = 2 review at `9258e2c`

Reviewed `9258e2cb4a98ca93047901687e4941210dc9c50b` against [the previous review](review-ab186ff.md), with attention to practical run failures and serious scientific mistakes.

The Stage 1 pin crash is fixed in the CPU CLI fixture. The loader now checks the declared generator, code digest, and IEEE precision. I found no additional concrete blocker in the Stage 1-to-training path. The GPU path and production artifact restore remain unverified.

## P2 — The Gold–K2 comparison rejects the existing Gold cache

The comparison requires both caches to contain revision metadata and then compares their full asset-hash dictionaries. The existing 25 MB Gold cache at `notebooks/cache/down_proj_decoded.pt` has neither revision nor asset hashes. I exercised the real comparison function against that file and a K2-shaped cache with the new metadata. It raised `ValueError: cannot compare caches when either cache lacks 'model_revision'`, so the requested comparison HTML was not produced.

There is a second compatibility case in the same guard. A cache built from local model weights records `model.safetensors`, while HTTP streaming records `model.safetensors:header`. Their hash dictionaries differ even when both use the same model. This prevents a pod-local K2 decode from being compared with a Mac-streamed Gold decode.

Support the existing Gold cache with its known provenance, or supply a practical migration or regeneration path. Use a comparable model identity across local and streamed decodes. Test the comparison with the actual existing Gold cache and with local-versus-streamed representations of the same model.

[Comparison guard](../../notebooks/build_down_proj_report.py#L154) · [Local identity](../../notebooks/decode_down_proj.py#L150) · [Streamed identity](../../notebooks/decode_down_proj.py#L165)

## P2 — The Gold report presents an untested causal explanation as a result

The generated report says the decoded vectors strongly confirm that style and length compression drive CE recovery and the accuracy drop, and says changing roughly a dozen neurons is enough to cause that tradeoff. The report decodes frozen output directions and shows correlations with token scores. Those observations do not establish the active effect of each neuron or the cause of the accuracy change. The study explicitly deferred the causal checks.

Present the decoded formatting and digit associations and the observed model-level metrics, and label the proposed explanation as a hypothesis pending the relevant interventions. Update both the generator and published HTML together.

[Report source](../../notebooks/_gold_down_proj_report.py#L1004) · [Generated HTML](../../docs/sparse-rewiring-down-proj-report.html#L512) · [Deferred causal checks](../sparse-rewiring-gsm8k/decision-log.md#L254)

## Verification

- The focused K2 launcher and K2/Gold plan tests passed **68 tests** on CPU.
- `git diff --check ab186ff...9258e2c` passed.
- The full suite stopped at collection because PEFT and PyArrow are unavailable in the local environment. The focused reference fixture uses a stand-in when PEFT is unavailable.
- A separate CPU probe reproduced both cache compatibility failures. No GPU run or production artifact restore was performed.
