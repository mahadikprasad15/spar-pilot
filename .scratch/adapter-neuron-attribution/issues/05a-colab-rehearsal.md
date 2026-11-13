# 05a: Colab bug-finding notebook and GPU utilization instrument

**Status:** needs-triage — draft for adversarial review
**Type:** task
**Blocked by:** 01, 03, 04

**What to build:** `notebooks/adapter-attribution-colab.ipynb` and `pilot_eval/attribution_gpu.py`. The notebook runs every stage once on the real model and a small covering subset to find bugs cheaply. Outputs are labelled `rehearsal-colab` and can never feed production.

- [ ] Environment cells carry the joint audit's Colab lessons: stream child stdout/stderr; uninstall unused preinstalled packages (torchvision) before pinning torch; record device, driver, torch/CUDA build; `pad_token_id` from the pinned tokenizer config.
- [ ] Subset: `gold-sft`, `online-grpo`, untuned; 16 discovery + 8 confirmation base-correct rows chosen to cover short and long rows; 10 generation items.
- [ ] Real-model checks: finite differences (3 adapters, 3 neurons) per arm; Σ_j attr_j identity; hooks-attached logits == unhooked; removal hook == edited checkpoint; untuned → 0.
- [ ] One pass of each stage: A+B on the subset, 3 Stage C conditions, 2 generation runs × 10 items, a tiny report.
- [ ] `attribution_gpu.py`: NVML sampler thread (busy %, memory, power at 1 Hz), per-stage summary (median and p10 busy, peak memory, share < 50% busy), tokens/s real and padded, padding fraction, MFU from parameter count and device peak for the active precision, one `torch.profiler` steady-state trace per stage.
- [ ] Utilization sweep on the Colab device: token budgets 16k–256k for forward-only and forward+backward; generation batch 16/32/64. Record results; they inform but do not set the Pod budget (different device memory).
- [ ] Evidence upload to the private HF dataset through the real online export path; the receipt captured in the notebook.
- [ ] Every failure found is fixed with a regression test and added to `docs/agents/experiment-lessons.md`.

## Comments
