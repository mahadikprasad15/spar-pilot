"""Sparse rewiring: activation-shift pass (spec §Standalone analysis, D22; the GPU part of ticket 17).

For each arm's final checkpoint: run the 128 validation items (prompt + gold answer, teacher-forced, one item at a
time, no padding) twice, with every rewire module off (the exact base model) and on (deterministic gates). At every
token, record each selected neuron's post-activation ``SiLU(gate) * up`` in both passes; the shift is on - off.
Selected neurons: every final survivor, plus a fixed reference of 1,000 random non-survivors per layer
(``numpy`` seed ``REFERENCE_SEED``). Per neuron: mean signed shift and mean absolute shift over all tokens
(lesson 2026-11-11: magnitudes beside signed averages). Summaries use ``rewire_standalone.summarize_activation_shift``.

Reads only sealed run outputs (the arm's evidence and final checkpoint, verified) plus the run's own validation
set from its input bundle. Writes one sealed unit per arm under
``c-attribution/sparse-rewiring/analysis/activation-shift-<arm>/`` (``activation_shift.json``,
``activation_shift.npz``) in the given run root, local or on the volume.

    python -m pilot_eval.rewire_activation_shift --root $WORK/local-prod            # on the Pod, after the chain
    # then copy the sealed units to the volume run root (the command prints the cp line)
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pilot_eval import attribution_units as au

SCHEMA = 'sparse-rewiring/activation-shift-v1'
REFERENCE_PER_LAYER = 1000
REFERENCE_SEED = 20261125


def select_neurons(survivors: Sequence[Mapping[str, Any]], layers: int, neurons: int, *,
                   per_layer: int = REFERENCE_PER_LAYER, seed: int = REFERENCE_SEED) -> dict[int, dict[str, np.ndarray]]:
    """Per layer: survivor indices and ``per_layer`` random non-survivor indices (sorted, fixed by ``seed``)."""
    rng = np.random.default_rng(seed)
    out = {}
    for layer in range(layers):
        surv = np.array(sorted({int(s['neuron']) for s in survivors if int(s['layer']) == layer}), dtype=np.int64)
        pool = np.setdiff1d(np.arange(neurons), surv)
        ref = np.sort(rng.choice(pool, size=min(per_layer, pool.size), replace=False)).astype(np.int64)
        out[layer] = dict(survivors=surv, reference=ref)
    return out


def measure(model, items: Sequence[Mapping[str, Any]], selection: Mapping[int, Mapping[str, np.ndarray]],
            device) -> dict[int, dict[str, np.ndarray]]:
    """Per layer and group: per-neuron sum of signed and absolute shifts and the token count."""
    import torch
    from pilot_eval import rewire_module as rm
    mods = rm.modules(model)
    idx = {m.layer: torch.as_tensor(np.concatenate([selection[m.layer]['survivors'],
                                                    selection[m.layer]['reference']]), device=device)
           for m in mods}
    captured: dict[int, torch.Tensor] = {}

    def hook(module, inputs, _output):
        gate, up = module.pre_activations(inputs[0])
        captured[module.layer] = (module.act_fn(gate) * up)[..., idx[module.layer]].detach().to(torch.float64)

    handles = [m.register_forward_hook(hook) for m in mods]
    sums = {m.layer: dict(signed=torch.zeros(idx[m.layer].numel(), dtype=torch.float64, device=device),
                          absolute=torch.zeros(idx[m.layer].numel(), dtype=torch.float64, device=device))
            for m in mods}
    tokens = 0
    model.eval()
    try:
        with torch.no_grad():
            for item in items:
                ids = torch.as_tensor([list(item['input_ids'])], device=device)
                with rm.off(model):
                    model(input_ids=ids, use_cache=False)
                base = {k: v[0] for k, v in captured.items()}
                model(input_ids=ids, use_cache=False)
                for layer, act in captured.items():
                    delta = act[0] - base[layer]
                    sums[layer]['signed'] += delta.sum(0)
                    sums[layer]['absolute'] += delta.abs().sum(0)
                tokens += ids.shape[1]
    finally:
        for h in handles:
            h.remove()
    out = {}
    for layer in sorted(sums):
        n_s = selection[layer]['survivors'].size
        signed, absolute = (sums[layer][k].cpu().numpy() / tokens for k in ('signed', 'absolute'))
        out[layer] = dict(survivor_mean=signed[:n_s], survivor_mean_abs=absolute[:n_s],
                          reference_mean=signed[n_s:], reference_mean_abs=absolute[n_s:])
    out['tokens'] = tokens
    return out


def summarize(per_layer: Mapping[Any, Any], selection: Mapping[int, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    from pilot_eval import rewire_standalone as rs
    layers = sorted(k for k in per_layer if k != 'tokens')
    cat = lambda key: np.concatenate([per_layer[l][key] for l in layers]) if layers else np.zeros(0)
    signed = rs.summarize_activation_shift(cat('survivor_mean'), cat('reference_mean'))
    magnitude = rs.summarize_activation_shift(cat('survivor_mean_abs'), cat('reference_mean_abs'))
    survivors = [dict(layer=l, neuron=int(n), mean_shift=float(m), mean_abs_shift=float(a))
                 for l in layers for n, m, a in zip(selection[l]['survivors'], per_layer[l]['survivor_mean'],
                                                    per_layer[l]['survivor_mean_abs'])]
    return dict(per_neuron_mean_shift=signed, per_neuron_mean_abs_shift=magnitude, survivors=survivors,
                tokens=int(per_layer['tokens']))


def run(root: str | Path, *, device: str = 'cuda', items: Sequence[Mapping[str, Any]] | None = None,
        model_loader=None) -> dict[str, str]:
    """Measure every arm of the finished run at ``root`` (the directory that holds ``c-attribution/``)."""
    import torch
    from pilot_eval import rewire_chain as rc
    from pilot_eval import rewire_checkpoint as rckpt
    from pilot_eval import rewire_evaluation as reval
    from pilot_eval import rewire_launch as rl
    from pilot_eval import rewire_report as rr
    root = Path(root)
    record = rr.launch_record(root)
    if items is None:
        items = rl.load_inputs(record)['val_items']   # the bundle is re-verified against its seal
    if model_loader is None:
        def model_loader():
            from transformers import AutoModelForCausalLM
            from pilot_eval.neuron_audit_collector import set_ieee_precision
            set_ieee_precision()
            return AutoModelForCausalLM.from_pretrained(record['inputs']['model_dir'], local_files_only=True,
                                                        torch_dtype=torch.float32,
                                                        attn_implementation='eager').to(device)
    written = {}
    for arm, spec in record['arms'].items():
        arm_dir = rr.paths(root)['runs'] / spec['run_id']
        plan = json.loads((rr._sealed(arm_dir / rc.ARM_EVIDENCE_DIR) / 'plan.json').read_text())
        if (arm_dir / rc.ARM_EVIDENCE_DIR / 'complete.json').is_file():
            rc.restore_arm_evidence(arm_dir)
        seal = rckpt.verify_arm(arm_dir, plan)
        final = int(plan['checkpoints'][-1])
        export_dir = arm_dir / 'masks' / f'step-{final}'
        au.verify_run(export_dir)
        export = json.loads((export_dir / f'{final}.json').read_text())
        if export.get('frozen') is not True or export.get('survivors_omitted'):
            raise ValueError(f'{arm}: the final survivor export is not frozen with full records')
        model = model_loader()
        try:
            reval.load_checkpoint_for_eval(model, rckpt.checkpoint_dir(arm_dir, final), plan,
                                           runtime_sha256=seal['runtime_sha256'], code_commit=seal['code_commit'])
            mods = reval.rm.modules(model)
            selection = select_neurons(export['survivors'], len(mods), mods[0].intermediate_size)
            per_layer = measure(model, items, selection, next(model.parameters()).device)
        finally:
            model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        summary = summarize(per_layer, selection)
        body = dict(schema=SCHEMA, arm=arm, rank=int(plan['rank']), run_id=spec['run_id'], step=final,
                    plan_sha256=plan['plan_sha256'], checkpoint_code_commit=seal['code_commit'],
                    items=len(items), item_ids_sha256=__import__('hashlib').sha256(
                        '\n'.join(sorted(str(i['id']) for i in items)).encode()).hexdigest(),
                    positions='all tokens of prompt + gold answer (teacher-forced), one item at a time',
                    reference=dict(per_layer=REFERENCE_PER_LAYER, seed=REFERENCE_SEED),
                    shift='SiLU(gate)*up with rewire on (deterministic gates) minus rewire off', **summary)
        arrays = io.BytesIO()
        np.savez(arrays, **{f'layer{l}_{k}': v for l, d in per_layer.items() if l != 'tokens' for k, v in d.items()},
                 **{f'layer{l}_{g}_index': selection[l][g] for l in selection for g in ('survivors', 'reference')})
        out = rr.paths(root)['analysis'] / f'activation-shift-{arm}'
        au.seal_run(out, {'activation_shift.json': body, 'activation_shift.npz': arrays.getvalue()})
        written[arm] = str(out)
    return written


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--root', required=True, help='the finished run directory that holds c-attribution/')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    written = run(args.root, device=args.device)
    print(json.dumps(written, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
