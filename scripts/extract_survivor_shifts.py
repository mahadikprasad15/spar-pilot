"""Lightweight on-pod post-hoc exporter for sparse-rewiring survivors & val128 activation shifts.

Runs in ~30–60s on an A100 with sl-0 mounted and exports a compact JSON (<200 KB) containing:
  1. Final binary surviving neurons (layer, idx, log_alpha, z_det, gate_norm, up_norm).
  2. Exact paired-forward val128 completion-token activation shifts
     ``Δa_j(t) = a_rewired(t) - a_base(t)``, where ``a_base(t)`` comes from a full base-model
     forward pass (``with rm.off(model):``, matching ``pilot_eval.rewire_activation_shift.measure``)
     and ``a_rewired(t)`` comes from the full rewired forward pass on the exact same token sequence,
     so upstream rewiring never contaminates the base activation or flips the active pole.

Requires verified final checkpoint identity (``rewire_checkpoint.verify_arm`` /
``rewire_checkpoint.load_tensors``) unless an explicit toy-fixture bypass is used in unit tests.
"""
from __future__ import annotations

import argparse
import heapq
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pilot_eval import hard_concrete as hc
from pilot_eval import rewire_checkpoint as rckpt
from pilot_eval import rewire_module as rm


def _load_verified_final_checkpoint(
    model: torch.nn.Module,
    *,
    arm_dir: Path | None = None,
    ckpt_dir: Path | None = None,
    rank: int | None = None,
    val_items_path: Path | None = None,
    allow_unverified_fixture: bool = False,
) -> tuple[list[rm.RewireMLP], dict[str, Any]]:
    """Install ``RewireMLP`` and load a verified completed-arm final checkpoint into ``model``.

    Always requires a completed, sealed arm (``training-complete.json``, ``plan.json``,
    ``rewire_checkpoint.verify_arm``) and refuses intermediate checkpoints (such as ``checkpoint-0``
    or ``checkpoint-64``). ``allow_unverified_fixture=True`` is for toy-model unit tests only.
    """
    from pilot_eval import rewire_chain as rc
    from pilot_eval.training import file_hash

    if arm_dir is None and ckpt_dir is not None and not allow_unverified_fixture:
        # Derive parent arm directory from <arm_dir>/checkpoints/checkpoint-<step> and enforce full arm completion
        candidate_arm = ckpt_dir.parent.parent
        if ckpt_dir.parent.name != 'checkpoints' or not (candidate_arm / rckpt.ARM_SEAL).is_file():
            raise ValueError(
                f'{ckpt_dir}: refusing checkpoint without verified completed parent arm '
                f'(pass --arm-dir pointing to a completed arm with {rckpt.ARM_SEAL})'
            )
        arm_dir = candidate_arm

    if arm_dir is not None:
        ev_dir = arm_dir / rc.ARM_EVIDENCE_DIR
        if (ev_dir / 'complete.json').is_file():
            rc.restore_arm_evidence(arm_dir)
        arm_seal_path = arm_dir / rckpt.ARM_SEAL
        plan_path = arm_dir / 'plan.json'
        if not arm_seal_path.is_file() or not plan_path.is_file():
            raise ValueError(f'{arm_dir}: missing {rckpt.ARM_SEAL} or plan.json (refusing unverified arm)')
        arm_seal = json.loads(arm_seal_path.read_text(encoding='utf-8'))
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        rckpt.verify_arm(
            arm_dir,
            plan,
            runtime_sha256=arm_seal['runtime_sha256'],
            code_commit=arm_seal['code_commit'],
        )
        final_step = int(plan['steps'][-1]['step'])
        expected_ckpt_dir = rckpt.checkpoint_dir(arm_dir, final_step)
        if ckpt_dir is not None and ckpt_dir.resolve() != expected_ckpt_dir.resolve():
            raise ValueError(
                f'{ckpt_dir}: refusing non-final checkpoint (completed arm final step is {final_step}: '
                f'{expected_ckpt_dir})'
            )
        ckpt_dir = expected_ckpt_dir
        if val_items_path is not None and val_items_path.is_file():
            expected_val_sha = (plan.get('inputs') or {}).get('val128_k2_targets_sha256')
            if expected_val_sha is not None and file_hash(val_items_path) != expected_val_sha:
                raise ValueError(
                    f'{val_items_path}: SHA-256 differs from arm plan val128_k2_targets_sha256 ({expected_val_sha})'
                )
        rank = int(plan['rank'])
        mods = rm.install(model, rank=rank)
        seal = rckpt.load_into(
            model,
            ckpt_dir,
            plan,
            runtime_sha256=arm_seal['runtime_sha256'],
            code_commit=arm_seal['code_commit'],
        )
        rm.set_gate(model, None)
        provenance = {
            'verified': True,
            'arm_dir': str(arm_dir.resolve()),
            'checkpoint_dir': str(ckpt_dir.resolve()),
            'final_step': final_step,
            'plan_sha256': plan['plan_sha256'],
            'training_complete_sha256': file_hash(arm_seal_path),
            'checkpoint_sha256': seal['files'][rckpt.STATE_FILE],
            'code_commit': arm_seal['code_commit'],
            'val_items_sha256': file_hash(val_items_path) if (val_items_path and val_items_path.is_file()) else None,
            'rank': rank,
        }
        return mods, provenance

    if not allow_unverified_fixture or ckpt_dir is None or rank is None:
        raise ValueError('provide --arm-dir pointing to a verified completed arm')
    mods = rm.install(model, rank=int(rank))
    state = load_file(str(ckpt_dir / rckpt.STATE_FILE))
    missing, unexpected = model.load_state_dict(state, strict=False)
    rewire_missing = [k for k in missing if 'rewire_' in k]
    if rewire_missing or unexpected:
        raise ValueError(f'checkpoint tensor mismatch: missing={rewire_missing}, unexpected={unexpected}')
    rm.set_gate(model, None)
    return mods, {'verified': False, 'checkpoint_dir': str(ckpt_dir), 'rank': int(rank)}


def _extract_survivors(mods: list[rm.RewireMLP]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for mod in mods:
        la = mod.rewire_log_alpha.detach().float().cpu()
        z_det = hc.deterministic_gate(la)
        active = torch.nonzero(z_det > 0.0, as_tuple=False).flatten().tolist()
        if not active:
            continue
        if mod.rank == 0:
            gn = hc.row_change_norms(rank=0, w=mod.rewire_W_gate.detach().float().cpu(), z=z_det)
            un = hc.row_change_norms(rank=0, w=mod.rewire_W_up.detach().float().cpu(), z=z_det)
        else:
            gn = hc.row_change_norms(
                rank=mod.rank,
                a=mod.rewire_A_gate.detach().float().cpu(),
                b=mod.rewire_B_gate.detach().float().cpu(),
                z=z_det,
            )
            un = hc.row_change_norms(
                rank=mod.rank,
                a=mod.rewire_A_up.detach().float().cpu(),
                b=mod.rewire_B_up.detach().float().cpu(),
                z=z_det,
            )
        for idx in active:
            out.append({
                'layer': int(mod.layer),
                'idx': int(idx),
                'label': f'L{mod.layer} #{idx}',
                'log_alpha': float(la[idx]),
                'z_det': float(z_det[idx]),
                'gate_norm': float(gn[idx]),
                'up_norm': float(un[idx]),
            })
    return out


@torch.no_grad()
def measure_activation_shifts(
    model: torch.nn.Module,
    mods: list[rm.RewireMLP],
    survivors: list[dict[str, Any]],
    val_items: list[dict[str, Any]],
    tokenizer: Any,
    *,
    top_contexts: int = 5,
    context_window: int = 12,
) -> list[dict[str, Any]]:
    """Measure exact completion-token Δa_j = a_rewired(t) - a_base(t) via paired base & rewired forwards.

    Pass 1 runs with ``with rm.off(model):`` (exact fully-base model); Pass 2 runs with rewiring ON
    (deterministic gates, ``rm.set_gate(model, None)``) on the identical token sequence. This matches
    ``pilot_eval.rewire_activation_shift.measure`` so upstream rewiring is reflected properly in
    ``a_rewired - a_base`` rather than leaking into ``a_base``.
    """
    device = next(model.parameters()).device
    by_layer: dict[int, list[int]] = {}
    for s in survivors:
        by_layer.setdefault(int(s['layer']), []).append(int(s['idx']))

    mod_by_layer = {int(m.layer): m for m in mods}
    idx_tensors = {
        L: torch.tensor(idxs, dtype=torch.long, device=device)
        for L, idxs in by_layer.items()
    }
    stats = {
        (int(s['layer']), int(s['idx'])): {
            'n_tokens': 0,
            'sum_base': 0.0,
            'sum_rewired': 0.0,
            'sum_delta': 0.0,
            'sum_abs_delta': 0.0,
            'n_pos': 0,
            'n_neg': 0,
            'top_pos': [],   # min-heap of (delta, item_id, pos, ctx_str)
            'top_neg': [],   # min-heap of (-delta, item_id, pos, ctx_str)
        }
        for s in survivors
    }

    captured_act: dict[int, torch.Tensor] = {}

    def _make_hook(layer_id: int):
        idx_t = idx_tensors[layer_id]

        def _hook(module: rm.RewireMLP, inputs, _output):
            gate, up = module.pre_activations(inputs[0])
            captured_act[layer_id] = (module.act_fn(gate) * up)[0, :, idx_t].detach().to(torch.float64)

        return _hook

    hooks = [mod_by_layer[L].register_forward_hook(_make_hook(L)) for L in by_layer]
    prev_gates = [m.gate for m in mods]
    rm.set_gate(model, None)  # deterministic z_det during rewired pass
    was_training = model.training
    model.eval()
    try:
        for item in val_items:
            input_ids = item['input_ids']
            labels = item['labels']
            # Supervised completion positions (predicting labels[t+1] != -100)
            comp_pos = [t for t in range(len(labels) - 1) if int(labels[t + 1]) != -100]
            if not comp_pos:
                continue
            x_in = torch.tensor([input_ids], dtype=torch.long, device=device)
            pos_t = torch.tensor(comp_pos, dtype=torch.long, device=device)

            # Pass 1: fully base model (all RewireMLP modules OFF)
            captured_act.clear()
            with rm.off(model):
                _ = model(input_ids=x_in, use_cache=False)
            base_acts = {L: captured_act[L][pos_t].clone() for L in by_layer}

            # Pass 2: fully rewired model (deterministic gates ON)
            captured_act.clear()
            _ = model(input_ids=x_in, use_cache=False)
            rew_acts = {L: captured_act[L][pos_t] for L in by_layer}

            for L, idxs in by_layer.items():
                a_base_cpu = base_acts[L].cpu()
                a_rew_cpu = rew_acts[L].cpu()
                delta = a_rew_cpu - a_base_cpu

                for col_k, n_idx in enumerate(idxs):
                    st = stats[(L, n_idx)]
                    d_col = delta[:, col_k]
                    st['n_tokens'] += int(d_col.numel())
                    st['sum_base'] += float(a_base_cpu[:, col_k].sum().item())
                    st['sum_rewired'] += float(a_rew_cpu[:, col_k].sum().item())
                    st['sum_delta'] += float(d_col.sum().item())
                    st['sum_abs_delta'] += float(d_col.abs().sum().item())
                    st['n_pos'] += int((d_col > 0).sum().item())
                    st['n_neg'] += int((d_col < 0).sum().item())

                    max_val, max_i = torch.max(d_col, dim=0)
                    min_val, min_i = torch.min(d_col, dim=0)
                    for is_pos, val, local_i in [(True, float(max_val), int(max_i)),
                                                 (False, float(min_val), int(min_i))]:
                        heap = st['top_pos'] if is_pos else st['top_neg']
                        score = val if is_pos else -val
                        if len(heap) < top_contexts or score > heap[0][0]:
                            t_abs = comp_pos[local_i]
                            lo = max(0, t_abs - context_window)
                            hi = min(len(input_ids), t_abs + 4)
                            ctx = tokenizer.decode(input_ids[lo:t_abs + 1]) + ' ▶ ' + tokenizer.decode(input_ids[t_abs + 1:hi])
                            entry = (score, str(item.get('id', '')), int(t_abs), ctx)
                            if len(heap) < top_contexts:
                                heapq.heappush(heap, entry)
                            else:
                                heapq.heapreplace(heap, entry)
    finally:
        for h in hooks:
            h.remove()
        for m, g in zip(mods, prev_gates):
            m.gate = g
        model.train(was_training)

    enriched: list[dict[str, Any]] = []
    for s in survivors:
        st = stats[(int(s['layer']), int(s['idx']))]
        n_tok = max(st['n_tokens'], 1)
        mean_delta = st['sum_delta'] / n_tok
        enriched.append({
            **s,
            'activation_shift': {
                'n_tokens': st['n_tokens'],
                'base_mean': st['sum_base'] / n_tok,
                'rewired_mean': st['sum_rewired'] / n_tok,
                'mean_delta': mean_delta,
                'mean_abs_delta': st['sum_abs_delta'] / n_tok,
                'frac_pos': st['n_pos'] / n_tok,
                'frac_neg': st['n_neg'] / n_tok,
                'active_pole': '+v' if mean_delta >= 0.0 else '-v',
                'top_pos_contexts': [
                    {'delta': round(sc, 4), 'item_id': iid, 'pos': pos, 'context': ctx}
                    for sc, iid, pos, ctx in sorted(st['top_pos'], reverse=True)
                ],
                'top_neg_contexts': [
                    {'delta': round(-sc, 4), 'item_id': iid, 'pos': pos, 'context': ctx}
                    for sc, iid, pos, ctx in sorted(st['top_neg'], reverse=True)
                ],
            },
        })
    return enriched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Export sparse-rewiring survivors & paired-forward Δa_j shifts')
    parser.add_argument('--model-dir', required=True, help='local pinned Qwen2.5-1.5B-Instruct directory')
    parser.add_argument('--arm-dir', default=None, help='verified completed arm directory (recommended)')
    parser.add_argument('--ckpt-dir', default=None, help='sealed checkpoint directory (if --arm-dir omitted)')
    parser.add_argument('--rank', type=int, choices=[0, 1], default=None)
    parser.add_argument('--val-items', required=True, help='JSON file of validation items (val128 or val128-k2)')
    parser.add_argument('--arm-name', required=True, help='human label, e.g. sparse-k2-sft-r1-matched-s42')
    parser.add_argument('--out', required=True, help='output compact JSON path')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-items', type=int, default=None, help='optional cap for smoke tests')
    args = parser.parse_args(argv)

    raw_val = json.loads(Path(args.val_items).read_text(encoding='utf-8'))
    val_items = raw_val['items'] if isinstance(raw_val, dict) and 'items' in raw_val else raw_val
    if args.max_items is not None:
        val_items = val_items[:args.max_items]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.float32, attn_implementation='eager'
    )
    mods, provenance = _load_verified_final_checkpoint(
        model,
        arm_dir=Path(args.arm_dir) if args.arm_dir else None,
        ckpt_dir=Path(args.ckpt_dir) if args.ckpt_dir else None,
        rank=args.rank,
        val_items_path=Path(args.val_items),
        allow_unverified_fixture=False,
    )
    from pilot_eval.training import file_hash
    cfg_file = Path(args.model_dir) / 'config.json'
    if cfg_file.is_file():
        provenance['model_config_sha256'] = file_hash(cfg_file)
    model.to(args.device)

    survivors = _extract_survivors(mods)
    enriched = measure_activation_shifts(model, mods, survivors, val_items, tokenizer)

    payload = {
        'schema': 'sparse-rewiring/survivor-shifts-v2',
        'arm': args.arm_name,
        'rank': provenance['rank'],
        'provenance': provenance,
        'val_items_path': str(args.val_items),
        'n_val_items': len(val_items),
        'survivor_count': len(enriched),
        'survivors': enriched,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Wrote {out_path} ({len(enriched)} survivors, {out_path.stat().st_size / 1024:.1f} KB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
