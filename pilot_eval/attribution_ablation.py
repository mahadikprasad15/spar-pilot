"""Ticket 04: ablation engine and condition builder (spec §Selection and analysis freeze, §Stage C, §Stage D).

- **Removal** sets the gate g = 0 on a set of adapters through a forward hook on ``lora_B`` (the scaled branch
  s·lora_B(...) then contributes 0). ``edit_checkpoint`` writes an edited PEFT checkpoint with the chosen b
  zeroed (hashed); removal equals that checkpoint and Stage D generates from it.
- **Natural 64-term read** at a ``down_proj`` adapter: c_t -> Σ_{j in keep} a_j z_{j,t}, no rescaling.
- **Constant read**: c_t -> c̄ at every position (right padding: padded positions cannot affect real ones).
- **Matched controls** per target: same module type, discovery write within ±tolerance, sampled without
  replacement with a seeded generator from adapters outside both metrics' top-10 sets; the group control is
  the union; an unavailable match is recorded and never loosened; a group with any unavailable member is
  unavailable and its condition is a recorded skip.
- **Freeze**: top-10 / top-5 by |mean attribution| (ties: layer, then module order), controls, the top
  ``down_proj`` adapter, its neuron orders and c̄, plus the analysis settings; built only from discovery
  inputs and sealed. ``conditions`` reads only a sealed freeze.
- **Stage C**: ΔL, ΔS and KL(full arm ‖ condition) per example against the *same arm unablated*, FP32
  forwards, logits only at the read positions in chunks.
"""
import hashlib
import json
from contextlib import contextmanager, ExitStack
from pathlib import Path

import numpy as np

from pilot_eval import attribution_metrics as am
from pilot_eval import attribution_units as au
from pilot_eval.attribution_instrument import PROJECTIONS, adapter_modules

FREEZE_DIR = 'attribution/freeze'
METRICS = ('trace', 'stop')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def _hooks(registrations):
    handles = []
    try:
        for module, kind, fn in registrations:
            handles.append(getattr(module, kind)(fn))
        yield
    finally:
        for h in handles:
            h.remove()


# --- interventions -----------------------------------------------------------------------------

def removal(model, indices, adapter='default'):
    """Gate g = 0 on the chosen adapters' branch (``lora_B`` output x 0)."""
    modules = adapter_modules(model)
    return _hooks([(modules[i][2].lora_B[adapter], 'register_forward_hook', lambda m, a, o: o * 0.0)
                   for i in indices])


def natural_read(model, layer, keep, adapter='default'):
    """Replace the ``down_proj`` read with its natural sub-read Σ_{j in keep} a_j z_j (no rescaling)."""
    import torch
    module = model.get_base_model().model.layers[layer].mlp.down_proj
    keep = torch.tensor(sorted(int(j) for j in keep), dtype=torch.long)
    if len(set(keep.tolist())) != len(keep):
        raise ValueError('neuron indices must be unique')

    def hook(m, args, output):
        x = args[0]
        a = m.weight[0]
        index = keep.to(x.device)
        return (x[..., index] * a[index]).sum(-1, keepdim=True) if len(index) else output * 0.0

    return _hooks([(module.lora_A[adapter], 'register_forward_hook', hook)])


def constant_read(model, index, cbar, adapter='default'):
    """Replace adapter ``index``'s read with the constant c̄ at every position."""
    module = adapter_modules(model)[index][2]
    return _hooks([(module.lora_A[adapter], 'register_forward_hook',
                    lambda m, a, o: o * 0.0 + float(cbar))])


def condition_context(model, condition, adapter='default'):
    kind = condition['kind']
    if kind == 'removal':
        return removal(model, condition['adapters'], adapter)
    if kind == 'read':
        return natural_read(model, condition['layer'], condition['keep'], adapter)
    if kind == 'constant':
        return constant_read(model, condition['adapter'], condition['cbar'], adapter)
    raise ValueError(f'unknown condition kind {kind!r}')


def _key(layer, projection):
    group = 'self_attn' if projection in PROJECTIONS[:4] else 'mlp'
    return f'base_model.model.model.layers.{layer}.{group}.{projection}.lora_B.weight'


def edit_checkpoint(adapter_dir, out_dir, adapters):
    """Copy a PEFT checkpoint with the chosen adapters' b zeroed; returns hashes and zeroed keys."""
    from safetensors.numpy import load_file, save_file
    adapter_dir, out_dir = Path(adapter_dir), Path(out_dir)
    tensors = load_file(str(adapter_dir / 'adapter_model.safetensors'))
    keys = [_key(l, p) for l, p in adapters]
    missing = [k for k in keys if k not in tensors]
    if missing or len(set(keys)) != len(keys):
        raise ValueError(f'adapters not in the checkpoint or repeated: {missing[:3]}')
    edited = {k: (np.zeros_like(v) if k in keys else v) for k, v in tensors.items()}
    out_dir.mkdir(parents=True, exist_ok=False)
    save_file(edited, str(out_dir / 'adapter_model.safetensors'))
    (out_dir / 'adapter_config.json').write_bytes((adapter_dir / 'adapter_config.json').read_bytes())
    record = dict(source_sha256=_sha(adapter_dir / 'adapter_model.safetensors'),
                  edited_sha256=_sha(out_dir / 'adapter_model.safetensors'),
                  config_sha256=_sha(out_dir / 'adapter_config.json'), zeroed_keys=sorted(keys),
                  adapters=[list(a) for a in adapters])
    (out_dir / 'edit_record.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    return record


def verify_edit(out_dir, record):
    """Re-hash the edited checkpoint; return its hash or raise."""
    from safetensors.numpy import load_file
    out_dir = Path(out_dir)
    if _sha(out_dir / 'adapter_model.safetensors') != record['edited_sha256']:
        raise ValueError('edited checkpoint hash differs from its record')
    tensors = load_file(str(out_dir / 'adapter_model.safetensors'))
    if any(np.any(tensors[k]) for k in record['zeroed_keys']):
        raise ValueError('an edited b is not zero')
    return record['edited_sha256']


# --- selection -----------------------------------------------------------------------------------

def top_k(mean_attr, k):
    """Indices by |mean attribution| descending; ties by index (= layer, then module order)."""
    values = np.abs(np.asarray(mean_attr, dtype=np.float64))
    if not np.isfinite(values).all():
        raise ValueError('mean attributions must be finite')
    return sorted(range(len(values)), key=lambda i: (-values[i], i))[:k]


def discovery_write(sum_abs_c, token_total, b_norms, scale):
    """‖s b‖ · mean |c_t| over unpadded discovery positions, per adapter."""
    return np.abs(scale) * np.asarray(b_norms, dtype=np.float64) * np.asarray(sum_abs_c) / np.asarray(token_total)


def matched_controls(targets, write, projections, *, exclude, tolerance, seed):
    rng = np.random.default_rng(seed)
    used, per, unavailable = set(), {}, []
    for target in targets:
        eligible = [i for i in range(len(write)) if projections[i] == projections[target] and i not in exclude
                    and i not in used and abs(write[i] - write[target]) <= tolerance * abs(write[target])]
        if not eligible:
            per[str(target)] = None
            unavailable.append(target)
            continue
        choice = int(eligible[int(rng.integers(len(eligible)))])
        used.add(choice)
        per[str(target)] = choice
    group = None if unavailable else [per[str(t)] for t in targets]
    return dict(per_target=per, group=group, unavailable=unavailable, tolerance=tolerance, seed=seed,
                rule='same module type; |write - target| <= tolerance * target; without replacement; '
                     'outside both metrics\' top-10 sets; never loosened')


def neuron_orders(attr_j, *, attr_i, term_size, random_seeds):
    if not random_seeds or len(random_seeds) != 3 or len(set(random_seeds)) != 3:
        raise ValueError('three distinct random-order seeds must come from the launch record')
    attr_j = np.asarray(attr_j, dtype=np.float64)
    sign = 1.0 if attr_i >= 0 else -1.0
    size = np.asarray(term_size, dtype=np.float64)
    width = len(attr_j)
    return dict(attribution=sorted(range(width), key=lambda j: (-sign * attr_j[j], j)),
                term_size=sorted(range(width), key=lambda j: (-size[j], j)),
                random=[np.random.default_rng(s).permutation(width).tolist() for s in random_seeds],
                random_seeds=list(random_seeds))


def build_freeze(arms, *, sources, analysis, tolerance, seed, random_seeds, top_group, top_singles, read_size):
    """Selection and analysis freeze from discovery-only inputs (per arm: mean attributions, writes, reads)."""
    if not sources or any(s.get('split') != 'discovery' for s in sources):
        raise ValueError('the freeze is built from discovery inputs only')
    if not random_seeds:
        raise ValueError('random-order seeds must come from the launch record')
    out = dict(schema=f'{au.STUDY}/freeze-v1', sources=sources, analysis=analysis,
               settings=dict(tolerance=tolerance, seed=seed, random_seeds=list(random_seeds), top_group=top_group,
                             top_singles=top_singles, read_size=read_size), arms={})
    for arm, x in sorted(arms.items()):
        adapters = [tuple(a) for a in x['adapters']]
        projections = [p for _, p in adapters]
        tops = {m: top_k(x['mean_attr'][m], top_group) for m in METRICS}
        exclude = set(tops['trace']) | set(tops['stop'])
        downs = [i for i, (_, p) in enumerate(adapters) if p == 'down_proj']
        layers = [adapters[i][0] for i in downs]
        out['arms'][arm] = {}
        for m in METRICS:
            mean = np.asarray(x['mean_attr'][m], dtype=np.float64)
            down = max(downs, key=lambda i: (abs(mean[i]), -i))
            layer = adapters[down][0]
            position = layers.index(layer)
            out['arms'][arm][m] = dict(
                top10=tops[m], top5=tops[m][:top_singles],
                predicted={str(i): float(mean[i]) for i in tops[m]}, predicted_all=[float(v) for v in mean],
                controls=matched_controls(tops[m], np.asarray(x['write'], dtype=np.float64), projections,
                                          exclude=exclude, tolerance=tolerance, seed=seed),
                down_proj=dict(index=down, layer=layer, predicted=float(mean[down]),
                               cbar=float(x['signed_sum_c'][position] / x['token_total'][position]),
                               orders=neuron_orders(x['neuron'][m][layer], attr_i=float(mean[down]),
                                                    term_size=x['term_size'][layer], random_seeds=random_seeds)))
    return out


def seal_freeze(root, freeze):
    directory = Path(root) / FREEZE_DIR
    au.seal_run(directory, {'freeze.json': freeze})
    return directory


def conditions(freeze_dir, arm, metric):
    """Stage C conditions for one arm and metric, read only from a sealed freeze (≤ 19)."""
    au.verify_run(freeze_dir)
    freeze = json.loads((Path(freeze_dir) / 'freeze.json').read_text())
    if any(s.get('split') != 'discovery' for s in freeze['sources']):
        raise ValueError('freeze lists non-discovery inputs')
    sel = freeze['arms'][arm][metric]
    read = freeze['settings']['read_size']
    out = []

    def add(name, **condition):
        out.append(dict(name=name, status='run', reason=None, **condition))

    def skip(name, reason):
        out.append(dict(name=name, kind='removal', adapters=None, status='skipped', reason=reason))

    for r, target in enumerate(sel['top5'], 1):
        add(f'top{r}-{metric}', kind='removal', adapters=[target])
    for r, target in enumerate(sel['top5'], 1):
        control = sel['controls']['per_target'][str(target)]
        if control is None:
            skip(f'match{r}-{metric}', 'matched-control-unavailable')
        else:
            add(f'match{r}-{metric}', kind='removal', adapters=[control])
    add(f'top10-{metric}', kind='removal', adapters=list(sel['top10']))
    if sel['controls']['group'] is None:
        skip(f'match10-{metric}', 'matched-control-unavailable')
    else:
        add(f'match10-{metric}', kind='removal', adapters=list(sel['controls']['group']))
    down = sel['down_proj']
    if down['index'] not in sel['top5']:
        add(f'down-full-{metric}', kind='removal', adapters=[down['index']])
    orders = down['orders']
    for label, order in [('attr', orders['attribution']), ('size', orders['term_size'])] + \
            [(f'rand{k + 1}', o) for k, o in enumerate(orders['random'])]:
        add(f'read64-{label}-{metric}', kind='read', layer=down['layer'], keep=list(order[:read]))
    add(f'cbar-{metric}', kind='constant', adapter=down['index'], cbar=down['cbar'])
    if len(out) > 19:
        raise ValueError('more than 19 conditions per arm and metric')
    return out


# --- Stage C -------------------------------------------------------------------------------------

class StageC:
    """Per-example ΔL, ΔS and KL(full arm ‖ condition) on fixed rows; the reference is the same arm unablated."""

    def __init__(self, model, *, stop_ids, pad_token_id, head_chunk=1024, adapter='default'):
        from pilot_eval.attribution_instrument import Instrument
        import torch
        self.torch, self.model, self.adapter = torch, model, adapter
        self.inst = Instrument(model, stop_ids=stop_ids, pad_token_id=pad_token_id, identity_rtol=1.0,
                               adapter=adapter, head_chunk=head_chunk)
        self.stop_ids, self.chunk = list(stop_ids), head_chunk

    def _hidden(self, ids, attention, positions, context):
        torch = self.torch
        with context, torch.no_grad(), torch.autocast(ids.device.type, enabled=False):
            return self.inst.body(input_ids=ids, attention_mask=attention, position_ids=positions,
                                  use_cache=False).last_hidden_state

    def evaluate(self, rows, condition):
        torch = self.torch
        self.inst.check_routing()
        ids, attention, positions = self.inst.batch(rows)
        full = self._hidden(ids, attention, positions, ExitStack())
        cut = self._hidden(ids, attention, positions, condition_context(self.model, condition, self.adapter))
        out = dict(delta_trace=[], delta_stop=[], kl=[], trace=[], stop=[])
        with torch.no_grad():
            for r, row in enumerate(rows):
                length = len(row['input_ids'])
                trace = am.logit_positions(row['masks']['trace_targets'], length=length)
                stop = am.logit_positions(row['masks']['stop_targets'], length=length)
                positions_r = sorted(set(trace) | set(stop))
                where = {p: k for k, p in enumerate(positions_r)}
                f_terms, c_terms, kl = [], [], {}
                for s in range(0, len(positions_r), self.chunk):
                    index = torch.tensor(positions_r[s:s + self.chunk], device=ids.device)
                    targets = ids[r, index + 1]
                    lf, lc = self.inst.head(full[r, index]), self.inst.head(cut[r, index])
                    f_terms.append(am.position_terms(lf, targets, self.stop_ids))
                    c_terms.append(am.position_terms(lc, targets, self.stop_ids))
                    p, q = torch.log_softmax(lf, -1), torch.log_softmax(lc, -1)
                    for k, value in zip(index.tolist(), (p.exp() * (p - q)).double().sum(-1).tolist()):
                        kl[k] = value
                f, c = torch.cat(f_terms), torch.cat(c_terms)
                lt = [where[p] for p in trace]
                full_l, cut_l = f[lt, 0].double().mean(), c[lt, 0].double().mean()
                out['trace'].append(float(cut_l))
                out['delta_trace'].append(float(cut_l - full_l))
                out['kl'].append(float(np.mean([kl[p] for p in trace])))
                if stop:
                    ls = [where[p] for p in stop]
                    full_s, cut_s = f[ls, 1].double().mean(), c[ls, 1].double().mean()
                    out['stop'].append(float(cut_s))
                    out['delta_stop'].append(float(cut_s - full_s))
                else:
                    out['stop'].append(float('nan'))
                    out['delta_stop'].append(float('nan'))
        result = {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}
        result['example_ids'] = [r['id'] for r in rows]
        result['reference'] = 'same arm, unablated (full)'
        result['kl_direction'] = 'KL(full arm || condition)'
        return result
