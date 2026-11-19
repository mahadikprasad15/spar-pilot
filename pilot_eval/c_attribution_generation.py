"""C-attribution greedy generation-ablation stage (ticket 07). Exploratory; greedy only.

Conditions per arm come only from a sealed freeze: intact, the four target groups and their matched
controls, plus a second intact pass (the nondeterminism floor) and, once, the untuned base. Adapter and
read removals are never combined. A condition whose control is unavailable is recorded as skipped.

Every condition and arm uses one fixed question order and one fixed batch membership. Each (arm,
condition, batch) is a sealed unit holding ``responses.jsonl`` and ``config.json``; a sealed unit is
verified and reused on resume, never regenerated. The restored LoRA hash is checked after every condition.
"""
import json
import math
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from pilot_eval import attribution_ablation as ab
from pilot_eval import attribution_units as au
from pilot_eval.attribution_instrument import frozen_disable_adapter
from pilot_eval.c_attribution_selection import CONTROL, GENERATION_DIR, read_freeze
from pilot_eval.c_attribution_stability import BOOTSTRAP_SEED, RESAMPLES, resample_counts
from pilot_eval.scoring import score_gsm8k_flexible_v3

EOS = 151645
CAP = 1024
LABELS = ('exploratory: components were selected on these questions',
          'greedy accuracy is a different target from the sampled J used for attribution',
          'predicted effects are local first-order sampled-J approximations')
ORDER = ('intact', 'top10-adapters', 'match10-adapters', 'global64-reads', 'match-global64',
         'hot64-reads', 'match-hot64', 'layer64-reads', 'match-layer64', 'intact-repeat')


class GenerationError(RuntimeError):
    pass


# --- conditions and batches ---------------------------------------------------------------------

def conditions(freeze):
    """The ten per-arm conditions (nine plus the intact repeat) from a freeze, with skips recorded."""
    groups = freeze['groups']
    out = [{'name': 'intact', 'kind': 'none', 'status': 'run', 'reason': None}]
    for target in ('top10-adapters', 'global64-reads', 'hot64-reads', 'layer64-reads'):
        g = groups[target]
        out.append(_condition(target, g['kind'], g['targets'], g))
        control = g['control']
        if control['available']:
            out.append(_condition(CONTROL[target], g['kind'], control['group'], control))
        else:
            out.append({'name': CONTROL[target], 'kind': g['kind'], 'status': 'skipped',
                        'reason': 'matched-control-unavailable',
                        'deficient_strata': control.get('deficient_strata'),
                        'deficient_targets': control.get('deficient_targets')})
    out.append({'name': 'intact-repeat', 'kind': 'none', 'status': 'run', 'reason': None})
    if [c['name'] for c in out] != list(ORDER):
        raise GenerationError('condition order differs from the spec')
    return out


def _condition(name, kind, members, record):
    c = {'name': name, 'kind': kind, 'status': 'run', 'reason': None, 'prediction': record.get('prediction')}
    if kind == 'adapters':
        c['adapters'] = [int(i) for i in members]
    else:
        c['reads'] = record['reads']
    if not members:
        c.update(status='skipped', reason='empty-group')
    return c


def untuned_condition():
    """The untuned base (adapter disabled), generated once in the same batches."""
    return {'name': 'untuned', 'kind': 'none', 'status': 'run', 'reason': None}


def batches(question_ids, batch_size):
    """One fixed membership: consecutive chunks of the fixed question order."""
    ids = list(question_ids)
    if len(set(ids)) != len(ids) or batch_size < 1:
        raise GenerationError('unique question ids and a positive batch size are required')
    return [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]


def intervention(model, condition):
    kind = condition['kind']
    if condition['name'] == 'untuned':
        return frozen_disable_adapter(model)
    if kind == 'none':
        return ExitStack()
    if kind == 'adapters':
        return ab.adapter_zeroing(model, condition['adapters'])
    if kind == 'reads':
        return ab.neuron_read_removal(model, {int(l): js for l, js in condition['reads'].items()})
    raise GenerationError(f'unknown condition kind {kind!r}')


# --- greedy generation ----------------------------------------------------------------------------

GREEDY_ARGS = dict(do_sample=False, num_beams=1, num_return_sequences=1, repetition_penalty=1.0,
                   no_repeat_ngram_size=0, use_cache=True)


def greedy(model, prompts, *, pad_token_id, eos_token_id=EOS, cap=CAP):
    """Greedy continuations for prompt-id lists: left padding, explicit arguments over generation_config."""
    import torch
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(eos_token_id=eos_token_id, pad_token_id=pad_token_id, **GREEDY_ARGS)
    device = next(model.parameters()).device
    width = max(len(p) for p in prompts)
    ids = torch.tensor([[pad_token_id] * (width - len(p)) + list(p) for p in prompts], device=device)
    attention = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=device)
    with torch.inference_mode():
        out = model.generate(input_ids=ids, attention_mask=attention, max_new_tokens=cap, eos_token_id=eos_token_id,
                             pad_token_id=pad_token_id, **GREEDY_ARGS)
    results = []
    for tokens in out[:, width:].tolist():
        end = next((i for i, t in enumerate(tokens) if t == eos_token_id), None)
        generated = tokens if end is None else tokens[:end + 1]
        results.append({'generated_token_ids': generated, 'content_token_ids': generated if end is None else generated[:-1],
                        'stop_reason': 'cap' if end is None else 'eos'})
    return results


def score(text, gold, capped):
    s = score_gsm8k_flexible_v3(text, gold, capped=capped)
    return {'flexible': s, 'reward': 0 if capped else int(s['correct']), 'invalid': s['status'] != 'valid'}


# --- units ----------------------------------------------------------------------------------------

def unit_dir(root, arm, condition, batch_index):
    return Path(root) / GENERATION_DIR / arm / condition / f'batch-{batch_index:04d}'


def run_condition(root, arm, condition, model, questions, plan, *, decode, pad_token_id, eos_token_id=EOS,
                  cap=CAP, adapter='default'):
    """Generate one condition over the fixed batches; sealed units are verified and reused.

    ``questions`` = {question_id: {'prompt_ids', 'prompt', 'gold'}}; ``plan`` = output of ``batches``.
    Returns per-unit status ('written' | 'reused' | 'skipped').
    """
    if condition['status'] != 'run':
        return {'status': 'skipped', 'reason': condition['reason']}
    before = ab.lora_hash(model, adapter)
    statuses = []
    for b, members in enumerate(plan):
        directory = unit_dir(root, arm, condition['name'], b)
        if (directory / 'complete.json').is_file():
            au.verify_run(directory)
            config = json.loads((directory / 'config.json').read_text())
            if config['question_ids'] != members:
                raise GenerationError(f'{directory}: sealed batch membership differs from the plan')
            statuses.append('reused')
            continue
        with intervention(model, condition):  # exactly one frozen group per unit
            outputs = greedy(model, [questions[q]['prompt_ids'] for q in members], pad_token_id=pad_token_id,
                             eos_token_id=eos_token_id, cap=cap)
        if ab.lora_hash(model, adapter) != before:  # checked after every unit, before it is sealed
            raise GenerationError(f"{arm} {condition['name']} batch {b}: adapter not restored")
        rows = []
        for q, o in zip(members, outputs):
            capped = o['stop_reason'] == 'cap'
            text = decode(o['content_token_ids'])
            rows.append({'arm': arm, 'condition': condition['name'], 'question_id': q,
                         'prompt': questions[q]['prompt'], 'generated_token_ids': o['generated_token_ids'],
                         'text': text, 'stop_reason': o['stop_reason'], 'capped': capped,
                         'n_tokens': len(o['generated_token_ids']), **score(text, questions[q]['gold'], capped)})
        config = {'arm': arm, 'condition': {k: v for k, v in condition.items() if k != 'prediction'},
                  'batch_index': b, 'question_ids': members, 'cap': cap, 'eos_token_id': eos_token_id,
                  'pad_token_id': pad_token_id, 'greedy_args': GREEDY_ARGS, 'padding': 'left',
                  'lora_hash_before': before, 'lora_hash_after': before}
        statuses.append(au.seal_run(directory, {'responses.jsonl': ''.join(json.dumps(r) + '\n' for r in rows),
                                                'config.json': config}))
    return {'status': 'run', 'units': statuses}


def run_arm(root, arm, model, questions, plan, **kw):
    """All conditions for one arm, from its sealed freeze; refuses to start without it, or with a batch plan
    other than the frozen one."""
    freeze = read_freeze(root, arm)
    if 'generation_plan' in freeze and freeze['generation_plan'] != [list(b) for b in plan]:
        raise GenerationError(f'{arm}: the batch plan differs from the frozen generation plan')
    return {c['name']: run_condition(root, arm, c, model, questions, plan, **kw) for c in conditions(freeze)}


def load_condition(root, arm, condition, plan):
    rows = {}
    for b, members in enumerate(plan):
        directory = unit_dir(root, arm, condition, b)
        au.verify_run(directory)
        for line in (directory / 'responses.jsonl').read_text().splitlines():
            r = json.loads(line)
            rows[r['question_id']] = r
        if set(members) - set(rows):
            raise GenerationError(f'{directory}: missing questions')
    return rows


# --- summary --------------------------------------------------------------------------------------

def mcnemar_exact(gains, losses):
    """Two-sided exact McNemar p from the discordant counts (binomial, p = 0.5)."""
    n = gains + losses
    if n == 0:
        return 1.0
    k = min(gains, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _interval(diff, counts):
    q = len(diff)
    values = counts @ np.asarray(diff, dtype=np.float64) / q
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {'estimate': float(np.mean(diff)), 'p2.5': float(lo), 'p97.5': float(hi)}


def summarize_arm(results, question_ids, conditions_, *, counts=None):
    """Per-condition summaries against the same arm's intact run. ``results`` = {condition: {qid: row}}."""
    ids = list(question_ids)
    counts = resample_counts(len(ids), RESAMPLES, BOOTSTRAP_SEED) if counts is None else counts
    reward = {c: np.array([results[c][q]['reward'] for q in ids], dtype=np.float64) for c in results}
    intact = reward['intact']
    acc_intact = float(intact.mean())
    out = {'labels': list(LABELS), 'questions': len(ids), 'conditions': {}}
    by_name = {c['name']: c for c in conditions_}
    for c in conditions_:
        name = c['name']
        if c['status'] != 'run':
            out['conditions'][name] = {'status': 'skipped', 'reason': c['reason'],
                                       **{k: c[k] for k in ('deficient_strata', 'deficient_targets') if k in c}}
            continue
        r = reward[name]
        rows = [results[name][q] for q in ids]
        tokens = [row['n_tokens'] for row in rows]
        gains = int(((r == 1) & (intact == 0)).sum())
        losses = int(((r == 0) & (intact == 1)).sum())
        entry = {'status': 'run', 'accuracy': float(r.mean()),
                 'delta_pp_vs_intact': {k: 100 * v for k, v in _interval(r - intact, counts).items()},
                 'percent_of_intact': 100 * float(r.mean()) / acc_intact if acc_intact > 0 else None,
                 'gains': gains, 'losses': losses, 'mcnemar_exact_p': mcnemar_exact(gains, losses),
                 'invalid': sum(row['invalid'] for row in rows), 'capped': sum(row['capped'] for row in rows),
                 'tokens_mean': float(np.mean(tokens)), 'tokens_median': float(np.median(tokens)),
                 'predicted_removal_effect': c.get('prediction')}
        control_name = CONTROL.get(name)
        if control_name and by_name.get(control_name, {}).get('status') == 'run':
            entry['target_minus_control_pp'] = {k: 100 * v for k, v in
                                                _interval(r - reward[control_name], counts).items()}
            entry['control_predicted_removal_effect'] = by_name[control_name].get('prediction')
        elif control_name:
            entry['target_minus_control_pp'] = None
            entry['target_minus_control_reason'] = 'matched-control-unavailable'
        out['conditions'][name] = entry
    if 'intact-repeat' in results:
        flips = [q for q in ids if results['intact'][q]['reward'] != results['intact-repeat'][q]['reward']
                 or results['intact'][q]['generated_token_ids'] != results['intact-repeat'][q]['generated_token_ids']]
        out['nondeterminism_floor'] = {'questions_differing': len(flips),
                                       'correctness_flips': sum(results['intact'][q]['reward'] != results['intact-repeat'][q]['reward']
                                                                for q in ids),
                                       'question_ids': flips}
    return out
