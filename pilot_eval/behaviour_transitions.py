"""Paired answer outcomes from a verified, immutable saved-scoring bundle."""
import io
import json
import re
from pathlib import Path

import numpy as np

from pilot_eval.behaviour_breakdown import verify_behaviour, _jsonl
from pilot_eval.recovery import _inside, _name
from pilot_eval.rescoring import _save_bytes
from pilot_eval.run import _write_state
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _save_frozen


SCORERS = ('strict', 'flexible_v3', 'last_number')
BINS = ('C', 'W', 'U')
LIMITS = ('Answer outcomes are scorer-defined, not validated reasoning gains. '
          'Intervals condition on these trained checkpoints and observed responses; '
          'they do not measure training-seed variability or new-decoding uncertainty.')


def _bin(score):
    return 'C' if score['correct'] else ('W' if score['status'] == 'valid' else 'U')


def _bootstrap(directory, ids):
    """Persist sorted item order and actual draws, shared by all metrics on it."""
    import hashlib
    digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    relative = f'bootstrap/{digest}.npy'
    indices = np.random.default_rng(42).integers(0, len(ids), (10000, len(ids)), dtype=np.int32)
    buffer = io.BytesIO()
    np.save(buffer, indices, allow_pickle=False)
    _save_bytes(directory / relative, buffer.getvalue())
    _save_frozen(directory / f'bootstrap/{digest}.json', dict(item_ids=ids, draws=10000,
        seed=42, generator='numpy.default_rng/PCG64', numpy_version=np.__version__,
        indices_path=relative, indices_sha256=file_hash(directory / relative)))
    return indices, relative


def _difference(before, after, indices):
    delta = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    return dict(estimate=float(delta.mean()),
                interval_95=np.percentile(delta[indices].mean(axis=1), [2.5, 97.5]).tolist())


def _flow(before, after, scorer, ids, directory):
    cells = {a + ':' + b: [] for a in BINS for b in BINS}
    for item in ids:
        cells[_bin(before[item]['scores'][scorer]) + ':' + _bin(after[item]['scores'][scorer])].append(item)
    gains = len(cells['W:C']) + len(cells['U:C'])
    losses = len(cells['C:W']) + len(cells['C:U'])
    indices, path = _bootstrap(directory, ids)
    return dict(n=len(ids), item_ids=ids, bin_order=list(BINS), cell_ids=cells,
        counts=[[len(cells[a + ':' + b]) for b in BINS] for a in BINS],
        gross_gains=gains, gross_losses=losses, churn=gains + losses, net_change=gains - losses,
        net_wrong_flow=len(cells['W:C']) - len(cells['C:W']),
        net_invalid_flow=len(cells['U:C']) - len(cells['C:U']),
        difference=_difference([before[i]['scores'][scorer]['correct'] for i in ids],
                               [after[i]['scores'][scorer]['correct'] for i in ids], indices),
        resampling_path=path)


def _stage_config(parent_path, root, parent_config):
    directory = _inside(root, parent_config['run_path'])
    return dict(protocol='pilot4-answer-transitions-v1', parent_path=str(parent_path.relative_to(root)),
        parent_sha256=file_hash(parent_path), parent_seal_sha256=file_hash(directory / 'complete.json'),
        implementation_sha256=file_hash(Path(__file__)), numpy_version=np.__version__,
        bootstrap=dict(draws=10000, seed=42, percentiles=[2.5, 97.5], unit='paired question ID'))


def _unit(directory, name, compute):
    target = directory / 'units' / (name + '.json')
    marker = target.with_suffix('.complete.json')
    if marker.exists():
        if json.loads(marker.read_text()) != dict(sha256=file_hash(target)):
            raise ValueError('transition report unit changed')
        return json.loads(target.read_text())
    value = compute()
    _save_frozen(target, value)
    _save_frozen(marker, dict(sha256=file_hash(target)))
    return value


def _verify_files(directory):
    seal = json.loads((directory / 'complete.json').read_text())
    if seal.get('config_sha256') != file_hash(directory / 'config.json'):
        raise ValueError('transition report config changed')
    required = {'results/results.json', 'results/paired-items.jsonl', 'results/report.md',
                'plots/accuracy-differences.svg'}
    if not required.issubset(seal.get('files', {})):
        raise ValueError('transition report inventory incomplete')
    for relative, digest in seal['files'].items():
        if file_hash(_inside(directory, relative)) != digest:
            raise ValueError('transition report output changed')
    return json.loads((directory / 'results/results.json').read_text())


def _descriptive(rows):
    groups = {}
    for row in rows:
        if row['method'] != 'sft':
            continue
        for measure in ['gold_calculator_annotations', 'gold_nonempty_lines']:
            key = f'{measure}={row[measure]}'
            groups.setdefault(key, []).append(row)
    result = {}
    for key, selected in groups.items():
        result[key] = dict(n=len(selected), mean_tokens_0=float(np.mean([r['tokens_0'] for r in selected])),
            mean_tokens_64=float(np.mean([r['tokens_64'] for r in selected])),
            flexible_v3_cells={a + ':' + b: sum(r['bins']['flexible_v3'] == [a,b] for r in selected)
                               for a in BINS for b in BINS})
    return result


def _plot(result, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels, values, intervals = [], [], []
    for method in ['sft', 'grpo']:
        for scorer in SCORERS:
            row = result['within_method'][method][scorer]
            labels.append(f'{method.upper()} · {scorer}')
            values.append(row['difference']['estimate'] * 100)
            intervals.append([v * 100 for v in row['difference']['interval_95']])
    for scorer, row in result['direct_comparison'].items():
        if row.get('available'):
            labels.append('GRPO64 − SFT64 · ' + scorer)
            values.append(row['difference']['estimate'] * 100)
            intervals.append([v * 100 for v in row['difference']['interval_95']])
    with plt.rc_context({'svg.hashsalt':'pilot4-answer-transitions-v1', 'font.size':10}):
        fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
        for index, (value, limits) in enumerate(zip(values, intervals)):
            ax.plot(limits, [index, index], color='#647875', linewidth=2)
            ax.plot(value, index, 'o', color='#36798a' if value >= 0 else '#b46d59')
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.axvline(0, color='#b9c1bd', linewidth=1)
        ax.set_xlabel('Accuracy difference (percentage points); paired question bootstrap 95% interval')
        ax.set_title('Saved greedy endpoints: within-method changes and direct endpoint comparison')
        ax.spines[['top', 'right']].set_visible(False)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='svg', metadata={'Date':None})
        plt.close(fig)
    _save_bytes(directory / 'plots/accuracy-differences.svg', buffer.getvalue())


def _markdown(result):
    lines = ['# Paired greedy answer transitions', '', LIMITS, '',
        'Original endpoint denominators: ' + ', '.join(f'{k}={v}' for k,v in result['original_endpoint_totals'].items()) + '.',
        'Paired tables use the explicitly saved common IDs; excluded endpoint IDs are retained separately.', '',
        'Bins: C = correct, W = valid but wrong, U = scorer-invalid. Rows are checkpoint 0; columns checkpoint 64.', '',
        '| Method / scorer | N | Gross gains | Gross losses | Churn | Net | Net W flow | Net U flow | Change, 95% interval |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for method, scorers in result['within_method'].items():
        for scorer, flow in scorers.items():
            estimate = flow['difference']; lo, hi = estimate['interval_95']
            lines.append(f"| {method} / {scorer} | {flow['n']} | {flow['gross_gains']} | {flow['gross_losses']} | {flow['churn']} | {flow['net_change']} | {flow['net_wrong_flow']} | {flow['net_invalid_flow']} | {estimate['estimate']:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    for method, scorers in result['within_method'].items():
        for scorer, flow in scorers.items():
            lines += ['', f'## {method.upper()} · {scorer}', '', '| 0 → 64 | C | W | U |', '|---|---:|---:|---:|']
            for label, counts in zip(BINS, flow['counts']):
                lines.append('| ' + label + ' | ' + ' | '.join(map(str, counts)) + ' |')
    lines += ['', '## Direct paired GRPO64 − SFT64', '',
              'Uses explicitly listed common IDs. It does not compare two separate within-method intervals.']
    for scorer, row in result['direct_comparison'].items():
        lines.append(f"- {scorer}: N={row['n']}; " +
            (str(row['difference']) if row['available'] else 'blocked: ' + row['reason']))
    lines += ['', '## SFT descriptive strata', '',
        'Exact gold calculator-annotation counts and nonempty gold-solution lines are descriptive, not validated difficulty or causal measures.', '',
        '| Stratum | N | Mean response tokens 0 | Mean response tokens 64 |', '|---|---:|---:|---:|']
    for label, row in result['sft_descriptive'].items():
        lines.append(f"| {label} | {row['n']} | {row['mean_tokens_0']:.2f} | {row['mean_tokens_64']:.2f} |")
    lines += ['', 'All cell IDs, paired records, original/common cohorts and actual bootstrap draws are retained in this report bundle.', '',
              '![Accuracy differences](../plots/accuracy-differences.svg)', '']
    return '\n'.join(lines).encode()


def report_transitions(path, root, name, *, max_new_units=None):
    if max_new_units is not None and max_new_units < 1:
        raise ValueError('max-new-units must be positive')
    root = Path(root).resolve()
    path = Path(path).resolve()
    source_result = verify_behaviour(path, root)
    comparison = source_result['comparison']
    if not comparison['declared_settings_complete']:
        raise ValueError('paired transitions require complete declared input/decoding settings')
    for method in ['sft','grpo']:
        settings = comparison['declared_settings']
        if settings[method+':0'] != settings[method+':64']:
            raise ValueError('within-method endpoint input/decoding contracts differ: '+method)
    parent = json.loads(path.read_text())
    records = _jsonl(_inside(root, parent['run_path']) / 'results/responses.jsonl')
    directory = root / 'reports' / _name(name)
    config = _stage_config(path, root, parent)
    with run_lock(directory):
        _save_frozen(directory / 'config.json', config)
        _save_frozen(directory / 'meta/run_manifest.json', dict(stage='paired-answer-transitions', **config))
        if (directory / 'complete.json').exists():
            result = _verify_files(directory)
            _write_state(directory, 'completed', 9, 9)
            return result
        groups = {(m,s): {r['item_id']:r for r in records if r['method']==m and r['step']==s}
                  for m in ['sft','grpo'] for s in [0,64]}
        within, paired = {}, []
        new_units = 0
        for method in ['sft','grpo']:
            before, after = groups[method,0], groups[method,64]
            ids = sorted(set(before) & set(after))
            within[method] = {}
            for scorer in SCORERS:
                unit_name = method+'-'+scorer
                is_new = not (directory/'units'/(unit_name+'.complete.json')).exists()
                if is_new and max_new_units is not None and new_units >= max_new_units:
                    _write_state(directory, 'incomplete', sum(len(v) for v in within.values()), 9)
                    return dict(state='incomplete', execution_kind='cpu-saved-output-only')
                within[method][scorer] = _unit(directory, unit_name,
                    lambda scorer=scorer: _flow(before, after, scorer, ids, directory))
                new_units += int(is_new)
            for item in ids:
                gold = before[item]['gold']
                paired.append(dict(method=method, item_id=item, gold=gold,
                    gold_calculator_annotations=len(re.findall(r'<<[^<>]*>>', gold)),
                    gold_nonempty_lines=sum(bool(line.strip()) for line in gold.splitlines()),
                    tokens_0=before[item]['token_count'], tokens_64=after[item]['token_count'],
                    bins={s:[_bin(before[item]['scores'][s]), _bin(after[item]['scores'][s])] for s in SCORERS},
                    before=before[item], after=after[item]))
            _write_state(directory, 'running', len(within)*3, 9)
        direct = {}
        common = comparison['common_ids']
        for scorer in SCORERS:
            if not comparison['paired_comparison_eligible']:
                direct[scorer] = dict(available=False, n=len(common), item_ids=common,
                    reason='incompatible or missing declared endpoint settings, prompt/gold, or common IDs')
                continue
            unit_name = 'direct-'+scorer
            is_new = not (directory/'units'/(unit_name+'.complete.json')).exists()
            if is_new and max_new_units is not None and new_units >= max_new_units:
                _write_state(directory, 'incomplete', 6+len(direct), 9)
                return dict(state='incomplete', execution_kind='cpu-saved-output-only')
            indices, relative = _bootstrap(directory, common)
            direct[scorer] = _unit(directory, unit_name, lambda scorer=scorer: dict(available=True,
                n=len(common), item_ids=common, resampling_path=relative,
                cohort_label='common-ID endpoint cohort; original cohorts retained separately',
                difference=_difference([groups['sft',64][i]['scores'][scorer]['correct'] for i in common],
                                       [groups['grpo',64][i]['scores'][scorer]['correct'] for i in common], indices)))
            new_units += int(is_new)
        result = dict(within_method=within, direct_comparison=direct,
            source_comparison=comparison, bootstrap=config['bootstrap'], limitations=LIMITS,
            original_endpoint_totals={f'{m}:{s}':len(g) for (m,s),g in groups.items()},
            excluded_from_within_method={m:{str(s):sorted(set(groups[m,s])-(set(groups[m,0])&set(groups[m,64])))
                for s in [0,64]} for m in ['sft','grpo']},
            sft_descriptive=_descriptive(paired), parent_identity=config,
            execution_kind='cpu-saved-output-only')
        _save_frozen(directory / 'results/results.json', result)
        _save_bytes(directory / 'results/paired-items.jsonl', ''.join(json.dumps(r,sort_keys=True)+'\n' for r in paired).encode())
        _plot(result, directory)
        _save_bytes(directory / 'results/report.md', _markdown(result))
        files = {str(p.relative_to(directory)):file_hash(p) for folder in ['units','results','plots','bootstrap']
                 for p in (directory/folder).glob('*') if p.is_file()}
        _save_frozen(directory / 'complete.json', dict(config_sha256=file_hash(directory/'config.json'), files=files))
        _write_state(directory, 'completed', 9, 9)
        return result
