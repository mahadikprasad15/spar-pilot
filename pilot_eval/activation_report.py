"""CPU-only reconstruction and paired example uncertainty for Pilot 3."""
import json
from pathlib import Path

import numpy as np

from pilot_eval.activation_math import derive_measurements


def direction_resolution(relative_write, agreement):
    floor = agreement.get('resolution', {}).get('block_delta_vector')
    return {'direction_resolution_relative': floor,
            'direction_resolved': None if floor is None or relative_write is None else bool(relative_write > floor)}


def bootstrap_measurement(summary, weighting, draws, *, multiplicities=None):
    """Recompute primary nonlinear ratios; NaN marks undefined replicates only."""
    if weighting not in ('token', 'example'):
        raise ValueError('unknown weighting')
    count = np.asarray(summary['count'], dtype=np.float64)
    draws = np.asarray(draws)
    if draws.ndim != 2 or not np.issubdtype(draws.dtype, np.integer) or (draws < 0).any() or (draws >= len(count)).any():
        raise ValueError('invalid example bootstrap draws')
    if multiplicities is None:
        multiplicities = np.stack([np.bincount(row, minlength=len(count)) for row in draws]).astype(float)
    if multiplicities.shape != (len(draws), len(count)):
        raise ValueError('bootstrap multiplicity shape mismatch')
    nonempty = count > 0
    factor = np.ones_like(count) if weighting == 'token' else np.divide(
        1., count, out=np.zeros_like(count), where=nonempty)
    base = np.asarray(summary['base_norm_sum']) * factor
    delta = np.asarray(summary.get('delta_sum', summary['delta_norm_sum']))
    delta = delta * factor.reshape((-1,) + (1,) * (delta.ndim - 1))
    values = np.full(len(draws), np.nan)
    # Bound temporary resampled vector storage independently of hidden width.
    chunk_size = 256 if delta.ndim == 2 else max(1, len(draws))
    for start in range(0, len(draws), chunk_size):
        weights = multiplicities[start:start + chunk_size]
        denominator = weights @ base
        numerator = weights @ delta
        if delta.ndim == 2:
            numerator = np.linalg.norm(numerator, axis=1)
        if not np.isfinite(numerator).all() or not np.isfinite(denominator).all():
            raise ValueError('nonfinite bootstrap reduction')
        np.divide(numerator, denominator, out=values[start:start + len(weights)], where=denominator != 0)
    valid = values[np.isfinite(values)]
    return {'replicates': values, 'defined_replicates': len(valid),
            'undefined_replicates': int(len(values) - len(valid)),
            'interval_95': None if not len(valid) else np.quantile(valid, [.025, .975]).tolist()}


def _denominators(summary):
    count = summary['count']
    nonempty = count > 0
    values = np.divide(summary['base_norm_sum'], count, out=np.zeros_like(count), where=nonempty)[nonempty]
    return {'nonempty_examples': int(nonempty.sum()),
            'zero_mean_norm_examples': int((values == 0).sum()),
            'example_mean_norm_min': None if not len(values) else float(values.min()),
            'example_mean_norm_max': None if not len(values) else float(values.max()),
            'example_mean_norm_median': None if not len(values) else float(np.median(values))}


def report_activation(config_path, output_root, *, name):
    """Require verified complete evidence, then save an immutable scientific report."""
    import re
    from pilot_eval.activation_measurement import load_execution, verify_measurement, iter_measurement_batches
    from pilot_eval.activation_prepare import measurement_steps, VIEWS, PROJECTIONS
    from pilot_eval.activation_workflow import _save_arrays
    from pilot_eval.training import file_hash, run_lock
    from pilot_eval.run import _write_json, _write_state
    from pilot_eval.workflow import _hash, _save_frozen

    root = Path(output_root).resolve()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError('invalid report name')
    source = verify_measurement(config_path, root)
    execution, prepared, rows = load_execution(config_path, root)
    steps = measurement_steps(prepared)
    source_dir = root / execution['run_path']
    from importlib.metadata import version
    identity = {'report_implementation': {filename: file_hash(Path(__file__).with_name(filename))
                                         for filename in ['activation_report.py', 'activation_plots.py', 'activation_math.py']},
                'report_runtime': {'numpy': np.__version__, 'matplotlib': version('matplotlib')},
                'schema_version': 1, 'execution': execution, 'prepared': prepared,
                'measurement_complete_sha256': file_hash(source_dir / 'complete.json'),
                'analysis': {'replicates': 2000, 'seed': 42, 'interval': 'percentile-95',
                             'weightings': ['token', 'example'], 'resampling_unit': 'example',
                             'pairing': 'GSM8K joint views/checkpoints; FineWeb independent'}}
    directory = root / 'reports' / name
    with run_lock(directory):
        _save_frozen(directory / 'config.json', identity)
        marker_path = directory / 'complete.json'
        if marker_path.exists():
            marker = json.loads(marker_path.read_text())
            if marker['identity_sha256'] != _hash(identity):
                raise ValueError('report identity mismatch')
            for path, digest in marker['files'].items():
                target = (directory / path).resolve()
                if not target.is_relative_to(directory.resolve()) or file_hash(target) != digest:
                    raise ValueError('report payload hash mismatch')
            required = {'config.json', 'results/results.json', 'results/measurements.csv',
                        'results/mean-vectors.npz', 'results/bootstrap-draws.npz',
                        'results/bootstrap-primary.npz', 'results/validation.jsonl',
                        'results/validation-summary.json', 'results/report.md', 'plots/manifest.json'}
            required.update('plots/' + filename for filename in json.loads((directory / 'plots/manifest.json').read_text())['files'])
            if set(marker['files']) != required:
                raise ValueError('report completion payload inventory mismatch')
            return json.loads((directory / 'results/results.json').read_text())
        _write_state(directory, 'running', 0, len(steps))
        try:
            rng = np.random.default_rng(42)
            cohorts = {corpus: [i for i, row in enumerate(rows) if row['corpus'] == corpus]
                       for corpus in ['gsm8k', 'fineweb']}
            draws = {c: rng.integers(0, len(ids), (2000, len(ids))) for c, ids in cohorts.items()}
            weights = {c: np.stack([np.bincount(row, minlength=len(cohorts[c])) for row in values]).astype(float)
                       for c, values in draws.items()}
            _save_arrays(directory / 'results/bootstrap-draws.npz', draws)
            records, vectors, replicates, diagnostics = [], {}, {}, []
            for step in steps:
                chunks = list(iter_measurement_batches(config_path, root, step=step))
                arrays = {key: np.concatenate([chunk['arrays'][key] for chunk in chunks])
                          for key in chunks[0]['arrays']}
                for chunk in chunks:
                    for evidence in chunk['validation']['positions']:
                        diagnostics.append(dict(step=step, batch_index=chunk['batch_index'], **evidence))
                del chunks
                for view_index, view in enumerate(VIEWS):
                    corpus = 'fineweb' if view == 'user' else 'gsm8k'
                    indices = cohorts[corpus]
                    for layer in range(28):
                        for kind, projections in [('block', [None]), ('module', PROJECTIONS)]:
                            for projection in projections:
                                selection = (indices, view_index, layer) if kind == 'block' else (
                                    indices, view_index, layer, PROJECTIONS.index(projection))
                                fields = ['count', 'base_norm_sum', 'delta_norm_sum'] + (
                                    ['base_sum', 'delta_sum'] if kind == 'block' else ['ratio_sum', 'defined_count'])
                                summary = {field: arrays[f'{kind}_{field}'][selection] for field in fields}
                                for weighting in ['token', 'example']:
                                    key = f'{step}-{view}-{layer}-{projection or "block"}-{weighting}'
                                    derived = derive_measurements(summary, weighting)
                                    interval = bootstrap_measurement(summary, weighting, draws[corpus], multiplicities=weights[corpus])
                                    replicates[key] = interval.pop('replicates')
                                    if kind == 'block':
                                        for field in ['mean_base', 'mean_delta']:
                                            vectors[key + '-' + field] = np.asarray(derived.pop(field), dtype=float)
                                    records.append(dict(step=step, view=view, corpus=corpus, layer=layer,
                                                        projection=projection, kind=kind, weighting=weighting,
                                                        **derived, **interval,
                                                        **(direction_resolution(derived['relative_write'], execution['agreement']) if kind == 'block' else {}),
                                                        zero_write=derived['relative_write'] == 0,
                                                        undefined_write=derived['relative_write'] is None,
                                                        denominator_diagnostics=_denominators(summary)))
                del arrays
                _write_state(directory, 'running', steps.index(step) + 1, len(steps))
            _save_arrays(directory / 'results/mean-vectors.npz', vectors)
            # Undefined values are explicit masks in finite NPZ payloads.
            _save_arrays(directory / 'results/bootstrap-primary.npz', {
                suffix + key: value for key, values in replicates.items()
                for suffix, value in [('defined-', np.isfinite(values).astype(float)),
                                      ('value-', np.nan_to_num(values))]})
            result = {'schema_version': 1, 'report_complete': True, 'name': name,
                      'checkpoint_steps': steps, 'variant_labels': prepared.get('variant_labels', {}),
                      'source_run': execution['run_path'], 'source_completion': source,
                      'provenance': identity, 'measurements': records,
                      'bootstrap_cohort_ids': {c: [rows[i]['id'] for i in ids] for c, ids in cohorts.items()},
                      'limits': ['One training seed; intervals condition on frozen cohorts.',
                                 'FineWeb is a bounded convenience control, not a representative web sample.',
                                 'Contexts and lengths differ across views; fixed gold lengths are not generated lengths.',
                                 'Direction resolution flags use an empirical engineering envelope, not a statistical confidence bound.',
                                 'Geometry is descriptive, not causal evidence or a layer-placement recommendation.'],
                      'pinned_question': 'Do longer gold solutions dominate token-weighted write profiles, and does equal-example weighting change them?'}
            if prepared.get('control_note'):
                result['limits'].append(prepared['control_note'])
            _write_json(directory / 'results/results.json', result)
            with (directory / 'results/validation.jsonl').open('w') as handle:
                for item in diagnostics:
                    handle.write(json.dumps(item) + '\n')
            validation_summary = []
            for step in steps:
                for view in VIEWS:
                    evidence = [r for r in diagnostics if r['step'] == step and r['view'] == view]
                    coordinates = sum(r['coordinate_count'] for r in evidence)
                    below = sum(r['below_resolution_coordinates'] for r in evidence)
                    validation_summary.append(dict(step=step, view=view, sampled_coordinates=coordinates,
                        below_resolution_coordinates=below, below_resolution_fraction=None if not coordinates else below / coordinates,
                        **{key: max((r[key] for r in evidence), default=None) for key in
                           ['branch_max_error', 'subtraction_max_error', 'branch_max_fraction', 'subtraction_max_fraction']}))
            result['validation_summary'] = validation_summary
            _write_json(directory / 'results/results.json', result)
            _write_json(directory / 'results/validation-summary.json', validation_summary)
            _write_tables(directory, result, rows)
            from pilot_eval.activation_plots import plot_report
            plot_report(directory, result)
            # Reverify source after reporting; never bind an altered source to a report.
            verify_measurement(config_path, root)
            if file_hash(source_dir / 'complete.json') != identity['measurement_complete_sha256']:
                raise ValueError('measurement changed during reporting')
            files = [p for p in directory.rglob('*') if p.is_file()
                     and p.relative_to(directory).parts[0] in ['results', 'plots']] + [directory / 'config.json']
            _write_json(marker_path, {'identity_sha256': _hash(identity),
                                     'files': {str(p.relative_to(directory)): file_hash(p) for p in files}})
            _write_state(directory, 'completed', len(steps), len(steps))
            return result
        except Exception as error:
            _write_state(directory, 'failed', 0, len(steps), str(error))
            raise


def _write_tables(directory, result, rows):
    import csv
    fields = ['step', 'view', 'layer', 'projection', 'kind', 'weighting', 'relative_write',
              'mean_base_norm', 'mean_delta_norm', 'mean_base_vector_norm', 'relative_write_alternative',
              'mean_token_ratio', 'token_count', 'example_count', 'empty_example_count',
              'defined_token_count', 'undefined_token_count', 'ratio_example_count',
              'direction_resolved', 'direction_resolution_relative', 'defined_replicates', 'undefined_replicates', 'interval_95', 'zero_write', 'undefined_write', 'denominator_diagnostics']
    with (directory / 'results/measurements.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(result['measurements'])
    lengths = {view: [sum(r['masks'][view]) for r in rows if any(r['masks'][view])] for view in ['question', 'solution', 'user']}
    _write = ['# ' + result['name'], '', 'Complete verified forward-only measurements; no answer generation or scoring.', '',
              result['pinned_question'], '', '## Interpretation limits', ''] + ['- ' + text for text in result['limits']]
    if result.get('variant_labels'):
        _write += ['', '## Intervention labels', ''] + [f"- {key}: {label}" for key, label in result['variant_labels'].items()]
    _write += ['', '## Reconstruction', '', 'Primary block: norm of mean delta / mean baseline norm. Primary module: mean direct norm / mean ordinary norm.',
               'Token and equal-example weights are applied to both terms. Secondary module ratios condition on defined tokens.',
               '2,000 paired example bootstrap replicates, seed 42; percentile intervals use defined replicates only. Undefined values remain null.',
               '', '## Counted lengths', '', '| View | Examples | Min | Median | Max |', '|---|---:|---:|---:|---:|']
    _write += [f'| {view} | {len(values)} | {min(values)} | {np.median(values):g} | {max(values)} |' for view, values in lengths.items()]
    _write += ['', '## Sampled numerical validation', '',
               'Threshold fractions must be ≤1. Below-resolution coverage refers to sampled coordinates, not all measurement tokens.', '',
               '| Step | View | Branch max error | Subtraction max error | Max threshold fraction | Below resolution / sampled |',
               '|---:|---|---:|---:|---:|---:|']
    _write += [f"| {r['step']} | {r['view']} | {r['branch_max_error']:.3g} | {r['subtraction_max_error']:.3g} | {max(r['branch_max_fraction'], r['subtraction_max_fraction']):.3g} | {r['below_resolution_coordinates']} / {r['sampled_coordinates']} |"
               for r in result['validation_summary']]
    _write += ['', '## Evidence', '', '- `measurements.csv` and `results.json`: all ratios, coverage, intervals and denominator statistics.',
               '- `mean-vectors.npz`: mean untuned and change vectors, keyed by checkpoint/view/layer/weighting.',
               '- `bootstrap-draws.npz` and `bootstrap-primary.npz`: exact resampling draws, scalar values and defined masks.',
               '- `validation.jsonl`: sampled rank-1 errors, threshold fractions and below-resolution coordinate counts.',
               '- `plots/manifest.json`: scientific figures and scales; config.json binds source checkpoint/input identities.']
    (directory / 'results/report.md').write_text('\n'.join(_write) + '\n')
