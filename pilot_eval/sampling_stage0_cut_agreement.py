"""Batch-versus-singleton agreement for principal-cut variants (the original ADR 0014 protocol).

As in the original Stage 0 batch16 qualification: every variant gets teacher-forced
FP32 answer-position logits and the cached decode step compared with singleton
inference at the full 16 and tail 6 shapes, within the unchanged 1e-3 gate; two
principal output cuts (first and last arm, as V1/V5 were for Gold/GRPO) get
left-padded batched greedy replay against fresh singleton references, exact or a
measured near tie. A variant batches only if its numeric units pass and its arm's
replay (where one exists) passes; anything else generates one question at a time.
"""
import json
from pathlib import Path

from pilot_eval.run import _write_json, _write_state
from pilot_eval.sampling_stage0_agreement import replay, token_rows
from pilot_eval.sampling_stage0_cuts import GROUP, IntegrityFailure, mark_integrity_failure
from pilot_eval.stage0_batching import BATCH_GATE, numeric_ok
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash, _save_frozen

SIZES = (GROUP, 150 % GROUP) # 16 and the tail 6


def replay_variants(variants):
    arms = list(dict.fromkeys(v['source_arm'] for v in variants.values()))
    return [f'{arms[0]}-output', f'{arms[-1]}-output'] if len(arms) > 1 else [f'{arms[0]}-output']


def configuration(sources, items):
    config = dict(protocol='sampling-cut-batch-agreement-v1', sources_sha256=_hash(sources),
        variants=sources['variants'], variant_order=sources['variant_order'], decoding=sources['decoding'],
        replay_variants=replay_variants(sources['variants']), sequence_ids=[i['id'] for i in items[:GROUP]],
        sizes=list(SIZES), batch_gate=BATCH_GATE, seed=42,
        decision='per variant: numeric pass and, if its arm was replayed, that replay passes; otherwise singleton')
    config['units'] = units(config)
    return config


def units(config):
    """Variant-major order, so each adapter is switched in once."""
    ids = config['sequence_ids']; result = []
    for variant in config['variant_order']:
        result += [dict(key=f'numeric-b{s}-{variant}', kind='numeric', variant=variant, ids=ids[:s]) for s in config['sizes']]
        if variant in config['replay_variants']:
            result += [dict(key=f'reference-{variant}-{i:02d}', kind='reference', variant=variant, ids=[q]) for i, q in enumerate(ids)]
            result += [dict(key=f'greedy-b{s}-{variant}', kind='greedy', variant=variant, ids=ids[:s]) for s in config['sizes']]
    return result


def assess(config, values):
    by_key = {u['key']: v for u, v in zip(config['units'], values)}; variants = {}; arms = {}
    for replayed in config['replay_variants']:
        checks = [c for s in config['sizes'] for c in by_key[f'greedy-b{s}-{replayed}']['checks']]
        arms[config['variants'][replayed]['source_arm']] = dict(variant=replayed,
            passed=all(by_key[f'greedy-b{s}-{replayed}']['acceptable'] for s in config['sizes']),
            exact_answers=sum(c['exact'] for c in checks), answers=len(checks),
            near_ties=sum(not c['exact'] and c['near_tie'] for c in checks))
    for variant in config['variant_order']:
        numeric = [by_key[f'numeric-b{s}-{variant}'] for s in config['sizes']]
        arm = config['variants'][variant]['source_arm']
        passed = all(numeric_ok(v, config['batch_gate']) for v in numeric)
        variants[variant] = dict(numeric_passed=passed, max_absolute_error=max(v['max_absolute_error'] for v in numeric),
            arm_replay_passed=arms.get(arm, {}).get('passed'),
            batching_qualified=passed and arms.get(arm, {}).get('passed', True))
    return dict(variants=variants, replays=arms, batch_gate=config['batch_gate'], sizes=config['sizes'],
        batch_sizes={v: GROUP if variants[v]['batching_qualified'] else 1 for v in config['variant_order']})


def _sealed(config, directory, unit):
    folder = directory/'units'/unit['key']
    if not (folder/'complete.json').exists():return None
    if json.loads((folder/'complete.json').read_text()) != dict(config_sha256=_hash(config), unit=unit,
            runtime_sha256=file_hash(directory/'meta/runtime.json'), sha256=file_hash(folder/'result.json')):
        raise ValueError(f'Changed agreement unit: {unit["key"]}')
    return json.loads((folder/'result.json').read_text())


def run(config, items, directory, runtime, engine):
    """Collect every sealed agreement unit; caller owns the lock, deadline and engine lifetime."""
    import time
    from pilot_eval.sampling_stage0_evaluation import environment, no_prior_integrity_failure, require_intact
    directory = Path(directory); no_prior_integrity_failure(directory)
    env = environment(runtime); runtime_path = directory/'meta/runtime.json'
    for path, value, label in [(directory/'config.json', config, 'config'), (runtime_path, env, 'runtime')]:
        if path.exists() and json.loads(path.read_text()) != value:raise ValueError(f'Resume {label} differs from saved {label}.')
        _save_frozen(path, value)
    if (directory/'complete.json').exists():return verify(config, directory)
    by_id = {i['id']: i for i in items}; selected = [by_id[q] for q in config['sequence_ids']]
    engine.prepare_inputs(selected); tokens = token_rows(engine, selected)
    _save_frozen(directory/'inputs/token-rows.json', tokens)
    values = [_sealed(config, directory, u) for u in config['units']]
    try:
        for i, unit in enumerate(config['units']):
            if values[i] is not None:continue
            rows = [by_id[q] for q in unit['ids']]; begin = time.monotonic()
            print(f'Cut batch agreement {i+1}/{len(values)}: {unit["key"]}', flush=True)
            if unit['kind'] == 'reference':
                response = engine.generate(rows, unit['variant'], config['seed'])
                if len(response) != 1:raise ValueError('Wrong singleton response count.')
                value = dict(id=unit['ids'][0], **response[0])
            elif unit['kind'] == 'numeric':
                value = engine.compare(rows, unit['variant'], tokens); numeric_ok(value, config['batch_gate'])
            else:
                references = {u['ids'][0]: values[j] for j, u in enumerate(config['units'])
                              if u['kind'] == 'reference' and u['variant'] == unit['variant']}
                responses = engine.generate(rows, unit['variant'], config['seed'])
                if len(responses) != len(rows):raise ValueError('Wrong batched response count.')
                value = replay(config, unit, rows, responses, references, engine)
            value['seconds'] = time.monotonic()-begin
            require_intact(directory, engine.check_integrity(), message='Model integrity failed during cut batch agreement.')
            folder = directory/'units'/unit['key']; _write_json(folder/'result.json', value)
            _write_json(folder/'complete.json', dict(config_sha256=_hash(config), unit=unit,
                runtime_sha256=file_hash(runtime_path), sha256=file_hash(folder/'result.json')))
            values[i] = value; _write_state(directory, 'running', sum(v is not None for v in values), len(values))
    except IntegrityFailure as exc:
        mark_integrity_failure(directory, exc); raise
    proof = require_intact(directory, engine.check_integrity(full=True), message='Final model integrity failed.')
    _write_json(directory/'meta/final-integrity.json', proof)
    result = {**assess(config, values), 'environment': env}
    _write_json(directory/'results/agreement.json', result)
    files = ['config.json', 'meta/runtime.json', 'meta/final-integrity.json', 'inputs/token-rows.json',
             'results/agreement.json'] + [f'units/{u["key"]}/complete.json' for u in config['units']]
    _write_json(directory/'complete.json', dict(config_sha256=_hash(config), files={f: file_hash(directory/f) for f in files}))
    _write_state(directory, 'complete', len(values), len(values))
    return result


def verify(config, directory):
    from pilot_eval.sampling_stage0_evaluation import no_prior_integrity_failure
    directory = Path(directory); no_prior_integrity_failure(directory)
    if json.loads((directory/'config.json').read_text()) != config:raise ValueError('Cut agreement config changed.')
    seal = json.loads((directory/'complete.json').read_text())
    if seal['config_sha256'] != _hash(config):raise ValueError('Cut agreement completion identity changed.')
    for name, digest in seal['files'].items():
        if file_hash(directory/name) != digest:raise ValueError(f'Cut agreement evidence changed: {name}')
    values = [_sealed(config, directory, u) for u in config['units']]
    if any(v is None for v in values):raise ValueError('Cut agreement evidence is incomplete.')
    result = json.loads((directory/'results/agreement.json').read_text())
    if {k: v for k, v in result.items() if k != 'environment'} != json.loads(json.dumps(assess(config, values))):
        raise ValueError('Cut agreement summary differs from sealed units.')
    if result['environment'] != json.loads((directory/'meta/runtime.json').read_text()):
        raise ValueError('Cut agreement environment evidence changed.')
    return result
