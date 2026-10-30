"""Matched-128 historical cohort for E0.6a/E0.5 (ADR 0016); CPU only, no model."""
import json
from pathlib import Path

from pilot_eval.run import _write_json
from pilot_eval.sft import safe_name
from pilot_eval.stage0_evaluation import _jsonl
from pilot_eval.stage0_numeric import agrees, dependency_fingerprint
from pilot_eval.stage0_rollout_math import cohorts
from pilot_eval.stage0_rollout_plan import validate_training_membership
from pilot_eval.stage0_rollout_sources import read_sources
from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_workflow import _writer, _state, _verify_parents
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash, _save_frozen

PROTOCOL = 'stage0-matched-cohort-v1'
COHORT = 'matched-128-historical'
FILES = ['config.json', 'results/rollouts.jsonl', 'results/gold.jsonl', 'results/datasets.json', 'inputs/special-tokens.json']


def implementation():
    folder = Path(__file__).parent
    return dependency_fingerprint(['stage0_matched.py'], folder)


def _order(draws, order):
    index = {identifier: n for n, identifier in enumerate(order)}
    return sorted(draws, key=lambda r: (index[r['group_id']], r['draw_index']))


def prepare(audit_path, baseline_path, weights_dir, output_root, name, *, dependencies=None):
    root = Path(output_root).resolve(); safe_name(name)
    source = read_sources(audit_path, baseline_path, root)
    if source['state'] != 'audited':
        raise ValueError('Historical raw groups/runtime are incomplete; restore the missing files listed by the R0 source audit.')
    if source['policy_matches'] is not True:
        # The own_all gate and the temperature-1 premise depend on the sampling distribution.
        raise ValueError('Historical sampling settings differ from temperature 1, top-k 0, top-p 1, cap 1024, one question per group.')
    if dependencies is None:
        from pilot_eval.stage0_weights import LocalWeights
        dependencies = LocalWeights()
    weights = inside(root, weights_dir)
    inventory = dependencies.inventory(weights, source['source_audit']['sources']['sft'])
    fixture = inventory['execution_kind'] == 'synthetic-cpu-fixture'; parents = dict(source['parents'])
    for value, digest in inventory['parents'].items():
        parents[str(inside(root, value).relative_to(root))] = digest
    if inventory['base_sha256'] != source['identity']['base_sha256']:
        raise ValueError('Selected weights differ from the audited untuned base.')
    if not fixture:
        validate_training_membership(source, root)
        for asset in ['tokenizer.json', 'tokenizer_config.json']:
            parents[str((weights / asset).relative_to(root))] = file_hash(weights / asset)
    runtime = source['legacy_runtime']
    special = dict(eos_token_id=runtime.get('eos_token_id'), pad_token_id=runtime.get('pad_token_id'))
    if any(type(v) is not int for v in special.values()):
        raise ValueError('Historical sampling runtime must record EOS and pad token IDs.')
    draws = _order(source['historical'], source['training_order'])
    ids = {r['group_id'] for r in draws}
    if len(ids) != 128 or len(draws) != 1024:
        raise ValueError('Expected the 128 verified historical eight-draw groups.')
    training = [r for r in source['training'] if r['id'] in ids]
    datasets = cohorts(draws, source['training_order'], selection_seed=42)
    directory = root / f'runs/stage0/{name}/cohort'; path = root / 'plans' / name / 'stage0.cohort.json'
    config = dict(protocol=PROTOCOL, schema_version=1, name=name, run_path=str(directory.relative_to(root)), cohort=COHORT,
        audit_path=str(inside(root, audit_path).relative_to(root)), baseline_path=str(inside(root, baseline_path).relative_to(root)),
        parents=parents, model_identity=source['identity'], weights_dir=str(weights.relative_to(root)),
        execution_kind='synthetic-cpu-fixture' if fixture else 'real-cuda', implementation=implementation(),
        legacy=source['legacy'], sampled_runtime=runtime, training_order=source['training_order'],
        question_ids=[r['id'] for r in training], counts=datasets['counts'], selection_seed=42, special=special,
        scope='The 128 training questions with original untuned eight-draw groups; matched across gold and own datasets (ADR 0016).')
    with _writer(directory):
        _save_frozen(path, config); _save_frozen(directory / 'config.json', config)
        _jsonl(directory / 'results/rollouts.jsonl', draws); _jsonl(directory / 'results/gold.jsonl', training)
        _write_json(directory / 'results/datasets.json', datasets); _write_json(directory / 'inputs/special-tokens.json', special)
        _write_json(directory / 'complete.json', dict(config_sha256=_hash(config), files={f: file_hash(directory / f) for f in FILES}))
        _state(directory, 'complete', 128, 128)
    return path


def load(path, root):
    root = Path(root).resolve(); path = inside(root, path); config = json.loads(path.read_text())
    directory = inside(root, config['run_path'])
    if config.get('protocol') != PROTOCOL or config.get('implementation') != implementation():
        raise ValueError('Matched cohort code changed; prepare a new named cohort.')
    if json.loads((directory / 'config.json').read_text()) != config:
        raise ValueError('Matched cohort plan/run identity changed.')
    seal = json.loads((directory / 'complete.json').read_text())
    if seal != dict(config_sha256=_hash(config), files={f: file_hash(directory / f) for f in FILES}):
        raise ValueError('Matched cohort payload changed.')
    _verify_parents(root, config['parents'])
    read = lambda name: [json.loads(x) for x in (directory / name).read_text().splitlines()]
    draws, training = read('results/rollouts.jsonl'), read('results/gold.jsonl')
    datasets = json.loads((directory / 'results/datasets.json').read_text())
    if (not agrees(datasets, json.loads(json.dumps(cohorts(draws, config['training_order'], selection_seed=config['selection_seed']))))
            or [r['id'] for r in training] != config['question_ids'] or {r['group_id'] for r in draws} != set(config['question_ids'])):
        raise ValueError('Matched cohort memberships do not reconstruct from its sealed draws.')
    return dict(kind='matched', cohort=COHORT, plan_path=path, config=config, directory=directory,
        complete_path=directory / 'complete.json', training=training, draws=draws, datasets=datasets,
        special=json.loads((directory / 'inputs/special-tokens.json').read_text()), sampled_runtime=config['sampled_runtime'])


def verify(path, output_root):
    cohort = load(path, output_root)
    return dict(cohort=cohort['cohort'], counts=cohort['config']['counts'], verified=True)
