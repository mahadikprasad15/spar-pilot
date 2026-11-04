"""Evidence-selected pool route: extend verified historical answers or sample all 512.

The historical 128 groups count only after the source audit, the recorded runtime and
an exact two-group replay all agree. Any other outcome records why and samples 512.
"""
import json
from pathlib import Path

from pilot_eval import offline_grpo_pool
from pilot_eval.offline_grpo_support import IntegrityFailure, attempt_scope, attempt_start
from pilot_eval.run import _write_json
from pilot_eval.stage0_evaluation import _jsonl, _reservation
from pilot_eval.stage0_rollout_math import normalize, replay_comparison
from pilot_eval.stage0_rollout_plan import replay_policy
from pilot_eval.stage0_rollout_sources import POLICY, read_sources
from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_workflow import _writer
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash, _save_frozen

FRESH = 'uniform-fresh-512'
REUSE = 'reuse-historical-128'


def load_decision(directory, root):
    """Return the sealed route decision in a pool directory, or None before selection."""
    folder = directory/'route'; path = folder/'decision.json'; seal = folder/'decision.complete.json'
    if not path.exists() and not seal.exists():
        return None
    if json.loads(seal.read_text()) != dict(sha256=file_hash(path)):
        raise ValueError('Route decision changed')
    decision = json.loads(path.read_text())
    if decision['implementation_sha256'] != file_hash(Path(__file__)):
        raise ValueError('Route implementation changed; prepare a new plan')
    if decision['route'] == REUSE and file_hash(folder/'historical.jsonl') != decision['historical_sha256']:
        raise ValueError('Historical pool snapshot changed')
    if decision['comparison'] is not None:
        _verify_replay(folder, decision)
    return decision


def _verify_replay(folder, decision):
    """A replay outcome counts only with its saved draws and passed integrity check."""
    attempt = folder/'attempts'/decision['replay_attempt']; replay = folder/'replay.jsonl'; final = attempt/'final-integrity.json'
    if (not replay.is_file() or not final.is_file() or file_hash(replay) != decision['replay_sha256']
            or file_hash(final) != decision['replay_integrity_sha256']
            or json.loads(final.read_text()).get('base_unchanged') is not True):
        raise ValueError('Replay evidence changed')
    try:
        attempt_start(attempt, ['base_unchanged'])
    except ValueError as exc:
        raise ValueError('Replay evidence changed') from exc
    if decision['route'] == REUSE:
        current = [json.loads(line) for line in replay.read_text().splitlines()]
        ids = list(dict.fromkeys(row['group_id'] for row in current))
        historical = [json.loads(line) for line in (folder/'historical.jsonl').read_text().splitlines()]
        reference = [row for identifier in ids for row in historical if row['group_id'] == identifier]
        if replay_comparison(reference, current) != decision['comparison'] or not decision['comparison']['exact']:
            raise ValueError('Replay evidence changed')


def historical_records(directory):
    return [json.loads(line) for line in (directory/'route/historical.jsonl').read_text().splitlines()]


def _source_reasons(source, pool):
    reasons = []
    if source['state'] != 'audited':
        reasons.append(f"Historical raw groups or runtime incomplete: {source['verified_historical_groups']} of 128 verified")
    if source['training_order'] != pool['optimizer_order']:
        reasons.append('Historical training order differs from the frozen optimizer order')
    if any(source['identity'][k] != pool['source_identity'][k] for k in pool['source_identity']):
        reasons.append('Historical model identity differs from the frozen training source')
    if type((source['legacy_runtime'] or {}).get('eos_token_id')) is not int:
        reasons.append('Historical EOS identity was not recorded')
    return reasons


def _replay(source, weights, runtime, policy, root, target, deps, reservation, remaining, by_id):
    config = dict(weights_dir=str(weights.relative_to(root)), model_identity=source['identity'], runtime=runtime,
                  generation_settings={**POLICY, 'seed': source['legacy']['settings']['seed']},
                  execution_kind=runtime['execution_kind'], run_path=str(target.relative_to(root)))
    with attempt_scope(target, reservation, runtime, remaining, real=runtime['execution_kind'] == 'real-cuda',
                       on_failure=lambda exc: None) as run:
        if hasattr(deps, 'preflight'):
            deps.preflight(config, root)
        engine = run.engine = deps.engine(config, root)
        eos = engine.special_tokens()['eos_token_id']
        if eos != source['legacy_runtime']['eos_token_id']:
            return None, eos, run.path
        rows = [by_id[u['id']] for u in policy['units']]
        if engine.prepare_inputs(rows) != [dict(id=r['id'], input_ids=r['input_ids'][:r['prompt_tokens']]) for r in rows]:
            raise ValueError('Replay prompt token identity differs from the frozen training source')
        current = []
        for unit, row in zip(policy['units'], rows):
            values = engine.sample([row], config['generation_settings'], unit['seed'])
            if len(values) != 8:
                raise ValueError('Expected eight replay draws')
            current.extend(normalize(row, v, n, origin='replay', seed=unit['seed'], limit=1024, eos=eos) for n, v in enumerate(values))
        integrity = engine.check_integrity(full=True)
        _write_json(run.path/'final-integrity.json', integrity)
        if integrity.get('base_unchanged') is not True:
            raise IntegrityFailure('Full base integrity failed during replay')
        return current, eos, run.path


def select(pool_path, audit_path, baseline_path, weights_dir, runtime_path, output_root, *,
           dependencies=None, approval_path=None, reservation_provider=None):
    from pilot_eval.stage0_rollout_backend import RolloutDependencies
    root = Path(output_root).resolve()
    pool, directory, training = offline_grpo_pool._plan(pool_path, root)
    existing = load_decision(directory, root)
    if existing is not None:
        return existing
    target = directory/'route'
    audit_path, baseline_path, runtime_path = [inside(root, p) for p in [audit_path, baseline_path, runtime_path]]
    weights = inside(root, weights_dir)
    source = read_sources(audit_path, baseline_path, root)
    reasons = _source_reasons(source, pool)
    comparison = None; policy = None; runtime = None; current = None
    deps = dependencies or RolloutDependencies()
    with _writer(target):
        if not reasons:
            runtime = deps.runtime()
            if runtime != json.loads(runtime_path.read_text()):
                raise ValueError('Actual runtime differs from the explicitly approved runtime')
            policy = replay_policy(dict(seeds_recovered=source['seeds_recovered'], policy_matches=source['policy_matches'],
                historical_runtime=source['legacy_runtime'], legacy=source['legacy'],
                historical_order=source['historical_order'], uniform_units=[]), runtime)
            reasons.extend(policy['reasons'])
        if not reasons:
            by_id = {r['id']: r for r in training}
            job = dict(protocol='offline-grpo-pool-route-v1', pool_sha256=file_hash(inside(root, pool_path)), units=policy['units'])
            execution = {'reservation': {'reservation_id': None, 'job_sha256': _hash(job)}}
            reservation, remaining = (reservation_provider or _reservation)(job, execution, target, root, runtime, approval_path)
            current, eos, attempt = _replay(source, weights, runtime, policy, root, target, deps, reservation, remaining, by_id)
            if current is None:
                reasons.append(f'Current tokenizer EOS {eos} differs from the recorded historical EOS')
            else:
                _jsonl(target/'replay.jsonl', current)
                reference = [r for u in policy['units'] for r in source['historical'] if r['group_id'] == u['id']]
                comparison = replay_comparison(reference, current)
                if not comparison['exact']:
                    reasons.append('Exact two-group replay differed from the recorded historical draws')
        chosen = FRESH if reasons else REUSE
        decision = dict(protocol='offline-grpo-pool-route-v1', route=chosen, reasons=reasons, comparison=comparison,
            runtime=runtime, pool_sha256=file_hash(inside(root, pool_path)), audit_path=str(audit_path.relative_to(root)),
            baseline_path=str(baseline_path.relative_to(root)), implementation_sha256=file_hash(Path(__file__)),
            verified_historical_groups=source['verified_historical_groups'], missing=source['missing'])
        if comparison is not None:
            decision.update(replay_attempt=attempt.name, replay_sha256=file_hash(target/'replay.jsonl'),
                            replay_integrity_sha256=file_hash(attempt/'final-integrity.json'))
        if chosen == REUSE:
            _jsonl(target/'historical.jsonl', source['historical'])
            decision.update(historical_order=source['historical_order'], historical_sha256=file_hash(target/'historical.jsonl'),
                            extension_seed_start=source['legacy']['settings']['seed']+128, parents=source['parents'])
        _save_frozen(target/'decision.json', decision)
        _save_frozen(target/'decision.complete.json', dict(sha256=file_hash(target/'decision.json')))
    print(f"Offline pool route: {chosen}" + (f"; {'; '.join(reasons)}" if reasons else ''), flush=True)
    return load_decision(directory, root)
