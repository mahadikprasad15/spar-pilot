"""Evidence-selected 128-plus-384 or fresh-512 pool route; CPU fixtures only."""
import json

import pytest

from test_stage0_rollouts import Sampling, sources
from pilot_eval import offline_grpo_fresh as fresh, offline_grpo_pool as pool, offline_grpo_route as route


Engine = Sampling


@pytest.fixture
def world(tmp_path, monkeypatch):
    audit, baseline, weights = sources(tmp_path)
    (weights/'config.json').write_text(json.dumps(dict(eos_token_id=99)))
    for name in ['tokenizer.json', 'tokenizer_config.json']:
        (weights/name).write_text('{}')
    rows = json.loads((tmp_path/'training.json').read_text())
    order = [r['id'] for r in rows]
    identity = {k: 'Qwen/Qwen2.5-1.5B-Instruct' if k == 'model' else 'pinned' if 'revision' in k else 'float32'
                for k in ['model', 'model_revision', 'tokenizer_revision', 'dtype']}
    prepared = dict(optimizer_prompt_order=order, **identity)
    monkeypatch.setattr(pool, '_prepared', lambda path, root: (prepared, rows, [dict(id='heldout')]))
    monkeypatch.setattr(pool, 'load_frozen', lambda path, root: dict(
        source_path='prepared.json', settings=dict(completion_limit=1024, top_p=1., top_k=0),
        optimizer_groups=[order[i:i+8] for i in range(0, 512, 8)]))
    monkeypatch.setattr(fresh, '_reservation', lambda *args: (dict(reservation_id='fixture'), 1000))
    (tmp_path/'prepared.json').write_text('{}'); (tmp_path/'frozen.json').write_text('{}')
    plan = pool.prepare(tmp_path/'frozen.json', tmp_path, 'example')
    runtime = tmp_path/'runtime.json'
    runtime.write_text(json.dumps(dict(execution_kind='synthetic-cpu-fixture', session_id='session-a', device='tiny CPU')))
    return dict(root=tmp_path, audit=audit, baseline=baseline, weights=weights, plan=plan, runtime=runtime, order=order)


def provider(*args):
    return dict(reservation_id='fixture'), 1000


def choose(world, engine):
    return route.select(world['plan'], world['audit'], world['baseline'], world['weights'], world['runtime'], world['root'],
                        dependencies=engine, reservation_provider=provider)


def prepare_fresh(world, engine):
    return fresh.prepare(world['plan'], world['audit'], world['weights'], world['runtime'], world['root'], dependencies=engine)


def test_exact_replay_keeps_128_and_samples_only_the_remaining_384(world):
    engine = Engine()
    decision = choose(world, engine)
    assert decision['route'] == 'reuse-historical-128' and decision['comparison']['exact']
    assert len(engine.calls) == 2
    historical = decision['historical_order']
    assert engine.calls == [(historical[0], 42), (historical[1], 43)]
    plan = prepare_fresh(world, engine)
    config = json.loads(plan.read_text())
    remaining = [i for i in world['order'] if i not in historical]
    assert [u['id'] for u in config['units']] == remaining and len(remaining) == 384
    assert [u['seed'] for u in config['units'][:2]] == [42+128, 42+128+1]
    assert {u['origin'] for u in config['units']} == {'new-extension'}
    engine.calls.clear()
    assert fresh.run(plan, world['root'], dependencies=engine)['complete']
    assert [i for i, _ in engine.calls] == remaining
    pool.import_rollouts(world['plan'], plan, world['root'])
    windows = pool.load_pool(world['plan'], world['root'])
    draws = [d for w in windows for q in w for d in q['draws']]
    assert len(draws) == 4096 and len({d['draw_id'] for d in draws}) == 4096
    assert sum(d['origin'] == 'historical' for d in draws) == 1024
    assert sum(d['origin'] == 'new-extension' for d in draws) == 3072
    assert [[q['id'] for q in w] for w in windows] == [world['order'][i:i+8] for i in range(0, 512, 8)]
    receipt = json.loads((world['root']/'runs/offline-grpo/example/pool/handoff.json').read_text())
    assert receipt['metadata']['route'] == 'reuse-historical-128'


def test_replay_mismatch_falls_back_to_all_512_and_keeps_evidence(world):
    engine = Engine(mismatch=True)
    decision = choose(world, engine)
    assert decision['route'] == 'uniform-fresh-512' and not decision['comparison']['exact']
    assert decision['comparison']['differences']
    assert (world['root']/'runs/offline-grpo/example/pool/route/replay.jsonl').exists()
    config = json.loads(prepare_fresh(world, engine).read_text())
    assert len(config['units']) == 512 and {u['origin'] for u in config['units']} == {'new-uniform'}
    assert config['route']['route'] == 'uniform-fresh-512'


def test_missing_raw_group_selects_fresh_without_loading_a_model(world):
    (world['root']/'runs/historical/batches/0001.json').unlink()
    engine = Engine()
    decision = choose(world, engine)
    assert decision['route'] == 'uniform-fresh-512' and decision['comparison'] is None
    assert engine.loads == 0 and any('incomplete' in reason for reason in decision['reasons'])


def test_different_runtime_selects_fresh_without_sampling(world):
    world['runtime'].write_text(json.dumps(dict(execution_kind='synthetic-cpu-fixture', session_id='session-a', device='other GPU')))
    class Other(Engine):
        def runtime(self):return dict(execution_kind='synthetic-cpu-fixture', session_id='session-a', device='other GPU')
    engine = Other()
    decision = choose(world, engine)
    assert decision['route'] == 'uniform-fresh-512' and engine.loads == 0
    assert any('GPU type' in reason for reason in decision['reasons'])


def test_actual_runtime_must_match_the_approved_one(world):
    with pytest.raises(ValueError, match='approved runtime'):
        choose(world, Engine(session='another'))


def test_decision_is_reused_on_resume_and_tamper_is_rejected(world):
    engine = Engine()
    first = choose(world, engine)
    assert choose(world, engine) == first and len(engine.calls) == 2
    plan = prepare_fresh(world, engine)
    fresh._load(plan, world['root'])
    folder = world['root']/'runs/offline-grpo/example/pool/route'
    saved = (folder/'decision.json').read_text()
    (folder/'decision.json').write_text(saved.replace('reuse-historical-128', 'uniform-fresh-512'))
    with pytest.raises(ValueError, match='Route decision changed'):
        fresh._load(plan, world['root'])
    with pytest.raises(ValueError, match='Route decision changed'):
        choose(world, engine)
    (folder/'decision.json').write_text(saved)
    fresh._load(plan, world['root'])
    (folder/'historical.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='Historical pool snapshot changed'):
        route.load_decision(world['root']/'runs/offline-grpo/example/pool', world['root'])


def test_failed_base_check_during_replay_stops_without_a_decision(world):
    class FailedBase(Engine):
        def check_integrity(self, *, full=False):
            return dict(base_unchanged=not full)
    with pytest.raises(ValueError, match='integrity'):
        choose(world, FailedBase())
    assert route.load_decision(world['root']/'runs/offline-grpo/example/pool', world['root']) is None


def test_replay_evidence_is_bound_to_the_decision(world):
    choose(world, Engine())
    directory = world['root']/'runs/offline-grpo/example/pool'
    replay = directory/'route/replay.jsonl'
    saved = replay.read_bytes()
    replay.unlink()
    with pytest.raises(ValueError, match='Replay evidence changed'):
        route.load_decision(directory, world['root'])
    replay.write_bytes(saved)
    route.load_decision(directory, world['root'])
    final = next((directory/'route/attempts').glob('*/final-integrity.json'))
    final.write_text(json.dumps(dict(base_unchanged=False)))
    with pytest.raises(ValueError, match='Replay evidence changed'):
        route.load_decision(directory, world['root'])


def test_units_from_an_attempt_that_failed_its_integrity_check_are_not_reused(world):
    class Failing(Engine):
        checks = 0
        def check_integrity(self, *, full=False):
            type(self).checks += 1
            return dict(base_unchanged=type(self).checks < 4)
    plan = prepare_fresh(world, Engine(mismatch=True))
    with pytest.raises(ValueError, match='integrity'):
        fresh.run(plan, world['root'], dependencies=Failing())
    sealed = list((world['root']/'runs/offline-grpo/example/pool/fresh/units').glob('*/complete.json'))
    assert sealed
    with pytest.raises(ValueError, match='invalidates'):
        fresh.run(plan, world['root'], dependencies=Engine())
