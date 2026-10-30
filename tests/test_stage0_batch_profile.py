"""Batch profiling is bounded evidence, not automatic batch qualification."""
from pathlib import Path
import runpy
import json
from types import SimpleNamespace

import pytest


def helpers():
    return runpy.run_path(str(Path(__file__).parents[1] / 'scripts/stage0_batch_profile.py'))


def test_batches_reuse_singleton_baseline_and_stop_at_first_memory_failure():
    profile = helpers()['profile_cases']
    rows = [dict(id='q0', prompt='first', gold='2'), dict(id='q1', prompt='second', gold='3')]
    baseline = [dict(id=r['id'], variant=v, seconds=1., generated_token_ids=[i + 2])
                for v in ['V0a', 'V0b'] for i, r in enumerate(rows)]
    writes = {}
    probes = []

    class Engine:
        def __init__(self): self.calls = []
        def generate(self, batch, variant, seed):
            self.calls.append((len(batch), variant, [r['id'] for r in batch]))
            return [dict(text='answer', generated_token_ids=[2 if r['id'] == 'q0' else 3]) for r in batch]

    def memory(size):
        probes.append(size)
        return {'fits': size < 4}

    engine = Engine()
    result = profile(engine, rows, baseline, [1, 2, 4, 8],
                     lambda path, value: writes.update({path: value}), memory)
    assert probes == [1, 2, 4]
    assert engine.calls == [(2, 'V0a', ['q0', 'q1']), (2, 'V0b', ['q0', 'q1'])]
    assert [r['batch_size'] for r in result] == [1, 2, 4]
    assert result[1]['exact_singleton_token_agreement'] is True
    assert result[1]['production_authorized'] is False
    assert result[2]['memory']['fits'] is False
    assert 'b2-V0a-answers.json' in writes


def test_disagreement_is_saved_without_claiming_qualification():
    profile = helpers()['profile_cases']
    rows = [dict(id='q0', prompt='first', gold='2'), dict(id='q1', prompt='second', gold='3')]
    baseline = [dict(id=r['id'], variant=v, seconds=1., generated_token_ids=[2])
                for v in ['V0a', 'V0b'] for r in rows]

    class Engine:
        def generate(self, batch, variant, seed):
            return [dict(generated_token_ids=[999]) for _ in batch]

    writes = {}
    result = profile(Engine(), rows, baseline, [2],
                     lambda path, value: writes.update({path: value}), lambda size: {'fits': True})
    assert result[0]['exact_singleton_token_agreement'] is False
    assert result[0]['production_authorized'] is False
    assert writes['b2-V0a-answers.json'][0]['matches_saved_singleton'] is False


@pytest.mark.parametrize('settle_failed', [False, True])
def test_saved_workflow_settles_only_baseline_reserves_and_writes_raw_outputs(tmp_path, monkeypatch, settle_failed):
    from test_stage0_workflow import prepared
    from pilot_eval.stage0_workflow import record_budget
    from pilot_eval.stage0_budget import reserve, reservations, settle
    from pilot_eval.workflow import _hash
    from pilot_eval.training import file_hash
    from pilot_eval import stage0_evaluation_backend as backend
    helpers = globals()['helpers']()
    main = helpers['main']
    root = tmp_path
    audit = prepared(root)
    usage = root / 'initial-usage.json'
    usage.write_text(json.dumps(dict(event_id='start', session_id='fixture', total_units=5.,
                                    observed_at='2026-10-29T00:00:00+00:00', note='fixture')))
    record_budget(audit, root, usage)
    runtime = dict(execution_kind='synthetic-cpu-fixture', session_id='fixture')
    base = root / 'baseline'
    (base / 'answers').mkdir(parents=True)
    config = dict(runtime=runtime, decoding=dict(use_cache=True, max_new_tokens=1024),
                  qualification_sha256='qhash')
    (base / 'config.json').write_text(json.dumps(config))
    cohort = [dict(id=f'q{i}', prompt=f'prompt{i}', gold='2') for i in range(2)]
    for i, (variant, row) in enumerate((v, r) for v in ['V0a', 'V0b'] for r in cohort):
        (base / 'answers' / f'{i}.json').write_text(json.dumps(dict(
            variant=variant, id=row['id'], generated_token_ids=[2], seconds=1.)))
    (base / 'complete.json').write_text(json.dumps(dict(config_sha256=_hash(config), files={
        str(p.relative_to(base)): file_hash(p) for p in (base / 'answers').glob('*.json')})))
    approval = dict(reviewer='Prasad', notes='fixture', session_id='fixture',
                    max_units_per_hour=1.6, max_seconds=600., source_reviewed=True)
    previous = reserve(root, _hash(config), approval)
    (base / 'reservation.json').write_text(json.dumps(previous))
    failed_dir = None
    failed_reservation = None
    if settle_failed:
        usage.write_text(json.dumps(dict(event_id='after-singleton', session_id='fixture', total_units=5.05,
                                        observed_at='2026-10-29T00:05:00+00:00', note='fixture')))
        record_budget(audit, root, usage)
        settle(root, previous['reservation_id'])
        failed_dir = root / 'failed-profile'
        (failed_dir / 'meta').mkdir(parents=True)
        failed_config = dict(protocol='stage0-batch-performance-profile-v1', runtime=runtime)
        failed_reservation = reserve(root, _hash(failed_config), approval)
        (failed_dir / 'config.json').write_text(json.dumps(failed_config))
        (failed_dir / 'reservation.json').write_text(json.dumps(failed_reservation))
        (failed_dir / 'meta/status.json').write_text(json.dumps(dict(state='failed', completed=0)))
    (root / 'edits.json').write_text(json.dumps({'spectral_path': 'spectral.json'}))
    (root / 'spectral.json').write_text(json.dumps({'audit_path': str(audit.relative_to(root))}))
    q = dict(sequence_ids=['q0', 'q1'], variants={'V0a': {}, 'V0b': {}}, edits_path='edits.json',
             decoding={'max_new_tokens': 1024})
    shared = dict(reserve_units=lambda rate, seconds: rate * seconds / 3600,
                  load_sources=lambda *args: (q, {'runtime': runtime}, cohort, 'qhash', 'dhash'))
    monkeypatch.setitem(main.__globals__, 'runpy', SimpleNamespace(run_path=lambda _: shared))

    class Engine:
        def __init__(self): self.closed = False; self.inputs = {}; self.loads = 0
        def runtime(self): return runtime
        def engine(self, config, root): self.loads += 1; return self
        def prepare_inputs(self, rows):
            self.inputs = {r['id']: [1, 2] for r in rows}
            return rows
        def generate(self, rows, variant, seed):
            return [dict(text='2', generated_token_ids=[2]) for _ in rows]
        def check_integrity(self, full=False): return dict(base_unchanged=True, adapters_unchanged=True)
        def close(self): self.closed = True

    engine = Engine()
    monkeypatch.setattr(backend, 'EvaluationDependencies', lambda: engine)
    helper_path = root / 'helper.py'
    helper_path.write_text('# tiny fixture helper')
    args = SimpleNamespace(repo_dir=str(root), cached_helper=str(helper_path), output_root=str(root), name='batches',
                           rate=1.6, seconds=600., baseline=str(base), qualification='q.json',
                           settle_baseline=True, observed_units=5.1, batch_sizes=[1, 2])
    args.settle_previous_profile = str(failed_dir) if failed_dir else None
    main(args, memory_probe=lambda *args: {'fits': True})
    directory = root / 'runs/stage0/batches/batch-profile'
    assert engine.closed and engine.loads == 1
    assert len((directory / 'responses.jsonl').read_text().splitlines()) == 4
    product = json.loads((directory / 'complete.json').read_text())
    assert 'responses.jsonl' in product['files']
    assert 'writer.json' not in product['files']
    states = reservations(root)
    assert states[previous['reservation_id']]['state'] == 'settled'
    if failed_reservation:
        assert states[failed_reservation['reservation_id']]['state'] == 'settled'
    active = [v for v in states.values() if v['state'] == 'active']
    assert len(active) == 1 and active[0]['reserved_units'] == pytest.approx(0.2666666667)
    with pytest.raises(ValueError, match='Profile exists'):
        main(args, memory_probe=lambda *args: {'fits': True})
    assert engine.loads == 1
