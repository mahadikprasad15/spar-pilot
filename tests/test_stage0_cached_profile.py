"""A small performance probe must not silently start production or renew spend."""
from pathlib import Path
import runpy
import json

import pytest


def profile_helpers():
    return runpy.run_path(str(Path(__file__).parents[1] / 'scripts/stage0_cached_profile.py'))


def test_work_is_four_singleton_reference_answers_at_original_seeds():
    helpers = profile_helpers()
    rows = [dict(id=f'q{i}', prompt=f'prompt {i}') for i in range(150)]
    work = helpers['make_work'](rows, ['q0', 'q1'] + [f'q{i}' for i in range(2, 16)])
    assert [(w['variant'], w['row']['id'], w['seed']) for w in work] == [
        ('V0a', 'q0', 42), ('V0a', 'q1', 43), ('V0b', 'q0', 42), ('V0b', 'q1', 43)]
    assert len(work) == 4
    assert helpers['reserve_units'](1.6, 600) == pytest.approx(0.2666666667)
    with pytest.raises(ValueError):
        helpers['make_work'](rows, ['missing', 'q1'])


def test_summary_uses_single_completion_tokens_and_never_forecasts_all_variants():
    summarize = profile_helpers()['summarize']
    rows = [dict(variant=v, seconds=10., generated_token_ids=list(range(100)))
            for v in ['V0a', 'V0a', 'V0b', 'V0b']]
    result = summarize(rows, 35.)
    assert result['answers'] == 4
    assert result['generated_tokens'] == 400
    assert result['generation_seconds'] == 40
    assert result['aggregate_tokens_per_second'] == 10
    assert result['model_load_seconds'] == 35
    assert result['production_authorized'] is False
    assert result['all_variants_eta_seconds'] is None


@pytest.mark.parametrize('fail_second', [False, True])
def test_profile_saves_answers_enforces_reservation_and_preserves_failures(tmp_path, monkeypatch, fail_second):
    from test_stage0_workflow import prepared
    from pilot_eval.stage0_budget import reservations
    audit = prepared(tmp_path)
    (tmp_path / 'edit.json').write_text(json.dumps({'spectral_path': 'spectral.json'}))
    (tmp_path / 'spectral.json').write_text(json.dumps({'audit_path': str(audit.relative_to(tmp_path))}))
    helpers = profile_helpers()
    run = helpers['run']
    runtime = dict(execution_kind='synthetic-cpu-fixture', session_id='fixture-session')
    q = dict(sequence_ids=['q0', 'q1'], variants={'V0a': {}, 'V0b': {}},
             weights_dir='weights', model_identity={}, decoding={'max_new_tokens': 1024},
             edits_path='edit.json')
    rows = [dict(id=f'q{i}', prompt=f'question {i}', gold='#### 2') for i in range(2)]
    monkeypatch.setitem(run.__globals__, 'load_sources',
                        lambda *args: (q, {'runtime': runtime}, rows, 'source-hash', 'diagnostic-hash'))

    class Fake:
        def __init__(self): self.calls = []; self.closed = False
        def runtime(self): return runtime
        def engine(self, config, root):
            assert config['decoding']['use_cache'] is True
            assert config['decoding']['max_new_tokens'] == 1024
            return self
        def prepare_inputs(self, items): return items
        def generate(self, items, variant, seed):
            assert len(items) == 1
            if fail_second and self.calls: raise RuntimeError('injected engine failure')
            self.calls.append((variant, items[0]['id'], seed))
            return [dict(text='2', token_ids=[2], generated_token_ids=[2, 3], stop_reason='eos')]
        def check_integrity(self, full=False):
            return dict(base_unchanged=True, adapters_unchanged=True)
        def close(self): self.closed = True

    engine = Fake()
    args = (tmp_path, tmp_path / 'q.json', 'probe', 1.6, 600., 5.)
    directory = tmp_path / 'runs/stage0/probe/cached-profile'
    if fail_second:
        with pytest.raises(RuntimeError, match='injected'):
            run(*args, dependencies=engine)
        assert len(list((directory / 'answers').glob('*.json'))) == 1
        assert not (directory / 'complete.json').exists()
        assert json.loads((directory / 'meta/status.json').read_text())['state'] == 'failed'
    else:
        assert run(*args, dependencies=engine) == directory
        assert len(list((directory / 'answers').glob('*.json'))) == 4
        assert len((directory / 'responses.jsonl').read_text().splitlines()) == 4
        assert json.loads((directory / 'summary.json').read_text())['production_authorized'] is False
        assert (directory / 'complete.json').is_file()
    assert engine.closed
    allocations = list(reservations(tmp_path).values())
    assert len(allocations) == 1 and allocations[0]['state'] == 'active'
    assert allocations[0]['reserved_units'] == pytest.approx(0.2666666667)
    with pytest.raises(ValueError, match='already exists'):
        run(*args, dependencies=engine)
    assert len(reservations(tmp_path)) == 1
