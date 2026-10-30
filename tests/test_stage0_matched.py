"""ADR 0016: matched-128 historical cohort for E0.6a and E0.5; CPU fixtures only."""
import json

import numpy as np
import pytest

from pilot_eval.training import file_hash


def mcmd(root, op, *args, dependencies=None):
    from pilot_eval.cli import main
    return main(['stage0-cohort-' + op, '--output-root', str(root), *map(str, args)], dependencies=dependencies)


def matched(root, name='matched'):
    from test_stage0_rollouts import Tiny
    assert mcmd(root, 'prepare', '--audit', root / 'plans/audit/stage0.plan.json',
                '--baseline', root / 'plans/baseline/baseline.config.json',
                '--weights-dir', root / 'weights', '--name', name, dependencies=Tiny()) == 0
    return root / 'plans' / name / 'stage0.cohort.json'


@pytest.fixture(scope='module')
def historical(tmp_path_factory):
    from test_stage0_rollouts import sources
    root = tmp_path_factory.mktemp('matched-sources'); sources(root, gold_eos=99)
    return root


@pytest.fixture
def source(tmp_path, historical):
    import shutil
    shutil.copytree(historical, tmp_path, dirs_exist_ok=True)
    return tmp_path


def test_matched_cohort_packages_only_the_verified_historical_questions(source):
    from pilot_eval.stage0_cohort_source import load_cohort
    plan = matched(source)
    cohort = load_cohort(plan, source)
    ids = {r['group_id'] for r in cohort['draws']}
    assert cohort['kind'] == 'matched' and cohort['cohort'] == 'matched-128-historical'
    assert len(ids) == 128 and len(cohort['draws']) == 1024
    assert {r['id'] for r in cohort['training']} == ids
    assert cohort['datasets']['counts']['questions'] == 128
    assert set(cohort['datasets']['gold']) == ids
    assert cohort['special'] == dict(eos_token_id=99, pad_token_id=98)
    assert cohort['sampled_runtime']['device'] == 'tiny CPU'
    assert mcmd(source, 'verify', '--config', plan) == 0
    rollouts = source / 'runs/stage0/matched/cohort/results/rollouts.jsonl'
    rollouts.write_text(rollouts.read_text().replace('#### 2', '#### 3', 1))
    with pytest.raises(ValueError):
        load_cohort(plan, source)


def test_numeric_agreement_tolerates_cross_cpu_rounding_only():
    from pilot_eval.stage0_numeric import agrees
    assert agrees({'x': [1.0, 2.5], 'n': 3}, {'x': [1.0 * (1 + 2e-16), np.float64(2.5)], 'n': 3})
    assert not agrees({'x': 1.0}, {'x': 1.0 + 1e-9})
    assert not agrees({'n': 3}, {'n': 4})
    assert not agrees({'x': [1.0]}, {'x': [1.0, 2.0]})


def test_reconstruction_allowance_is_the_larger_of_legacy_and_fp32_bound():
    from pilot_eval.stage0_push_math import reconstruction_allowance, gamma
    u = 2.0 ** -24
    assert gamma(10) == pytest.approx(10 * u / (1 - 10 * u))
    oracle = np.array([1e-3, -1e-3])
    terms = np.array([4.0, 4.0]); dots = np.array([0.0, 0.0])
    allowed = reconstruction_allowance(oracle, terms, dots, tokens=1000, sequences=8, width=1536)
    assert allowed == pytest.approx(max(1e-5 * np.linalg.norm(oracle), gamma(1008) * np.linalg.norm(terms)))
    assert allowed > 1e-5 * np.linalg.norm(oracle)  # GRPO-style cancellation: bound dominates
    tiny = reconstruction_allowance(np.array([5.0]), np.array([1e-9]), np.array([0.0]), tokens=4, sequences=1, width=3)
    assert tiny == pytest.approx(5e-5)


def test_consistency_reports_its_pure_noise_reference():
    from pilot_eval.stage0_push_math import summarize_vectors
    result = summarize_vectors(np.ones((16, 3)), [1] * 16, [8] * 16)
    assert result['full_batches'] == 16 and result['consistency_noise_reference'] == pytest.approx(0.25)


def test_distance_on_matched_cohort_freezes_its_first_scoring_session(source):
    from test_stage0_distance import cmd, Scores, approval, production
    from test_stage0_rollouts import usage
    plan = matched(source)
    assert cmd(source, 'prepare', '--rollouts', plan, '--name', 'distance') == 0
    config = json.loads((source / 'plans/distance/stage0.distance.json').read_text())
    assert config['cohort'] == 'matched-128-historical' and config['runtime'] is None
    assert config['sampled_runtime']['device'] == 'tiny CPU'
    dplan = source / 'plans/distance/stage0.distance.json'
    usage(source, 'start', 4.)
    first = Scores()
    assert cmd(source, 'run', '--config', dplan, '--approval', approval(source), '--max-new-units', 1, dependencies=first) == 0
    directory = source / 'runs/stage0/distance/e06a_distance'
    assert json.loads((directory / 'meta/runtime.json').read_text())['session_id'] == 'session-a'
    assert cmd(source, 'run', '--config', dplan, '--max-new-units', 1, dependencies=Scores(session='other')) == 1

    class LaterRelease(Scores):
        def runtime(self): return dict(super().runtime(), git_commit='later')
    production(source, dplan)
    engine = LaterRelease()
    assert cmd(source, 'run', '--config', dplan, dependencies=engine) == 0
    unit = next(p for p in (directory / 'units').glob('*/metadata.json')
                if json.loads(p.read_text()).get('code_git_commit') == 'later')
    assert unit
    usage(source, 'scored', 4.05)
    assert cmd(source, 'report', '--config', dplan) == 0
    assert cmd(source, 'verify', '--config', dplan) == 0
    results = json.loads((directory / 'results/results.json').read_text())
    assert results['cohort'] == 'matched-128-historical'
    assert any('selection' in s for s in results['limitations'])
    assert 'matched-128-historical' in (directory / 'report.html').read_text()
    import csv
    import pyarrow.parquet as pq
    assert all(r['cohort'] == 'matched-128-historical' for r in pq.read_table(directory / 'summary.parquet').to_pylist())
    for name in ['summary.csv', 'per-problem.csv', 'position-windows.csv', 'top-gold-tokens.csv']:
        rows = list(csv.DictReader((directory / 'results' / name).open()))
        assert rows and all(r['cohort'] == 'matched-128-historical' for r in rows), name


@pytest.fixture
def push_ready(tmp_path, monkeypatch):
    import shutil
    from test_stage0_rollouts import sources
    from test_stage0_spectral import TinyWeights, MODULES, spectral
    from test_stage0_pushes import cmd
    from pilot_eval.cli import main
    from pilot_eval import stage0_pushes
    root = tmp_path; sources(root, gold_eos=99, width=64, output=64)
    np.savez(root / 'weights/tiny.npz', **{m: np.eye(64, dtype=np.float32) for m in MODULES})
    assert main(['stage0-spectral-prepare', '--audit', str(root / 'plans/audit/stage0.plan.json'), '--weights-dir', str(root / 'weights'),
                 '--name', 'basis', '--output-root', str(root)], dependencies=TinyWeights()) == 0
    sp = root / 'plans/basis/stage0.spectral.json'; assert spectral(root, 'run', sp, dependencies=TinyWeights()) == 0
    plan = matched(root)
    assert cmd(root, 'prepare', '--spectral', sp, '--rollouts', plan, '--name', 'pushes') == 0
    monkeypatch.setattr(stage0_pushes, 'instrument_check', lambda: dict(passed=True, scope='synthetic workflow fixture'))
    assert cmd(root, 'check', '--config', root / 'plans/pushes/stage0.pushes.json') == 0
    return root


def test_pushes_on_matched_cohort_use_only_its_questions(push_ready):
    config = json.loads((push_ready / 'plans/pushes/stage0.pushes.json').read_text())
    assert config['cohort'] == 'matched-128-historical'
    batches = [json.loads(x) for x in (push_ready / 'runs/stage0/pushes/e05_pushes/inputs/batches.jsonl').read_text().splitlines()]
    grpo = [b for b in batches if b['objective'] == 'grpo']
    case = grpo[0]['case']
    assert sum(b['question_count'] for b in grpo if b['case'] == case) == 128
    gold = [b['problem_ids'] for b in batches if b['objective'] == 'gold' and b['case'] == case]
    own = [b['problem_ids'] for b in batches if b['objective'] == 'own_correct_one' and b['case'] == case]
    assert gold == own


def test_push_gradient_session_allows_only_a_recorded_code_release_change(push_ready):
    from test_stage0_pushes import cmd, Gradients
    from test_stage0_rollouts import usage
    from test_stage0_workflow import write_json
    plan = push_ready / 'plans/pushes/stage0.pushes.json'
    usage(push_ready, 'push-start', 4.)
    review = push_ready / 'push-budget.json'
    write_json(review, dict(reviewer='test', notes='bounded profile', session_id='session-a', source_reviewed=True,
                            max_units_per_hour=1., max_seconds=600.))
    assert cmd(push_ready, 'run', '--config', plan, '--approval', review, '--max-new-units', 1, dependencies=Gradients()) == 0
    assert cmd(push_ready, 'run', '--config', plan, '--max-new-units', 1, dependencies=Gradients(session='other')) == 1

    class LaterRelease(Gradients):
        def runtime(self): return dict(super().runtime(), git_commit='later')
    assert cmd(push_ready, 'run', '--config', plan, '--max-new-units', 1, dependencies=LaterRelease()) == 0
    commits = [json.loads(p.read_text()).get('code_git_commit')
               for p in (push_ready / 'runs/stage0/pushes/e05_pushes/units').glob('*/metadata.json')]
    assert 'later' in commits


# --- Review findings on 5c72c88 (Codex), ADR 0016 follow-up ---

def _helper(kind):
    from pathlib import Path
    notebook = json.loads((Path(__file__).parents[1] / f'notebooks/stage-0-{kind}-colab.ipynb').read_text())
    return next(''.join(c['source']) for c in notebook['cells'] if 'def run_stage0' in ''.join(c['source']))


@pytest.mark.parametrize('kind', ['cpu', 'gpu'])
def test_interrupted_notebook_helper_stops_and_reaps_its_worker(kind):
    import signal, subprocess, types
    source = _helper(kind)
    start = source.index('def run_stage0'); end = source.find('\n\n', start)
    events = []

    class Worker:
        returncode = None
        def __init__(self, *args, **kwargs):
            self.stdout = self._lines()
        def _lines(self):
            yield 'started\n'
            raise KeyboardInterrupt
        def send_signal(self, value): events.append(('signal', value))
        def terminate(self): events.append(('terminate',))
        def kill(self): events.append(('kill',))
        def wait(self, timeout=None):
            events.append(('wait', timeout))
            if len([e for e in events if e[0] == 'wait']) < 3:
                raise subprocess.TimeoutExpired('worker', timeout)
            self.returncode = -9
            return -9
    fake = types.SimpleNamespace(Popen=Worker, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT,
                                 TimeoutExpired=subprocess.TimeoutExpired, CompletedProcess=subprocess.CompletedProcess)
    namespace = dict(subprocess=fake, sys=__import__('sys'), signal=signal, ARTIFACT_ROOT='root', REPO_DIR='repo')
    exec(source[start:end if end > 0 else None], namespace)
    with pytest.raises(KeyboardInterrupt):
        namespace['run_stage0']('status')
    assert events[0] == ('signal', signal.SIGINT)
    assert ('terminate',) in events and ('kill',) in events and events[-1][0] == 'wait'


def test_fingerprints_cover_the_whole_numerical_dependency_chain(tmp_path):
    from pilot_eval.stage0_numeric import dependency_fingerprint
    package = tmp_path / 'pilot_eval'; package.mkdir()
    (package / 'top.py').write_text('from pilot_eval.middle import f\n')
    (package / 'middle.py').write_text('def f():\n    from pilot_eval import leaf\n')
    (package / 'leaf.py').write_text('ATTENTION = "eager"\n')
    before = dependency_fingerprint(['top.py'], package)
    assert set(before) == {'top.py', 'middle.py', 'leaf.py'}
    (package / 'leaf.py').write_text('ATTENTION = "sdpa"\n')
    assert dependency_fingerprint(['top.py'], package) != before
    from pilot_eval import stage0_distance, stage0_push_plan
    for implementation in [stage0_distance.implementation(), stage0_push_plan.implementation()]:
        assert {'stage0_qualification_backend.py', 'stage0_evaluation_backend.py', 'stage0_sources.py'} <= set(implementation)
    assert 'stage0_rollout_backend.py' in stage0_distance.implementation()


def test_loaded_model_must_actually_use_eager_attention():
    from types import SimpleNamespace
    from pilot_eval.stage0_numeric import require_eager_attention
    require_eager_attention(SimpleNamespace(config=SimpleNamespace(_attn_implementation='eager')))
    with pytest.raises(ValueError, match='eager'):
        require_eager_attention(SimpleNamespace(config=SimpleNamespace(_attn_implementation='sdpa')))


def test_matched_reload_tolerates_cross_cpu_float_rounding_only(source, monkeypatch):
    import pilot_eval.stage0_matched as matched_module
    from pilot_eval.stage0_cohort_source import load_cohort
    plan = matched(source)
    original = matched_module.cohorts

    def perturbed(scale):
        def inner(*args, **kwargs):
            value = original(*args, **kwargs)
            key = next(k for k, v in value['neg_log_p_correct'].items() if v)
            value['neg_log_p_correct'][key] *= 1 + scale
            return value
        return inner
    monkeypatch.setattr(matched_module, 'cohorts', perturbed(2e-16))
    load_cohort(plan, source)
    monkeypatch.setattr(matched_module, 'cohorts', perturbed(1e-6))
    with pytest.raises(ValueError, match='reconstruct'):
        load_cohort(plan, source)


def test_matched_cohort_rejects_an_incompatible_sampling_policy(source, monkeypatch):
    import pilot_eval.stage0_matched as matched_module
    original = matched_module.read_sources
    monkeypatch.setattr(matched_module, 'read_sources', lambda *a, **k: dict(original(*a, **k), policy_matches=False))
    from test_stage0_rollouts import Tiny
    assert mcmd(source, 'prepare', '--audit', source / 'plans/audit/stage0.plan.json',
                '--baseline', source / 'plans/baseline/baseline.config.json',
                '--weights-dir', source / 'weights', '--name', 'matched', dependencies=Tiny()) == 1
    assert not (source / 'plans/matched/stage0.cohort.json').exists()


def test_push_report_describes_actual_counts_and_carries_the_cohort(push_ready):
    from test_stage0_pushes import cmd, Gradients
    from test_stage0_rollouts import usage
    from test_stage0_workflow import write_json
    root = push_ready; plan = root / 'plans/pushes/stage0.pushes.json'; directory = root / 'runs/stage0/pushes/e05_pushes'
    config = json.loads(plan.read_text())
    assert '512' not in config['grpo']
    usage(root, 'push-start', 4.)
    approval = root / 'push-budget.json'
    write_json(approval, dict(reviewer='test', notes='bounded profile', session_id='session-a', source_reviewed=True,
                              max_units_per_hour=1., max_seconds=600.))
    engine = Gradients()
    assert cmd(root, 'run', '--config', plan, '--approval', approval, dependencies=engine) == 0
    assert cmd(root, 'forecast', '--config', plan) == 0
    usage(root, 'push-profiled', 4.01)
    review = root / 'push-production.json'
    write_json(review, dict(reviewer='test', notes='production', session_id='session-a', source_reviewed=True,
        max_units_per_hour=1., max_seconds=6000., forecast_sha256=file_hash(directory / 'results/forecast.json')))
    assert cmd(root, 'freeze', '--config', plan, '--review', review) == 0
    assert cmd(root, 'run', '--config', plan, '--no-progress', dependencies=engine) == 0
    assert cmd(root, 'report', '--config', plan) == 0
    summary = json.loads((directory / 'results/summary.json').read_text())
    assert summary['cohort'] == 'matched-128-historical'
    page = (directory / 'report.html').read_text()
    assert 'matched-128-historical' in page and '512' not in page and '128 questions' in page
    assert all('512' not in c['scope'] for c in summary['comparisons'])


def test_matched_cohort_rejects_decoding_that_contradicts_settings(source, monkeypatch):
    import pilot_eval.stage0_matched as matched_module
    from test_stage0_rollouts import Tiny
    original = matched_module.read_sources

    def contradictory(*a, **k):
        value = original(*a, **k)
        legacy = dict(value['legacy'], decoding=dict(value['legacy']['decoding'], temperature=0.7, do_sample=False))
        return dict(value, legacy=legacy)
    monkeypatch.setattr(matched_module, 'read_sources', contradictory)
    assert mcmd(source, 'prepare', '--audit', source / 'plans/audit/stage0.plan.json',
                '--baseline', source / 'plans/baseline/baseline.config.json',
                '--weights-dir', source / 'weights', '--name', 'matched', dependencies=Tiny()) == 1
    assert not (source / 'plans/matched/stage0.cohort.json').exists()


def test_every_standalone_export_names_its_cohort(push_ready):
    import csv
    from test_stage0_pushes import cmd, Gradients
    from test_stage0_rollouts import usage
    from test_stage0_workflow import write_json
    root = push_ready; plan = root / 'plans/pushes/stage0.pushes.json'; directory = root / 'runs/stage0/pushes/e05_pushes'
    usage(root, 'push-start', 4.)
    approval = root / 'push-budget.json'
    write_json(approval, dict(reviewer='test', notes='bounded profile', session_id='session-a', source_reviewed=True,
                              max_units_per_hour=1., max_seconds=600.))
    engine = Gradients()
    assert cmd(root, 'run', '--config', plan, '--approval', approval, dependencies=engine) == 0
    assert cmd(root, 'forecast', '--config', plan) == 0
    usage(root, 'push-profiled', 4.01)
    review = root / 'push-production.json'
    write_json(review, dict(reviewer='test', notes='production', session_id='session-a', source_reviewed=True,
        max_units_per_hour=1., max_seconds=6000., forecast_sha256=file_hash(directory / 'results/forecast.json')))
    assert cmd(root, 'freeze', '--config', plan, '--review', review) == 0
    assert cmd(root, 'run', '--config', plan, '--no-progress', dependencies=engine) == 0
    assert cmd(root, 'report', '--config', plan) == 0
    predictions = root / 'predictions.json'
    write_json(predictions, dict(reviewer='test', notes='frozen first', predictions='fixture', prior_exposure=False))
    assert cmd(root, 'predict', '--config', plan, '--review', predictions) == 0
    assert cmd(root, 'reveal', '--config', plan) == 0
    assert cmd(root, 'verify', '--config', plan) == 0
    label = 'matched-128-historical'
    endpoints = json.loads((directory / 'results/endpoints.json').read_text())
    assert endpoints['cohort'] == label and endpoints['questions']['grpo'] == 128
    for name in ['modules.jsonl', 'directions.jsonl', 'endpoint-alignments.jsonl']:
        rows = [json.loads(x) for x in (directory / 'results' / name).read_text().splitlines()]
        assert rows and all(r['cohort'] == label for r in rows), name
    for name in ['summary.csv', 'endpoints.csv']:
        rows = list(csv.DictReader((directory / 'results' / name).open()))
        assert rows and all(r['cohort'] == label for r in rows), name
    page = (directory / 'endpoints.html').read_text()
    assert label in page and '128' in page
