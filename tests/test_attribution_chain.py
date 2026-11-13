"""Ticket 11: overnight chain runner; every external call mocked (CPU, no GPU, no Hub, no runpodctl)."""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from pilot_eval import attribution_chain as ch
from pilot_eval import attribution_units as au

ARMS = ['gold-sft', 'online-grpo', 'offline-grpo', 'sampling-sft-k0', 'sampling-sft-k2', 'sampling-sft-k5']
TEARDOWN = ['writers_joined', 'volume_copy', 'hf_export', 'summary_sealed', 'summary_on_volume', 'stop_requested']


def record(**overrides):
    base = dict(
        schema=ch.LAUNCH_SCHEMA, run_id='overnight-r1', mode='production', code_commit='abc123', pod_id='pod-1',
        pod_stop_commands=['runpodctl stop pod {id}', 'runpodctl pod stop {id}'], expected_arms=list(ARMS),
        volume_identity='sl-0:vol-uuid', export_repo='owner/private', hourly_cost_usd=2.0,
        caps_seconds=dict(total=1000, qualification=100, base_traces=100, attribution_and_freeze=300, stage_c=200,
                          stage_d=200, report=50, preflight=50),
        tolerances=dict(finite_difference=dict(epsilon=1e-2, median_rel_err_max=0.05, max_rel_err_max=0.20),
                        neuron_sum_identity=1e-4, batch_invariance=1e-4, resolution_floor='10x batch invariance'),
        minimums=dict(discovery=120, heldout=60), selection=dict(top_group=10, top_singles=5, tolerance=0.2,
                                                                 seed=20000, random_seeds=[1, 2, 3], read_size=64),
        statistics=dict(resamples=10000, seed=1234, alpha=0.05, margin=0.25, sparse_lower=0.7, dense_upper=0.3,
                        gate_singles_min=4, null_orders=1000, null_seed=4),
        generation_batch_rule=dict(candidates=[1, 8, 16, 32, 64], items=20, max_flips=1, max_stop_changes=1),
        token_budget_sweep=[2048, 4096, 8192, 16384, 32768, 65536], watchdog_margin_seconds=1800,
        marker_stale_seconds=600, compute=dict(fp32_peak_flops=19.5e12),
        stage_d=dict(early_stop_rule=dict(kind='quantile-of-untuned-eos-tokens', quantile=0.1),
                     pilot_results={'gold-sft': 'p/gold.jsonl', 'online-grpo': 'p/online.jsonl'}),
        timeouts_seconds=dict(copy=60, export_attempt=60, stop_attempt=30, writer_join=10))
    base.update(overrides)
    return base


def sealed_record(tmp_path, **overrides):
    path = tmp_path / 'launch' / 'launch_record.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    return ch.seal_launch_record(record(**overrides), path)


class Calls:
    def __init__(self):
        self.order, self.stops = [], []


def harness(tmp_path, stages, *, copier=None, exporter=None, runner=None, now=None, writers=(), **record_overrides):
    calls = Calls()
    path = sealed_record(tmp_path, **record_overrides)

    def default_copier(local, volume, *, sealed_only):
        calls.order.append('volume_copy')
        assert sealed_only
        return dict(verified=True, copied=['x'], problems=[])

    def default_exporter(local, *, label, expected_units, expected_runs, repo):
        calls.order.append('hf_export')
        return dict(revision='a' * 40, completeness='complete', readback_verified=True)

    def default_runner(cmd, **kw):
        calls.order.append('stop')
        calls.stops.append((cmd, kw.get('timeout')))
        return subprocess.CompletedProcess(cmd, 0, 'stopped', '')

    chain = ch.Chain(ch.load_launch_record(path, code_commit='abc123'), local_root=tmp_path / 'local',
                     volume_root=tmp_path / 'volume', stages=stages, copier=copier or default_copier,
                     exporter=exporter or default_exporter, runner=runner or default_runner,
                     now=now or time.monotonic, writers=list(writers), volume_identity='sl-0:vol-uuid')
    return chain, calls


def ok_stages(order=None, **replace):
    order = order if order is not None else []

    def make(name):
        def stage(ctx):
            order.append(name)
            if name == 'qualification':
                return dict(checks=dict(finite_differences=True, batch_invariance=True, neuron_identity=True),
                            review_flags=[], arms=list(ctx.record['expected_arms']), budget=8192)
            if name == 'report':
                return dict(gates={'gold-sft|trace': True}, primaries=dict(primary1='inconclusive'))
            return dict(status='complete')
        return stage

    stages = {name: make(name) for name in ch.STAGES}
    stages.update(replace)
    return stages


# --- launch record --------------------------------------------------------------------------------

def test_launch_record_refuses_null_or_proposed_values(tmp_path):
    path = sealed_record(tmp_path)
    assert ch.load_launch_record(path, code_commit='abc123')['run_id'] == 'overnight-r1'
    for bad in [dict(caps_seconds=dict(total=None)), dict(minimums=dict(discovery='[proposed] 120', heldout=60)),
                dict(pod_id=None), dict(selection=dict(random_seeds=None))]:
        p = sealed_record(tmp_path / 'b', **bad)
        with pytest.raises(ch.LaunchRefused, match='unresolved'):
            ch.load_launch_record(p, code_commit='abc123')
    for bad in [dict(stage_d=dict(early_stop_rule=None, pilot_results={})), dict(compute=dict(fp32_peak_flops=None))]:
        with pytest.raises(ch.LaunchRefused, match='unresolved'):
            ch.load_launch_record(sealed_record(tmp_path / 'c', **bad), code_commit='abc123')
    no_pilot = sealed_record(tmp_path / 'd', stage_d=dict(early_stop_rule=dict(kind='quantile-of-untuned-eos-tokens',
                                                                               quantile=0.1), pilot_results={}))
    with pytest.raises(ch.LaunchRefused, match='pilot'):
        ch.load_launch_record(no_pilot, code_commit='abc123')
    with pytest.raises(ch.LaunchRefused, match='commit'):
        ch.load_launch_record(path, code_commit='other')
    path.write_text(path.read_text().replace('overnight-r1', 'overnight-r2'))
    with pytest.raises(ch.LaunchRefused, match='seal'):
        ch.load_launch_record(path, code_commit='abc123')


def test_production_refuses_missing_arms_rehearsal_uses_its_labelled_set(tmp_path):
    with pytest.raises(ch.LaunchRefused, match='arms'):
        ch.load_launch_record(sealed_record(tmp_path / 'p', expected_arms=ARMS[:2]), code_commit='abc123')
    rec = ch.load_launch_record(sealed_record(tmp_path / 'r', mode='rehearsal', expected_arms=ARMS[:2],
                                              rehearsal_label='rehearsal-pod-2arm'), code_commit='abc123')
    assert rec['expected_arms'] == ARMS[:2]
    with pytest.raises(ch.LaunchRefused, match='label'):
        ch.load_launch_record(sealed_record(tmp_path / 'r2', mode='rehearsal', expected_arms=ARMS[:2]),
                              code_commit='abc123')


# --- exit paths ---------------------------------------------------------------------------------------

def test_success_path_runs_every_stage_signs_off_and_ends_in_teardown_order(tmp_path):
    order = []
    chain, calls = harness(tmp_path, ok_stages(order))
    summary = chain.run()
    assert order == ch.STAGES
    assert summary['teardown'] == TEARDOWN and summary['state'] == 'complete'
    assert summary['signoff']['by'] == ch.AUTO_REVIEWER and 'user' not in summary['signoff']['by'].split()
    assert calls.stops[0][0] == ['runpodctl', 'stop', 'pod', 'pod-1'] and calls.stops[0][1] == 30
    assert summary['stop'] == dict(requested=True, externally_verified=False, attempts=summary['stop']['attempts'])
    restored = json.loads((tmp_path / 'volume' / au.SUMMARY_DIR / 'run_summary.json').read_text())
    assert restored['backups']['volume_copy']['verified'] and restored['backups']['hf_export']['verified']


def test_qualification_failure_stops_before_production(tmp_path):
    order = []

    def failing(ctx):
        order.append('qualification')
        return dict(checks=dict(finite_differences=False), review_flags=[], arms=list(ARMS))

    chain, calls = harness(tmp_path, ok_stages(order, qualification=failing))
    summary = chain.run()
    assert 'attribution_and_freeze' not in order and summary['state'] == 'stopped-before-production'
    assert summary['teardown'] == TEARDOWN and 'finite_differences' in summary['stop_reason']


@pytest.mark.parametrize('problem', ['review_flag', 'missing_arm', 'minimum', 'budget'])
def test_listed_preproduction_failures_stop_the_chain(tmp_path, problem):
    order = []

    def qual(ctx):
        order.append('qualification')
        out = dict(checks=dict(fd=True), review_flags=[], arms=list(ARMS))
        if problem == 'review_flag':
            out['review_flags'] = ['layer 3 discrepancy']
        if problem == 'missing_arm':
            out['arms'] = ARMS[:-1]
        if problem == 'budget':
            raise ch.StopBeforeProduction('no feasible token budget')
        return out

    def traces(ctx):
        order.append('base_traces')
        if problem == 'minimum':
            from pilot_eval.attribution_traces import MinimumUnmet
            raise MinimumUnmet('discovery: 100 < 120')
        return {}

    chain, _ = harness(tmp_path, ok_stages(order, qualification=qual, base_traces=traces))
    summary = chain.run()
    assert summary['state'] == 'stopped-before-production' and 'attribution_and_freeze' not in order
    assert summary['teardown'] == TEARDOWN


def test_analysis_outcomes_never_stop_the_run(tmp_path):
    order = []

    def stage_c(ctx):
        order.append('stage_c')
        return dict(gate_failed=True, skipped=['match10-stop'], primaries='inconclusive')

    chain, _ = harness(tmp_path, ok_stages(order, stage_c=stage_c))
    summary = chain.run()
    assert order == ch.STAGES and summary['state'] == 'complete'


def test_exception_path_still_tears_down_in_order(tmp_path):
    def boom(ctx):
        raise RuntimeError('CUDA error: device-side assert')

    chain, calls = harness(tmp_path, ok_stages(stage_c=boom))
    summary = chain.run()
    assert summary['state'] == 'failed' and 'device-side assert' in summary['failures'][0]['error']
    assert summary['failures'][0]['log'] and summary['teardown'] == TEARDOWN
    assert calls.order == ['volume_copy', 'hf_export', 'stop']


def test_hf_failure_still_stops_and_is_reported(tmp_path):
    def exporter(*a, **k):
        raise ConnectionError('hub down')

    chain, calls = harness(tmp_path, ok_stages(), exporter=exporter)
    summary = chain.run()
    assert summary['backups']['hf_export']['verified'] is False and len(summary['backups']['hf_export']['attempts']) == 3
    assert summary['stop']['requested'] is True and summary['teardown'] == TEARDOWN


def test_hung_copy_and_hung_stop_time_out_and_neither_backup_verified_is_reported(tmp_path):
    def hung_copy(*a, **k):
        time.sleep(30)

    def exporter(*a, **k):
        raise PermissionError('401')

    attempts = []

    def runner(cmd, **kw):
        attempts.append(cmd)
        raise subprocess.TimeoutExpired(cmd, kw['timeout'])

    chain, _ = harness(tmp_path, ok_stages(), copier=hung_copy, exporter=exporter, runner=runner,
                       timeouts_seconds=dict(copy=0.2, export_attempt=0.2, stop_attempt=0.1, writer_join=0.1))
    started = time.monotonic()
    summary = chain.run()
    assert time.monotonic() - started < 10
    assert summary['backups']['volume_copy'] == dict(verified=False, outcome='timeout', timeout_seconds=0.2)
    assert 'NO VERIFIED BACKUP' in summary['backups']['statement']
    assert len(attempts) == 2 and summary['stop']['requested'] is False
    assert [a['outcome'] for a in summary['stop']['attempts']] == ['timeout', 'timeout']
    local = json.loads((tmp_path / 'local' / au.SUMMARY_DIR / 'run_summary.json').read_text())
    assert local['backups']['volume_copy']['verified'] is False  # never labelled verified


def test_stop_command_fallback(tmp_path):
    seen = []

    def runner(cmd, **kw):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 1 if len(seen) == 1 else 0, '', 'unknown command')

    chain, _ = harness(tmp_path, ok_stages(), runner=runner)
    summary = chain.run()
    assert seen == [['runpodctl', 'stop', 'pod', 'pod-1'], ['runpodctl', 'pod', 'stop', 'pod-1']]
    assert summary['stop']['requested'] is True and summary['stop']['attempts'][0]['returncode'] == 1


def test_writers_are_stopped_and_joined_first(tmp_path):
    events = []

    class Writer:
        def stop(self):
            events.append('stop')

        def join(self, timeout):
            events.append(('join', timeout))
            return True

    def copier(*a, **k):
        events.append('copy')
        return dict(verified=True, problems=[])

    chain, _ = harness(tmp_path, ok_stages(), copier=copier, writers=[Writer()])
    chain.run()
    assert events == ['stop', ('join', 10), 'copy']


# --- caps ---------------------------------------------------------------------------------------------

def test_overrun_ends_the_stage_at_a_unit_boundary_and_goes_to_report_and_stop(tmp_path):
    clock = [0.0]
    order, units = [], []

    def attribution(ctx):
        order.append('attribution_and_freeze')
        for unit in range(10):
            clock[0] += 50
            units.append(unit)
            ctx.boundary()
        return {}

    chain, _ = harness(tmp_path, ok_stages(order, attribution_and_freeze=attribution), now=lambda: clock[0])
    summary = chain.run()
    assert units == list(range(6)) and summary['stages']['attribution_and_freeze']['status'] == 'cap-reached'
    assert 'stage_c' not in order and order[-1] == 'report' and summary['state'] == 'cap-reached'
    assert summary['teardown'] == TEARDOWN


def test_longest_first_packing_does_not_stop_a_healthy_run(tmp_path):
    clock = [0.0]
    projections = []

    def attribution(ctx):
        for seconds in [100, 60, 30, 20, 10, 10, 10, 10, 10, 10]:  # 270 s < 300 s cap
            clock[0] += seconds
            ctx.boundary()
            projections.append(ctx.projection(done=len(projections) + 1, total=10))
        return {}

    chain, _ = harness(tmp_path, ok_stages(attribution_and_freeze=attribution), now=lambda: clock[0])
    summary = chain.run()
    assert projections[0]['projected_seconds'] > 300 and summary['state'] == 'complete'


def test_elapsed_budget_persists_across_a_restart(tmp_path):
    clock = [0.0]

    def slow(ctx):
        clock[0] += 80
        ctx.boundary()
        raise RuntimeError('killed')

    chain, _ = harness(tmp_path, ok_stages(attribution_and_freeze=slow), now=lambda: clock[0])
    chain.run()
    budget = json.loads((tmp_path / 'volume' / ch.BUDGET_FILE).read_text())
    assert budget['elapsed']['attribution_and_freeze'] == pytest.approx(80)


# --- resume and marker ---------------------------------------------------------------------------------

def marker(tmp_path, **fields):
    path = tmp_path / 'volume' / ch.MARKER_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    base = dict(run_id='overnight-r1', host=socket.gethostname(), pid=999999, state='running',
                heartbeat_unix=time.time() - 3600)
    base.update(fields)
    path.write_text(json.dumps(base))
    return path


def test_killed_worker_stale_marker_resumes_and_skips_completed_stages(tmp_path):
    order = []
    chain, _ = harness(tmp_path, ok_stages(order))
    state = tmp_path / 'volume' / ch.STATE_FILE
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps(dict(run_id='overnight-r1', completed={
        'preflight': {}, 'qualification': dict(checks=dict(fd=True), review_flags=[], arms=ARMS)},
        signoff=dict(by=ch.AUTO_REVIEWER))))
    marker(tmp_path)
    summary = chain.run()
    assert summary['resume']['decision'] == 'take-over' and order[0] == 'base_traces'


def test_live_writer_stops_for_investigation(tmp_path):
    marker(tmp_path, pid=os.getpid(), heartbeat_unix=time.time())
    order = []
    chain, calls = harness(tmp_path, ok_stages(order))
    with pytest.raises(ch.ResumeRefused, match='live'):
        chain.run()
    assert order == [] and calls.stops == []  # never tears down another writer's run


@pytest.mark.parametrize('fields,reason', [(dict(host='other-host'), 'ambiguous'),
                                           (dict(heartbeat_unix=time.time()), 'ambiguous'),
                                           (dict(run_id='someone-else'), 'foreign')])
def test_ambiguous_or_foreign_marker_stops(tmp_path, fields, reason):
    marker(tmp_path, **fields)
    chain, _ = harness(tmp_path, ok_stages())
    with pytest.raises(ch.ResumeRefused, match=reason):
        chain.run()


def test_volume_identity_mismatch_refuses_resume(tmp_path):
    chain, _ = harness(tmp_path, ok_stages())
    chain.volume_identity = 'sl-1:other'
    with pytest.raises(ch.ResumeRefused, match='volume'):
        chain.run()


def test_resume_skips_sealed_units(tmp_path):
    units = au.Units(tmp_path / 'local', plan_sha256='p', code_sha256='c', runtime_sha256='r')
    calls = []

    def compute(batch):
        calls.append(batch)
        return dict(blocks={('examples', 0, 'x'): np.ones((1, 2), dtype=np.float32)}, counts={'examples': [1]})

    batches = [['e0'], ['e1'], ['e2']]
    ch.run_units(units, 'gold-sft', batches, compute, boundary=lambda: None, input_sha256='i')
    assert calls == batches
    calls.clear()
    ch.run_units(units, 'gold-sft', batches, compute, boundary=lambda: None, input_sha256='i')
    assert calls == []


def test_unavailable_matched_group_is_skipped_not_fatal(tmp_path):
    units = au.Units(tmp_path / 'local', plan_sha256='p', code_sha256='c', runtime_sha256='r')
    conditions = [dict(name='top1-trace', kind='removal', adapters=[1], status='run', reason=None),
                  dict(name='match10-trace', kind='removal', adapters=None, status='skipped',
                       reason='matched-control-unavailable')]
    seen = []

    def evaluate(condition):
        seen.append(condition['name'])
        return dict(blocks={('examples', 0, 'delta_trace'): np.zeros((2, 1), dtype=np.float32)},
                    counts={'examples': [1, 1]})

    out = ch.run_conditions(units, 'gold-sft', conditions, evaluate, example_ids=['q0', 'q1'], boundary=lambda: None,
                            input_sha256='i')
    assert seen == ['top1-trace'] and out['skipped'] == [dict(name='match10-trace', reason='matched-control-unavailable')]


# --- watchdog ----------------------------------------------------------------------------------------

WATCHDOG = Path(__file__).resolve().parents[1] / 'scripts' / 'attribution_watchdog.sh'


def test_watchdog_deadline_uses_the_persisted_remaining_cap(tmp_path):
    budget = au.Budget(tmp_path / 'b.json', stage_caps=dict(x=100), total_cap=1000, now=lambda: 0)
    budget.charge('x', 400)
    deadline = ch.watchdog_deadline(budget, margin_seconds=1800, wall_now=1_000_000.0)
    assert deadline == pytest.approx(1_000_000 + 600 + 1800)


def test_watchdog_fires_on_a_hung_stage(tmp_path):
    hung = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    log = tmp_path / 'actions.log'
    env = dict(os.environ, WATCHDOG_COPY=f'echo copy >> {log}', WATCHDOG_STOP_1=f'echo stop1 >> {log}; exit 1',
               WATCHDOG_STOP_2=f'echo stop2 >> {log}', WATCHDOG_SUMMARY=str(tmp_path / 'watchdog_summary.json'))
    deadline = time.time() + 1
    result = subprocess.run(['bash', str(WATCHDOG), str(deadline), str(hung.pid), str(tmp_path / 'done'), '2', '5', '5'],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert hung.wait(timeout=10) is not None
    assert log.read_text().split() == ['copy', 'stop1', 'stop2']
    summary = json.loads((tmp_path / 'watchdog_summary.json').read_text())
    assert summary['fired'] is True and summary['stop'] == 'requested' and summary['externally_verified'] is False


def test_watchdog_cancelled_cleanly_on_normal_completion(tmp_path):
    log = tmp_path / 'actions.log'
    (tmp_path / 'done').write_text('complete')
    env = dict(os.environ, WATCHDOG_COPY=f'echo copy >> {log}', WATCHDOG_STOP_1=f'echo stop1 >> {log}',
               WATCHDOG_STOP_2='true', WATCHDOG_SUMMARY=str(tmp_path / 's.json'))
    result = subprocess.run(['bash', str(WATCHDOG), str(time.time() + 3600), str(os.getpid()), str(tmp_path / 'done'),
                             '2', '5', '5'], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and not log.exists() and 'cancelled' in result.stdout
