"""Ticket 09: the C-attribution launch record, runner, teardown and resume (fake stages, CPU)."""
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest
import yaml

from pilot_eval import attribution_units as au
from pilot_eval import c_attribution_chain as cc
from pilot_eval.attribution_chain import LaunchRefused, ResumeRefused, StopBeforeProduction

REPO = Path(__file__).resolve().parents[1]
DRAFT = REPO / '.scratch/c-attribution-pilot/launch-record-rehearsal.yaml'
FILL = {'inputs.artifact_root': '/w/inputs', 'inputs.bundle': '/w/bundle.json', 'inputs.model_dir': '/w/model',
        'inputs.work_dir': '/w'}


def _draft(tmp_path, **changes):
    body = yaml.safe_load(DRAFT.read_text())
    body['hourly_cost_usd'] = 1.5
    for dotted, value in changes.items():
        parent, keys = body, dotted.split('.')
        for k in keys[:-1]:
            parent = parent[k]
        parent[keys[-1]] = value
    path = tmp_path / 'draft.yaml'
    path.write_text(yaml.safe_dump(body))
    return path


def sealed(tmp_path, **changes):
    return cc.finalize_draft(_draft(tmp_path, **changes), tmp_path / 'launch_record.json', code_commit='c0ffee',
                             pod_id='podtest1', fill=FILL)


# --- launch record --------------------------------------------------------------------------------

def test_the_actual_rehearsal_draft_seals_and_every_leaf_is_typed(tmp_path):
    path = sealed(tmp_path)
    record = cc.load_launch_record(path, code_commit='c0ffee')
    assert record['run_id'] == 'c-rehearsal-pod-1' and record['pod_id'] == 'podtest1'
    assert record['hot_layers']['layers'] == [11, 15, 16, 24]
    assert record['tolerances']['cached_vs_forced']['per_token_abs_max'] == cc.MEASURE_ONLY
    assert [' '.join(c) for c in cc.stop_commands(record)] == ['runpodctl stop pod podtest1', 'runpodctl pod stop podtest1']
    with pytest.raises(LaunchRefused, match='code commit'):
        cc.load_launch_record(path, code_commit='other')


def test_unfilled_draft_is_refused(tmp_path):
    with pytest.raises(LaunchRefused, match='inputs.bundle'):
        cc.finalize_draft(DRAFT, tmp_path / 'r.json', code_commit='c0ffee', pod_id='p1',
                          fill={k: v for k, v in FILL.items() if k != 'inputs.bundle'})


@pytest.mark.parametrize('dotted,value,match', [
    ('tolerances.neuron_sum_identity', '1e-4', 'neuron_sum_identity'),             # YAML number read as text
    ('qualification.fd_rows', 4.0, 'fd_rows'),                                       # a count that is not an integer
    ('selection.seeds.top10-adapters', True, 'top10-adapters'),                      # a bool where an int belongs
    ('scope.attribution_questions.online-grpo', [1, 2], 'online-grpo'),              # a nested leaf of the wrong type
    ('hot_layers.layers', ['11', 15], r'hot_layers.layers\[0\]'),                    # nested list leaf
    ('attribution.extra', 3, 'unknown key'),                                         # a key the schema does not have
    ('generation.cap', '[proposed] 1024', 'unresolved'),
])
def test_a_wrong_type_anywhere_refuses_the_seal(tmp_path, dotted, value, match):
    with pytest.raises(LaunchRefused, match=match):
        sealed(tmp_path, **{dotted: value})


def test_record_rules(tmp_path):
    with pytest.raises(LaunchRefused, match='both syntaxes'):
        sealed(tmp_path, pod_stop_commands=['runpodctl stop pod {id}'])
    with pytest.raises(LaunchRefused, match='only in a rehearsal'):
        sealed(tmp_path, mode='production', label='production')
    with pytest.raises(LaunchRefused, match='FP64'):
        sealed(tmp_path, **{'tolerances.finite_difference.precision': 'fp32'})
    with pytest.raises(LaunchRefused, match='one seed per group'):
        sealed(tmp_path, **{'selection.seeds': {'top10-adapters': 1}})
    numbers = {'tolerances.cached_vs_forced': {'per_token_abs_max': 1e-3, 'per_row_abs_M_max': 0.5}}
    with pytest.raises(LaunchRefused, match='scope: all'):
        sealed(tmp_path, mode='production', label='production', **numbers)
    path = sealed(tmp_path, mode='production', label='production', **numbers,
                  **{'scope.attribution_questions': {'online-grpo': 'all', 'offline-grpo': 'all'},
                     'scope.generation_questions': {'online-grpo': 'all', 'offline-grpo': 'all'},
                     'scope.generation_conditions': 'all'})
    assert cc.load_launch_record(path, code_commit='c0ffee')['mode'] == 'production'
    record = json.loads(path.read_text())
    record['hourly_cost_usd'] = 2.0
    path.write_text(json.dumps(record))
    with pytest.raises(LaunchRefused, match='seal'):
        cc.load_launch_record(path, code_commit='c0ffee')


# --- chain with fake stages -----------------------------------------------------------------------

class Exporter:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def __call__(self, local_root, *, label, include, expected_units, expected_runs, repo):
        self.calls.append(dict(include=include, expected_runs=list(expected_runs)))
        if self.fail:
            raise RuntimeError('hub down')
        # the pre-export summary must already be sealed and on disk when the export runs
        for run in expected_runs:
            if 'summaries' in run:
                au.verify_run(Path(local_root) / run)
        return dict(readback_verified=True, completeness='complete')


class Runner:
    def __init__(self, codes=(1, 0)):
        self.codes, self.commands = list(codes), []

    def __call__(self, command, **kw):
        self.commands.append(command)
        return subprocess.CompletedProcess(command, self.codes.pop(0) if self.codes else 0, 'out', '')


def _stages(log, *, fail=None, stop=None, outputs=True):
    def make(name):
        def stage(ctx):
            log.append(name)
            if name == stop:
                raise StopBeforeProduction(f'{name} integrity failure')
            if name == fail:
                raise RuntimeError(f'{name} crashed')
            ctx.boundary(detail='unit 1')
            out = {}
            if outputs:
                rel = f'{cc.SUBTREE}/{name}'
                au.seal_run(ctx.local_root / rel, {'result.json': {'stage': name}})
                out['outputs'] = [rel]
            if name == 'qualification':
                out.update(checks={'fd': True, 'identity': True}, review_flags=[], arms=list(ctx.record['arms']))
            if name == 'aggregate_freeze':
                out['analysis'] = {'stability': 'unstable', 'controls_unavailable': ['layer64-reads']}
            return out
        return stage
    return {name: make(name) for name in cc.STAGES}


def _chain(tmp_path, stages, *, exporter=None, runner=None, local='local'):
    record = cc.load_launch_record(sealed(tmp_path), code_commit='c0ffee')
    return cc.Chain(record, local_root=tmp_path / local, volume=tmp_path / 'vol', stages=stages,
                    exporter=exporter or Exporter(), runner=runner or Runner(), volume_identity='sl-0',
                    current_pod='podtest1')


def test_a_complete_run_tears_down_in_order_and_stops_with_the_second_syntax(tmp_path):
    log, exporter, runner = [], Exporter(), Runner(codes=(1, 0))
    chain = _chain(tmp_path, _stages(log), exporter=exporter, runner=runner)
    summary = chain.run()
    assert log == cc.STAGES and summary['state'] == 'complete'
    assert summary['stages']['aggregate_freeze']['analysis']['stability'] == 'unstable'  # analysis never stops it
    assert summary['teardown'] == ['volume_copy', 'summary_sealed', 'hf_export', 'summary_on_volume', 'stop_requested']
    assert summary['backups']['volume_copy']['verified'] and summary['backups']['hf_export']['verified']
    # the pre-export summary is inside the export's expected set (and verified on disk at export time)
    assert any('pre-export' in r for r in exporter.calls[0]['expected_runs'])
    assert [' '.join(c) for c in runner.commands] == ['runpodctl stop pod podtest1', 'runpodctl pod stop podtest1']
    assert summary['stop']['requested'] and not summary['stop']['externally_verified']
    root = cc.run_root(tmp_path / 'vol', 'c-rehearsal-pod-1')
    assert (root / cc.DONE_FILE).read_text() == 'complete\n'
    assert set(summary['elapsed_seconds']) == set(cc.STAGES)
    progress = json.loads((root / cc.PROGRESS_FILE).read_text())
    assert progress['stage'] == 'report' and progress['last_update_unix'] > 0
    # everything sealed reached the volume under this run's subtree
    for name in cc.STAGES:
        au.verify_run(root / cc.SUBTREE / name)
    au.verify_run(root / cc.launch_dir(chain.record))


def test_integrity_failure_stops_before_later_stages_and_still_tears_down(tmp_path):
    log = []
    summary = _chain(tmp_path, _stages(log, stop='rows')).run()
    assert log == ['preflight', 'qualification', 'rows'] and summary['state'] == 'stopped'
    assert 'integrity' in summary['stop_reason'] and summary['stop']['requested']


def test_failed_qualification_signoff_stops(tmp_path):
    log = []
    stages = _stages(log)
    good = stages['qualification']
    stages['qualification'] = lambda ctx: dict(good(ctx), checks={'fd': False})
    summary = _chain(tmp_path, stages).run()
    assert summary['state'] == 'stopped' and 'check failed: fd' in summary['stop_reason']
    assert 'rows' not in log


def test_a_crash_is_reported_and_the_export_failure_is_never_labelled_verified(tmp_path):
    log = []
    summary = _chain(tmp_path, _stages(log, fail='generation'), exporter=Exporter(fail=True)).run()
    assert summary['state'] == 'failed' and summary['failures'][0]['stage'] == 'generation'
    assert summary['backups']['hf_export']['verified'] is False
    assert summary['backups']['complete'] == []  # a failed run is never a complete backup (review 06-10)
    assert 'volume copy: verified bytes, partial' in summary['backups']['statement']
    assert summary['stop']['requested']


def test_resume_from_an_empty_local_disk_reverifies_before_skipping(tmp_path):
    log = []
    first = _chain(tmp_path, _stages(log, fail='attribution'))
    assert first.run()['state'] == 'failed'
    root = cc.run_root(tmp_path / 'vol', 'c-rehearsal-pod-1')
    (root / cc.DONE_FILE).unlink(missing_ok=True)                  # an interrupted run (the Pod died before teardown finished)
    marker = json.loads((root / cc.MARKER_FILE).read_text())
    dead = subprocess.Popen(['true'])
    dead.wait()  # the old writer's process is gone
    (root / cc.MARKER_FILE).write_text(json.dumps(dict(marker, state='running', pid=dead.pid,
                                                       heartbeat_unix=time.time() - 10_000)))
    log.clear()
    second = _chain(tmp_path, _stages(log), local='fresh-local')   # empty local disk
    summary = second.run()
    assert summary['state'] == 'complete' and summary['resume']['decision'] == 'take-over'
    # preflight always runs again (review 06-10); completed later stages are re-verified and skipped
    assert log == ['preflight', 'attribution', 'aggregate_freeze', 'generation', 'report']
    assert summary['stages']['rows'] == {'status': 'complete', 'resumed': True}
    assert 'c-attribution/rows' in summary['restored_from_volume']['runs']
    assert summary['attempt'] == 'attempt-002'


def test_a_completed_stage_whose_outputs_are_gone_is_rerun_and_a_tampered_copy_is_refused(tmp_path):
    import shutil
    log = []
    _chain(tmp_path, _stages(log, fail='attribution')).run()
    root = cc.run_root(tmp_path / 'vol', 'c-rehearsal-pod-1')
    (root / cc.DONE_FILE).unlink(missing_ok=True)
    marker = json.loads((root / cc.MARKER_FILE).read_text())
    (root / cc.MARKER_FILE).write_text(json.dumps(dict(marker, state='released')))
    shutil.rmtree(root / cc.SUBTREE / 'rows')               # expected work with no verified artifact anywhere
    log.clear()
    summary = _chain(tmp_path, _stages(log), local='fresh-local').run()
    assert log == ['preflight', 'rows', 'attribution', 'aggregate_freeze', 'generation', 'report']
    assert summary['stages']['qualification'] == {'status': 'complete', 'resumed': True}
    # a hash-mismatched volume copy is never restored or reused
    other = tmp_path / 'again'
    other.mkdir()
    _chain(other, _stages([], fail='attribution')).run()
    root = cc.run_root(other / 'vol', 'c-rehearsal-pod-1')
    (root / cc.DONE_FILE).unlink(missing_ok=True)
    (root / cc.MARKER_FILE).write_text(json.dumps(dict(json.loads((root / cc.MARKER_FILE).read_text()), state='released')))
    (root / cc.SUBTREE / 'rows/result.json').write_text('tampered')
    with pytest.raises(ValueError, match='failed its hash'):
        _chain(other, _stages([]), local='fresh-local')._start()


def test_a_finished_run_id_is_refused_and_a_live_writer_is_never_taken_over(tmp_path):
    _chain(tmp_path, _stages([])).run()
    with pytest.raises(ResumeRefused, match='new run ID'):
        cc.precheck(cc.load_launch_record(tmp_path / 'launch_record.json', code_commit='c0ffee'), tmp_path / 'vol',
                    current_pod='podtest1')
    other = tmp_path / 'other'
    other.mkdir()
    record = cc.load_launch_record(sealed(other), code_commit='c0ffee')
    root = cc.run_root(other / 'vol', record['run_id'])
    (root / cc.CHAIN_DIR).mkdir(parents=True)
    live = subprocess.Popen(['sleep', '30'])
    try:
        (root / cc.MARKER_FILE).write_text(json.dumps(dict(run_id=record['run_id'], host=socket.gethostname(),
                                                           pid=live.pid, state='running', heartbeat_unix=time.time())))
        with pytest.raises(ResumeRefused, match='live writer'):
            cc.precheck(record, other / 'vol', current_pod='podtest1')
    finally:
        live.kill()


def test_a_different_record_under_the_same_run_id_is_refused(tmp_path):
    _chain(tmp_path, _stages([], fail='rows')).run()
    root = cc.run_root(tmp_path / 'vol', 'c-rehearsal-pod-1')
    (root / cc.DONE_FILE).unlink(missing_ok=True)
    marker = json.loads((root / cc.MARKER_FILE).read_text())
    (root / cc.MARKER_FILE).write_text(json.dumps(dict(marker, state='released')))
    record = cc.load_launch_record(sealed(tmp_path, hourly_cost_usd=9.0), code_commit='c0ffee')
    chain = cc.Chain(record, local_root=tmp_path / 'local', volume=tmp_path / 'vol', stages=_stages([]),
                     exporter=Exporter(), runner=Runner(), volume_identity='sl-0', current_pod='podtest1')
    with pytest.raises(ResumeRefused, match='different launch record'):
        chain._start()


def test_wrong_volume_identity_still_tears_down_and_requests_the_stop(tmp_path):
    record = cc.load_launch_record(sealed(tmp_path), code_commit='c0ffee')
    runner = Runner(codes=(0,))
    chain = cc.Chain(record, local_root=tmp_path / 'local', volume=tmp_path / 'vol', stages=_stages([]),
                     exporter=Exporter(), runner=runner, volume_identity='other', current_pod='podtest1')
    summary = chain.run()
    assert summary['state'] == 'failed' and 'volume identity' in summary['failures'][0]['error']
    assert summary['stop']['requested'] and len(runner.commands) == 1


# --- review 06-10 -------------------------------------------------------------------------------------

def _released(root):
    marker = json.loads((root / cc.MARKER_FILE).read_text())
    assert marker['state'] == 'released'
    return marker


def test_sealed_work_reaches_the_volume_at_each_boundary_not_only_at_teardown(tmp_path):
    """Review P1: until teardown, sealed work lived only on the local disk; a lost Pod lost it."""
    log, seen = [], {}
    stages = _stages(log)

    def attribution(ctx):
        rel = f'{cc.SUBTREE}/attribution/unit-0000'
        au.seal_run(ctx.local_root / rel, {'meta.json': {'unit': 0}})
        ctx.boundary(unit=0)
        # what would survive if the Pod vanished right now (no teardown, no watchdog copy)
        seen['unit'] = au.verify_run(ctx.volume_root / rel)
        seen['rows'] = au.verify_run(ctx.volume_root / cc.SUBTREE / 'rows')
        seen['progress'] = json.loads((ctx.volume_root / cc.PROGRESS_FILE).read_text())
        return dict(outputs=[rel])
    stages['attribution'] = attribution
    assert _chain(tmp_path, stages).run()['state'] == 'complete'
    assert seen['unit'] and seen['rows']
    assert seen['progress']['mirror']['verified'] is True


def test_a_handled_failure_ends_the_attempt_not_the_run_and_resumes(tmp_path):
    """Review P1: teardown wrote ``done`` for failed and stopped runs, so precheck refused the resume."""
    log = []
    first = _chain(tmp_path, _stages(log, fail='attribution')).run()
    assert first['state'] == 'failed'
    root = cc.run_root(tmp_path / 'vol', 'c-rehearsal-pod-1')
    assert not (root / cc.DONE_FILE).exists()
    assert json.loads((root / cc.ATTEMPT_END_FILE).read_text())['state'] == 'failed'
    _released(root)
    record = cc.load_launch_record(tmp_path / 'launch_record.json', code_commit='c0ffee')
    assert cc.precheck(record, tmp_path / 'vol', current_pod='podtest1')['decision'] == 'fresh'
    log.clear()
    second = _chain(tmp_path, _stages(log), local='fresh-local').run()
    assert second['state'] == 'complete' and second['attempt'] == 'attempt-002'
    assert log == ['preflight', 'attribution', 'aggregate_freeze', 'generation', 'report']
    assert (root / cc.DONE_FILE).read_text() == 'complete\n'
    with pytest.raises(ResumeRefused, match='new run ID'):
        cc.precheck(record, tmp_path / 'vol', current_pod='podtest1')


def _identity_stages(log, identity, **kw):
    stages = _stages(log, **kw)
    inner = stages['preflight']
    stages['preflight'] = lambda ctx: dict(inner(ctx) or {}, identity=identity)
    return stages


def test_resume_reruns_preflight_and_refuses_qualification_from_another_runtime(tmp_path):
    """Review P1: resume skipped preflight and qualification on their old seals alone."""
    log = []
    runtime = {'device_name': 'NVIDIA A100-SXM4-80GB', 'torch': '2.9.0+cu128', 'precision': 'ieee'}
    assert _chain(tmp_path, _identity_stages(log, runtime, fail='attribution')).run()['state'] == 'failed'
    # today's preflight fails: it runs and stops the chain (it is never skipped)
    log.clear()
    failing = _identity_stages(log, runtime, stop='preflight')
    summary = _chain(tmp_path, failing, local='l2').run()
    assert summary['state'] == 'stopped' and log == ['preflight']
    # another GPU/runtime: qualification's receipt does not cover it
    log.clear()
    other = dict(runtime, device_name='NVIDIA A40')
    summary = _chain(tmp_path, _identity_stages(log, other), local='l3').run()
    assert summary['state'] == 'stopped' and log == ['preflight']
    assert 'qualified under' in summary['stop_reason'] and 'device_name' in summary['stop_reason']
    # the same runtime resumes
    log.clear()
    summary = _chain(tmp_path, _identity_stages(log, runtime), local='l4').run()
    assert summary['state'] == 'complete'
    assert log == ['preflight', 'attribution', 'aggregate_freeze', 'generation', 'report']
    assert summary['stages']['qualification'] == {'status': 'complete', 'resumed': True}


def test_teardown_ends_a_timed_out_copy_before_requesting_the_stop(tmp_path):
    """Review P1: a timed-out copy thread was still running when the stop was requested."""
    import os
    from c_attribution_teardown_fakes import slow_copy
    record = cc.load_launch_record(sealed(tmp_path, **{'timeouts_seconds.copy': 2}), code_commit='c0ffee')
    local = tmp_path / 'local'
    alive_at_stop = []

    class StopRunner(Runner):
        def __call__(self, command, **kw):
            pid = int((local / 'copy.pid').read_text())
            try:
                os.kill(pid, 0)
                alive_at_stop.append(True)
            except ProcessLookupError:
                alive_at_stop.append(False)
            return super().__call__(command, **kw)

    chain = cc.Chain(record, local_root=local, volume=tmp_path / 'vol', stages=_stages([]), exporter=Exporter(),
                     runner=StopRunner(codes=(0,)), volume_identity='sl-0', copier=slow_copy, current_pod='podtest1')
    summary = chain.run()
    copy = summary['backups']['volume_copy']
    assert copy['outcome'] == 'timeout' and copy['ended'] is True and copy['verified'] is False
    assert alive_at_stop == [False]


def test_an_empty_copy_with_missing_expected_work_is_not_a_complete_backup(tmp_path):
    """Review P1: a copy of nothing was labelled verified despite missing expected runs."""
    from c_attribution_teardown_fakes import empty_copy
    record = cc.load_launch_record(sealed(tmp_path), code_commit='c0ffee')
    chain = cc.Chain(record, local_root=tmp_path / 'local', volume=tmp_path / 'vol', stages=_stages([]),
                     exporter=Exporter(), runner=Runner(), volume_identity='sl-0', copier=empty_copy,
                     current_pod='podtest1')
    summary = chain.run()
    copy = summary['backups']['volume_copy']
    assert copy['coverage'] == 'partial' and f'{cc.SUBTREE}/rows' in copy['missing']
    assert 'volume copy' not in summary['backups']['complete']
    assert 'volume copy: verified bytes, partial' in summary['backups']['statement']


def test_a_failed_run_reports_which_stages_never_produced_outputs(tmp_path):
    summary = _chain(tmp_path, _stages([], fail='generation')).run()
    assert summary['backups']['stages_not_completed'] == ['generation', 'report']
    assert summary['backups']['complete'] == []
    assert 'run did not complete' in summary['backups']['statement']


def test_the_record_pod_id_must_be_the_pod_running_the_launch(tmp_path):
    """Review P1: a stale record on another Pod would stop the old Pod and leave this one billing."""
    record = cc.load_launch_record(sealed(tmp_path), code_commit='c0ffee')
    with pytest.raises(ResumeRefused, match='podtest1.*current-pod-other|current-pod-other.*podtest1'):
        cc.precheck(record, tmp_path / 'vol', current_pod='current-pod-other')
    with pytest.raises(ResumeRefused, match='RUNPOD_POD_ID'):
        cc.precheck(record, tmp_path / 'vol', current_pod='')
    runner = Runner(codes=(0,))
    chain = cc.Chain(record, local_root=tmp_path / 'local', volume=tmp_path / 'vol', stages=_stages([]),
                     exporter=Exporter(), runner=runner, volume_identity='sl-0', current_pod='current-pod-other')
    summary = chain.run()
    assert summary['state'] == 'failed' and 'current-pod-other' in summary['failures'][0]['error']
    assert runner.commands == []  # never stop a Pod the record names when it is not this one
    assert summary['stop'] == dict(requested=False, reason=summary['stop']['reason'], attempts=[])


# --- review 06-10 follow-up -------------------------------------------------------------------------------

def test_no_complete_backup_withholds_the_stop_and_a_relaunch_backs_up_and_closes_the_run(tmp_path):
    """Follow-up P1: every stage succeeded, both backups failed, and the Pod was still stopped and `done` written."""
    from c_attribution_teardown_fakes import empty_copy
    record = cc.load_launch_record(sealed(tmp_path), code_commit='c0ffee')
    log, runner = [], Runner(codes=(0,))
    chain = cc.Chain(record, local_root=tmp_path / 'local', volume=tmp_path / 'vol', stages=_stages(log),
                     exporter=Exporter(fail=True), runner=runner, volume_identity='sl-0', copier=empty_copy,
                     current_pod='podtest1')
    summary = chain.run()
    root = cc.run_root(tmp_path / 'vol', 'c-rehearsal-pod-1')
    assert summary['state'] == 'complete' and summary['backups']['safe'] == []
    assert runner.commands == [] and summary['stop']['requested'] is False and summary['stop']['withheld'] is True
    assert not (root / cc.DONE_FILE).exists()             # the run stays open for backup recovery
    withheld = json.loads((root / cc.STOP_WITHHELD_FILE).read_text())
    assert 'no complete verified backup' in withheld['reason'] and 'c_attribution_launch.sh' in withheld['next']
    assert 'STOP WITHHELD' in summary['backups']['statement']
    end = json.loads((root / cc.ATTEMPT_END_FILE).read_text())
    assert end['stop'] == 'withheld' and end['writer_pid'] == os.getpid()
    # the Pod is still up with its local disk: relaunch with working backups; nothing is computed again
    log.clear()
    runner = Runner(codes=(0,))
    again = cc.Chain(record, local_root=tmp_path / 'local', volume=tmp_path / 'vol', stages=_stages(log),
                     exporter=Exporter(), runner=runner, volume_identity='sl-0', current_pod='podtest1')
    summary = again.run()
    assert summary['state'] == 'complete' and log == ['preflight']
    assert summary['backups']['safe'] == ['volume copy', 'HF export'] and runner.commands
    assert (root / cc.DONE_FILE).read_text() == 'complete\n' and not (root / cc.STOP_WITHHELD_FILE).exists()


def test_unsealed_local_work_counts_for_the_stop_decision(tmp_path):
    """A sealed unit of an unfinished stage is not in the expected outputs, but stopping would still lose it."""
    log = []
    stages = _stages(log)

    def attribution(ctx):
        au.seal_run(ctx.local_root / cc.SUBTREE / 'attribution/unit-0000', {'meta.json': {'unit': 0}})
        raise RuntimeError('crash after one unit')
    stages['attribution'] = attribution
    from c_attribution_teardown_fakes import empty_copy
    record = cc.load_launch_record(sealed(tmp_path), code_commit='c0ffee')
    runner = Runner(codes=(0,))
    summary = cc.Chain(record, local_root=tmp_path / 'local', volume=tmp_path / 'vol', stages=stages,
                       exporter=Exporter(fail=True), runner=runner, volume_identity='sl-0', copier=empty_copy,
                       current_pod='podtest1').run()
    assert f'{cc.SUBTREE}/attribution/unit-0000' in summary['backups']['volume_copy']['missing']
    assert runner.commands == []


# --- setup fingerprint and rebind (user, 2026-11-20) ----------------------------------------------------

QUALIFIED = {'runtime': {'device_name': 'NVIDIA A100-SXM4-80GB', 'torch': '2.9.0+cu128', 'cuda_runtime': '12.8',
                         'cudnn': 91002, 'cuda_libraries': {'nvidia-cudnn-cu12': '9.10.2.21',
                                                            'nvidia-cublas-cu12': '12.8.4.1'}},
             'bundle_sha256': 'b' * 64}


def _with(identity, dotted, value):
    out = json.loads(json.dumps(identity))
    parent, keys = out, dotted.split('.')
    for k in keys[:-1]:
        parent = parent[k]
    parent[keys[-1]] = value
    return out


@pytest.mark.parametrize('dotted,value,remedy', [
    ('runtime.cuda_libraries.nvidia-cudnn-cu12', '9.1.0.70', 'pip install nvidia-cudnn-cu12==9.10.2.21'),
    ('runtime.torch', '2.9.1+cu128', 'pip install torch==2.9.0+cu128'),
    ('runtime.device_name', 'NVIDIA A40', 'the same GPU type (NVIDIA A100-SXM4-80GB)'),
])
def test_a_resume_refusal_names_what_to_install_or_rent(tmp_path, dotted, value, remedy):
    log = []
    assert _chain(tmp_path, _identity_stages(log, QUALIFIED, fail='attribution')).run()['state'] == 'failed'
    summary = _chain(tmp_path, _identity_stages([], _with(QUALIFIED, dotted, value)), local='l2').run()
    assert summary['state'] == 'stopped' and remedy in summary['stop_reason'], summary['stop_reason']


def test_the_driver_is_recorded_but_does_not_refuse_a_resume(tmp_path):
    log = []
    stages = _identity_stages(log, QUALIFIED, fail='attribution')
    assert _chain(tmp_path, stages).run()['state'] == 'failed'
    log.clear()
    inner = _identity_stages(log, QUALIFIED)
    first = inner['preflight']
    inner['preflight'] = lambda ctx: dict(first(ctx), host={'driver': '570.86.10'})  # another host's driver
    summary = _chain(tmp_path, inner, local='l2').run()
    assert summary['state'] == 'complete'
    assert summary['stages']['preflight']['host'] == {'driver': '570.86.10'}


def test_a_rebound_record_continues_the_run_on_a_replacement_pod(tmp_path):
    """The old Pod died; on the new Pod one command reseals the record for it and the run continues."""
    log = []
    assert _chain(tmp_path, _stages(log, fail='attribution')).run()['state'] == 'failed'
    old = tmp_path / 'launch_record.json'
    new = cc.rebind(old, tmp_path / 'rebound.json', code_commit='c0ffee', pod_id='podnew2',
                    fill={'inputs.work_dir': '/new/work'})
    record = cc.load_launch_record(new, code_commit='c0ffee')
    assert record['pod_id'] == 'podnew2' and record['inputs']['work_dir'] == '/new/work'
    assert cc.run_sha256(record) == cc.run_sha256(cc.load_launch_record(old, code_commit='c0ffee'))
    log.clear()
    runner = Runner(codes=(0,))
    summary = cc.Chain(record, local_root=tmp_path / 'new-local', volume=tmp_path / 'vol', stages=_stages(log),
                       exporter=Exporter(), runner=runner, volume_identity='sl-0', current_pod='podnew2').run()
    assert summary['state'] == 'complete', summary['failures']
    assert log == ['preflight', 'attribution', 'aggregate_freeze', 'generation', 'report']   # nothing redone
    assert summary['stages']['rows'] == {'status': 'complete', 'resumed': True}
    assert [' '.join(c) for c in runner.commands] == ['runpodctl stop pod podnew2']      # never the old Pod
    state = json.loads((cc.run_root(tmp_path / 'vol', record['run_id']) / cc.STATE_FILE).read_text())
    assert [b['pod_id'] for b in state['bindings']] == ['podtest1', 'podnew2']


def test_rebind_changes_only_the_pod_binding(tmp_path):
    _chain(tmp_path, _stages([], fail='attribution')).run()
    old = tmp_path / 'launch_record.json'
    with pytest.raises(LaunchRefused, match='not a binding field'):
        cc.rebind(old, tmp_path / 'r.json', code_commit='c0ffee', pod_id='podnew2', fill={'hourly_cost_usd': 9})
    with pytest.raises(LaunchRefused, match='code commit'):
        cc.rebind(old, tmp_path / 'r.json', code_commit='other', pod_id='podnew2')
    with pytest.raises(LaunchRefused, match='RUNPOD_POD_ID'):
        cc.rebind(old, tmp_path / 'r.json', code_commit='c0ffee', pod_id='')
    # a record changed in anything else is still refused under the same run ID, on any Pod
    record = cc.load_launch_record(sealed(tmp_path, hourly_cost_usd=9.0, pod_id='podnew2'), code_commit='c0ffee')
    chain = cc.Chain(record, local_root=tmp_path / 'l2', volume=tmp_path / 'vol', stages=_stages([]),
                     exporter=Exporter(), runner=Runner(), volume_identity='sl-0', current_pod='podnew2')
    with pytest.raises(ResumeRefused, match='different launch record'):
        chain._start()


def test_rebind_on_a_pod_whose_old_writer_may_still_be_alive_is_refused(tmp_path):
    """A replacement Pod never takes over a writer whose heartbeat is fresh: the old Pod may still be running."""
    _chain(tmp_path, _stages([], fail='attribution')).run()
    root = cc.run_root(tmp_path / 'vol', 'c-rehearsal-pod-1')
    marker = json.loads((root / cc.MARKER_FILE).read_text())
    (root / cc.MARKER_FILE).write_text(json.dumps(dict(marker, state='running', host='old-pod-host',
                                                       heartbeat_unix=time.time())))
    new = cc.rebind(tmp_path / 'launch_record.json', tmp_path / 'r.json', code_commit='c0ffee', pod_id='podnew2')
    with pytest.raises(ResumeRefused, match='ambiguous writer'):
        cc.precheck(cc.load_launch_record(new, code_commit='c0ffee'), tmp_path / 'vol', current_pod='podnew2')


def test_the_approved_rehearsal_draft_seals_with_cost_not_tracked_and_untuned(tmp_path):
    """User 2026-11-20: cost is not tracked; untuned is a rehearsal condition; the [for approval] values stand."""
    path = cc.finalize_draft(DRAFT, tmp_path / 'r.json', code_commit='c0ffee', pod_id='podtest1', fill=FILL)
    record = cc.load_launch_record(path, code_commit='c0ffee')
    assert record['hourly_cost_usd'] == 'not-tracked'
    assert record['scope']['generation_conditions'] == ['intact', 'top10-adapters', 'global64-reads', 'untuned']
    assert record['qualification']['noop_greedy_tokens'] == 16
    assert '[for approval]' not in DRAFT.read_text()
    summary = cc.Chain(record, local_root=tmp_path / 'local', volume=tmp_path / 'vol', stages=_stages([]),
                       exporter=Exporter(), runner=Runner(), volume_identity='sl-0', current_pod='podtest1').run()
    assert summary['cost_estimate_usd'] is None
    assert 'cost not tracked' in (tmp_path / 'local' / cc.SUMMARY_DIR / summary['attempt'] / 'final' /
                                  'run_summary.md').read_text()
    with pytest.raises(LaunchRefused, match='hourly_cost_usd'):
        sealed(tmp_path, hourly_cost_usd='free')


# --- rehearsal finding (2026-11-22): a real Pod stop kills the runner inside the stop call -------------

class PodGone(BaseException):
    """The container ends while ``runpodctl stop`` runs: nothing after the call executes."""


class KillingRunner(Runner):
    def __call__(self, command, **kw):
        self.commands.append(command)
        raise PodGone('container stopped')


def test_the_closing_records_are_on_the_volume_before_the_stop_is_requested(tmp_path):
    runner = KillingRunner()
    chain = _chain(tmp_path, _stages([]), runner=runner)
    with pytest.raises(PodGone):
        chain.run()
    assert runner.commands                                      # the stop was requested
    root = cc.run_root(tmp_path / 'vol', 'c-rehearsal-pod-1')
    assert (root / cc.DONE_FILE).read_text() == 'complete\n'    # the run ID is closed
    end = json.loads((root / cc.ATTEMPT_END_FILE).read_text())
    assert end['state'] == 'complete' and end['writer_pid'] == os.getpid() and end['stop'] == 'requesting'
    stop = json.loads((root / cc.STOP_RECORD_FILE).read_text())
    assert stop['stop']['requested'] == 'in progress' and stop['steps'][-1] == 'stop_requesting'
    assert stop['stop']['commands'] == ['runpodctl stop pod podtest1', 'runpodctl pod stop podtest1']
    _released(root)
    record = cc.load_launch_record(tmp_path / 'launch_record.json', code_commit='c0ffee')
    with pytest.raises(ResumeRefused, match='already finished'):
        cc.precheck(record, tmp_path / 'vol', current_pod='podtest1')


def test_a_stop_that_returns_updates_the_closing_records_with_its_outcome(tmp_path):
    chain = _chain(tmp_path, _stages([]), runner=Runner(codes=(1, 1)))
    summary = chain.run()
    root = cc.run_root(tmp_path / 'vol', 'c-rehearsal-pod-1')
    assert not summary['stop']['requested']
    assert json.loads((root / cc.ATTEMPT_END_FILE).read_text())['stop'] == 'not-requested'
    stop = json.loads((root / cc.STOP_RECORD_FILE).read_text())
    assert stop['stop']['requested'] is False and len(stop['stop']['attempts']) == 2
    assert stop['steps'][-1] == 'stop_not_requested'
