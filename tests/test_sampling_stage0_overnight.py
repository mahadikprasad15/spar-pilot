import json
import subprocess

import pytest

from pilot_eval import sampling_stage0_overnight as overnight


class Ops:
    stop_requested = False
    def __init__(self, fail_at=None):
        self.calls = []; self.fail_at = fail_at
    def preflight(self): self.calls.append('preflight')
    def plans(self): self.calls.append('plans'); return {'agreement': 'a.json'}
    def stage(self, stage):
        self.calls.append(stage)
        if stage == self.fail_at: raise RuntimeError(stage+' child failed')
        return {'plan': stage}
    def export(self, partial=False):
        self.calls.append('export'); return {'repo': 'r', 'revision': '0'*40, 'manifest_sha256': 'm'}
    def stop(self):
        self.calls.append('stop')
    def reverify(self, stage, value):
        self.calls.append('reverify-'+stage)
        if stage == getattr(self, 'tampered', None): raise ValueError(stage+' evidence changed')
    def reverify_export(self, receipt): self.calls.append('reverify-export')


def test_stages_run_in_order_and_resume_after_failure_without_repeats(tmp_path):
    ops = Ops(fail_at='drift')
    with pytest.raises(RuntimeError, match='drift'):
        overnight.orchestrate(tmp_path, ops)
    assert ops.calls == ['preflight', 'plans', 'agreement', 'behaviour', 'drift']
    assert not (tmp_path/'drift.complete.json').exists()
    ops.fail_at = None; ops.calls = []
    overnight.orchestrate(tmp_path, ops)
    assert ops.calls == ['preflight', 'reverify-agreement', 'reverify-behaviour', 'drift', 'geometry', 'export']
    ops.calls = []; overnight.orchestrate(tmp_path, ops)
    # Completed work is never repeated, but its sealed evidence is re-checked before it is trusted.
    assert ops.calls == ['preflight']+['reverify-'+s for s in overnight.STAGES]+['reverify-export']


def test_changed_completed_result_stops_before_export_and_pod_stop(tmp_path):
    ops = Ops(); ops.stop_requested = True
    ops.export = lambda: (_ for _ in ()).throw(ValueError('HF down'))
    with pytest.raises(ValueError):
        overnight.orchestrate(tmp_path, ops)
    ops.tampered = 'behaviour'; ops.calls = []
    ops.export = lambda: ops.calls.append('export') or {'repo': 'r', 'revision': '0'*40, 'manifest_sha256': 'm'}
    with pytest.raises(ValueError, match='behaviour evidence changed'):
        overnight.orchestrate(tmp_path, ops)
    assert 'export' not in ops.calls and 'stop' not in ops.calls


def test_completion_record_declares_its_scope(tmp_path):
    overnight.orchestrate(tmp_path, Ops())
    scope = json.loads((tmp_path/'complete.json').read_text())['scope']
    assert scope['stages'] == list(overnight.STAGES)
    assert 'historical Base/Gold/GRPO behaviour join' in scope['not_included']


MOUNTS = '''overlay / overlay rw 0 0
/dev/sda1 /etc/hosts ext4 rw 0 0
sl-0 /workspace fuse.geesefs rw 0 0
'''


def test_persistent_mount_is_established_or_the_run_stops(tmp_path):
    assert overnight.persistent_mount('/workspace/artifacts', MOUNTS, None) == {'mount': '/workspace', 'fstype': 'fuse.geesefs'}
    assert overnight.persistent_mount('/workspace/artifacts', MOUNTS, '/workspace')['mount'] == '/workspace'
    with pytest.raises(ValueError, match='container disk'):
        overnight.persistent_mount('/root/artifacts', MOUNTS, None)
    with pytest.raises(ValueError, match='mount'):
        overnight.persistent_mount('/workspace/artifacts', MOUNTS, '/runpod-volume')


def test_pod_stop_only_after_verified_export(tmp_path):
    ops = Ops(); ops.stop_requested = True
    def failing_export(): raise ValueError('HF unavailable')
    ops.export = failing_export
    with pytest.raises(ValueError, match='HF'):
        overnight.orchestrate(tmp_path, ops)
    assert 'stop' not in ops.calls and not (tmp_path/'complete.json').exists()
    ops.export = lambda: {'repo': 'r', 'revision': '0'*40, 'manifest_sha256': 'm'}
    overnight.orchestrate(tmp_path, ops)
    assert ops.calls[-1] == 'stop' and json.loads((tmp_path/'complete.json').read_text())['external_copy_verified'] is True
    states = [json.loads(l)['state'] for l in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert states[-3:] == ['exported-verified', 'pod-stop-requested', 'pod-stop-command-returned']


def _child(tmp_path):
    (tmp_path/'plan.json').write_text(json.dumps({'run_path': 'runs/x'}))
    return {'root': str(tmp_path), 'retries': 1, 'hours': {'drift': '1'}}


def test_crashed_child_is_relaunched_once_to_resume(tmp_path):
    cfg = _child(tmp_path); codes = iter([1, 0]); seen = []
    def runner(command, **kwargs):
        seen.append(command); assert kwargs['env']['HF_HUB_OFFLINE'] == '1' and kwargs['timeout'] == 3600
        return subprocess.CompletedProcess(command, next(codes))
    overnight.run_child('drift', tmp_path/'plan.json', cfg, tmp_path/'logs/drift.log', runner=runner)
    assert len(seen) == 2 and seen[0][1:4] == ['-m', 'pilot_eval.sampling_stage0_drift', 'run']


def test_integrity_failure_and_timeout_are_never_retried(tmp_path):
    cfg = _child(tmp_path); calls = []
    def crash(command, **kwargs):
        calls.append(1); (tmp_path/'runs/x/meta').mkdir(parents=True, exist_ok=True)
        (tmp_path/'runs/x/meta/integrity-failure.json').write_text('{}')
        return subprocess.CompletedProcess(command, 1)
    with pytest.raises(RuntimeError, match='drift child failed'):
        overnight.run_child('drift', tmp_path/'plan.json', cfg, tmp_path/'logs/drift.log', runner=crash)
    assert calls == [1]
    (tmp_path/'runs/x/meta/integrity-failure.json').unlink(); calls.clear()
    def hang(command, **kwargs):
        calls.append(1); raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    with pytest.raises(RuntimeError, match='timeout'):
        overnight.run_child('drift', tmp_path/'plan.json', cfg, tmp_path/'logs/drift.log', runner=hang)
    assert calls == [1]


def test_sources_manifest_reads_training_wrapper_record(tmp_path):
    (tmp_path/'gen.json').write_text(json.dumps({'settings': {'local_model': 'inputs/models/rev'}}))
    (tmp_path/'data.json').write_text(json.dumps({'generation_plan': 'gen.json'}))
    entries = []
    for arm in (0, 2, 5):
        (tmp_path/f'k{arm}.json').write_text(json.dumps({'gold_config': 'gold.json', 'data_plan': 'data.json'}))
        entries.append({'arm': arm, 'config': f'k{arm}.json', 'job': f'jobs/k{arm}/job.plan.json'})
    (tmp_path/'jobs.json').write_text(json.dumps(entries))
    assert overnight.sources_manifest(tmp_path, 'jobs.json') == {'gold_config': 'gold.json', 'weights_dir': 'inputs/models/rev',
        'training_plans': {'k0': 'k0.json', 'k2': 'k2.json', 'k5': 'k5.json'}}
    (tmp_path/'jobs.json').write_text(json.dumps(entries[:2]))
    with pytest.raises(ValueError, match='k0/k2/k5'):
        overnight.sources_manifest(tmp_path, 'jobs.json')


def _args(tmp_path):
    (tmp_path/'jobs.json').write_text('[]'); (tmp_path/'inputs.json').write_text('{}')
    return ['--root', str(tmp_path), '--training-jobs', 'jobs.json', '--inputs', 'inputs.json', '--name', 'night']


def test_second_wrapper_never_overwrites_live_status(tmp_path):
    from pilot_eval.training import run_lock
    folder = tmp_path/'runs/stage0/overnight/night'
    with run_lock(folder):
        (folder/'status.json').write_text(json.dumps({'state': 'behaviour'}))
        assert overnight.main(_args(tmp_path)) == 2
        assert json.loads((folder/'status.json').read_text()) == {'state': 'behaviour'}


def test_crash_records_failed_stage_and_traceback(tmp_path, monkeypatch):
    def crash(folder, ops):
        overnight.record(folder, 'drift'); raise TypeError('bug in an unexpected place')
    monkeypatch.setattr(overnight, 'orchestrate', crash)
    assert overnight.main(_args(tmp_path)) == 1
    folder = tmp_path/'runs/stage0/overnight/night'
    value = json.loads((folder/'status.json').read_text())
    assert value['state'] == 'stopped' and value['failed_stage'] == 'drift' and value['error_type'] == 'TypeError'
    assert 'bug in an unexpected place' in (folder/'failure-traceback.txt').read_text()


def test_stop_pod_requires_volume_identity(tmp_path):
    with pytest.raises(SystemExit):
        overnight.main(_args(tmp_path)+['--stop-pod'])


def test_status_is_read_only(tmp_path):
    out = overnight.status(tmp_path, 'night')
    assert out['overnight'] is None and set(out['stages']) == {'agreement', 'behaviour', 'drift', 'geometry'}
    assert not (tmp_path/'runs').exists()


def test_stage_wiring_passes_each_plan_to_the_next(tmp_path, monkeypatch):
    from pilot_eval import sampling_stage0_agreement as agreement, sampling_stage0_evaluation as evaluation
    from pilot_eval import sampling_stage0_drift as drift, sampling_stage0_geometry as geometry
    calls = []
    def plan(name):
        path = tmp_path/'plans/night'/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('{}'); return path
    monkeypatch.setattr(overnight, 'sources_manifest', lambda root, jobs: {'training_plans': {}})
    monkeypatch.setattr(agreement, 'prepare', lambda manifest, root, name, gpu: calls.append(('agreement', manifest.name, gpu)) or plan('a.json'))
    monkeypatch.setattr(evaluation, 'prepare', lambda manifest, root, name, gpu, agreement_plan:
                        calls.append(('behaviour', agreement_plan.name)) or plan('sampling-stage0.evaluation.json'))
    monkeypatch.setattr(drift, 'prepare', lambda behaviour, inputs, root, name: calls.append(('drift', behaviour.name, inputs.name)) or plan('d.json'))
    monkeypatch.setattr(geometry, 'prepare', lambda behaviour, root, name: calls.append(('geometry', behaviour.name)) or plan('g.json'))
    monkeypatch.setattr(overnight, 'run_child', lambda stage, plan, cfg, log: calls.append(('child', stage, plan.name)))
    for module in (agreement, evaluation, drift, geometry):
        monkeypatch.setattr(module, 'verify', lambda plan, root: {'batch_sizes': {'k0': 16}} if plan.name == 'a.json' else {'variants': {}})
    (tmp_path/'inputs.json').write_text('{}')
    cfg = {'root': str(tmp_path), 'state_dir': 'runs/stage0/overnight/night', 'name': 'night', 'gpu': 'L4',
           'training_jobs': 'jobs.json', 'inputs': 'inputs.json'}
    ops = overnight.Operations(cfg); ops.state.mkdir(parents=True)
    (ops.state/'plans.complete.json').write_text(json.dumps(ops.plans()))
    results = {s: ops.stage(s) for s in overnight.STAGES}
    assert calls == [('agreement', 'sources.json', 'L4'), ('child', 'agreement', 'a.json'),
                     ('behaviour', 'a.json'), ('child', 'behaviour', 'sampling-stage0.evaluation.json'),
                     ('drift', 'sampling-stage0.evaluation.json', 'inputs.json'), ('child', 'drift', 'd.json'),
                     ('geometry', 'sampling-stage0.evaluation.json'), ('child', 'geometry', 'g.json')]
    assert results['agreement']['summary'] == {'batch_sizes': {'k0': 16}}


def test_reverify_detects_changed_plan_or_result(tmp_path, monkeypatch):
    from pilot_eval import sampling_stage0_drift as drift
    from pilot_eval.training import file_hash
    plan = tmp_path/'plans/night/d.json'; plan.parent.mkdir(parents=True); plan.write_text('{}')
    ops = overnight.Operations({'root': str(tmp_path), 'state_dir': 's', 'name': 'night'})
    value = {'plan': 'plans/night/d.json', 'plan_sha256': file_hash(plan), 'summary': {'controls': {'untuned': {'exact_zero': True}}}}
    monkeypatch.setattr(drift, 'verify', lambda p, r: {'controls': {'untuned': {'exact_zero': True}}, 'cells': []})
    ops.reverify('drift', value)
    monkeypatch.setattr(drift, 'verify', lambda p, r: {'controls': {'untuned': {'exact_zero': False}}})
    with pytest.raises(ValueError, match='evidence changed'):
        ops.reverify('drift', value)
    plan.write_text('{"edited": true}')
    with pytest.raises(ValueError, match='plan changed'):
        ops.reverify('drift', value)


def test_geometry_failure_still_exports_gpu_results_but_never_stops_the_pod(tmp_path):
    ops = Ops(fail_at='geometry'); ops.stop_requested = True
    with pytest.raises(RuntimeError, match='geometry'):
        overnight.orchestrate(tmp_path, ops)
    assert 'export' in ops.calls and 'stop' not in ops.calls
    partial = json.loads((tmp_path/'partial-export.json').read_text())
    assert partial['state'] == 'exported-partial' and partial['missing_stages'] == ['geometry']
    assert not (tmp_path/'complete.json').exists() and not (tmp_path/'export.complete.json').exists()
    ops.fail_at = None; ops.calls = []
    overnight.orchestrate(tmp_path, ops) # rerun finishes geometry and makes the full export
    assert ops.calls[-2:] == ['export', 'stop'] and json.loads((tmp_path/'complete.json').read_text())['state'] == 'exported-verified'


def test_gpu_environment_is_frozen_across_stages(tmp_path):
    state = tmp_path/'state'
    env = {'device': 'NVIDIA L4', 'driver': '580', 'session_id': 'boot-1', 'git_commit': 'a'}
    overnight.same_environment(state, env)
    overnight.same_environment(state, {**env, 'session_id': 'boot-2', 'git_commit': 'b'}) # restart or code commit
    with pytest.raises(ValueError, match='environment'):
        overnight.same_environment(state, {**env, 'driver': '570'})


def test_unreported_mount_must_be_the_runpod_network_volume():
    with pytest.raises(ValueError, match='network volume'):
        overnight.persistent_mount('/data/artifacts', '/dev/sdb1 /data ext4 rw 0 0\n', None)
    with pytest.raises(ValueError, match='network volume'):
        overnight.persistent_mount('/workspace/artifacts', '/dev/sdb1 /workspace ext4 rw 0 0\n', None)


def test_export_resume_checks_the_saved_download_verification(tmp_path, monkeypatch):
    from pilot_eval import sampling_sft_exports as exports
    folder = tmp_path/'s/export'; folder.mkdir(parents=True)
    monkeypatch.setattr(exports, 'verify_manifest', lambda path, root: {'sha256': 'm', 'files': {'a': {}, 'b': {}}})
    (folder/'hf-upload.json').write_text(json.dumps({'revision': '0'*40, 'archive_sha256': 'x'}))
    receipt = {'manifest_sha256': 'm', 'revision': '0'*40}
    ops = overnight.Operations({'root': str(tmp_path), 'state_dir': 's', 'name': 'night'})
    (folder/'hf-download-verification.json').write_text(json.dumps({'archive_sha256': 'x', 'verified_files': 3}))
    ops.reverify_export(receipt)
    (folder/'hf-download-verification.json').write_text(json.dumps({'archive_sha256': 'y', 'verified_files': 3}))
    with pytest.raises(ValueError, match='Export evidence'):
        ops.reverify_export(receipt)
    (folder/'hf-download-verification.json').write_text(json.dumps({'archive_sha256': 'x', 'verified_files': 2}))
    with pytest.raises(ValueError, match='Export evidence'):
        ops.reverify_export(receipt)
