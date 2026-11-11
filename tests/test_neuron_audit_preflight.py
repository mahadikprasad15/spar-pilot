"""Ticket 03b worker preflight: environment order, interpreter, storage ops, capacity, online HF check."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pilot_eval import neuron_audit_preflight as pf

GOOD_ENV = dict(PYTHONUNBUFFERED='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')


def test_offline_flags_set_and_late_import_detected():
    env = dict(GOOD_ENV, HF_TOKEN='secret-value')
    record, problems = pf.prepare_environment(env, modules={})
    assert problems == [] and env['HF_HUB_OFFLINE'] == '1' and env['HF_HUB_ENABLE_HF_TRANSFER'] == '0'
    assert record['hf_token_set'] is True and 'secret-value' not in json.dumps(record)
    _, problems = pf.prepare_environment(dict(GOOD_ENV), modules={'huggingface_hub': object()})
    assert any('after importing' in p for p in problems)
    _, problems = pf.prepare_environment(dict(GOOD_ENV, HF_HUB_OFFLINE='1'), modules={'huggingface_hub': object()})
    assert problems == []


def test_missing_hf_transfer_unbuffered_and_cublas_reported(monkeypatch):
    import builtins
    real = builtins.__import__
    monkeypatch.setattr(builtins, '__import__', lambda name, *a, **k: (_ for _ in ()).throw(ImportError())
                        if name == 'hf_transfer' else real(name, *a, **k))
    _, problems = pf.prepare_environment({'HF_HUB_ENABLE_HF_TRANSFER': '1'}, modules={})
    assert any('hf_transfer' in p for p in problems)
    assert any('PYTHONUNBUFFERED' in p for p in problems) and any('CUBLAS' in p for p in problems)


def test_interpreter_and_package_pins():
    versions = {'torch': '2.9.0', 'peft': '0.18.0'}
    record, problems = pf.check_interpreter('/venv/bin/python', '/venv', versions, executable='/venv/bin/python',
                                            prefix='/venv', version=versions.get)
    assert problems == [] and record['packages'] == versions
    _, problems = pf.check_interpreter('/venv/bin/python', '/venv', versions, executable='/usr/bin/python3',
                                       prefix='/usr', version=lambda n: '2.7.1' if n == 'torch' else '0.18.0')
    assert len(problems) == 3
    _, problems = pf.check_interpreter(None, None, {}, executable='x', prefix='y', version=versions.get)
    assert any('not recorded' in p for p in problems)


def test_storage_probe_runs_writer_operations_and_cleans_up(tmp_path):
    record, problems = pf.storage_probe(tmp_path, 3 << 20, label='volume')
    assert problems == [] and list(record['operations']) == ['create_write_fsync', 'read_back_hash', 'rename',
                                                              'copyfile', 'delete']
    assert not list(tmp_path.iterdir())


def test_storage_probe_reports_failure(tmp_path, monkeypatch):
    def no_copy(*a, **k):
        raise PermissionError('Operation not permitted')  # e.g. FUSE refusing metadata changes
    monkeypatch.setattr(pf.shutil, 'copyfile', no_copy)
    record, problems = pf.storage_probe(tmp_path, label='volume')
    assert problems == ['volume: copyfile failed (PermissionError)']
    assert record['operations']['delete']['ok'] is True and not list(tmp_path.iterdir())


def test_capacity_uses_provisioned_size_and_60_percent_rule(tmp_path):
    (tmp_path / 'existing.bin').write_bytes(b'x' * 1000)
    capacity, problems = pf.usable_capacity(tmp_path, 11_000)
    assert problems == [] and capacity['usable_bytes'] == 10_000
    ok, problems = pf.confirm_capacity(tmp_path, capacity, 5_000)
    assert problems == [] and ok['probe']['bytes'] == 250
    _, problems = pf.confirm_capacity(tmp_path, capacity, 7_000)
    assert any('60%' in p for p in problems)
    _, problems = pf.usable_capacity(tmp_path, None)
    assert problems == ['storage.provisioned_volume_bytes is not recorded']


class FakeHub:
    def __init__(self, private=True, corrupt=False):
        self.private, self.corrupt, self.store = private, corrupt, {}

    def info(self, repo):
        return SimpleNamespace(private=self.private)

    def publish(self, repo, files):
        self.store.update({n: Path(p).read_bytes() for n, p in files.items()})
        return 'f' * 40

    def fetch_to(self, repo, name, revision, directory):
        path = Path(directory) / Path(name).name
        path.write_bytes(self.store[name] + (b'x' if self.corrupt else b''))
        return path


def test_online_hf_check_private_write_readback(tmp_path):
    record, problems = pf.hf_online_check(FakeHub(), 'o/r', tmp_path)
    assert problems == [] and record['readback_matches'] is True and record['revision'] == 'f' * 40
    assert pf.hf_online_check(FakeHub(private=False), 'o/r', tmp_path)[1] == ['HF export repository is not private']
    assert pf.hf_online_check(FakeHub(corrupt=True), 'o/r', tmp_path)[1] == ['HF readback hash differs']


def test_online_check_runs_in_a_separate_online_process(tmp_path):
    calls = []

    def runner(command, env, **kw):
        calls.append((command, env))
        return SimpleNamespace(returncode=0, stdout=json.dumps(dict(record=dict(ok=1), problems=[])) + '\n', stderr='')

    record, problems = pf.hf_online_subprocess('o/r', tmp_path, runner=runner)
    command, env = calls[0]
    assert problems == [] and record == dict(ok=1)
    assert command[1:4] == ['-m', 'pilot_eval.neuron_audit_preflight', 'hf-check'] and env['HF_HUB_OFFLINE'] == '0'
    failing = lambda *a, **k: SimpleNamespace(returncode=1, stdout='', stderr='401 Unauthorized')
    assert pf.hf_online_subprocess('o/r', tmp_path, runner=failing)[1] == ['online HF export check process failed']


def test_run_collects_every_problem_and_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, 'precision_and_device', lambda: (dict(device='cpu'), []))
    plan = dict(preflight=dict(expected_sys_executable='a', expected_sys_prefix='b', export_repo='o/r'),
                runtime=dict(packages={}, expected_device='NVIDIA A100-SXM4-80GB'),
                storage=dict(provisioned_volume_bytes=None))
    with pytest.raises(pf.PreflightFailure) as info:
        pf.run(plan, volume_dir=tmp_path, local_temp=tmp_path, offline_load=lambda: 1 / 0,
               environ=dict(GOOD_ENV), interpreter=lambda *a: ({}, []))
    problems = info.value.problems
    assert any('device cpu' in p for p in problems) and any('offline model' in p for p in problems)
    assert 'storage.provisioned_volume_bytes is not recorded' in problems
    assert 'HF online export check was not run' in problems
