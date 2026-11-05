"""Finishing a partial export after the driver stopped at its export step."""
import importlib.util
import json
from pathlib import Path

import pytest

from test_offline_grpo_overnight import real_export_run


def _script():
    path = Path(__file__).parent.parent/'scripts/offline_grpo_finish_export.py'
    spec = importlib.util.spec_from_file_location('offline_grpo_finish_export', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _failed_export(tmp_path, monkeypatch):
    workflow, path, auth, deps, hub, directory = real_export_run(tmp_path, monkeypatch)
    real = workflow._export

    def fail(*args, **kwargs):
        raise ValueError('Missing required export payload: tokens.tmp.npz')
    monkeypatch.setattr(workflow, '_export', fail)
    state = workflow.run(path, tmp_path, auth, dependencies=deps)
    monkeypatch.setattr(workflow, '_export', real)
    assert state['state'] == 'incomplete' and state['export']['state'] == 'failed'
    return workflow, path, deps, hub, directory


def test_finish_export_leaves_out_temporary_files_and_records_the_result(tmp_path, monkeypatch, capsys):
    script = _script()
    workflow, path, deps, hub, directory = _failed_export(tmp_path, monkeypatch)
    leftover = tmp_path/'runs/child/units/a/tokens.tmp.npz'
    leftover.parent.mkdir(parents=True)
    leftover.write_bytes(b'gone')
    (leftover.parent/'tokens.npz').write_bytes(b'sealed')

    manifest, skipped = script.build_manifest(path, tmp_path)
    value = json.loads(manifest.read_text())
    assert skipped == ['runs/child/units/a/tokens.tmp.npz']
    assert 'runs/child/units/a/tokens.npz' in value['files']
    assert 'runs/child/units/a/tokens.tmp.npz' not in value['files']
    assert value['partial'] is True and not hub.published
    assert workflow.status(path, tmp_path)['export']['state'] == 'failed'
    assert 'GB' in script.summary(manifest)

    leftover.unlink()
    assert script.main(['--root', str(tmp_path), '--config', str(path), '--upload', '--manifest', str(manifest)],
                       dependencies=deps) == 0
    state = workflow.status(path, tmp_path)
    assert state['state'] == 'incomplete'
    assert state['export']['state'] == 'partial_verified'
    assert state['export']['manifest_sha256'] == value['sha256']
    assert hub.published


def test_finish_export_refuses_a_second_export_and_a_foreign_manifest(tmp_path, monkeypatch):
    script = _script()
    workflow, path, deps, hub, directory = _failed_export(tmp_path, monkeypatch)
    manifest, _ = script.build_manifest(path, tmp_path)
    outside = tmp_path/'manifest.json'
    outside.write_text(manifest.read_text())
    with pytest.raises(ValueError):
        script.publish(path, tmp_path, outside, dependencies=deps)
    script.publish(path, tmp_path, manifest, dependencies=deps)
    with pytest.raises(ValueError, match='already exported'):
        script.publish(path, tmp_path, manifest, dependencies=deps)
    with pytest.raises(ValueError, match='already exported'):
        script.build_manifest(path, tmp_path)
