"""Ticket 10: the C-attribution chain end to end through the real backend on CPU.

``Chain`` -> ``Stages`` -> the real ``TransformersBackend`` (``from_pretrained`` base, PEFT arms, the pinned-format
tokenizer, the bundle restored through the export path) on ``c_attribution_realpath_fixture``. Stubbed, and only
these: CUDA (device cpu; memory counters and availability), the volume mount/identity check and disk facts, the Hub
(private check; export through the fake Hub), and the Pod stop command. Sizes are reduced (cap, token budget,
resamples). torch, transformers, PEFT and tokenizers must import: a missing one fails this test, never skips it.
"""
import json
import os
import shutil
import subprocess
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest
import torch  # noqa: F401  (a hard import: this test fails, not skips, without the real dependencies)
import peft  # noqa: F401
import tokenizers  # noqa: F401
import transformers  # noqa: F401

import c_attribution_realpath_fixture as world
from pilot_eval import attribution_units as au
from pilot_eval import c_attribution_chain as cc
from pilot_eval import c_attribution_stages as cs
from pilot_eval.attribution_chain import LaunchRefused
from pilot_eval.c_attribution_backend import TransformersBackend

Usage = namedtuple('Usage', 'total used free')
DISK = Usage(120e9, 40e9, 80e9)
DEVICE_BYTES = 80 * 2 ** 30


def versions():
    return {m.__name__: m.__version__ for m in (torch, transformers, peft, tokenizers)}


def backend(w, volume, **probes):
    base = dict(cuda_available=True, volume_identity=lambda: [], private_repo=lambda repo: None,
                disk_usage=lambda path: DISK, reset_peak=lambda: None, peak_memory=lambda: 10 ** 9,
                total_memory=lambda: DEVICE_BYTES)
    base.update(probes)
    return TransformersBackend(w['record'], volume, device='cpu', probes=base, repo_root=w['code'])


class Runner:
    def __init__(self):
        self.commands = []

    def __call__(self, command, **kw):
        self.commands.append(command)
        return subprocess.CompletedProcess(command, 0, 'stopped', '')


def exporter(hub):
    def run(local_root, *, label, include, expected_units, expected_runs, repo):
        receipt = au.export(local_root, label=label, include=include, expected_units=expected_units,
                            expected_runs=expected_runs, client=hub, repo=repo, attempts=1,
                            temp_dir=Path(local_root) / 'tmp', on_event=lambda e: None)
        return dict(receipt, readback_verified=True)
    return run


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    return world.build(tmp_path_factory.mktemp('world'))


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv('HF_HOME', str(tmp_path / 'hf-home'))
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')


def chain(w, tmp_path, *, local='local', stages=None, be=None):
    volume = tmp_path / 'vol'
    be = be or backend(w, volume)
    runner = Runner()
    c = cc.Chain(w['record'], local_root=tmp_path / local, volume=volume, volume_identity='sl-0', runner=runner,
                 stages=stages or cs.Stages(w['record'], tmp_path / local, be).mapping(), exporter=exporter(w['hub']))
    return c, runner


def test_the_whole_chain_runs_through_the_real_backend(built, tmp_path, env):
    c, runner = chain(built, tmp_path)
    summary = c.run()
    assert summary['state'] == 'complete', (summary['failures'], summary.get('stop_reason'))
    print(json.dumps(versions()))
    local = tmp_path / 'local'
    qual = json.loads((local / cs.QUAL_DIR / 'qualification.json').read_text())
    assert all(qual['checks'].values()) and set(qual['arms']) == {'online-grpo', 'offline-grpo'}
    assert qual['details']['online-grpo']['capped_rows_available'] == 0
    assert qual['details']['offline-grpo']['capped_rows_available'] == 1
    report = json.loads((local / cs.REPORT_DIR / 'report.json').read_text())
    assert set(report['arms']) == {'online-grpo', 'offline-grpo'}
    assert [' '.join(c) for c in runner.commands] == ['runpodctl stop pod podcpu1']
    assert summary['backups']['hf_export']['verified'] and summary['backups']['volume_copy']['verified']


def test_old_metrics_are_bit_identical_with_and_without_seqlogp(built, tmp_path, env):
    from pilot_eval.attribution_instrument import Instrument
    be = backend(built, tmp_path / 'vol')
    stages = cs.Stages(built['record'], tmp_path / 'local', be)
    rows = []
    for r in stages.loaded('online-grpo')['rows'][:6]:  # old masks over the same positions: answer and its EOS
        targets = r['masks']['seq_targets']
        rows.append(dict(r, masks=dict(r['masks'], trace_targets=targets[:-1] or targets, stop_targets=targets[-1:])))
    model = be.load('online-grpo')
    kw = dict(stop_ids=[be.eos_token_id], pad_token_id=be.pad_token_id, identity_rtol=1e-4)
    old = Instrument(model, metric_order=('trace', 'stop'), **kw).attribute(rows)
    new = Instrument(model, metric_order=('trace', 'stop', 'seqlogp'), **kw).attribute(rows)
    for name in ('trace', 'stop'):
        assert np.array_equal(old['attr'][name], new['attr'][name], equal_nan=True)
        assert np.array_equal(old['metrics'][name], new['metrics'][name], equal_nan=True)
        for layer in old['neuron'][name]:
            assert np.array_equal(old['neuron'][name][layer], new['neuron'][name][layer], equal_nan=True)


# --- recorded Pod failures: each stops the chain before any GPU work ----------------------------------

def _stopped_at_preflight(w, tmp_path, **kw):
    c, runner = chain(w, tmp_path, **kw)
    summary = c.run()
    assert summary['state'] == 'stopped' and list(summary['stages']) == ['preflight'], summary['stages']
    assert not (tmp_path / 'local' / cs.QUAL_DIR).exists()
    assert runner.commands  # the Pod stop is still requested
    return summary['stop_reason']


def test_wrong_preflight_argument_starts_nothing(built, tmp_path):
    with pytest.raises(SystemExit):
        cc.main(['run', '--record', str(built['record_path']), '--volume', str(tmp_path)])  # --local missing
    assert not (tmp_path / 'c-attribution').exists()


@pytest.mark.parametrize('change,match', [
    ({'tolerances.neuron_sum_identity': '1e-4'}, 'neuron_sum_identity'),                     # YAML number as text
    ({'scope.generation_questions.online-grpo': [817, 611]}, 'online-grpo'),                 # nested type
    ({'pod_stop_commands': ['runpodctl stop {id}', 'runpodctl pod stop {id}']}, 'both syntaxes'),  # stop syntax
])
def test_record_failures_refuse_the_seal(built, tmp_path, change, match):
    import yaml
    body = yaml.safe_load(built['draft'].read_text())
    for dotted, value in change.items():
        parent, keys = body, dotted.split('.')
        for k in keys[:-1]:
            parent = parent[k]
        parent[keys[-1]] = value
    draft = tmp_path / 'draft.yaml'
    draft.write_text(yaml.safe_dump(body))
    fill = {k: built['record']['inputs'][k.split('.')[1]] for k in cc.USER_FILLED if k.startswith('inputs.')}
    with pytest.raises(LaunchRefused, match=match):
        cc.finalize_draft(draft, tmp_path / 'r.json', code_commit='x', pod_id='p', fill=fill)


def test_hf_cache_on_the_volume_stops(built, tmp_path, env, monkeypatch):
    monkeypatch.setenv('HF_HOME', str(tmp_path / 'vol' / 'hf'))
    assert 'HF_HOME' in _stopped_at_preflight(built, tmp_path)


def test_missing_catalog_stops(built, tmp_path, env):
    code = tmp_path / 'code-without-catalog'
    shutil.copytree(built['code'], code)
    shutil.rmtree(code / 'catalog')
    be = TransformersBackend(built['record'], tmp_path / 'vol', device='cpu', repo_root=code,
                             probes=dict(cuda_available=True, volume_identity=lambda: [],
                                         private_repo=lambda repo: None, disk_usage=lambda p: DISK))
    assert 'catalog' in _stopped_at_preflight(built, tmp_path, be=be)


def test_a_modified_adapter_after_restore_stops(built, tmp_path, env):
    restored = tmp_path / 'restored'
    shutil.copytree(built['restored'], restored)
    record = json.loads(json.dumps(built['record']))
    record['inputs']['artifact_root'] = str(restored)
    adapter = restored / world.ADAPTERS['offline-grpo'] / 'adapter_model.safetensors'
    data = bytearray(adapter.read_bytes())
    data[-1] ^= 1
    adapter.write_bytes(bytes(data))
    w = dict(built, record=record)
    reason = _stopped_at_preflight(w, tmp_path)
    assert 'differs: adapter:offline-grpo:64:adapter_model.safetensors' in reason


def test_the_real_volume_check_without_its_identity_file_stops(built, tmp_path, env):
    be = TransformersBackend(built['record'], tmp_path / 'vol', device='cpu', repo_root=built['code'],
                             probes=dict(cuda_available=True, private_repo=lambda repo: None, disk_usage=lambda p: DISK))
    (tmp_path / 'vol').mkdir()
    assert 'identity' in _stopped_at_preflight(built, tmp_path, be=be)


# --- resume ---------------------------------------------------------------------------------------

def test_an_interrupted_unit_and_a_deleted_local_root_resume_from_the_volume(built, tmp_path, env):
    import time
    be = backend(built, tmp_path / 'vol')
    stages = cs.Stages(built['record'], tmp_path / 'local', be)
    mapping = stages.mapping()
    real = mapping['attribution']

    class Interrupted(Exception):
        pass

    def interrupted(ctx):
        original = ctx.boundary

        def boundary(**kw):
            original(**kw)
            raise Interrupted('Pod lost after the first sealed unit')
        ctx.boundary = boundary
        return real(ctx)

    mapping['attribution'] = interrupted
    c, _ = chain(built, tmp_path, stages=mapping)
    assert c.run()['state'] == 'failed'
    root = cc.run_root(tmp_path / 'vol', built['record']['run_id'])
    first_unit = f'{cs.ATTR_DIR}/online-grpo/unit-0000'
    au.verify_run(root / first_unit)                              # the sealed unit reached the volume
    sealed_at = json.loads((root / first_unit / 'complete.json').read_text())['sealed_unix']
    # the Pod died: no done file, a stale marker of a dead writer, and the local root is gone
    (root / cc.DONE_FILE).unlink()
    dead = subprocess.Popen(['true'])
    dead.wait()
    marker = json.loads((root / cc.MARKER_FILE).read_text())
    (root / cc.MARKER_FILE).write_text(json.dumps(dict(marker, state='running', pid=dead.pid,
                                                       heartbeat_unix=time.time() - 3600)))
    shutil.rmtree(tmp_path / 'local')
    c2, runner = chain(built, tmp_path, local='local')
    summary = c2.run()
    assert summary['state'] == 'complete', summary['failures']
    assert summary['resume']['decision'] == 'take-over'
    assert first_unit in summary['restored_from_volume']['runs']
    assert summary['stages']['qualification'] == {'status': 'complete', 'resumed': True}
    # the restored unit was reused, not recomputed
    assert json.loads((tmp_path / 'local' / first_unit / 'complete.json').read_text())['sealed_unix'] == sealed_at
    assert runner.commands
