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
                total_memory=lambda: DEVICE_BYTES, pinned_base_model=w['pins'])
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
                 stages=stages or cs.Stages(w['record'], tmp_path / local, be).mapping(), exporter=exporter(w['hub']),
                 current_pod='podcpu1')
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
    assert summary['backups']['complete'] == ['volume copy', 'HF export']
    # review 06-10 P2: the results the spec asks for are in the sealed outputs, as numbers
    for arm in ('online-grpo', 'offline-grpo'):
        boot = json.loads((local / cs.AGG_DIR / arm / 'bootstrap.json').read_text())
        selected = boot['selected']
        from pilot_eval import c_attribution_selection as sel
        freeze = sel.read_freeze(local, arm)
        assert {r['group'] for r in selected['reads']} <= {'global64-reads', 'hot64-reads', 'layer64-reads'}
        assert selected['reads']
        roles = {r['role'] for r in selected['reads']}
        assert 'target' in roles
        intervals = selected['intervals']
        assert set(intervals) == {'p2.5', 'p97.5'}
        for values in intervals.values():
            assert len(values) == len(selected['reads']) and all(isinstance(v, float) for v in values)
        for values in boot['neurons']['per_layer'].values():   # were NumPy display strings before the fix
            assert isinstance(values, list) and all(isinstance(v, float) for v in values)
        assert freeze
    for arm in ('online-grpo', 'offline-grpo'):  # the rehearsal record's conditions leave untuned out: said so
        assert report['arms'][arm]['untuned_note'] == 'untuned not in the record scope'


def test_the_untuned_row_is_reported_alongside_when_in_scope(tmp_path, env):
    w = world.build(tmp_path / 'world', record_changes={
        'scope.generation_conditions': ['intact', 'top10-adapters', 'global64-reads', 'untuned']})
    c, _ = chain(w, tmp_path)
    summary = c.run()
    assert summary['state'] == 'complete', (summary['failures'], summary.get('stop_reason'))
    report = json.loads((tmp_path / 'local' / cs.REPORT_DIR / 'report.json').read_text())
    for arm in ('online-grpo', 'offline-grpo'):
        untuned = report['arms'][arm]['conditions']['untuned']
        assert untuned['status'] == 'run' and isinstance(untuned['accuracy'], float)
        assert set(untuned['delta_pp_vs_intact']) == {'estimate', 'p2.5', 'p97.5'}  # paired with intact
        assert 'untuned_note' not in report['arms'][arm]


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
                             probes=dict(cuda_available=True, volume_identity=lambda: [], pinned_base_model=built['pins'],
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
                             probes=dict(cuda_available=True, private_repo=lambda repo: None, disk_usage=lambda p: DISK,
                                         pinned_base_model=built['pins']))
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
    (root / cc.DONE_FILE).unlink(missing_ok=True)
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


# --- review 06-10 -----------------------------------------------------------------------------------

def test_backend_sets_reads_back_and_records_ieee_precision(built, tmp_path, env):
    from pilot_eval.neuron_audit_collector import PRECISION_BACKENDS, _backend
    for path in PRECISION_BACKENDS:          # start from TF32 everywhere: the backend must set IEEE itself
        _backend(path).fp32_precision = 'tf32'
    be = backend(built, tmp_path / 'vol')
    runtime = be.runtime()
    assert runtime['precision'] == dict({p: 'ieee' for p in PRECISION_BACKENDS}, matmul_precision='highest')
    assert be.preflight() == []
    _backend('cuda.matmul').fp32_precision = 'tf32'  # something changed it after the backend set it
    try:
        assert any('IEEE' in p and 'cuda.matmul' in p for p in be.preflight())
    finally:
        _backend('cuda.matmul').fp32_precision = 'ieee'


class _Ctx:
    def log(self, message):
        pass

    def boundary(self, **kw):
        pass

    def expect(self, *runs):
        pass


def test_the_fp64_reference_and_each_arms_model_are_freed_before_the_next_measurement(built, tmp_path, env):
    """Review 06-10 P1: release() must not leave the caller's FP64 reference or a previous arm's model live."""
    import gc
    import weakref
    live = []

    def alive(kind=''):
        gc.collect()
        return [name for name, ref in live if ref() is not None and kind in name]

    def reset_peak():  # the peak-memory measurement must not include the FP64 reference
        assert not alive('FP64'), f'peak-memory measurement with {alive("FP64")}'

    be = backend(built, tmp_path / 'vol', reset_peak=reset_peak)
    real_load = be.load

    def load(arm, dtype='float32'):
        if dtype == 'float32':
            assert not alive(), f'loading {arm} with {alive()} still live'
        model = real_load(arm, dtype)
        live.append((f'{arm} {"FP64 reference" if dtype == "float64" else "model"}', weakref.ref(model)))
        return model

    be.load = load
    stages = cs.Stages(built['record'], tmp_path / 'local', be)
    stages.qualification(_Ctx())
    stages.rows(_Ctx())
    stages.attribution(_Ctx())
    assert not alive(), alive()


def test_the_pod_preflight_checks_the_base_model_against_the_codes_pins(built, tmp_path, env):
    """Without the test's stand-in pins the backend uses HF's identities for 989aa798; the tiny world fails them."""
    from pilot_eval import c_attribution_bundle as cb
    be = backend(built, tmp_path / 'vol')
    del be.probes['pinned_base_model']
    assert TransformersBackend(built['record'], tmp_path / 'vol', device='cpu', probes={},
                               repo_root=built['code']).pinned_base_model == cb.PINNED_BASE_MODEL
    be.pinned_base_model = cb.PINNED_BASE_MODEL
    assert 'differs from HF revision 989aa798' in _stopped_at_preflight(built, tmp_path, be=be)
