"""Whole chain through the real backend on CPU (review round 6, "next step"; review round 7, recommendation).

``Chain`` -> ``Stages`` -> the real ``TransformersBackend`` (real loader, PEFT arms, ``ChatTokenizer``,
``bind_sources``, sealed sequences and inventory, ``HFGenerator``) on the fixture in
``attribution_realpath_fixture``, with the launch record sealed from a fresh copy of the rehearsal YAML.

Stubbed, and only these: CUDA (device ``cpu``; memory counters simulated from the tokens a forward sees, and a
simulated CUDA out-of-memory above a token count), the volume's mount/identity check, the Hub (private check and
export) and the Pod stop command. Sizes are reduced for CPU (token budgets, ``max_new_tokens``).
"""
import json
import subprocess
from pathlib import Path
from unittest import mock

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('peft')

import attribution_realpath_fixture as fx  # noqa: E402
from pilot_eval import attribution_chain as ch  # noqa: E402
from pilot_eval import attribution_stages as st  # noqa: E402
from pilot_eval import attribution_units as au  # noqa: E402
from pilot_eval.attribution_backend import TransformersBackend  # noqa: E402

BYTES_PER_TOKEN = 10 ** 6  # simulated activation memory
DEVICE_BYTES = 10 ** 9     # 0.8 x this fits 800 tokens
OOM_TOKENS = 2048          # a forward over more tokens raises a simulated CUDA OOM
SIZES = dict(token_budget_sweep=[16, 64, 256, 1024, 4096])


class CPUBackend(TransformersBackend):
    """The real backend on CPU; only CUDA's memory accounting and the Pod's external checks are replaced."""

    def __init__(self, record, volume_root):
        super().__init__(record, volume_root, device='cpu')
        self.max_tokens = 0

    def _watch(self, model):
        """Count the tokens of every forward at the embedding layer (attribution passes embeddings, not IDs,
        to the decoder, so a hook on the top-level forward would miss them)."""
        def pre(module, args):
            tokens = int(args[0].numel())
            if tokens > OOM_TOKENS:
                raise RuntimeError('CUDA out of memory (simulated)')
            self.max_tokens = max(self.max_tokens, tokens)
        embed = model.get_input_embeddings()
        if not getattr(embed, '_cpu_watch', False):
            embed.register_forward_pre_hook(pre)
            embed._cpu_watch = True
        return model

    def load(self, arm):
        return self._watch(super().load(arm))

    def untuned(self):
        return self._watch(super().untuned())

    def device_memory(self):
        return DEVICE_BYTES

    def reset_peak(self):
        self.max_tokens = 0

    def peak_memory(self):
        return self.max_tokens * BYTES_PER_TOKEN

    def synchronize(self):
        pass

    def preflight(self):
        from types import SimpleNamespace
        from pilot_eval import neuron_audit_export, neuron_audit_preflight
        hub = SimpleNamespace(info=lambda repo: SimpleNamespace(private=True))
        with mock.patch.object(torch.cuda, 'is_available', lambda: True), \
                mock.patch.object(neuron_audit_preflight, 'volume_identity', lambda *a, **k: ({}, [])), \
                mock.patch.object(neuron_audit_export, 'OnlineHubProcess', lambda **k: hub), \
                mock.patch.object(neuron_audit_preflight, 'local_disk',
                                  lambda d: dict(local_dir=str(d), total_bytes=128 * 10 ** 9, free_bytes=120 * 10 ** 9)):
            return super().preflight()


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('realpath')
    return dict(fx.build(tmp, sizes=SIZES), tmp=tmp)


def run_chain(world, name):
    tmp = world['tmp'] / name
    stops = []

    def exporter(local, **kw):
        return dict(revision='a' * 40, completeness='complete', readback_verified=True)

    def runner(cmd, **kw):
        stops.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, 'pod stopped', '')

    record, local, volume = world['record'], tmp / 'local', tmp / 'volume'
    volume.mkdir(parents=True)
    backend = CPUBackend(record, volume)
    stages = st.build(record, local, volume, backend=backend)
    chain = ch.Chain(record, local_root=local, volume_root=volume, stages=stages, exporter=exporter, runner=runner,
                     volume_identity=record['volume_identity'])
    with mock.patch.dict('os.environ', {'HF_HUB_OFFLINE': '1'}):
        summary = chain.run()
    return summary, local, volume, stops


def test_the_whole_rehearsal_chain_runs_through_the_real_backend(world):
    summary, local, volume, stops = run_chain(world, 'rehearsal')
    assert summary['state'] == 'complete', json.dumps(summary, default=str, indent=1)[:6000]
    assert all(s.get('status') == 'complete' for s in summary['stages'].values()), summary['stages']
    assert summary['signoff']['by'] == ch.AUTO_REVIEWER
    qual = json.loads((local / st.QUAL_DIR / 'qualification.json').read_text())
    details = json.loads((local / st.QUAL_DIR / 'details.json').read_text())
    outcomes = {c['budget']: c['outcome'] for c in details['sweep']['candidates']}
    assert outcomes[16] == 'longest-example-exceeds-budget' and outcomes[4096] == 'oom'  # r6 (2) path exercised
    assert qual['budget'] == 256 and all(qual['checks'].values()), qual['checks']
    for directory in [st.QUAL_DIR, 'attribution/traces/calibration/rows', st.REPORT_DIR,
                      'attribution/stage_d/untuned/reference/config']:
        au.verify_run(local / directory)
    assert summary['backups']['volume_copy']['verified'] is True
    assert stops and stops[0][:3] == ['runpodctl', 'stop', 'pod']


PRODUCTION = fx.REPO / '.scratch/adapter-neuron-attribution/launch-record-production.yaml'


@pytest.fixture(scope='module')
def production_world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('realpath-production')
    arms = ('gold-sft', 'online-grpo', 'offline-grpo', 'sampling-sft-k0', 'sampling-sft-k2', 'sampling-sft-k5')
    sizes = dict(SIZES, minimums=dict(discovery=5, heldout=5))
    return dict(fx.build(tmp, sizes=sizes, draft_path=PRODUCTION, arms=arms,
                         math=dict(discovery=8, heldout=8, calibration=14)), tmp=tmp)


def test_the_whole_production_chain_runs_through_the_real_backend(production_world):
    """The production record as approved (6 arms, blind held-out rows, the 150-item ADR 0001 cohort, pilot
    comparison deferred); only sizes reduced for CPU."""
    summary, local, volume, stops = run_chain(production_world, 'production')
    assert summary['state'] == 'complete', json.dumps(summary, default=str, indent=1)[:6000]
    assert all(s.get('status') == 'complete' for s in summary['stages'].values()), summary['stages']
    report = json.loads((local / st.REPORT_DIR / 'report.json').read_text())
    assert report['rehearsal'] is False and 'deferred' in report['stage_d']['pilot_comparison']['reason']
    for directory in ['attribution/traces/discovery/rows', 'attribution/traces/heldout/rows',
                      'attribution/stage_d/gold-sft/full/b00000']:
        au.verify_run(local / directory)
