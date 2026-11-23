"""Review round 7: the real launcher (``scripts/attribution_overnight.sh``) and watchdog at their boundary.

Two worker processes, a mock ``runpodctl`` on PATH; no GPU, no Hub, no real Pod. The chain itself is not run
to completion: the owned-failure case uses a volume without its identity file, which the real CLI refuses."""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from pilot_eval import attribution_chain as ch

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / 'scripts' / 'attribution_overnight.sh'
DRAFT = REPO / '.scratch/adapter-neuron-attribution/launch-record-pod-rehearsal.yaml'


@pytest.fixture
def pod(tmp_path):
    """A sealed rehearsal record (from the real draft) for this checkout, a mock runpodctl, local and volume."""
    seq = tmp_path / 'prep' / yaml.safe_load(DRAFT.read_text())['inputs']['sequences_dir']
    seq.mkdir(parents=True)
    (seq / 'manifest.json').write_text(json.dumps(dict(sources=dict(gsm8k=dict(test_parquet='t.parquet')))))
    inputs = tmp_path / 'inputs.json'
    inputs.write_text(json.dumps(dict(prep=str(tmp_path / 'prep'), model_dir='/m', tokenizer_dir='/t',
                                      adapter_dirs={'gold-sft': '/a', 'online-grpo': '/b'})))
    draft = tmp_path / 'draft.yaml'
    draft.write_text(yaml.safe_dump(dict(yaml.safe_load(DRAFT.read_text()), hourly_cost_usd=1.0)))
    record = ch.finalize_draft(draft, tmp_path / 'launch_record.json', code_commit=ch._commit(),
                               inputs_json=inputs, pod_id='podtest1', work_dir=str(tmp_path))
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    stops = tmp_path / 'runpodctl.log'
    (bin_dir / 'runpodctl').write_text(f'#!/bin/sh\necho "$@" >> {stops}\n')
    (bin_dir / 'runpodctl').chmod(0o755)
    env = dict(os.environ, PATH=f'{bin_dir}:{Path(sys.executable).parent}:{os.environ["PATH"]}',
               RUNPOD_POD_ID='podtest1', PYTHONPATH=str(REPO))
    volume = tmp_path / 'volume'
    volume.mkdir()
    return dict(record=record, local=tmp_path / 'local', volume=volume, env=env, stops=stops,
                chain_dir=volume / 'attribution/chain')


def launch(pod, record=None):
    return subprocess.run(['bash', str(LAUNCHER), str(record or pod['record']), str(pod['local']), str(pod['volume'])],
                          env=pod['env'], capture_output=True, text=True, timeout=120)


def wait_for(path, seconds=40):
    end = time.time() + seconds
    while time.time() < end:
        if path.exists() and path.read_text().strip():
            return json.loads(path.read_text())
        time.sleep(0.5)
    raise AssertionError(f'{path} never appeared')


def test_r7_2_a_bad_record_starts_nothing_and_says_the_pod_is_still_running(pod):
    result = launch(pod, record=pod['record'].parent / 'missing.json')
    assert result.returncode != 0
    assert 'NO RUN STARTED' in result.stderr and 'runpodctl stop pod podtest1' in result.stderr
    assert not (pod['chain_dir'] / 'watchdog_owner').exists()  # ownership never taken
    assert not (pod['chain_dir'] / 'watchdog.log').exists() and not pod['stops'].exists()


def test_r7_1_a_second_launch_never_disarms_or_stops_a_live_writer(pod):
    live = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
    try:
        pod['chain_dir'].mkdir(parents=True)
        (pod['chain_dir'] / 'watchdog_owner').write_text('first-launch\n')
        rec = json.loads(pod['record'].read_text())
        (pod['volume'] / ch.MARKER_FILE).write_text(json.dumps(dict(
            run_id=rec['run_id'], host=socket.gethostname(), pid=live.pid, state='running',
            heartbeat_unix=time.time())))
        result = launch(pod)
        assert result.returncode != 0 and 'NO RUN STARTED' in result.stderr
        assert 'live writer' in result.stderr + result.stdout
        assert (pod['chain_dir'] / 'watchdog_owner').read_text() == 'first-launch\n'
        time.sleep(4)  # long enough for a wrongly started watchdog to have acted
        assert not pod['stops'].exists() and not (pod['chain_dir'] / 'watchdog_summary.json').exists()
        assert live.poll() is None
    finally:
        live.kill()


def test_r7_1_a_second_launch_on_the_same_pod_is_refused_by_the_local_lock(pod):
    pod['local'].mkdir(parents=True)
    holder = subprocess.Popen(['flock', str(pod['local'] / 'launch.lock'), 'sleep', '60'])
    try:
        time.sleep(0.5)
        result = launch(pod)
        assert result.returncode != 0 and 'another launch' in result.stderr
        assert not (pod['chain_dir'] / 'watchdog_owner').exists() and not pod['stops'].exists()
    finally:
        holder.kill()


def test_r7_1_an_owned_startup_failure_still_stops_the_pod_promptly(pod):
    """The launch takes ownership, then the real chain CLI refuses (this volume has no identity file): the
    watchdog sees its writer exit without completing and requests the stop within seconds."""
    started = time.time()
    result = launch(pod)
    assert 'NO RUN STARTED' not in result.stderr and 'chain pid' in result.stdout
    summary = wait_for(pod['chain_dir'] / 'watchdog_summary.json')
    deadline_summary = None
    for _ in range(40):  # the summary is rewritten with the stop outcome
        deadline_summary = json.loads((pod['chain_dir'] / 'watchdog_summary.json').read_text())
        if deadline_summary['stop'] != 'pending':
            break
        time.sleep(0.5)
    assert summary['reason'] == 'writer-exited-without-completing'
    assert deadline_summary['stop'] == 'requested' and time.time() - started < 60
    assert pod['stops'].read_text().split() == ['stop', 'pod', 'podtest1']


AFTER = REPO / 'scripts' / 'attribution_after_rehearsal.sh'


def after_rehearsal(tmp_path, state):
    vol, bin_dir = tmp_path / 'vol', tmp_path / 'bin'
    (vol / 'attribution/chain').mkdir(parents=True)
    (vol / 'attribution/chain/done').write_text(state + '\n')
    bin_dir.mkdir()
    held = bin_dir / 'runpodctl.held'  # held aside during the rehearsal
    held.write_text(f'#!/bin/sh\necho "$@" >> {tmp_path / "stops.log"}\n')
    held.chmod(0o755)
    env = dict(os.environ, HF_TOKEN='x', RUNPOD_POD_ID='podtest1', VOL=str(vol), WORK=str(tmp_path),
               REPO=str(tmp_path / 'not-a-repo'), RUNPODCTL=str(bin_dir / 'runpodctl'), POLL='1')
    result = subprocess.run(['bash', str(AFTER)], env=env, capture_output=True, text=True, timeout=60)
    return result, vol, tmp_path / 'stops.log', bin_dir


def test_after_a_failed_rehearsal_production_never_starts_and_the_pod_stops(tmp_path):
    result, vol, stops, bin_dir = after_rehearsal(tmp_path, 'failed')
    assert result.returncode != 0 and 'production not started' in result.stdout
    assert stops.read_text().split() == ['stop', 'pod', 'podtest1']
    assert (vol / 'attribution').exists() and (bin_dir / 'runpodctl').exists()  # nothing moved; stop restored


def test_after_a_complete_rehearsal_a_failed_handover_still_stops_the_pod(tmp_path):
    result, vol, stops, bin_dir = after_rehearsal(tmp_path, 'complete')
    assert 'git pull failed' in result.stdout and stops.read_text().split() == ['stop', 'pod', 'podtest1']
    assert (vol / 'attribution-rehearsal').exists() and not (vol / 'attribution').exists()
