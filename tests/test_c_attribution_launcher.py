"""Ticket 09: the real launcher and watchdog at their boundary, deadline disabled (no time caps).

Writers are plain processes; ``runpodctl`` is a mock on PATH. On macOS, ``tests/shims`` stands in for the
Linux-only ``timeout``, ``flock`` and ``setsid`` (only when the real tool is missing)."""
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from pilot_eval import attribution_chain as pc
from pilot_eval import c_attribution_chain as cc

REPO = Path(__file__).resolve().parents[1]
WATCHDOG = REPO / 'scripts' / 'attribution_watchdog.sh'
LAUNCHER = REPO / 'scripts' / 'c_attribution_launch.sh'
DRAFT = REPO / '.scratch/c-attribution-pilot/launch-record-rehearsal.yaml'
SHIMS = REPO / 'tests' / 'shims'


def _path():
    extra = [str(SHIMS)] if any(shutil.which(t) is None for t in ('timeout', 'flock', 'setsid')) else []
    return ':'.join(extra + [str(Path(sys.executable).parent), os.environ['PATH']])


def _watchdog(tmp_path, pid, *, progress=None, status_every=1):
    log = tmp_path / 'actions.log'
    env = dict(os.environ, PATH=_path(), WATCHDOG_COPY=f'echo copy >> {log}', WATCHDOG_STOP_1=f'echo stop1 >> {log}; exit 1',
               WATCHDOG_STOP_2=f'echo stop2 >> {log}', WATCHDOG_SUMMARY=str(tmp_path / 'summary.json'),
               WATCHDOG_STATUS_EVERY=str(status_every))
    if progress:
        env['WATCHDOG_PROGRESS'] = str(progress)
    (tmp_path / 'owner').write_text('attempt-A\n')
    proc = subprocess.Popen(['bash', str(WATCHDOG), 'none', str(pid), str(tmp_path / 'done'), '2', '5', '5',
                             str(tmp_path / 'owner'), 'attempt-A'], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    return proc, log


def test_deadline_disabled_a_live_writer_is_never_stopped_and_progress_is_visible(tmp_path):
    writer = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    progress = tmp_path / 'progress.json'
    progress.write_text(json.dumps(dict(stage='attribution', units_done=3, last_update_unix=time.time() - 120)))
    try:
        proc, log = _watchdog(tmp_path, writer.pid, progress=progress)
        time.sleep(5)
        assert proc.poll() is None and writer.poll() is None and not log.exists()
        (tmp_path / 'owner').write_text('attempt-B\n')  # a newer launch: this watchdog stands down untouched
        out, _ = proc.communicate(timeout=20)
        assert 'standing down' in out and not log.exists() and writer.poll() is None
        assert 'last progress update 12' in out and 'stage attribution, units 3' in out  # ~120 s stale, visible
    finally:
        writer.kill()


def test_deadline_disabled_a_writer_crash_is_copied_and_stopped_at_once(tmp_path):
    import threading
    writer = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2); raise SystemExit(1)'])
    threading.Thread(target=writer.wait, daemon=True).start()  # reap it, as the launcher's shell would
    started = time.time()
    proc, log = _watchdog(tmp_path, writer.pid)
    proc.communicate(timeout=60)
    assert time.time() - started < 20
    assert log.read_text().split() == ['copy', 'stop1', 'stop2']   # copy completes before any stop request
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['reason'] == 'writer-exited-without-completing' and summary['stop'] == 'requested'
    assert summary['stop_returncodes'] == [1, 0] and summary['externally_verified'] is False


def test_a_bad_deadline_value_is_refused(tmp_path):
    env = dict(os.environ, PATH=_path(), WATCHDOG_SUMMARY=str(tmp_path / 's.json'))
    (tmp_path / 'owner').write_text('a\n')
    result = subprocess.run(['bash', str(WATCHDOG), '3h', str(os.getpid()), str(tmp_path / 'done'), '2', '5', '5',
                             str(tmp_path / 'owner'), 'a'], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 2 and "epoch or 'none'" in result.stdout


# --- launcher -------------------------------------------------------------------------------------

@pytest.fixture
def pod(tmp_path):
    body = yaml.safe_load(DRAFT.read_text())
    body['hourly_cost_usd'] = 1.0
    draft = tmp_path / 'draft.yaml'
    draft.write_text(yaml.safe_dump(body))
    record = cc.finalize_draft(draft, tmp_path / 'launch_record.json', code_commit=pc._commit(), pod_id='podtest1',
                               fill={'inputs.artifact_root': str(tmp_path), 'inputs.bundle': str(tmp_path / 'b.json'),
                                     'inputs.model_dir': str(tmp_path / 'm'), 'inputs.work_dir': str(tmp_path)})
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    stops = tmp_path / 'runpodctl.log'
    (bin_dir / 'runpodctl').write_text(f'#!/bin/sh\necho "$@" >> {stops}\n')
    (bin_dir / 'runpodctl').chmod(0o755)
    env = dict(os.environ, PATH=f'{bin_dir}:{_path()}', RUNPOD_POD_ID='podtest1', PYTHONPATH=str(REPO),
               PY=sys.executable)
    volume = tmp_path / 'volume'
    volume.mkdir()
    chain_dir = cc.run_root(volume, 'c-rehearsal-pod-1') / cc.CHAIN_DIR
    return dict(record=record, local=tmp_path / 'local', volume=volume, env=env, stops=stops, chain_dir=chain_dir)


def launch(pod, record=None):
    return subprocess.run(['bash', str(LAUNCHER), str(record or pod['record']), str(pod['local']), str(pod['volume'])],
                          env=pod['env'], capture_output=True, text=True, timeout=120)


def test_a_bad_record_starts_nothing_and_says_the_pod_is_still_running(pod):
    result = launch(pod, record=pod['record'].parent / 'missing.json')
    assert result.returncode != 0 and 'NO RUN STARTED' in result.stderr and 'runpodctl stop pod podtest1' in result.stderr
    assert not (pod['chain_dir'] / 'watchdog_owner').exists() and not pod['stops'].exists()


def test_a_second_launch_never_disarms_or_stops_a_live_writer(pod):
    live = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
    try:
        pod['chain_dir'].mkdir(parents=True)
        (pod['chain_dir'] / 'watchdog_owner').write_text('first-launch\n')
        (pod['chain_dir'] / 'writer_marker.json').write_text(json.dumps(dict(
            run_id='c-rehearsal-pod-1', host=socket.gethostname(), pid=live.pid, state='running',
            heartbeat_unix=time.time())))
        result = launch(pod)
        assert result.returncode != 0 and 'NO RUN STARTED' in result.stderr
        assert 'live writer' in result.stderr + result.stdout
        assert (pod['chain_dir'] / 'watchdog_owner').read_text() == 'first-launch\n'
        time.sleep(3)
        assert not pod['stops'].exists() and live.poll() is None
    finally:
        live.kill()


def test_a_second_launch_on_the_same_pod_is_refused_by_the_local_lock(pod):
    pod['local'].mkdir(parents=True)
    holder = subprocess.Popen(['flock', str(pod['local'] / 'launch.lock'), 'sleep', '60'], env=pod['env'])
    try:
        time.sleep(1)
        result = launch(pod)
        assert result.returncode != 0 and 'another launch' in result.stderr
        assert not (pod['chain_dir'] / 'watchdog_owner').exists() and not pod['stops'].exists()
    finally:
        holder.kill()


def test_a_finished_run_id_is_refused_before_ownership(pod):
    pod['chain_dir'].mkdir(parents=True)
    (pod['chain_dir'] / 'done').write_text('complete\n')
    (pod['chain_dir'] / 'watchdog_owner').write_text('old\n')
    result = launch(pod)
    assert result.returncode != 0 and 'new run ID' in result.stderr
    assert (pod['chain_dir'] / 'watchdog_owner').read_text() == 'old\n' and not pod['stops'].exists()


def test_an_owned_startup_crash_still_stops_the_pod_promptly(pod):
    """Ownership taken, then the real chain CLI refuses (no volume identity file): the deadline-disabled watchdog
    sees the writer exit without completing and requests the stop within seconds."""
    started = time.time()
    result = launch(pod)
    assert 'NO RUN STARTED' not in result.stderr and 'deadline disabled' in result.stdout
    summary_path = pod['chain_dir'] / 'watchdog_summary.json'
    for _ in range(80):
        if summary_path.exists() and json.loads(summary_path.read_text())['stop'] != 'pending':
            break
        time.sleep(0.5)
    summary = json.loads(summary_path.read_text())
    assert summary['reason'] == 'writer-exited-without-completing' and summary['stop'] == 'requested'
    assert pod['stops'].read_text().split() == ['stop', 'pod', 'podtest1'] and time.time() - started < 60
