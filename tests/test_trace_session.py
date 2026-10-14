"""Pilot 5 public bootstrap workflow; no GPU or model downloads."""
import json
import subprocess
import sys


def test_fresh_restore_is_read_only_and_distinguishes_source_states(tmp_path):
    from pilot_eval.trace_session import TraceSession

    session = TraceSession(tmp_path, tmp_path, 'pilot5-test')
    restored = session.restore()
    assert restored['smoke']['state'] == 'missing'
    assert restored['grading'] == {'schema_version': 1, 'status': 'not-requested'}
    assert restored['sources'] == {}
    assert list(tmp_path.iterdir()) == []
    assert restored['sites'][0]['name'] == 'embedding-output'
    assert restored['sites'][-1]['name'] == 'block-27-output'
    assert restored['sites'][-1]['normalization'] == 'before-final-norm'

    source = tmp_path / 'prepared.json'
    source.write_text('{}')
    session.select_sources({'sft': str(source), 'grpo': str(tmp_path / 'absent.json')})
    fresh = TraceSession(tmp_path, tmp_path, 'pilot5-test').restore()
    assert fresh['sources']['sft']['state'] == 'pending-audit'
    assert fresh['sources']['grpo']['state'] == 'missing'
    script = 'from pilot_eval.trace_session import TraceSession; import sys; TraceSession(sys.argv[1], sys.argv[1], "pilot5-test").restore(); assert not any(x in sys.modules for x in ("torch", "transformers", "peft"))'
    subprocess.run([sys.executable, '-c', script, str(tmp_path)], check=True)


def test_reconnect_requires_new_session_and_detects_tampering(tmp_path):
    import pytest
    from pilot_eval.trace_session import TraceSession

    runtime = {'device': 'fixture GPU', 'versions': {'torch': 'fixture'}}
    session = TraceSession(tmp_path, tmp_path, 'smoke', environment_probe=lambda: runtime,
                           process_probe=lambda: [])
    session.write_smoke()
    assert session.restore()['smoke']['state'] == 'pending-reconnect'
    with pytest.raises(ValueError, match='fresh notebook kernel'):
        session.verify_reconnect()
    # The public operation runs in a genuinely different interpreter.
    script = 'from pilot_eval.trace_session import TraceSession; import sys; assert TraceSession(sys.argv[1], sys.argv[1], "smoke").verify_reconnect()["smoke"]["state"] == "completed"'
    subprocess.run([sys.executable, '-c', script, str(tmp_path)], check=True)
    smoke = tmp_path / 'runs/pilot-5/smoke/bootstrap/smoke.json'
    smoke.write_text('{}')
    with pytest.raises(ValueError, match='hash mismatch'):
        TraceSession(tmp_path, tmp_path, 'smoke').verify_reconnect()
    # Restoration must also verify the seal while reconnect is still pending.
    other = TraceSession(tmp_path, tmp_path, 'pending', environment_probe=lambda: runtime,
                         process_probe=lambda: [])
    other.write_smoke()
    (tmp_path / 'runs/pilot-5/pending/bootstrap/smoke.json').write_text('{}')
    with pytest.raises(ValueError, match='hash mismatch'):
        other.restore()


def test_readiness_and_audit_fail_before_inference(tmp_path):
    import pytest
    from pilot_eval.trace_session import TraceSession

    probes = []
    session = TraceSession(tmp_path, tmp_path, 'guards',
                           process_probe=lambda: [{'pid': 123}],
                           environment_probe=lambda: probes.append('import'))
    with pytest.raises(RuntimeError, match='Active experiment child'):
        session.write_smoke()
    assert probes == []
    with pytest.raises(ValueError, match='Ambiguous'):
        session.select_sources({'sft': ['one', 'two']})
    source = tmp_path / 'source.json'
    source.write_text('{}')
    session.select_sources({'sft': source})
    with pytest.raises(ValueError, match='Select all three'):
        session.audit_sources()
    session = TraceSession(tmp_path, tmp_path, 'runtime', process_probe=lambda: [],
                           environment_probe=lambda: {'device': '40GB'})
    with pytest.raises(ValueError, match='Runtime changed'):
        session.readiness(expected_runtime={'device': '80GB'})
