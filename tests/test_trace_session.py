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
