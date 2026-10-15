"""Public prereveal review workflow; CPU only and no learned readouts."""
import pytest


def test_exposure_history_is_durable_and_cannot_be_relabelled(tmp_path):
    from pilot_eval.trace_review import record_exposure, inspect_review
    event = dict(schema_version=1, id='first-look', reviewer='Prasad',
        experiment='pilot-4', output_kind='scalar-results',
        description='Saw behavioral accuracy plots; token-list history unknown.',
        observed_at='unknown', unknown_history='Earlier token-list exposure not established.')
    record_exposure(tmp_path, 'review-test', event)
    restored = inspect_review(tmp_path, 'review-test')
    assert restored['state'] == 'awaiting-review'
    assert restored['reveal_allowed'] is False
    assert restored['exposures'] == [event]
    assert restored['learned_lists_exposed'] is False
    record_exposure(tmp_path, 'review-test', event)
    assert len(inspect_review(tmp_path, 'review-test')['exposures']) == 1
    with pytest.raises(ValueError, match='different'):
        record_exposure(tmp_path, 'review-test', dict(event, output_kind='learned-lens-token-lists'))
    learned = dict(event, id='second-look', experiment='pilot-5',
        output_kind='learned-lens-token-lists', description='Opened adapted preview before freeze.')
    record_exposure(tmp_path, 'review-test', learned)
    assert inspect_review(tmp_path, 'review-test')['learned_lists_exposed'] is True


def test_public_review_commands_restore_without_model_imports_and_reject_tampering(tmp_path):
    import json
    import subprocess
    import sys
    event = dict(schema_version=1, id='reviewer-history', reviewer='Researcher',
        experiment='pilot-5', output_kind='other', description='No learned lists observed.',
        observed_at='unknown', unknown_history='Earlier history is unknown.')
    path = tmp_path / 'event.json'
    path.write_text(json.dumps(event))
    command = [sys.executable, '-m', 'pilot_eval', 'traces-review-exposure',
               '--name', 'cli-review', '--event', str(path), '--output-root', str(tmp_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    script = '''
import sys
from pilot_eval.cli import main
main(['traces-review-inspect', '--name', 'cli-review', '--output-root', sys.argv[1]])
assert not any(name in sys.modules for name in ['torch', 'transformers', 'peft'])
'''
    restored = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
                              capture_output=True, text=True, check=True)
    assert json.loads(restored.stdout)['reveal_allowed'] is False
    payload = tmp_path / 'runs/pilot-5/cli-review/review/exposures/reviewer-history.json'
    value = json.loads(payload.read_text())
    value['event']['description'] = 'Edited after recording.'
    payload.write_text(json.dumps(value))
    result = subprocess.run([sys.executable, '-m', 'pilot_eval', 'traces-review-inspect',
        '--name', 'cli-review', '--output-root', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 1
    assert 'Exposure record hash mismatch' in result.stdout + result.stderr
