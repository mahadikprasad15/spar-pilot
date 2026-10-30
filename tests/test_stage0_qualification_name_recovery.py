"""Bounded metadata recovery must neither rehash large payloads nor waive gates."""
import json
from pathlib import Path
import runpy

import pytest


def test_recover_unstarted_plan_preserves_science_and_sources(tmp_path, monkeypatch):
    from test_stage0_qualification import qualification
    from pilot_eval import stage0_qualification as q
    from pilot_eval.training import file_hash
    from pilot_eval.workflow import _hash, _save_frozen
    from pilot_eval.stage0_workflow import _state
    plan, _ = qualification(tmp_path)
    config = json.loads(plan.read_text())
    # Reproduce the persisted bug, which overwrote a safe requested name only
    # AFTER safe_name had run. Passing the unsafe name directly is not equivalent.
    wrong = tmp_path / 'plans/tokenizer_config.json/stage0.qualification.json'
    malformed = {**config, 'name': 'tokenizer_config.json',
                 'run_path': 'runs/stage0/tokenizer_config.json/qualification'}
    _save_frozen(wrong, malformed)
    old_dir = tmp_path / malformed['run_path']
    _save_frozen(old_dir / 'meta/manifest.json', {'config_sha256': _hash(malformed)})
    _state(old_dir, 'prepared', 0, malformed['total_units'])
    source_bytes = wrong.read_bytes()
    old_dir = tmp_path / json.loads(source_bytes)['run_path']
    manifest_bytes = (old_dir / 'meta/manifest.json').read_bytes()
    recover = runpy.run_path(str(Path(__file__).parents[1] /
                                'scripts/recover_stage0_qualification_name.py'))['recover']
    monkeypatch.setattr(q, 'load_edits', lambda *args: pytest.fail('No expensive source verification in metadata recovery.'))
    monkeypatch.setattr(q, 'verify_edits', lambda *args: pytest.fail('No expensive catalogue verification in metadata recovery.'))
    target = recover(tmp_path, 'proper-name')
    corrected = json.loads(target.read_text())
    original = json.loads(source_bytes)
    assert corrected == {**original, 'name': 'proper-name',
                          'run_path': 'runs/stage0/proper-name/qualification'}
    directory = tmp_path / corrected['run_path']
    assert json.loads((directory / 'meta/manifest.json').read_text())['config_sha256'] == _hash(corrected)
    receipt = json.loads((directory / 'meta/name-recovery.json').read_text())
    assert receipt['source_sha256'] == file_hash(wrong)
    assert receipt['measurement_verified'] is False
    assert wrong.read_bytes() == source_bytes
    assert (old_dir / 'meta/manifest.json').read_bytes() == manifest_bytes
    with pytest.raises(ValueError, match='Destination'):
        recover(tmp_path, 'proper-name')
    (old_dir / 'meta/runtime.json').write_text('{}')
    with pytest.raises(ValueError, match='Execution'):
        recover(tmp_path, 'another-name')
    assert not (tmp_path / 'plans/another-name').exists()
