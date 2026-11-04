"""Read-only source chain inventory; CPU only."""
import importlib.util
import json
from pathlib import Path


def module():
    spec = importlib.util.spec_from_file_location('source_check', Path(__file__).parent.parent/'scripts/offline_grpo_source_check.py')
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def test_scan_follows_json_references_and_lists_missing_paths(tmp_path):
    check = module()
    (tmp_path/'plans/a').mkdir(parents=True)
    (tmp_path/'plans/a/one.json').write_text(json.dumps(dict(next='plans/a/two.json', note='not a path', gone='runs/x/missing.json')))
    (tmp_path/'plans/a/two.json').write_text(json.dumps(dict(items=['inputs/models/w/model.bin'], back='plans/a/one.json', parents={'runs/y/file.bin': 'hash'})))
    (tmp_path/'runs/y').mkdir(parents=True); (tmp_path/'runs/y/file.bin').write_text('x')
    present, missing = check.scan(tmp_path, ['plans/a/one.json'])
    assert missing == ['inputs/models/w/model.bin', 'runs/x/missing.json']
    assert {'plans/a/one.json', 'plans/a/two.json', 'runs/y/file.bin'} <= present


def test_scan_reports_a_missing_starting_plan(tmp_path):
    present, missing = module().scan(tmp_path, ['plans/none.json'])
    assert missing == ['plans/none.json'] and not present


def test_checks_report_failures_without_stopping(tmp_path, capsys):
    check = module()
    outcomes = check.run_checks([('good', lambda: 'fine'), ('bad', lambda: (_ for _ in ()).throw(FileNotFoundError('absent')))])
    assert outcomes == dict(good=True, bad=False)
    text = capsys.readouterr().out
    assert 'OK    good: fine' in text and 'FAIL  bad: FileNotFoundError: absent' in text
