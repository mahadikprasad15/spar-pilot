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


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent.parent/'scripts'/f'{name}.py')
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def test_prepared_inputs_supply_every_template_requirement(tmp_path):
    from pilot_eval import offline_grpo_overnight as workflow
    script = _script('offline_grpo_prepare_inputs')
    (tmp_path/'plans/a100-stage0-v1').mkdir(parents=True)
    (tmp_path/'plans/run-qualification').mkdir(parents=True)
    (tmp_path/'plans/run-qualification/stage0.qualification.json').write_text(json.dumps(dict(model_identity=dict(model='m',dtype='float32'))))
    plan = dict(model_revision='r', source_evidence=dict(checkpoints={'0': dict(path='zero')}))
    inputs = script.build_inputs(tmp_path, name='run', qualification='run-qualification', tolerance=.001, repo='o/r', plan=plan, training=dict(eos_token_id=7),
                                 snapshot={'a': 'h'}, cache='cache', versions=dict(trl='x'), file_hash=lambda path: 'h')
    assert inputs['model_identity'] == dict(model='m', dtype='float32')
    assert inputs['fidelity_tolerance_nats'] == .001 and inputs['qualification'] == 'plans/run-qualification/stage0.qualification.json'
    import inspect, re
    source = inspect.getsource(workflow.template)
    needed = set(re.findall(r"need\('([a-z0-9_]+)'\)", source))
    assert needed and needed <= set(inputs)


def test_fidelity_summary_uses_absolute_differences():
    script = _script('offline_grpo_measure_fidelity')
    values = script.summary(dict(observations=[dict(differences=[-.5, .1], generated_token_ids=[1, 2], eos=True)]))
    assert values['max'] == .5 and values['tokens'] == 2 and values['eos'] == 1 and values['lengths'] == [2]
