"""Pod driver helpers that do not need a GPU: configs and the small rehearsal plan."""
import json
import subprocess
from pathlib import Path

import yaml

from pilot_eval import neuron_audit_plan as plan_mod
from pilot_eval import neuron_audit_pod as pod

INPUTS = dict(prep='/w/prep', model_dir='/w/hf/model', model_bytes=10, tokenizer_dir='/w/prep/tok',
              adapter_dirs={'gold-sft': '/w/prep/a'}, hf_home='/w/hf')


def test_rehearsal_and_production_configs(tmp_path):
    _, p_path, qualify, production = pod.write_configs(tmp_path, '/vol/audit', inputs=INPUTS, tag='t', rehearsal=True,
                                                       max_batches=2, interrupt_after=3)
    assert qualify['run_id'] == 'rehearsal-q-t' and production['run_id'] == 'rehearsal-p-t'
    assert production['max_batches_per_arm'] == 2 and production['rehearsal_interrupt_after_units'] == 3
    assert 'stop_command' not in production and production['volume_root'] == qualify['volume_dir'] == '/vol/audit'
    _, _, _, real = pod.write_configs(tmp_path, '/vol/audit', inputs=INPUTS, tag='v1', rehearsal=False,
                                      stop_command=pod.stop_command('abc'))
    assert real['run_id'] == 'p-v1' and 'runpodctl stop pod abc' in real['stop_command'][-1] and 'max_batches_per_arm' not in real
    assert json.loads(Path(p_path).read_text())['label'] == 'pod-t'


def test_small_rehearsal_plan_keeps_fixed_rules(tmp_path):
    out = plan_mod.make_rehearsal_plan(pod.PLAN, tmp_path / 'r.yaml', small=True)
    value = yaml.safe_load(Path(out).read_text())
    assert value['status'] == 'rehearsal' and value['operations']['qualification_max_gpu_hours'] == 1
    assert value['calibration_allocation']['fp64_reference'] == dict(math=2, general=2)
    assert value['benchmark']['token_budgets_padded'] == [4096, 8192]
    assert plan_mod.problems(value, '03b', rehearsal=True) == []
    real = yaml.safe_load(pod.PLAN.read_text())
    assert real['operations']['qualification_max_gpu_hours'] == 5
    assert real['operations']['stop_after_verified_partial_backup'] is True


def test_disk_problems_flags_a_default_50_gb_container_disk():
    assert pod.disk_problems(dict(total_bytes=50 * 10**9, free_bytes=43 * 10**9))
    assert pod.disk_problems(dict(total_bytes=120 * 10**9, free_bytes=113 * 10**9)) == []


def _chain_world(tmp_path, monkeypatch, *, status='qualified', flags=(), returncode=0):
    from test_neuron_audit_plan import approved
    from pilot_eval import neuron_audit_qualify as q
    work, vol = tmp_path / 'work', tmp_path / 'vol'
    work.mkdir()
    (work / 'inputs.json').write_text(json.dumps(INPUTS))
    plan = approved(tmp_path)
    calls, stops, signoffs = [], [], []

    def runner(args, check=True, log=None):
        calls.append(args[0])
        if args[0] == 'pilot_eval.neuron_audit_qualify':
            run_dir = work / 'out' / q.RUN_SUBDIR / 'q-t'
            run_dir.mkdir(parents=True)
            (run_dir / 'qualification.json').write_text(json.dumps(dict(status=status,
                                                                       review_flags_raised=list(flags))))
        return dict(command=args[:2], returncode=returncode if args[0].endswith('qualify') else 0, tail='')

    monkeypatch.setattr(q, 'signoff', lambda run_dir, **k: signoffs.append(k) or k)
    run = lambda **kw: pod.chain(work, vol, tag='t', plan_path=plan, runner=runner,  # noqa: E731
                                 stop_runner=lambda cmd, **k: stops.append(cmd), pod_id='pod123', **kw)
    return run, calls, stops, signoffs, vol


def test_chain_signs_off_by_rule_and_runs_04_when_03b_is_clean(tmp_path, monkeypatch):
    run, calls, stops, signoffs, _ = _chain_world(tmp_path, monkeypatch)
    report = run()
    assert calls == ['pilot_eval.neuron_audit_qualify', 'pilot_eval.neuron_audit_production']
    assert signoffs[0]['reviewer'].startswith('automatic-rule') and signoffs[0]['decision'] == 'accept'
    assert stops == []  # 04 stops the Pod itself, only after its verified backup
    assert report['decision']['continue_to_04'] is True


def test_chain_stops_before_04_on_flags_and_stops_pod_after_verified_volume_copy(tmp_path, monkeypatch):
    run, calls, stops, signoffs, vol = _chain_world(tmp_path, monkeypatch, flags=['early_layer'])
    report = run(uploader=lambda d: dict(revision='r'))
    assert calls == ['pilot_eval.neuron_audit_qualify'] and signoffs == []
    assert stops == [pod.stop_command('pod123')]
    assert (vol / 'runs/joint-neuron-adapter-audit/qualification-stopped/q-t/qualification.json').is_file()
    assert report['volume_copy_problems'] == [] and report['hf_package'] == dict(revision='r')


def test_chain_stops_on_failure_even_if_hf_upload_fails(tmp_path, monkeypatch):
    run, calls, stops, _, _ = _chain_world(tmp_path, monkeypatch, status='failed', returncode=1)

    def broken(_):
        raise subprocess.SubprocessError('network')
    report = run(uploader=broken)
    assert 'hf_error' in report and stops  # the verified volume copy is the backup
    assert any('exited 1' in r for r in report['decision']['reasons'])


def test_chain_refuses_partial_qualification_and_plan_without_rule(tmp_path, monkeypatch):
    run, calls, stops, _, _ = _chain_world(tmp_path, monkeypatch, status='partial')
    assert run(uploader=lambda d: {})['decision']['continue_to_04'] is False and calls == ['pilot_eval.neuron_audit_qualify']
    from test_neuron_audit_plan import approved
    import pytest
    bad = approved(tmp_path / 'x' if (tmp_path / 'x').mkdir() is None else tmp_path,
                   **{'review_flags.automatic_signoff': None})
    with pytest.raises(SystemExit, match='automatic_signoff'):
        pod.chain(tmp_path / 'work', tmp_path / 'vol', tag='t', plan_path=bad, pod_id='p')


def test_stop_command_tries_the_syntax_that_worked_on_the_pod_first():
    assert pod.stop_command('h97utghb7tl3gr') == ['sh', '-c', 'runpodctl stop pod h97utghb7tl3gr || runpodctl pod stop h97utghb7tl3gr']
    import pytest
    with pytest.raises(ValueError):
        pod.stop_command('x; rm -rf /')
