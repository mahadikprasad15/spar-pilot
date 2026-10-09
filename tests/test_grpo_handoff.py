"""Final Pilot 4 handoff through the public CPU workflow and CLI."""
import pytest
from pilot_eval.cli import main


def test_final_report_commands_are_cpu_safe_and_missing_evidence_cannot_publish(tmp_path, capsys, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'torch', None)
    with pytest.raises(SystemExit) as finished:
        main(['--help'])
    assert finished.value.code == 0
    help_text = capsys.readouterr().out
    assert 'grpo-final-report' in help_text
    assert 'grpo-final-verify' in help_text
    assert main(['grpo-final-report', '--selection', str(tmp_path/'missing.json'),
                 '--name', 'final', '--output-root', str(tmp_path)]) == 1
    assert not (tmp_path/'reports/final/complete.json').exists()


def test_existing_but_unsealed_sources_cannot_be_called_a_complete_report(tmp_path):
    import json
    from pilot_eval.grpo_handoff import report_handoff
    choices={key:key for key in ['grpo_evaluation','sft_writes','grpo_writes','control_writes','tokens_execution']}
    for value in choices.values(): (tmp_path/value).mkdir()
    selection=tmp_path/'selection.json';selection.write_text(json.dumps(choices))
    with pytest.raises((ValueError, OSError)):
        report_handoff(selection,tmp_path,'invalid-final')
    assert not (tmp_path/'reports/invalid-final/complete.json').exists()


def finish_controlled_handoff(root, training, sft_execution, grpo_execution):
    """Continue the real preparation/training/write workflow with controlled models."""
    import json
    from test_grpo_evaluation import Dependencies as EvalDependencies
    from pilot_eval.grpo_evaluation import evaluate_grpo
    from pilot_eval.grpo_prepare import load_grpo_prepared
    training_config=json.loads(training.read_text())
    frozen=json.loads((root/training_config['frozen_path']).read_text())
    plan,*_=load_grpo_prepared(root/frozen['source_path'],root)
    from pilot_eval.sft_evaluation import evaluate_sft
    from test_sft_training import TrainingBoundary
    for step in ['baseline','0','8','16','32','64']:
        evaluate_sft(root/plan['sources']['sft']['path'],root,step,dependencies=TrainingBoundary())
    eval_deps=EvalDependencies();eval_deps.fail_at=3
    with pytest.raises(RuntimeError,match='interrupted'):
        evaluate_grpo(training,root,'handoff-evaluation',dependencies=eval_deps)
    eval_deps.fail_at=None
    evaluate_grpo(training,root,'handoff-evaluation',dependencies=eval_deps)
    from pilot_eval.token_measurement_workflow import prepare_tokens,profile_tokens,freeze_tokens,measure_tokens
    from test_token_measurements import TokenDependencies
    token=prepare_tokens(sft_execution,grpo_execution,root,'handoff-tokens',context_chunk=4,workspace_bytes=1000000)
    profile_tokens(token,root,dependencies=TokenDependencies())
    execution=freeze_tokens(token,root,review_notes='Reviewed controlled supplemental evidence.')
    with pytest.raises(KeyboardInterrupt): measure_tokens(execution,root,dependencies=TokenDependencies(fail_at=8))
    measure_tokens(execution,root,dependencies=TokenDependencies())
    selection=root/'handoff-selection.json'
    choices=dict(grpo_evaluation='plans/handoff-evaluation/grpo.evaluation.json',sft_writes='reports/sft-report',
                 grpo_writes='reports/grpo-report',control_writes='reports/control-report',
                 tokens_execution=str(execution.relative_to(root)))
    selection.write_text(json.dumps(choices))
    from pilot_eval.grpo_handoff import report_handoff,verify_handoff
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    import sys
    with pytest.MonkeyPatch.context() as offline:
        offline.setitem(sys.modules,'torch',None)  # Final report/verify/export must not load Torch.
        result=report_handoff(selection,root,'final-handoff')
        assert result['report_complete'] and len(result['behaviour']['sft']['trajectory'])==5
        assert len(result['writes']['learned_direction_trajectory'])==840
        assert len(result['kl']['kl_trajectory'])==30
        assert result['random_control']['realizations']==1 and len(result['random_control']['modules'])==196
        assert result['matching']['objective_only_causal_comparison'] is False
        assert result['unavailable']['sft_gradient_consistency']['reason']=='not_saved'
        assert result['behaviour']['grpo']['drops']['greedy']['drop']==1.
        assert result['behaviour']['sft']['trajectory'][0]['flexible_accuracy']==1.
        assert all((root/r['path']).is_file() for r in result['download_inventory'])
        assert len(result['plot_files'])>=4
        inventory_paths={r['path'] for r in result['download_inventory']}
        assert {'plans/flight/grpo.preflight.json','plans/frozen/freeze-complete.json',
                'plans/sampling/baseline.config.json'} <= inventory_paths
        assert any('/sampling-baseline/' in path and path.endswith('responses.jsonl') for path in inventory_paths)
        assert any('/runs/diagnostics/' in '/'+path and path.endswith('complete.json') for path in inventory_paths)
        assert verify_handoff(selection,root,'final-handoff')==result
        assert report_handoff(selection,root,'final-handoff')==result
        assert all(p.read_bytes()==value for p,value in before.items())
        from pilot_eval.grpo_handoff import export_handoff
        bundle=export_handoff(selection,root,'final-handoff','full-download')
        assert (root/bundle['archive_path']).is_file()
        assert any(r['path']=='reports/final-handoff/complete.json' for r in bundle['files'])
        assert export_handoff(selection,root,'final-handoff','full-download')==bundle
        # A deleted raw shard invalidates even a previously complete final report.
        shard=next(r for r in result['download_inventory'] if '/shards/' in r['path'] and r['path'].endswith('arrays.npz'))
        saved=(root/shard['path']).read_bytes();(root/shard['path']).unlink()
        with pytest.raises((ValueError,OSError)):verify_handoff(selection,root,'final-handoff')
        (root/shard['path']).write_bytes(saved)



def test_signed_direction_figure_keeps_unresolved_gaps(tmp_path):
    # Public plotting surface: standalone figures from known report quantities.
    from pilot_eval.grpo_handoff_plots import plot_handoff
    row=dict(step=0,strict_accuracy=1.,flexible_accuracy=1.,mean_tokens=10.,
             flexible_change=dict(estimate=0.,interval_95=[0.,0.]),
             mean_token_change=dict(estimate=0.,interval_95=[0.,0.]))
    result=dict(behaviour=dict(sft=dict(trajectory=[row]),grpo=dict(
        summaries={k:dict(strict_accuracy=1.,flexible_accuracy=1.,mean_tokens=10.) for k in ['greedy-0','sampled-0','sampled-64']},
        changes=[dict(mode='greedy',**row)])),
        writes=dict(magnitudes={k:[] for k in ['sft','grpo','random']},learned_direction_trajectory=[]),
        kl=dict(kl_trajectory=[]))
    outputs=plot_handoff(result,tmp_path)
    assert len(outputs)==11
    assert all((tmp_path/p).read_text().startswith('<?xml') for p in outputs)
    assert 'Signed SFT' in (tmp_path/'direction.svg').read_text()


def test_export_requires_verified_complete_sources(tmp_path):
    from pilot_eval.grpo_handoff import export_handoff
    with pytest.raises(OSError):export_handoff(tmp_path/'absent.json',tmp_path,'report','archive')
    assert not (tmp_path/'exports/archive/complete.json').exists()


def test_archive_help_is_public_and_cpu_only(capsys):
    with pytest.raises(SystemExit):main(['grpo-final-export','--help'])
    output=capsys.readouterr().out
    assert '--archive-name' in output and '--selection' in output
