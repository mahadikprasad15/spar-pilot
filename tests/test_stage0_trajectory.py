"""Trajectory source audit, same-product pairing, CPU oracles and resume."""
import json
from pathlib import Path

import numpy as np
import pytest


def test_document_bootstrap_preserves_token_and_example_weighting():
    from pilot_eval.stage0_trajectory_math import interval
    draws=np.array([[0,0],[1,1],[0,1]])
    result=interval([1.,9.],[1,3],draws)
    assert result['equal_example_mean']==2. and result['token_mean']==2.5
    assert result['bootstrap_unit']=='whole document' and result['interval_95']==[1.05,2.95]
    assert result['token_interval_95']==[1.075,2.975]


def test_size_ratio_and_slopes_use_rss_size_without_zero_epsilon():
    from pilot_eval.stage0_trajectory_math import log_slopes
    points=[dict(step=0,size=0.,kl=0.),dict(step=8,size=1.,kl=.2),dict(step=16,size=2.,kl=.8),dict(step=64,size=4.,kl=3.2)]
    result=log_slopes(points)
    assert result['fitted_slope']==pytest.approx(2.) and all(r['slope']==pytest.approx(2.) for r in result['adjacent_slopes'])
    assert result['excluded_nonpositive_steps']==[0] and not result['direction_sensitivity_claim']


def test_runtime_groups_require_session_identity_and_do_not_infer_compatibility():
    from pilot_eval.stage0_trajectory_math import comparison_group
    common=dict(runtime=dict(device='A100',session_id='one'),inputs_sha256='inputs',model_identity={'base_sha256':'base'},scheduling={'batch_size':1})
    assert comparison_group(common,'product-a')==comparison_group(common,'product-b')
    changed={**common,'runtime':dict(device='A100',session_id='two')}
    assert comparison_group(common,'product-a')!=comparison_group(changed,'product-b')
    unknown={**common,'runtime':dict(device='A100')}
    assert comparison_group(unknown,'product-a')!=comparison_group(unknown,'product-b')


def test_lengths_are_paired_medians_and_keep_cap_status():
    from pilot_eval.stage0_trajectory_math import lengths
    # Median of paired changes (1) differs from difference of medians (9).
    before=[dict(id=str(i),tokens=t,capped=False,prompt_sha256=str(i)) for i,t in enumerate([1,10,100])]
    after=[dict(id=str(i),tokens=t,capped=i==2,prompt_sha256=str(i)) for i,t in enumerate([2,100,101])]
    result=lengths(before,after,np.array([[0,1,2],[0,0,0],[2,2,2]]))
    assert result['median_paired_change']==1 and result['shorter_share']==0
    assert result['reference_cap_rate']==0 and result['edited_cap_rate']==1/3
    with pytest.raises(ValueError):lengths(before,after[:-1],np.array([[0,1,2]]))


def test_complementary_kl_and_writes_join_only_in_verified_same_session():
    from pilot_eval.stage0_trajectory_sources import merge_products
    row=dict(alias='sft-64',id='q',corpus='fineweb',view='user',kl_values=[.1],writes=None,product='kl',kl_positions=[0],write_positions=None)
    other=dict(row,kl_values=None,writes=[dict(module='m',count=1,ratio_sum=.2)],product='writes',kl_positions=None,write_positions=[0])
    result=merge_products([dict(identity={'comparison_group':'same'},documents=[row]),dict(identity={'comparison_group':'same'},documents=[other])])
    assert len(result)==1 and result[0]['kl_values']==[.1] and result[0]['writes'][0]['ratio_sum']==.2
    result=merge_products([dict(identity={'comparison_group':'one'},documents=[row]),dict(identity={'comparison_group':'two'},documents=[other])])
    assert len(result)==2 and not any(r['kl_values'] is not None and r['writes'] is not None for r in result)
    with pytest.raises(ValueError):merge_products([dict(identity={'comparison_group':'same'},documents=[row,dict(row,kl_values=[.2])])])


def command(root,op,*args,dependencies=None):
    from pilot_eval.cli import main
    return main(['stage0-trajectory-'+op,'--output-root',str(root),*map(str,args)],dependencies=dependencies)


@pytest.fixture(scope='module')
def trajectory_sources(tmp_path_factory):
    from test_stage0_spectral import fixture,spectral
    from pilot_eval.stage0_edits import prepare as prepare_edits,run as run_edits
    from pilot_eval.stage0_qualification import prepare,run,approve
    from test_stage0_qualification import TinyDiagnostic
    from test_stage0_workflow import write_json
    from pilot_eval.stage0_workflow import record_budget
    from pilot_eval.stage0_budget import settle
    from pilot_eval.training import file_hash
    root=tmp_path_factory.mktemp('trajectory-sources');sp,deps=fixture(root,isotropic=True,steps=(0,8,16,32,64))
    assert spectral(root,'run',sp,dependencies=deps)==0
    edits=prepare_edits(sp,root,'edits');run_edits(edits,root)
    cohort=root/'cohort.json';write_json(cohort,dict(items=[dict(id=f'gsm8k:test:{i}',prompt=f'question {i}',gold='#### 2') for i in range(150)],prompt_template=dict(decoding=dict(eos_token_id=99,pad_token_id=98))))
    qp=prepare(edits,cohort,root,'qualify')
    usage=root/'usage.json';write_json(usage,dict(event_id='before',session_id='session-a',total_units=1.,observed_at='2026-10-28T08:00:00+00:00',note='synthetic cumulative'));record_budget(root/'plans/audit/stage0.plan.json',root,usage)
    review=root/'diagnostic.json';write_json(review,dict(reviewer='test',notes='fixture',session_id='session-a',max_units_per_hour=1.,max_seconds=600.,source_reviewed=True))
    result=run(qp,root,review,dependencies=TinyDiagnostic())
    write_json(usage,dict(event_id='after',session_id='session-a',total_units=1.02,observed_at='2026-10-28T08:10:00+00:00',note='synthetic cumulative'));record_budget(root/'plans/audit/stage0.plan.json',root,usage);settle(root,result['reservation_id'])
    write_json(review,dict(reviewer='test',notes='fixture gate',diagnostic_sha256=file_hash(root/'runs/stage0/qualify/qualification/complete.json'),absolute_gate=result['proposed_absolute_gate']));approve(qp,root,review)
    rows=[]
    for corpus in ['gsm8k','fineweb']:
        for i in range(150):
            masks=dict(question=[False,True,False],solution=[False,False,True],user=[False]*3) if corpus=='gsm8k' else dict(question=[False]*3,solution=[False]*3,user=[True]*3)
            rows.append(dict(id=f'{corpus}:{i}',corpus=corpus,input_ids=[1,2,3],attention_mask=[1]*3,masks=masks))
    from pilot_eval.workflow import _hash
    write_json(root/'fixed/items.json',rows);write_json(root/'fixed/activation.prepared.json',dict(items_path='fixed/items.json',items_sha256=_hash(rows),model='Qwen/Qwen2.5-1.5B-Instruct',model_revision='pinned',tokenizer_revision='pinned',dtype='float32'))
    return root


@pytest.fixture
def ready(tmp_path,trajectory_sources):
    import shutil
    from test_stage0_workflow import write_json
    shutil.copytree(trajectory_sources,tmp_path,dirs_exist_ok=True)
    selection=tmp_path/'trajectory-selection.json';write_json(selection,dict(products=[],random='missing-random.json'))
    assert command(tmp_path,'prepare','--spectral',tmp_path/'plans/basis/stage0.spectral.json','--inputs',tmp_path/'fixed/activation.prepared.json','--selection',selection,'--name','trajectory')==0
    return tmp_path


def test_public_cpu_geometry_resume_missing_sources_and_report(ready):
    root=ready;plan=root/'plans/trajectory/stage0.trajectory.json';directory=root/'runs/stage0/trajectory/trajectory'
    config=json.loads(plan.read_text());assert len([s for s in config['states'] if s['state']=='verified-adapter-payload'])==10
    endpoint=next(s for s in config['states'] if s['alias']=='sft-64')
    assert endpoint['sum_factor_norm_products']/endpoint['total_frobenius_size']==pytest.approx(np.sqrt(196))
    assert command(root,'run','--config',plan,'--max-new-units',1)==0
    first=next((directory/'units').glob('*/complete.json'));before=first.read_bytes()
    assert command(root,'run','--config',plan)==0 and first.read_bytes()==before
    assert command(root,'report','--config',plan,'--name','cpu-only')==0
    assert command(root,'verify','--config',plan,'--name','cpu-only')==0
    result=json.loads((root/'reports/cpu-only/results/results.json').read_text())
    assert result['observational_only'] and not result['drift_complete'] and result['missing_states']['random-64']=='missing'
    assert len(result['geometry'])==10*2*3
    assert command(root,'status','--config',plan)==0
    next((directory/'units').glob('*/result.json')).write_text('[]')
    assert command(root,'verify','--config',plan,'--name','cpu-only')==1


def test_public_missing_collection_profile_budget_resume_and_report(ready):
    from test_stage0_drift import TinyDrift
    from test_stage0_workflow import write_json
    from pilot_eval.stage0_workflow import record_budget
    from pilot_eval.stage0_budget import settle
    from pilot_eval.training import file_hash
    root=ready;plan=root/'plans/trajectory/stage0.trajectory.json'
    assert command(root,'run','--config',plan)==0
    engine=TinyDrift()
    assert command(root,'runtime','--config',plan,dependencies=engine)==0
    assert command(root,'collection-prepare','--config',plan,'--qualification',root/'plans/qualify/stage0.qualification.json',
        '--name','collection','--states','sft-0','sft-64','grpo-64',dependencies=engine)==0
    cp=root/'plans/collection/stage0.trajectory.collection.json';directory=root/'runs/stage0/collection/trajectory-collection'
    config=json.loads(cp.read_text());assert len(config['profile_keys'])==6
    predictions=root/'predictions.json';write_json(predictions,dict(reviewer='test',notes='fixture',predictions='measured trajectories only',prior_exposure=True))
    assert command(root,'collection-predict','--config',cp,'--review',predictions)==0
    approval=root/'collection-budget.json';write_json(approval,dict(reviewer='test',notes='bounded missing profile',session_id='session-a',source_reviewed=True,max_units_per_hour=1.,max_seconds=600.))
    engine.fail_after=1
    assert command(root,'collection-run','--config',cp,'--approval',approval,dependencies=engine)==1
    first=next((directory/'units').glob('*/complete.json'));before=first.read_bytes();engine.fail_after=None
    assert command(root,'collection-run','--config',cp,dependencies=engine)==0 and first.read_bytes()==before
    assert len(engine.calls)==6
    assert command(root,'collection-forecast','--config',cp)==0
    usage=root/'usage-profile.json';write_json(usage,dict(event_id='trajectory-profile',session_id='session-a',total_units=1.03,observed_at='2026-10-28T09:00:00+00:00',note='synthetic cumulative'));record_budget(root/'plans/audit/stage0.plan.json',root,usage)
    settle(root,json.loads((directory/'meta/active-reservation.json').read_text())['reservation_id'])
    forecast=json.loads((directory/'results/forecast.json').read_text());review=root/'collection-production.json'
    write_json(review,dict(reviewer='test',notes='measured missing work',session_id='session-a',source_reviewed=True,max_units_per_hour=1.,max_seconds=max(600.,forecast['conservative_total_seconds']),forecast_sha256=file_hash(directory/'results/forecast.json')))
    assert command(root,'collection-freeze','--config',cp,'--review',review)==0
    assert command(root,'collection-run','--config',cp,'--no-progress',dependencies=engine)==0
    assert command(root,'collection-verify','--config',cp)==0
    assert sum(len(ids) for _,ids in engine.calls)==900
    assert command(root,'report','--config',plan,'--collection',cp,'--name','with-collection')==0
    assert command(root,'verify','--config',plan,'--name','with-collection')==0
    result=json.loads((root/'reports/with-collection/results/results.json').read_text())
    zero=[c for c in result['cells'] if c['alias']=='sft-0'];assert all(c['kl']['equal_example_mean']==0 for c in zero)
    assert all(c['kl']['interval_95']==[.125,.125] for c in result['cells'] if c['alias']!='sft-0')
    assert all(c['paired_write_available'] for c in result['cells'])
    assert all(c['effective_write_mean']==.25 for c in result['cells'] if c['alias']!='sft-0')
    assert all(c['kl_over_size_squared']==pytest.approx(.125/c['total_frobenius_size']**2) for c in result['cells'] if c['alias']!='sft-0')
    assert not result['drift_complete'] and result['observational_only']
    # The identical saved session/source is reused without a model or allocation.
    selection=root/'reuse-selection.json';write_json(selection,dict(products=[dict(kind='trajectory-collection',config=str(cp.relative_to(root)))]))
    assert command(root,'prepare','--spectral',root/'plans/basis/stage0.spectral.json','--inputs',root/'fixed/activation.prepared.json','--selection',selection,'--name','reuse')==0
    rp=root/'plans/reuse/stage0.trajectory.json';assert command(root,'run','--config',rp)==0
    reused_engine=TinyDrift()
    assert command(root,'collection-prepare','--config',rp,'--qualification',root/'plans/qualify/stage0.qualification.json','--name','reused-collection','--states','sft-0','sft-64','grpo-64',dependencies=reused_engine)==0
    rcp=root/'plans/reused-collection/stage0.trajectory.collection.json';assert json.loads(rcp.read_text())['units']==[]
    assert command(root,'collection-predict','--config',rcp,'--review',predictions)==0
    assert command(root,'collection-run','--config',rcp,dependencies=reused_engine)==0 and reused_engine.loads==0
    assert command(root,'collection-verify','--config',rcp)==0
    # An otherwise identical GPU type in a new session must recollect paired values.
    different=TinyDrift(session='new-session')
    assert command(root,'collection-prepare','--config',rp,'--qualification',root/'plans/qualify/stage0.qualification.json','--name','separate-session','--states','sft-0','sft-64','grpo-64',dependencies=different)==0
    scp=root/'plans/separate-session/stage0.trajectory.collection.json'
    assert sum(len(u['ids']) for u in json.loads(scp.read_text())['units'])==900
    import os,shutil
    if os.environ.get('STAGE0_TRAJECTORY_VALIDATION_DEST'):
        destination=Path(os.environ['STAGE0_TRAJECTORY_VALIDATION_DEST']);assert not destination.exists();shutil.copytree(root,destination)
    next((directory/'units').glob('*/arrays.npz')).write_bytes(b'changed')
    assert command(root,'collection-verify','--config',cp)==1


def test_collection_rejects_gate_failure_session_change_and_budget_ceiling(ready):
    from test_stage0_drift import TinyDrift
    from test_stage0_workflow import write_json
    from pilot_eval.stage0_workflow import record_budget
    root=ready;plan=root/'plans/trajectory/stage0.trajectory.json';assert command(root,'run','--config',plan)==0
    engine=TinyDrift()
    assert command(root,'collection-prepare','--config',plan,'--qualification',root/'plans/qualify/stage0.qualification.json','--name','negative','--states','sft-0',dependencies=engine)==0
    cp=root/'plans/negative/stage0.trajectory.collection.json'
    review=root/'predictions.json';write_json(review,dict(reviewer='test',notes='fixture',predictions='zero test',prior_exposure=False));assert command(root,'collection-predict','--config',cp,'--review',review)==0
    assert command(root,'collection-run','--config',cp,dependencies=engine)==1 and engine.loads==0
    assert command(root,'collection-run','--config',cp,dependencies=TinyDrift(session='different'))==1
    approval=root/'bounded.json';write_json(approval,dict(reviewer='test',notes='negative fixture',session_id='session-a',source_reviewed=True,max_units_per_hour=1.,max_seconds=600.))
    def corrupt(value):value['arrays']['kl'][:]=.1
    bad=TinyDrift(corrupt=corrupt)
    assert command(root,'collection-run','--config',cp,'--approval',approval,dependencies=bad)==1
    directory=root/'runs/stage0/negative/trajectory-collection'
    assert list((directory/'attempts').glob('*/observations/*/arrays.npz')) and not list((directory/'units').glob('*/complete.json'))
    usage=root/'at-ceiling.json';write_json(usage,dict(event_id='trajectory-ceiling',session_id='session-a',total_units=20.,observed_at='2026-10-28T10:00:00+00:00',note='synthetic cumulative'));record_budget(root/'plans/audit/stage0.plan.json',root,usage)
    assert command(root,'collection-run','--config',cp,dependencies=engine)==1 and engine.loads==0


def test_notebooks_expose_cpu_trajectory_and_separate_missing_collection():
    import ast
    for name,required in [('stage-0-cpu-colab.ipynb',['trajectory-prepare','trajectory-run','trajectory-report','trajectory-verify']),
                          ('stage-0-gpu-colab.ipynb',['trajectory-runtime','trajectory-collection-prepare','trajectory-collection-predict','trajectory-collection-run','trajectory-collection-forecast','trajectory-collection-freeze','trajectory-collection-verify','budget-settle'])]:
        notebook=json.loads((Path('notebooks')/name).read_text());code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        for op in required:assert op in code
        for c in notebook['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']));assert c['outputs']==[] and c['execution_count'] is None


def test_legacy_token_and_write_products_preserve_policy_and_unknown_session_separation(tmp_path):
    from test_token_measurements import make_supplement_sources,TokenDependencies
    from pilot_eval.token_measurement_workflow import prepare_tokens,profile_tokens,freeze_tokens,measure_tokens,_execution
    from pilot_eval.activation_measurement import measure_activation
    from test_activation_measurement import MeasurementDependencies
    from pilot_eval.stage0_trajectory_sources import read_product,merge_products
    from pilot_eval.activation_prepare import PROJECTIONS
    sft,grpo=make_supplement_sources(tmp_path)
    prepared=prepare_tokens(sft,grpo,tmp_path,'tokens',context_chunk=8,workspace_bytes=1_000_000)
    profile_tokens(prepared,tmp_path,dependencies=TokenDependencies());execution=freeze_tokens(prepared,tmp_path,review_notes='fixture historical policy')
    measure_tokens(execution,tmp_path,dependencies=TokenDependencies());measure_activation(sft,tmp_path,dependencies=MeasurementDependencies())
    _,products=_execution(execution,tmp_path);source=products['sft'][1]
    states=[dict(alias=f'{arm}-{step}',state='verified-adapter-payload',adapter_sha256=p['source_evidence']['checkpoints'][str(step)]['adapter_sha256']) for arm,(_,p,_) in products.items() for step in [0,8,16,32,64]]
    config=dict(items_sha256=source['items_sha256'],model_identity={k:source[k] for k in ['model','model_revision','tokenizer_revision','dtype']},states=states,
        modules=[f'base_model.model.model.layers.{i}.{"self_attn" if p in PROJECTIONS[:4] else "mlp"}.{p}' for i in range(28) for p in PROJECTIONS])
    config['model_identity']['base_sha256']=source['source_evidence']['base_sha256']
    tokens=read_product(dict(key='tokens',kind='pilot4-tokens',config=str(execution.relative_to(tmp_path))),config,tmp_path)
    writes=read_product(dict(key='writes',kind='pilot3-writes',config=str(sft.relative_to(tmp_path))),config,tmp_path)
    assert tokens['coefficient_policy']['coefficient_batch']=='descriptive-sensitivity'
    assert all(r['writes'] is None for r in tokens['documents']) and all(r['kl_values'] is None for r in writes['documents'])
    assert tokens['identity']['comparison_group']!=writes['identity']['comparison_group']
    assert not any(r['kl_values'] is not None and r['writes'] is not None for r in merge_products([tokens,writes]))


def test_saved_behaviour_length_panel_runs_without_generation_or_human_labels(tmp_path):
    from test_behaviour_breakdown import source_selection,prepare,command as behaviour_command
    from pilot_eval.stage0_trajectory_report import endpoint_lengths
    selection=source_selection(tmp_path);root=tmp_path/'artifacts'
    assert prepare(selection,root)==0 and behaviour_command('score',root)==0
    output=root/'reports/lengths';(output/'results').mkdir(parents=True)
    config=dict(evaluation_state='ready',evaluation=dict(kind='behaviour',config='plans/followup/behaviour.analysis.json'))
    result=endpoint_lengths(config,root,output)
    assert len(result['comparisons'])==3 and all(c['median_paired_change']==0 for c in result['comparisons'])
    assert result['human_labels_used'] is False and result['joins_to_KL'] is False
    assert len(list((output/'results').glob('length-bootstrap-*.npy')))==3
