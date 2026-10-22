"""Combined findings through the public saved-workflow command boundary."""
import json
from pathlib import Path
import pytest

from pilot_eval.cli import main
from test_behaviour_sampling import prepared_sampled, sampling
from test_behaviour_transitions import transitions


def inputs(tmp_path):
    root,plan,_,_=prepared_sampled(tmp_path)
    assert transitions(root,plan)==0
    assert sampling(root,plan)==0
    return root,plan


def findings(root,plan,*extra):
    return main(['behaviour-findings','--config',str(plan),'--transitions',str(root/'reports/transitions'),
                 '--sampling',str(root/'reports/sampling'),'--name','findings','--output-root',str(root),*extra])


def test_partial_findings_reconciles_saved_evidence_and_reuses_completed_stage(tmp_path):
    root,plan=inputs(tmp_path)
    assert findings(root,plan)==0
    directory=root/'reports/findings'
    result=json.loads((directory/'results/results.json').read_text())
    assert result['execution_kind']=='cpu-saved-output-only'
    assert result['audit']['state']=='not-supplied'
    assert result['evidence_state']=='automatic-results-ready; human-audit-not-complete'
    assert result['recorded_flexible']['grpo:0']['available']==0
    assert result['recorded_flexible']['grpo:0']['correct'] is None
    assert result['greedy']['grpo:0']['flexible_v3']['correct']==3
    assert result['greedy']['grpo:64']['flexible_v3']['correct']==3
    assert result['sampling']['flexible_v3']['mean_draw_reward']['before']==9/48
    assert result['sampling']['flexible_v3']['observed_coverage']['after']==2/6
    assert (directory/'report.html').is_file()
    assert (directory/'plots/greedy-outcomes.svg').is_file()
    assert (directory/'results/summary.csv').is_file()
    receipt=(directory/'complete.json').read_bytes()
    assert findings(root,plan)==0
    assert (directory/'complete.json').read_bytes()==receipt


def test_full_selected_audit_report_requires_verified_freeze_and_reveal(tmp_path):
    root,plan=inputs(tmp_path)
    assert main(['behaviour-audit-prepare','--transitions',str(root/'reports/transitions'),
                 '--name','audit','--output-root',str(root)])==0
    audit_plan=root/'plans/audit/behaviour.audit.json'
    cfg=json.loads(audit_plan.read_text());directory=root/cfg['run_path']
    assert findings(root,plan,'--audit-config',str(audit_plan))==0
    partial=json.loads((root/'reports/findings/results/results.json').read_text())
    assert partial['audit']['state']=='awaiting-review'
    packet=json.loads((directory/'reviewer/packet.json').read_text())
    labels=[]
    for row in packet:
        answer=row['response'].split()[-1] if row['response'].startswith('####') else None
        labels.append(dict(response_id=row['response_id'],status='numeric' if answer else 'absent',answer=answer))
    path=tmp_path/'labels.json';path.write_text(json.dumps(dict(packet_sha256=cfg['packet_sha256'],reviewer='fixture',base_revision=0,labels=labels)))
    for op,extra in [('save',['--labels',str(path)]),('freeze',[]),('reveal',[])]:
        assert main(['behaviour-audit-'+op,'--config',str(audit_plan),'--output-root',str(root),*extra])==0
    # The previously bound partial report cannot silently become complete.
    assert findings(root,plan,'--audit-config',str(audit_plan))==1
    args=['behaviour-findings','--config',str(plan),'--transitions',str(root/'reports/transitions'),
          '--sampling',str(root/'reports/sampling'),'--name','full-findings','--output-root',str(root),
          '--audit-config',str(audit_plan)]
    assert main(args)==0
    result=json.loads((root/'reports/full-findings/results/results.json').read_text())
    assert result['audit']['complete_human_audit'] is True
    assert result['audit']['state']=='complete-selected-human-audit'
    assert result['audit']['explanation_review']=='pending'
    assert result['confirmed_case_overlay']['unconfirmed_proposals_used']==0
    assert result['confirmed_case_overlay']['original_audit_frozen'] is True
    (directory/'labels/frozen.json').write_bytes((directory/'labels/frozen.json').read_bytes()+b' ')
    assert main(args)==1


def test_assisted_report_is_partial_and_rechecks_confirmations(tmp_path):
    import subprocess
    import sys
    root,plan=inputs(tmp_path)
    assert main(['behaviour-audit-prepare','--transitions',str(root/'reports/transitions'),
                 '--name','audit','--output-root',str(root)])==0
    audit_plan=root/'plans/audit/behaviour.audit.json';cfg=json.loads(audit_plan.read_text());audit=root/cfg['run_path']
    packet=json.loads((audit/'reviewer/packet.json').read_text());numeric=next(r for r in packet if r['response']=='#### 3')
    backup=root/'backup.json';backup.write_text(json.dumps(dict(packet_sha256=cfg['packet_sha256'],reviewer='fixture',base_revision=0,
        labels=[dict(response_id=r['response_id'],status='numeric' if r['response'].startswith('####') else 'absent',
                     answer='3' if r['response_id']==numeric['response_id'] else None) for r in packet])))
    def command(module,*args):
        result=subprocess.run([sys.executable,'-m',module,*map(str,args)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
    command('pilot_eval.behaviour_assistance','--audit-config',audit_plan,'--backup',backup,'--output-root',root,'--name','proposals')
    assisted=audit/'assistance/proposals/results.json';report=root/'reports/assisted'
    command('pilot_eval.behaviour_assisted_report','--assistance',assisted,'--audit-directory',audit,
            '--transitions',root/'reports/transitions','--output',report)
    assert findings(root,plan,'--audit-config',str(audit_plan),'--assisted-report',str(report))==0
    result=json.loads((root/'reports/findings/results/results.json').read_text())
    assert result['audit']['complete_human_audit'] is False
    assert result['audit']['confirmed_numeric_answers']==1
    assert result['audit']['unconfirmed_numeric_proposals']>0
    assert result['confirmed_case_overlay']['unconfirmed_proposals_used']==0
    assert not (audit/'labels/frozen.json').exists()
    # Changing the source backup after the assisted report must be rejected,
    # even if the combined-report receipt itself has not changed.
    backup.write_text(backup.read_text()+' ')
    assert findings(root,plan,'--audit-config',str(audit_plan),'--assisted-report',str(report))==1


def test_interrupted_report_rejects_incomplete_unit_receipt(tmp_path):
    root,plan=inputs(tmp_path)
    assert findings(root,plan,'--max-new-units','1')==0
    directory=root/'reports/findings'
    assert json.loads((directory/'meta/status.json').read_text())['state']=='incomplete'
    receipt=directory/'units/synthesis.complete.json'
    receipt.write_text(json.dumps({'files':{}}))
    (directory/'results/results.json').unlink()
    assert findings(root,plan)==1
    assert not (directory/'complete.json').exists()


def test_resealed_aggregate_cannot_disagree_with_saved_per_question_counts(tmp_path):
    import hashlib
    root,plan=inputs(tmp_path)
    directory=root/'reports/sampling';path=directory/'results/results.json'
    data=json.loads(path.read_text())
    data['metrics']['flexible_v3']['coverage_gain_ids']=[]
    path.write_text(json.dumps(data))
    seal_path=directory/'complete.json';seal=json.loads(seal_path.read_text())
    seal['files']['results/results.json']=hashlib.sha256(path.read_bytes()).hexdigest()
    seal_path.write_text(json.dumps(seal))
    assert findings(root,plan)==1
    assert not (root/'reports/findings/complete.json').exists()


@pytest.mark.parametrize('target',['paired-interval','direct-contrast'])
def test_saved_uncertainty_must_reconcile_with_paired_outcomes(tmp_path,target):
    import hashlib
    root,plan=inputs(tmp_path)
    directory=root/'reports/transitions';path=directory/'results/results.json'
    data=json.loads(path.read_text())
    if target=='paired-interval':data['within_method']['grpo']['flexible_v3']['difference']['interval_95']=[.8,.9]
    else:data['direct_comparison']['flexible_v3']['difference']['estimate']=.9
    path.write_text(json.dumps(data))
    seal_path=directory/'complete.json';seal=json.loads(seal_path.read_text())
    seal['files']['results/results.json']=hashlib.sha256(path.read_bytes()).hexdigest()
    seal_path.write_text(json.dumps(seal))
    assert findings(root,plan)==1


def test_notebook_resumes_combined_report_in_fresh_cpu_process(tmp_path):
    import os
    import subprocess
    import sys
    root,plan=inputs(tmp_path)
    assert findings(root,plan,'--max-new-units','1')==0
    directory=root/'reports/findings';unit=(directory/'units/synthesis.complete.json').read_bytes()
    repo=Path(__file__).resolve().parents[1];book=repo/'notebooks/pilot-4-behaviour-breakdown.ipynb'
    blocker=tmp_path/'no-models';blocker.mkdir()
    (blocker/'sitecustomize.py').write_text("import sys\nfor n in ['torch','transformers','datasets','peft','accelerate','huggingface_hub']:\n sys.modules[n]=None\n")
    env={**os.environ,'PYTHONPATH':str(blocker)+os.pathsep+str(repo),'SPAR_REPO_DIR':str(repo),
         'SPAR_BEHAVIOUR_ROOT':str(root),'SPAR_BEHAVIOUR_NAME':'sample-parent',
         'SPAR_FINDINGS_NAME':'findings','SPAR_FINDINGS_TRANSITIONS':str(root/'reports/transitions'),
         'SPAR_FINDINGS_SAMPLING':str(root/'reports/sampling')}
    script="""
import json,sys
from pathlib import Path
book=json.loads(Path(sys.argv[1]).read_text());keep={'setup','settings','findings'};seen=set();namespace={}
for c in book['cells']:
 tags=set(c.get('metadata',{}).get('tags',[]))
 if c['cell_type']=='code' and tags&keep:
  exec(compile(''.join(c['source']),'<notebook>','exec'),namespace);seen.update(tags)
assert keep<=seen,(keep,seen)
"""
    command=[sys.executable,'-c',script,str(book)]
    first=subprocess.run(command,env=env,cwd=repo,capture_output=True,text=True)
    assert first.returncode==0,first.stderr
    assert (directory/'units/synthesis.complete.json').read_bytes()==unit
    receipt=(directory/'complete.json').read_bytes()
    for key in ['SPAR_FINDINGS_NAME','SPAR_FINDINGS_TRANSITIONS','SPAR_FINDINGS_SAMPLING']:env.pop(key)
    second=subprocess.run(command,env=env,cwd=repo,capture_output=True,text=True)
    assert second.returncode==0,second.stderr
    assert (directory/'complete.json').read_bytes()==receipt


@pytest.mark.parametrize('damage',['combined-output','parent-plan'])
def test_completed_findings_reject_changed_outputs_or_parent_identity(tmp_path,damage):
    root,plan=inputs(tmp_path)
    assert findings(root,plan)==0
    target=root/'reports/findings/report.html' if damage=='combined-output' else plan
    target.write_bytes(target.read_bytes()+b' ')
    assert findings(root,plan)==1


@pytest.mark.parametrize('aggregate',['churn','length-stratum'])
def test_descriptive_aggregates_must_match_the_paired_records(tmp_path,aggregate):
    import hashlib
    root,plan=inputs(tmp_path)
    directory=root/'reports/transitions';path=directory/'results/results.json'
    data=json.loads(path.read_text())
    if aggregate=='churn':data['within_method']['sft']['flexible_v3']['churn']=999
    else:next(iter(data['sft_descriptive'].values()))['mean_tokens_0']=999
    path.write_text(json.dumps(data))
    seal_path=directory/'complete.json';seal=json.loads(seal_path.read_text())
    seal['files']['results/results.json']=hashlib.sha256(path.read_bytes()).hexdigest()
    seal_path.write_text(json.dumps(seal))
    assert findings(root,plan)==1
