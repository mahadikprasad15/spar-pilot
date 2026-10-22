"""Combined findings through the public saved-workflow command boundary."""
import json
from pathlib import Path

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
