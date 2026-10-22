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
