"""Gold-independent proposals remain separate from human intended-answer labels."""
from pilot_eval.behaviour_assistance import propose_answer, build_assistance


def test_proposes_final_conclusion_without_recalculating():
    result = propose_answer('70 - 30 = 40 pounds. Final answer: initial weight was 40 pounds.')
    assert result['answer'] == '40'
    assert result['rule'] == 'final_phrase'


def test_intermediate_boxes_do_not_replace_final_conclusion():
    result = propose_answer(r'He sells $\boxed{3500}$ tomatoes. Saves $\boxed{50}$ daily. So he saves $\boxed{350}$ dollars a week.')
    assert result['answer'] == '350'
    assert result['rule'] == 'terminal_box'


def test_repeated_answer_and_conflicting_conclusions():
    assert propose_answer('Final answer: Each paid $34. 34.')['answer'] == '34'
    assert propose_answer('Final answer: 10 or 12.')['answer'] is None
    assert propose_answer('So they make 12*2=$<<384>>384>>384>>384')['answer'] is None


def test_preserves_labels_and_keeps_unconfirmed_proposals_separate():
    packet=[dict(response_id='r1',question='question',response='Final answer: 40 pounds.')]
    backup=dict(packet_sha256='sha',reviewer='P',base_revision=0,
                labels=[dict(response_id='r1',status='numeric',answer=None,evidence='',note='')])
    result=build_assistance(packet,backup,{'r1':[dict(capped=False,scores={
        'flexible_v3':dict(status='valid',extracted='34'),
        'last_number':dict(status='valid',extracted='40')})]})
    row=result['rows'][0]
    assert backup['labels'][0]['answer'] is None
    assert row['human_label']['answer'] is None
    assert row['proposal']['answer']=='40'
    assert row['proposal']['human_confirmed'] is False
    assert 'flexible_v3_disagreement' in row['review_reasons']
    assert 'gold' not in str(result)


def test_public_command_binds_sources_and_does_not_expose_gold(tmp_path):
    import hashlib
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    root=tmp_path/'artifacts';run=root/'runs/audit';(run/'reviewer').mkdir(parents=True);(run/'private').mkdir()
    packet=[dict(response_id='r1',question='question',response='Final answer: 40 pounds.')]
    packet_path=run/'reviewer/packet.json';packet_path.write_text(json.dumps(packet))
    digest=hashlib.sha256(packet_path.read_bytes()).hexdigest()
    config=root/'config.json';config.write_text(json.dumps(dict(run_path='runs/audit',packet_sha256=digest)))
    backup=root/'backup.json';backup.write_text(json.dumps(dict(packet_sha256=digest,reviewer='P',base_revision=0,
        labels=[dict(response_id='r1',status='numeric',answer=None,evidence='',note='')])) )
    mapping=run/'private/mapping.json';mapping.write_text(json.dumps([dict(response_id='r1',references=[dict(method='hidden',
        record=dict(gold='SECRET_GOLD',capped=False,scores=dict(flexible_v3=dict(status='valid',extracted='40',correct=False))))])]))
    original=backup.read_bytes()
    command=[sys.executable,'-m','pilot_eval.behaviour_assistance','--audit-config',str(config),'--backup',str(backup),
             '--output-root',str(root),'--name','test']
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    target=run/'assistance/test'
    assert backup.read_bytes()==original
    assert 'SECRET_GOLD' not in (target/'results.json').read_text()
    assert 'SECRET_GOLD' not in (target/'review.html').read_text()
    assert not (run/'labels/frozen.json').exists()
    assert subprocess.run(command,capture_output=True,text=True).returncode==0
    packet_path.write_text(packet_path.read_text()+' ')
    failure=subprocess.run(command,capture_output=True,text=True)
    assert failure.returncode!=0 and 'packet bytes changed' in failure.stderr
