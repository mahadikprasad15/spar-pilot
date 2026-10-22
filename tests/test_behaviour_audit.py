"""Single-reviewer answer audit at the notebook's public saved-workflow boundary."""
import json
from pathlib import Path

from pilot_eval.cli import main
from test_behaviour_transitions import save


def audit_parent(tmp_path):
    root=tmp_path/'artifacts'; sources=[]
    items=[dict(id=f'gsm8k:test:{i:02d}',prompt=f'Question {chr(65+i)}',
                gold='#### 1234' if i==2 else '#### 3') for i in range(28)]
    for method in ['sft','grpo']:
        for step in [0,64]:
            config=save(tmp_path/f'{method}-{step}-config.json',dict(model='tiny-fixture',
                model_revision='r1',tokenizer_revision='t1',prompt_template='fixture',items=items,
                decoding=dict(do_sample=False,max_new_tokens=1024,num_beams=1)))
            rows=[]
            for i,item in enumerate(items):
                text='\\boxed{3}'
                if i==0:text='\\boxed{3}' if method=='sft' and step==64 else '\\boxed{4}'
                if i==1:text='No answer.'
                if i==2:text='\\boxed{1, 234}'
                if i==3 and method=='sft' and step==64:text='\\boxed{4}'
                rows.append(dict(**item,generated_text=text,token_count=8,stop_reason='eos'))
            path=tmp_path/f'{method}-{step}.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            sources.append(dict(method=method,step=step,format='evaluation',responses=str(path),config=str(config)))
    selection=save(tmp_path/'selection.json',dict(schema_version=1,sources=sources,evidence_kind='fixture'))
    assert main(['behaviour-prepare','--selection',str(selection),'--name','scores','--output-root',str(root)])==0
    parent=root/'plans/scores/behaviour.analysis.json'
    assert main(['behaviour-score','--config',str(parent),'--output-root',str(root)])==0
    assert main(['behaviour-transitions','--config',str(parent),'--name','flows','--output-root',str(root)])==0
    return root,root/'reports/flows'


def prepare(root,parent,name='audit'):
    return main(['behaviour-audit-prepare','--transitions',str(parent),'--name',name,'--output-root',str(root)])


def action(operation,root,*extra,name='audit'):
    return main(['behaviour-audit-'+operation,'--config',str(root/'plans'/name/'behaviour.audit.json'),
                 '--output-root',str(root),*extra])


def test_selection_and_export_hide_private_fields_and_deduplicate(tmp_path):
    root,parent=audit_parent(tmp_path)
    assert prepare(root,parent)==0
    config=json.loads((root/'plans/audit/behaviour.audit.json').read_text())
    directory=root/config['run_path']
    selected=json.loads((directory/'private/selection.json').read_text())
    assert selected['sft']['mandatory_ids']==[f'gsm8k:test:{i:02d}' for i in [0,1,2,3]]
    assert selected['grpo']['mandatory_ids']==[f'gsm8k:test:{i:02d}' for i in [1,2]]
    assert selected['sft']['control_ids']==[f'gsm8k:test:{i:02d}' for i in [4,5,6,7,8,10,11,12,13,14,15,16,17,18,20,21,23,24,25,26]]
    assert len(selected['grpo']['control_ids'])==20
    assert len(selected['sft']['control_pool_ids'])==24
    assert 'numeric_disagreement' in selected['sft']['reasons']['gsm8k:test:02']
    packet=json.loads((directory/'reviewer/packet.json').read_text())
    assert all(set(row)=={'response_id','question','response'} for row in packet)
    assert len({r['response_id'] for r in packet})==len(packet)
    assert not any('gsm8k:' in r['response_id'] for r in packet)
    private=json.loads((directory/'private/mapping.json').read_text())
    assert any(len(row['references'])>=3 for row in private)
    assert sum(len(row['references']) for row in private)==92
    html=(directory/'reviewer/review.html').read_text()
    for secret in ['gsm8k:test:','flexible_v3','sample-parent','gold','checkpoint','mandatory_ids']:
        assert secret not in html
    status=json.loads((directory/'meta/status.json').read_text())
    assert status['state']=='awaiting-review'
    snapshot=(directory/'packet.complete.json').read_bytes()
    assert prepare(root,parent)==0
    assert (directory/'packet.complete.json').read_bytes()==snapshot


def test_incremental_labels_survive_reload_and_freeze_before_reveal(tmp_path):
    root,parent=audit_parent(tmp_path);assert prepare(root,parent)==0
    config=json.loads((root/'plans/audit/behaviour.audit.json').read_text());directory=root/config['run_path']
    packet=json.loads((directory/'reviewer/packet.json').read_text())
    assert action('reveal',root)==1
    patch=save(tmp_path/'answers.json',dict(packet_sha256=config['packet_sha256'],reviewer='fixture-reviewer',
        base_revision=0,labels=[dict(response_id=packet[0]['response_id'],status='numeric',answer='3',evidence='conclusion',note='')]))
    assert action('save',root,'--labels',str(patch))==0
    assert action('status',root)==0
    draft=json.loads((directory/'reviewer/draft.json').read_text())
    assert draft['revision']==1
    assert sum(r['status']!='pending' for r in draft['labels'])==1
    assert action('freeze',root)==1
    # Repeat the exact update safely after a lost process response.
    assert action('save',root,'--labels',str(patch))==0
    assert json.loads((directory/'reviewer/draft.json').read_text())['revision']==1
    patch=save(tmp_path/'remaining.json',dict(packet_sha256=config['packet_sha256'],reviewer='fixture-reviewer',
        base_revision=1,labels=[dict(response_id=r['response_id'],status='numeric',answer='3',evidence='',note='') for r in packet[1:]]))
    assert action('save',root,'--labels',str(patch))==0
    assert action('freeze',root)==0
    frozen=(directory/'labels/frozen.json').read_bytes()
    assert action('freeze',root)==0
    assert action('reveal',root)==0
    assert (directory/'labels/frozen.json').read_bytes()==frozen
    result=json.loads((directory/'reveal/results.json').read_text())
    assert result['review_scope']=='selected responses only'
    assert result['reviewer_count']==1
    assert result['labelled_unique_responses']==len(packet)
    assert result['post_reveal_revision'] is False
    assert len(result['responses'])==len(packet)
