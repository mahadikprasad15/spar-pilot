"""Offline tests at the same saved-workflow commands used by the notebook."""
import json
from pathlib import Path
import pytest

from pilot_eval.cli import main


def write_json(path, value):
    path.write_text(json.dumps(value))
    return path


def source_selection(tmp_path):
    ids=['gsm8k:test:1','gsm8k:test:2']
    items=[dict(id=i,prompt='Question '+i,gold='#### 3') for i in ids]
    config=write_json(tmp_path/'source-config.json',dict(model='tiny-fixture',model_revision='r1',
        tokenizer_revision='t1',prompt_template='fixture',items=items,
        policies=[dict(step=s,mode='greedy',do_sample=False,completion_limit=1024) for s in [0,64]]))
    grpo=[];sft=[]
    for step in [0,64]:
        for index,item in enumerate(items):
            before=dict(**item,generated_text='#### 3' if index==0 else 'No answer.',token_count=4,stop_reason='eos')
            after={**before,'generated_text':'#### 3' if step==0 or index==0 else '#### 4'}
            if step==0:after=before.copy()
            grpo.append(dict(item_id=item['id'],step=step,mode='greedy',draw_index=0,
                prompt=item['prompt'],gold=item['gold'],text=after['generated_text'],
                token_count=4,stop_reason='eos',capped=False))
            sft.append(dict(step=step,id=item['id'],baseline=before,adapted=after))
    gp=tmp_path/'grpo.jsonl';gp.write_text(''.join(json.dumps(r)+'\n' for r in grpo))
    sp=tmp_path/'sft.jsonl';sp.write_text(''.join(json.dumps(r)+'\n' for r in sft))
    selection=write_json(tmp_path/'selection.json',dict(schema_version=1,sources=[
        dict(method='grpo',format='grpo-behaviour',responses=str(gp),config=str(config)),
        dict(method='sft',format='paired-trajectory',responses=str(sp),config=str(config))]))
    return selection


def prepare(selection,root,name='followup'):
    return main(['behaviour-prepare','--selection',str(selection),'--name',name,'--output-root',str(root)])


def command(operation,root,name='followup',*extra):
    return main(['behaviour-'+operation,'--config',str(root/'plans'/name/'behaviour.analysis.json'),
                 '--output-root',str(root),*extra])


def test_public_saved_scoring_roundtrip_preserves_sources_and_restores(tmp_path):
    selection=source_selection(tmp_path);before={p:p.read_bytes() for p in tmp_path.glob('*.json*')}
    root=tmp_path/'artifacts'
    assert prepare(selection,root)==0
    assert command('score',root)==0
    assert command('verify',root)==0
    run=root/'runs/pilot-4/behaviour-breakdown/followup'
    result=json.loads((run/'results/results.json').read_text())
    assert result['groups']['grpo:0']['flexible_v3']['correct']==1
    assert result['groups']['grpo:64']['flexible_v3']['wrong']==1
    assert result['groups']['sft:0']['flexible_v3']['invalid']==1
    assert result['groups']['sft:64']['last_number']['accuracy']==.5
    assert result['comparison']['common_ids']==['gsm8k:test:1','gsm8k:test:2']
    assert result['execution_kind']=='cpu-saved-output-only'
    assert result['real_artifact_execution'] is False
    assert command('score',root)==0
    assert all(p.read_bytes()==content for p,content in before.items())


@pytest.mark.parametrize('text,capped,value,status',[
    ('Costs $1,234.50.',False,'2469/2','valid'),
    ('Answer: -3/2',False,'-3/2','valid'),
    ('First 0.5; finally 1/2.',False,'1/2','valid'),
    ('Answer .5',False,'1/2','valid'),
    ('50%',False,'50','valid'),
    ('x=3',False,'3','valid'),
    ('First 10; answer 1/0',False,None,'invalid'),
    ('First 10; answer 1e3',False,None,'invalid'),
    ('First 10; answer 1,00',False,None,'invalid'),
    ('First 10; answer 1.2.3',False,None,'invalid'),
    ('First 10; answer 1 / 2',False,None,'invalid'),
    ('abc12',False,None,'invalid'),
    ('twelve apples',False,None,'invalid'),
    ('#### 3',True,'3','invalid'),
])
def test_last_number_diagnostic_uses_whole_numeric_expression(tmp_path,text,capped,value,status):
    selection=source_selection(tmp_path)
    path=tmp_path/'grpo.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
    rows[0].update(text=text,capped=capped,stop_reason='cap' if capped else 'eos')
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    root=tmp_path/'artifacts';assert prepare(selection,root)==0;assert command('score',root)==0
    scored=[json.loads(line) for line in (root/'runs/pilot-4/behaviour-breakdown/followup/results/responses.jsonl').read_text().splitlines()]
    answer=next(r for r in scored if r['method']=='grpo' and r['step']==0 and r['item_id']=='gsm8k:test:1')['scores']['last_number']
    assert (answer['value'],answer['status'])==(value,status)
