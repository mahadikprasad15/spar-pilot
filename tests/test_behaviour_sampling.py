"""Observed eight-draw coverage and reward at the public saved-workflow seam."""
import json
from pathlib import Path
import pytest

from pilot_eval.cli import main
from test_behaviour_transitions import prepared_scores, save


def prepared_sampled(tmp_path):
    root,old=prepared_scores(tmp_path)
    config=tmp_path/'grpo-native.json'
    ids=[f'gsm8k:test:{i}' for i in range(6)]
    items=[dict(id=i,prompt='Question '+i,gold='Compute <<1+2=3>>3\n\n#### 3') for i in ids]
    policies=[dict(step=s,mode='greedy',do_sample=False,completion_limit=1024,num_beams=1,draws=1) for s in [0,64]]
    policies += [dict(step=s,mode='sampled',do_sample=True,completion_limit=1024,num_beams=1,
                      draws=8,temperature=1.0,top_p=1.0,top_k=0) for s in [0,64]]
    save(config,dict(model='fixture',model_revision='r1',tokenizer_revision='t1',prompt_template='fixture',
        items=items,policies=policies))
    rows=[]
    for step in [0,64]:
        for index,item in enumerate(items):
            greedy=json.loads((tmp_path/f'grpo-{step}.jsonl').read_text().splitlines()[index])
            rows.append(dict(item_id=item['id'],mode='greedy',step=step,draw_index=0,
                text=greedy['generated_text'],prompt=item['prompt'],gold=item['gold'],
                token_count=10,stop_reason='eos'))
            count=([0,1,8,0,0,0] if step==0 else [1,8,0,0,0,0])[index]
            for draw in range(8):
                rows.append(dict(item_id=item['id'],group_id=item['id'],draw_index=draw,
                    draw_id=f"{item['id']}:draw:{draw}",record_id=f"sampled:{step}:{item['id']}:draw:{draw}",
                    mode='sampled',step=step,prompt=item['prompt'],gold=item['gold'],
                    text='#### 3' if draw<count else '#### 4',token_count=10,stop_reason='eos'))
    raw=tmp_path/'sampled.jsonl';raw.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    selected=json.loads((tmp_path/'selection.json').read_text())
    selected['sources']=[r for r in selected['sources'] if r['method']=='sft']
    selected['sources'].append(dict(method='grpo',format='grpo-behaviour',responses=str(raw),config=str(config)))
    selection=save(tmp_path/'sampling-selection.json',selected)
    assert main(['behaviour-prepare','--selection',str(selection),'--name','sample-parent','--output-root',str(root)])==0
    parent=root/'plans/sample-parent/behaviour.analysis.json'
    assert main(['behaviour-score','--config',str(parent),'--output-root',str(root)])==0
    return root,parent,raw,config


def sampling(root,config,name='sampling',*extra):
    return main(['behaviour-sampling','--config',str(config),'--sampled-source-index','2',
                 '--name',name,'--output-root',str(root),*extra])


def test_fixed_eight_distinguishes_coverage_from_mean_reward(tmp_path):
    root,config,_,_=prepared_sampled(tmp_path)
    assert sampling(root,config)==0
    directory=root/'reports/sampling'
    result=json.loads((directory/'results/results.json').read_text())
    row=result['metrics']['flexible_v3']
    assert row['mean_draw_reward']['before']==9/48
    assert row['mean_draw_reward']['after']==9/48
    assert row['observed_coverage']['before']==2/6
    assert row['observed_coverage']['after']==2/6
    assert row['coverage_gain_ids']==['gsm8k:test:0']
    assert row['coverage_loss_ids']==['gsm8k:test:2']
    assert row['count_transition'][0][1]==1
    assert row['count_transition'][1][8]==1
    assert row['count_transition'][8][0]==1
    assert row['already_observed_solved']['item_ids']==['gsm8k:test:1']
    assert row['already_observed_solved']['mean_count_change']==7.0
    assert result['n']==6
    assert result['response_count']==96
    assert result['bootstrap']['unit']=='question with all eight observed draws'
    assert (directory/'plots/observed-coverage.svg').is_file()
    original=(directory/'complete.json').read_bytes()
    assert sampling(root,config)==0
    assert (directory/'complete.json').read_bytes()==original


@pytest.mark.parametrize('damage',['missing','duplicate','draw_alias','record_alias','prompt','policy','training'])
def test_fixed_eight_rejects_incomplete_or_incompatible_sampled_inputs(tmp_path,damage):
    root,parent,raw,cfg=prepared_sampled(tmp_path)
    rows=[json.loads(l) for l in raw.read_text().splitlines()]
    index=next(i for i,r in enumerate(rows) if r['mode']=='sampled')
    if damage=='missing':rows.pop(index)
    if damage=='duplicate':rows.append(rows[index])
    if damage=='draw_alias':rows[index]['draw_id']='wrong'
    if damage=='record_alias':rows[index+1]['record_id']=rows[index]['record_id']
    if damage=='prompt':rows[index]['prompt']='another question'
    raw.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    if damage=='policy':
        value=json.loads(cfg.read_text());value['policies'][-1]['temperature']=0.7;save(cfg,value)
    # Re-import changed exports under a distinct analysis identity so this checks
    # sample contracts rather than simply tripping the old parent source hash.
    selection=tmp_path/'sampling-selection.json'
    if damage=='training':
        for path in tmp_path.glob('*.json*'):
            path.write_text(path.read_text().replace('gsm8k:test:','gsm8k:train:'))
    assert main(['behaviour-prepare','--selection',str(selection),'--name','bad-parent','--output-root',str(root)])==0
    parent=root/'plans/bad-parent/behaviour.analysis.json'
    assert main(['behaviour-score','--config',str(parent),'--output-root',str(root)])==0
    assert sampling(root,parent)==1
    assert not (root/'reports/sampling/complete.json').exists()


def test_sampled_partial_resume_and_altered_parent_are_checked(tmp_path):
    root,parent,raw,_=prepared_sampled(tmp_path)
    assert sampling(root,parent,'resume','--max-new-units','1')==0
    directory=root/'reports/resume'
    assert not (directory/'complete.json').exists()
    before=(directory/'units/sampled-0.json').stat().st_mtime_ns
    assert sampling(root,parent,'resume')==0
    assert (directory/'units/sampled-0.json').stat().st_mtime_ns==before
    raw.write_text(raw.read_text()+' ')
    assert sampling(root,parent,'resume')==1
