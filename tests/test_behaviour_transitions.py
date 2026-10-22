"""Paired endpoint reports through the notebook's public saved-workflow seam."""
import json
from pathlib import Path
import pytest

from pilot_eval.cli import main


def save(path, value):
    path.write_text(json.dumps(value))
    return path


def prepared_scores(tmp_path, *, overlap=False):
    ids=[f'gsm8k:test:{i}' for i in range(6)]
    sources=[]
    before=['#### 4','No answer','#### 3','#### 3','#### 3','#### 4']
    after=['#### 3','#### 3','#### 4','No answer','#### 3','#### 4']
    for method in ['sft','grpo']:
        chosen=ids[:-1] if overlap and method=='sft' else ids
        for step in [0,64]:
            items=[dict(id=i,prompt='Question '+i,gold='Compute <<1+2=3>>3\n\n#### 3') for i in chosen]
            cfg=save(tmp_path/f'{method}-{step}-config.json',dict(model='fixture',model_revision='r1',
                tokenizer_revision='t1',prompt_template='fixture',items=items,
                decoding=dict(do_sample=False,max_new_tokens=1024,num_beams=1)))
            rows=[]
            for index,item in enumerate(items):
                text=(before if step==0 else after)[index]
                if method=='sft' and step==0 and index==0:text='#### 3'
                rows.append(dict(**item,generated_text=text,token_count=10+index-step//8,stop_reason='eos'))
            path=tmp_path/f'{method}-{step}.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            sources.append(dict(method=method,format='evaluation',step=step,responses=str(path),config=str(cfg)))
    selection=save(tmp_path/'selection.json',dict(schema_version=1,sources=sources))
    root=tmp_path/'artifacts'
    assert main(['behaviour-prepare','--selection',str(selection),'--name','scored','--output-root',str(root)])==0
    config=root/'plans/scored/behaviour.analysis.json'
    assert main(['behaviour-score','--config',str(config),'--output-root',str(root)])==0
    return root,config


def transitions(root,config,name='transitions',*extra):
    return main(['behaviour-transitions','--config',str(config),'--name',name,'--output-root',str(root),*extra])


def test_paired_report_reconciles_flows_and_keeps_paired_uncertainty(tmp_path):
    root,config=prepared_scores(tmp_path)
    assert transitions(root,config)==0
    directory=root/'reports/transitions'
    result=json.loads((directory/'results/results.json').read_text())
    flow=result['within_method']['grpo']['flexible_v3']
    assert flow['counts']==[[1,1,1],[1,1,0],[1,0,0]]
    assert (flow['gross_gains'],flow['gross_losses'],flow['churn'],flow['net_change'])==(2,2,4,0)
    assert (flow['net_wrong_flow'],flow['net_invalid_flow'])==(0,0)
    assert flow['cell_ids']['U:C']==['gsm8k:test:1']
    direct=result['direct_comparison']['flexible_v3']
    # Both endpoints agree on every question. Paired differences are all zero;
    # independently bootstrapping the endpoints would introduce false variance.
    assert direct['difference']=={'estimate':0.0,'interval_95':[0.0,0.0]}
    assert direct['n']==6
    assert result['bootstrap']['draws']==10000
    assert result['bootstrap']['seed']==42
    assert result['source_comparison']['step0_identity']['shared_reference_eligible'] is False
    rows=[json.loads(l) for l in (directory/'results/paired-items.jsonl').read_text().splitlines()]
    assert len(rows)==12
    assert rows[0]['gold_calculator_annotations']==1
    assert rows[0]['gold_nonempty_lines']==2
    before=(directory/'complete.json').read_bytes()
    assert transitions(root,config)==0
    assert (directory/'complete.json').read_bytes()==before
    assert (directory/'plots/accuracy-differences.svg').is_file()


def test_partial_report_resumes_sealed_units_and_discloses_overlap(tmp_path):
    root,config=prepared_scores(tmp_path,overlap=True)
    assert transitions(root,config,'partial','--max-new-units','1')==0
    directory=root/'reports/partial'
    assert not (directory/'complete.json').exists()
    sealed=directory/'units/sft-strict.json'
    old=sealed.stat().st_mtime_ns
    assert transitions(root,config,'partial')==0
    assert sealed.stat().st_mtime_ns==old
    result=json.loads((directory/'results/results.json').read_text())
    assert result['within_method']['sft']['strict']['n']==5
    assert result['within_method']['grpo']['strict']['n']==6
    assert result['direct_comparison']['strict']['n']==5
    assert result['source_comparison']['cohorts_identical'] is False


@pytest.mark.parametrize('damage',['parent','unit','plot'])
def test_reports_reject_changed_parent_or_derived_evidence(tmp_path,damage):
    root,config=prepared_scores(tmp_path)
    assert transitions(root,config)==0
    directory=root/'reports/transitions'
    path={'parent':root/'runs/pilot-4/behaviour-breakdown/scored/results/responses.jsonl',
          'unit':directory/'units/grpo-flexible_v3.json',
          'plot':directory/'plots/accuracy-differences.svg'}[damage]
    path.write_bytes(path.read_bytes()+b' ')
    assert transitions(root,config)==1


def test_within_method_comparison_blocks_changed_decoding_contract(tmp_path):
    root,_=prepared_scores(tmp_path)
    cfg=tmp_path/'sft-64-config.json'
    value=json.loads(cfg.read_text());value['decoding']['max_new_tokens']=512;save(cfg,value)
    assert main(['behaviour-prepare','--selection',str(tmp_path/'selection.json'),
                 '--name','changed-decoder','--output-root',str(root)])==0
    parent=root/'plans/changed-decoder/behaviour.analysis.json'
    assert main(['behaviour-score','--config',str(parent),'--output-root',str(root)])==0
    assert transitions(root,parent)==1
