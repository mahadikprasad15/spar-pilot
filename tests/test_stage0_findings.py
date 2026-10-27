"""CPU public findings workflow: pairing, review identity and independent statistics."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from pilot_eval.cli import main
from pilot_eval.workflow import _hash
from test_stage0_evaluation import frozen_root,TinyEvaluation
from test_stage0_drift import TinyDrift,release_greedy
from test_stage0_workflow import write_json


def command(root,operation,*args):
    return main(['stage0-findings-'+operation,'--output-root',str(root),*map(str,args)])


@pytest.fixture(scope='module')
def products(tmp_path_factory,frozen_root):
    from pilot_eval.stage0_evaluation import prepare as evaluation_prepare,run as evaluation_run
    from pilot_eval.stage0_drift import prepare as drift_prepare,run as drift_run
    root=tmp_path_factory.mktemp('findings-sources');shutil.copytree(frozen_root,root,dirs_exist_ok=True)
    execution=root/'plans/qualify/stage0.execution.json'
    evaluation=evaluation_prepare(execution,root,batch_size=64,batches_per_unit=1)
    rows=[]
    for corpus in ['gsm8k','fineweb']:
        for i in range(150):
            masks={v:[False,True,True] if v==('solution' if corpus=='gsm8k' else 'user') else [False]*3
                   for v in ['question','solution','user']}
            rows.append(dict(id=f'{corpus}:{i}',corpus=corpus,input_ids=[1,2,3],attention_mask=[1]*3,masks=masks))
    write_json(root/'fixed/items.json',rows)
    write_json(root/'fixed/activation.prepared.json',dict(items_path='fixed/items.json',items_sha256=_hash(rows),
        model='Qwen/Qwen2.5-1.5B-Instruct',model_revision='pinned',tokenizer_revision='pinned',dtype='float32'))
    drift=drift_prepare(execution,root/'fixed/activation.prepared.json',root,'drift',batch_size=64,batches_per_unit=1)
    assert command(root,'prepare','--evaluation',evaluation,'--drift',drift,'--name','primary')==0
    evaluation_run(evaluation,root,dependencies=TinyEvaluation(),progress=False)
    review=release_greedy(root);drift_run(drift,root,dependencies=TinyDrift(),approval_path=review,progress=False)
    return root


@pytest.fixture
def source(tmp_path,products):
    shutil.copytree(products,tmp_path,dirs_exist_ok=True)
    return tmp_path


def plan(root):return root/'plans/primary/stage0.findings.json'


def test_public_raw_findings_resume_and_reconciliation(source):
    config=json.loads(plan(source).read_text())
    assert config['registration']['status']=='before-collection'
    assert config['statistics']['draws']==10000 and config['statistics']['seed']==42
    assert config['noninferiority_margin']==-.05
    assert len(config['contrasts'])==14
    assert command(source,'run','--config',plan(source),'--max-new-units',1)==0
    directory=source/'runs/stage0/primary/findings'
    receipt=next((directory/'units').glob('*/complete.json'));before=receipt.read_bytes()
    assert command(source,'run','--config',plan(source))==0
    assert receipt.read_bytes()==before
    assert command(source,'verify','--config',plan(source))==0
    result=json.loads((directory/'results/results.json').read_text())
    assert not result['real_gpu_measurement'] and len(result['contrasts'])==14
    assert result['review']['state']=='not-prepared'
    for row in result['contrasts']:
        assert row['behaviour']['n']==150 and row['behaviour']['flexible']['difference']['estimate']==0
        assert row['behaviour']['flexible']['noninferiority']=='preserved'
        assert row['behaviour']['length']['median_paired_change']['estimate']==0
        assert len(row['drift'])==2
    assert (directory/'report.html').is_file()
    assert len(list((directory/'plots').glob('*.svg')))>=3
    assert (directory/'results/behaviour.csv').is_file()
    assert (directory/'results/paired-items.jsonl').is_file()
    assert command(source,'run','--config',plan(source))==0
    data=next((directory/'units').glob('*/result.json'));data.write_text('{}')
    assert command(source,'verify','--config',plan(source))==1


def test_statistics_match_independent_draws_and_boundary():
    from pilot_eval.stage0_findings_math import paired,noninferiority,mcnemar
    indices=np.random.default_rng(42).integers(0,4,(10000,4),dtype=np.int32)
    ref=np.array([1.,0.,0.,1.]);edited=np.array([0.,1.,1.,1.]);delta=edited-ref
    value=paired(ref,edited,indices)
    expected=np.percentile(np.array([np.mean(delta[row]) for row in indices]),[2.5,97.5])
    assert value['estimate']==.25
    np.testing.assert_array_equal(value['interval_95'],expected)
    assert noninferiority([-.05,.1])=='inconclusive'
    assert noninferiority([np.nextafter(-.05,0),.1])=='preserved'
    assert noninferiority([-.2,-.06])=='loss'
    assert mcnemar(0,0)==1 and mcnemar(0,4)==.125
    # Median paired change is deliberately different from marginal median difference.
    a=np.array([0.,1.,100.]);b=np.array([1.,100.,101.])
    draws=np.random.default_rng(42).integers(0,3,(10000,3))
    assert paired(a,b,draws,statistic='median')['estimate']==1
    assert np.median(b)-np.median(a)==99


def test_blinded_review_saved_revisions_freeze_and_report_overlay(source):
    assert command(source,'run','--config',plan(source))==0
    directory=source/'runs/stage0/primary/findings'
    raw=(directory/'report.html').read_bytes()
    assert command(source,'audit-prepare','--config',plan(source))==0
    audit=directory/'audit'
    packet=json.loads((audit/'reviewer/packet.json').read_text())
    assert packet and all(set(r)=={'response_id','question','response'} for r in packet)
    assert len({r['response_id'] for r in packet})==len(packet)
    assert len(packet)<len(json.loads((audit/'private/selection.json').read_text()))
    snapshot=(audit/'reviewer/packet.json').read_bytes()
    assert command(source,'audit-prepare','--config',plan(source))==0
    assert snapshot==(audit/'reviewer/packet.json').read_bytes()
    assert command(source,'audit-reveal','--config',plan(source))==1
    draft=json.loads((audit/'reviewer/draft.json').read_text())
    draft['reviewer']='human-fixture'
    draft['labels']=[dict(response_id=r['response_id'],status='numeric',answer='2',evidence='#### 2',note='') for r in packet]
    path=source/'labels.json';write_json(path,draft)
    assert command(source,'audit-save','--config',plan(source),'--labels',path)==0
    assert command(source,'audit-save','--config',plan(source),'--labels',path)==0 # exact retry is idempotent
    stale={**draft,'labels':[dict(draft['labels'][0],answer='9')]};write_json(path,stale)
    assert command(source,'audit-save','--config',plan(source),'--labels',path)==1
    assert command(source,'audit-freeze','--config',plan(source))==0
    assert command(source,'audit-reveal','--config',plan(source))==0
    revealed=json.loads((audit/'reveal/results.json').read_text())
    assert revealed['confirmed_evidence'] and not revealed['reasoning_validated']
    # Every fixture review is capped: numeric intended answer must not bypass the cap rule.
    assert all(r['capped'] and r['overlay_correct']==0 for r in revealed['confirmed_evidence'])
    assert command(source,'report','--config',plan(source))==0
    assert (directory/'report.html').read_bytes()==raw
    snapshots=list((directory/'reports').glob('*/results/results.json'))
    assert len(snapshots)==1
    combined=json.loads(snapshots[0].read_text())
    assert combined['review']['state']=='labels-frozen'
    for c in combined['contrasts']:
        assert c['behaviour']['targeted_overlay']['preservation_claim'] is False
    assert command(source,'report','--config',plan(source))==0
    assert len(list((directory/'reports').glob('*/complete.json')))==1


def test_partial_review_report_does_not_apply_draft_labels(source):
    assert command(source,'run','--config',plan(source))==0
    assert command(source,'audit-prepare','--config',plan(source))==0
    audit=source/'runs/stage0/primary/findings/audit'
    draft=json.loads((audit/'reviewer/draft.json').read_text());draft['reviewer']='test'
    draft['labels']=[dict(draft['labels'][0],status='numeric',answer='2')]
    file=source/'partial.json';write_json(file,draft)
    assert command(source,'audit-save','--config',plan(source),'--labels',file)==0
    assert command(source,'audit-freeze','--config',plan(source))==1
    assert command(source,'report','--config',plan(source))==0
    report=next((audit.parent/'reports').glob('*/results/results.json'))
    result=json.loads(report.read_text())
    assert result['review']['state']=='awaiting-review'
    assert result['review']['numeric_overlay_records']==0
    assert all('targeted_overlay' not in c['behaviour'] for c in result['contrasts'])


def test_identical_text_reuse_does_not_normalize_or_copy_across_questions():
    from pilot_eval.stage0_findings_review import identical_label
    row=dict(prompt='Question A',text='Final answer: 2.')
    candidate=dict(question='Question A',response='Final answer: 2.',
        label=dict(response_id='old',status='numeric',answer='2',evidence='2',note=''),
        reviewer='Prasad',source={'confirmed':'source'})
    assert identical_label(row,[candidate]) is candidate
    assert identical_label(dict(row,text='Final answer: 2. '),[candidate]) is None
    assert identical_label(dict(row,prompt='Question B'),[candidate]) is None
    conflict={**candidate,'label':dict(candidate['label'],answer='3')}
    with pytest.raises(ValueError,match='Conflicting'):
        identical_label(row,[candidate,conflict])
