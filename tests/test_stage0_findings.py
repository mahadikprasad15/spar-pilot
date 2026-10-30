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


def test_late_registration_and_changed_sources_are_reported(source):
    assert command(source,'prepare','--evaluation',source/'plans/qualify/stage0.evaluation.json',
                   '--drift',source/'plans/drift/stage0.drift.json','--name','late')==0
    late=json.loads((source/'plans/late/stage0.findings.json').read_text())
    assert late['registration']['status']=='after-collection-start; exploratory'
    assert command(source,'run','--config',plan(source),'--max-new-units',1)==0
    raw=source/'runs/stage0/qualify/paired-evaluation/results/responses.jsonl'
    raw.write_text(raw.read_text()+'{}\n')
    assert command(source,'run','--config',plan(source))==1
    assert command(source,'audit-prepare','--config',plan(source))==1


def test_incompatible_joins_and_inflight_registration_cannot_look_preregistered(monkeypatch,tmp_path):
    import pilot_eval.stage0_findings as workflow
    evaluation=dict(execution_sha256='same',runtime={'session_id':'one'},execution_kind='real-cuda',batch_size=1,
        variants={'V0a':{}},adapter_files={'V0a':{'factor':'same'}})
    drift={**evaluation,'runtime':{'session_id':'other'}}
    monkeypatch.setattr(workflow,'load_evaluation',lambda *args:(evaluation,{},tmp_path/'evaluation'))
    monkeypatch.setattr(workflow,'load_drift',lambda *args:(drift,tmp_path/'drift'))
    with pytest.raises(ValueError,match='matching'):workflow._parents('a','b',tmp_path)
    drift['runtime']=evaluation['runtime']
    (tmp_path/'evaluation/attempts/active').mkdir(parents=True)
    write_json(tmp_path/'evaluation/attempts/active/start.json',{'model_loading':True})
    write_json(tmp_path/'evaluation.json',{});write_json(tmp_path/'drift.json',{})
    saved=workflow.prepare('evaluation.json','drift.json',tmp_path,'inflight')
    assert json.loads(saved.read_text())['registration']['status']=='after-collection-start; exploratory'


def test_audit_selection_includes_wrong_to_wrong_answer_changes_but_not_format_only():
    from pilot_eval.stage0_findings_review import selection
    def row(v,text,answer,correct=False):
        return dict(record_id=v+':q',variant=v,item_id='q',text=text,capped=False,
            flexible=dict(status='valid',extracted=answer,correct=correct))
    contrasts=[dict(id='V1-minus-V0a',edited='V1',reference='V0a')]
    records=[row('V0a','#### 9','9'),row('V1','Final answer: 8','8')]
    found=selection(records,contrasts)
    assert set(found)=={'V0a:q','V1:q'}
    same=[row('V0a','#### 2','2',True),row('V1',r'\boxed{2}','2',True)]
    assert selection(same,contrasts)=={}


def test_cpu_only_verifier_and_full_seal_reconciliation(source):
    import subprocess,sys
    assert command(source,'run','--config',plan(source))==0
    code='''
import sys
class Reject:
    def find_spec(self,name,*args):
        if name.split('.')[0] in {'torch','transformers','peft'}:raise AssertionError(name)
sys.meta_path.insert(0,Reject())
from pilot_eval.stage0_findings import verify
assert verify(sys.argv[1],sys.argv[2])['analysis_complete']
'''
    subprocess.run([sys.executable,'-c',code,str(plan(source)),str(source)],check=True)
    directory=source/'runs/stage0/primary/findings'
    # Even a locally re-sealed altered estimate must fail numeric reconstruction.
    from pilot_eval.training import file_hash
    import pilot_eval.stage0_findings as workflow
    unit=directory/'units/V1-minus-V0a'
    value=json.loads((unit/'result.json').read_text());value['behaviour']['flexible']['difference']['estimate']=.7
    write_json(unit/'result.json',value)
    receipt=json.loads((unit/'complete.json').read_text());receipt['files']['result.json']=file_hash(unit/'result.json')
    write_json(unit/'complete.json',receipt)
    marker=json.loads((directory/'complete.json').read_text())
    marker['files']['units/V1-minus-V0a/result.json']=file_hash(unit/'result.json')
    marker['files']['units/V1-minus-V0a/complete.json']=file_hash(unit/'complete.json')
    write_json(directory/'complete.json',marker)
    assert command(source,'verify','--config',plan(source))==1


def test_notebooks_expose_cpu_findings_and_precollection_registration():
    import ast
    cpu=json.loads(Path('notebooks/stage-0-cpu-colab.ipynb').read_text())
    gpu=json.loads(Path('notebooks/stage-0-gpu-colab.ipynb').read_text())
    cpu_code='\n'.join(''.join(c['source']) for c in cpu['cells'] if c['cell_type']=='code')
    for name in ['findings-run','findings-audit-prepare','findings-audit-save','findings-audit-freeze','findings-report']:
        assert name in cpu_code
    gpu_code='\n'.join(''.join(c['source']) for c in gpu['cells'] if c['cell_type']=='code')
    assert gpu_code.index('findings-prepare')<gpu_code.index('evaluation-run')
    for notebook in [cpu,gpu]:
        for c in notebook['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']));assert c['execution_count'] is None and c['outputs']==[]


def test_historical_audit_loader_requires_its_real_frozen_receipts(tmp_path):
    from test_behaviour_audit import audit_parent,prepare,action
    from pilot_eval.stage0_findings_review import _historical
    base=tmp_path/'history';base.mkdir()
    root,parent=audit_parent(base,count=4)
    assert prepare(root,parent)==0
    config_path=root/'plans/audit/behaviour.audit.json'
    config=json.loads(config_path.read_text());directory=root/config['run_path']
    packet=json.loads((directory/'reviewer/packet.json').read_text())
    labels=base/'labels.json'
    write_json(labels,dict(packet_sha256=config['packet_sha256'],reviewer='fixture-reviewer',base_revision=0,
        labels=[dict(response_id=r['response_id'],status='absent',answer=None,evidence='',note='') for r in packet]))
    assert action('save',root,'--labels',str(labels))==0
    assert action('freeze',root)==0 and action('reveal',root)==0
    candidates,identity=_historical(root,config_path,None)
    assert len(candidates)==len(packet) and len(identity['result_sha256'])==64
    (directory/'labels/frozen.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):_historical(root,config_path,None)


def test_reused_labels_keep_the_approved_single_reviewer_identity(source,monkeypatch):
    import pilot_eval.stage0_findings_review as review
    assert command(source,'run','--config',plan(source))==0
    row=dict(question='question 3',response='#### 2',
        label=dict(response_id='old',status='numeric',answer='2',evidence='#### 2',note=''),
        reviewer='Prasad',source=dict(result_sha256='a'*64,response_id='old',post_reveal_exposed=True))
    # Public saved-workflow action, with a controlled historical-source backend.
    monkeypatch.setattr(review,'_historical',lambda *args:([row],None))
    assert command(source,'audit-prepare','--config',plan(source))==0
    audit=source/'runs/stage0/primary/findings/audit'
    draft=json.loads((audit/'reviewer/draft.json').read_text());draft['reviewer']='different reviewer'
    draft['labels']=[dict(draft['labels'][0],status='absent')]
    patch=source/'new-labels.json';write_json(patch,draft)
    assert command(source,'audit-save','--config',plan(source),'--labels',patch)==1
    state=review.status(plan(source),source)
    assert state['reviewer_count']==1 and state['reviewer_identities']==['Prasad']


def test_reidentified_drift_inherits_only_an_unchanged_precollection_registration(source):
    """ADR 0015: a new drift identity keeps the original pre-collection registration."""
    from pilot_eval.stage0_drift import prepare as drift_prepare
    execution=source/'plans/qualify/stage0.execution.json'
    evaluation=source/'plans/qualify/stage0.evaluation.json'
    redo=drift_prepare(execution,source/'fixed/activation.prepared.json',source,'drift-v2',batch_size=64,batches_per_unit=1)
    assert command(source,'prepare','--evaluation',evaluation,'--drift',redo,'--name','fresh')==0
    fresh=json.loads((source/'plans/fresh/stage0.findings.json').read_text())
    assert fresh['registration']['status']=='after-collection-start; exploratory'
    assert command(source,'prepare','--evaluation',evaluation,'--drift',redo,'--name','inherited',
                   '--registered-from',plan(source))==0
    inherited=json.loads((source/'plans/inherited/stage0.findings.json').read_text())
    original=json.loads(plan(source).read_text())
    registration=inherited['registration']
    assert registration['status']=='before-collection'
    assert registration['inherited_from']['path']=='plans/primary/stage0.findings.json'
    assert registration['inherited_from']['recorded_at']==original['registration']['recorded_at']
    assert inherited['contrasts']==original['contrasts'] and inherited['hypotheses']==original['hypotheses']
    # An exploratory plan cannot lend its registration.
    assert command(source,'prepare','--evaluation',evaluation,'--drift',redo,'--name','laundered',
                   '--registered-from',source/'plans/fresh/stage0.findings.json')==1
    # A tampered original cannot lend its registration either.
    tampered=dict(original,hypotheses='changed after results')
    (source/'plans/primary/stage0.findings.json').write_text(json.dumps(tampered))
    assert command(source,'prepare','--evaluation',evaluation,'--drift',redo,'--name','tampered',
                   '--registered-from',plan(source))==1
