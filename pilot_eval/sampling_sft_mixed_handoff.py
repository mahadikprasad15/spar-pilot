"""Conservative mixed-arm evidence assessment; external closure remains pending."""
import json
from pathlib import Path

from pilot_eval import sampling_sft as source
from pilot_eval import sampling_sft_mixed_generation as generation
from pilot_eval import sampling_sft_mixed_review as review
from pilot_eval import sampling_sft_mixed_gate as gate
from pilot_eval import sampling_sft_mixed_training_data as data
from pilot_eval import sampling_sft_mixed_training as training
from pilot_eval.sampling_sft_handoff import completion
from pilot_eval.sampling_sft_sources import inside
from pilot_eval.sft import safe_name
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash


def assess(output_root,selected):
    root=Path(output_root).resolve()
    required={'source_plan','generation_plan','review_plan','gate_plan','training_data_plan','training_selection','training_plans'}
    if not isinstance(selected,dict) or set(selected)!=required or set(selected['training_plans'])!={'0','2','5'}:
        raise ValueError('Explicit source, mixed stages, and all three training plans required.')
    for key in required-{'training_plans'}:inside(root,selected[key])
    for value in selected['training_plans'].values():inside(root,value)
    audit=source.status(root/selected['source_plan'],root)['audit']
    _,gen,_=generation.load(root/selected['generation_plan'],root)
    if gen['source_plan']!=selected['source_plan']:raise ValueError('Mixed source plan mismatch.')
    generation.validate_for_scoring(root,gen)
    _,rc,_=review.load(root/selected['review_plan'],root)
    if rc['generation_plan']!=selected['generation_plan']:raise ValueError('Mixed review plan mismatch.')
    coverage=review.verify(root/selected['review_plan'],root)
    if coverage['coverage_blocked']:raise ValueError('Coverage gate remains blocked.')
    _,gc,_=gate.load(root/selected['gate_plan'],root)
    if (gc['review_plan']!=selected['review_plan'] or gc['generation_plan']!=selected['generation_plan']):
        raise ValueError('Mixed gate plan mismatch.')
    report=gate.verify(root/selected['gate_plan'],root)
    _,dc,_=data.load(root/selected['training_data_plan'],root)
    if (dc['review_plan']!=selected['review_plan'] or dc['generation_plan']!=selected['generation_plan']):
        raise ValueError('Mixed training data mismatch.')
    data.verify(root/selected['training_data_plan'],root)
    ready=data.status(root/selected['training_data_plan'],root)['readiness']
    receipt,request=training.selection(root,root/selected['training_selection'],dc)
    if request['config_sha256']!=_hash(gc):raise ValueError('Training selection belongs to a different gate.')
    if set(receipt['arms'])!={0,2,5}:raise ValueError('All three selected training arms required.')
    results={}
    for key,path in selected['training_plans'].items():
        _,tc,_=training.load(root/path,root)
        if (tc['k']!=int(key) or tc['data_plan']!=selected['training_data_plan']
            or tc['selection']!=selected['training_selection']):raise ValueError('Mixed training plan mismatch.')
        results[key]=training.verify(root/path,root)
    stages=dict(source_audited=bool(audit and audit['source_ready']),real_qualified=True,
        generation_complete=True,generation_real=True,review_complete=True,gate_complete=True,
        gate_real=report['execution_kind']=='real-cuda',
        t15_t16_ready=bool(ready and ready['t15_passed'] and ready['t16_passed'] and ready['training_ready']),
        training_complete=all(r['successful_steps']==64 for r in results.values()),
        training_real=all(r['execution_kind']=='real-cuda' for r in results.values()),
        lifecycle_complete=False,budget_settled=False)
    return dict(state='complete' if completion(stages) else 'partial',stages=stages,
        pending=['lifecycle and independent external export verification','budget settlement'],
        training_authorized=False,scope='verified mixed scientific evidence; not a backup or stop authorization')


def build(selection_path,output_root,name):
    root=Path(output_root).resolve();safe_name(name)
    selected=json.loads(Path(selection_path).read_text());result=assess(root,selected)
    folder=root/'runs'/'sampling-sft'/'mixed-handoffs'/name
    parents={str(inside(root,p).relative_to(root)):file_hash(inside(root,p)) for p in
        [*(selected[k] for k in selected if k!='training_plans'),*selected['training_plans'].values()]}
    value=dict(protocol='sampling-sft-mixed-handoff-v1',selection=selected,parents=parents,
        implementation={Path(__file__).name:file_hash(Path(__file__))},assessment=result)
    source.frozen(folder/'report.json',value)
    source.frozen(folder/'complete.json',dict(report_sha256=file_hash(folder/'report.json'),content_sha256=_hash(value)))
    return folder/'report.json'
