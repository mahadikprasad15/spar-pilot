"""Reconciled saved-output findings. No model loading or new scoring protocol."""
import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pilot_eval.behaviour_breakdown import verify_behaviour, _jsonl
from pilot_eval.behaviour_transitions import _verify_files, _bin, SCORERS, BINS
from pilot_eval.behaviour_sampling import _verify as verify_sampling
from pilot_eval.recovery import _inside, _name
from pilot_eval.rescoring import _save_bytes
from pilot_eval.run import _write_state
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _save_frozen


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _seal_files(directory, marker, required=()):
    path=directory/marker
    value=json.loads(path.read_text())
    _require(set(required).issubset(value.get('files',{})), 'source receipt inventory incomplete')
    for relative,digest in value['files'].items():
        _require(file_hash(_inside(directory,relative))==digest, 'sealed source changed: '+relative)
    return value


def _close(left,right):
    return np.isclose(left,right,rtol=0,atol=1e-12)


def _interval_check(directory,relative,ids,delta,saved):
    path=_inside(directory,relative)
    manifest=json.loads(path.with_suffix('.json').read_text())
    _require(manifest['item_ids']==ids and manifest['draws']==10000 and manifest['seed']==42,
             'saved bootstrap order/procedure differs')
    indices=np.load(path,allow_pickle=False)
    _require(indices.shape==(10000,len(ids)) and np.issubdtype(indices.dtype,np.integer) and
             indices.min()>=0 and indices.max()<len(ids),'saved bootstrap indices invalid')
    delta=np.asarray(delta,dtype=float)
    limits=np.percentile(delta[indices].mean(axis=1),[2.5,97.5])
    _require(_close(delta.mean(),saved['estimate']) and
             np.allclose(limits,saved['interval_95'],rtol=0,atol=1e-12),
             'paired estimate/interval does not match saved outcomes and draws')


def _reconcile(scoring, records, transitions, pairs, sampled, samples,transition_path,sampling_path):
    groups=defaultdict(list)
    for record in records:groups[f'{record["method"]}:{record["step"]}'].append(record)
    for key,rows in groups.items():
        saved=scoring['groups'][key]
        _require(len({r['item_id'] for r in rows})==len(rows)==saved['total'], 'greedy denominator mismatch')
        for scorer in SCORERS:
            counts=Counter(_bin(r['scores'][scorer]) for r in rows)
            _require([counts[b] for b in BINS]==[saved[scorer][k] for k in ['correct','wrong','invalid']],
                     'greedy aggregate does not match per-item records')
    indexed={(r['method'],r['step'],r['item_id']):r for r in records}
    for pair in pairs:
        for endpoint in ['before','after']:
            row=pair[endpoint]
            _require(row==indexed[(row['method'],row['step'],row['item_id'])], 'transition endpoint differs from scored record')
    for method,scorers in transitions['within_method'].items():
        selected=[p for p in pairs if p['method']==method]
        for scorer,flow in scorers.items():
            cells={a+':'+b:sorted(p['item_id'] for p in selected
                if [_bin(p[e]['scores'][scorer]) for e in ['before','after']]==[a,b]) for a in BINS for b in BINS}
            _require(cells==flow['cell_ids'] and len(selected)==flow['n'], 'transition cell reconciliation failed')
            matrix=[[len(cells[a+':'+b]) for b in BINS] for a in BINS]
            gains=len(cells['W:C'])+len(cells['U:C']);losses=len(cells['C:W'])+len(cells['C:U'])
            _require(matrix==flow['counts'] and [gains,losses,gains-losses]==
                     [flow['gross_gains'],flow['gross_losses'],flow['net_change']], 'transition count reconciliation failed')
            ordered=sorted(selected,key=lambda p:p['item_id'])
            _require([p['item_id'] for p in ordered]==flow['item_ids'],'paired question order changed')
            _interval_check(transition_path,flow['resampling_path'],flow['item_ids'],
                [int(p['after']['scores'][scorer]['correct'])-int(p['before']['scores'][scorer]['correct']) for p in ordered],
                flow['difference'])
    for scorer,contrast in transitions['direct_comparison'].items():
        if not contrast.get('available'):continue
        before={r['item_id']:r for r in groups['sft:64']};after={r['item_id']:r for r in groups['grpo:64']}
        common=sorted(set(before)&set(after))
        _require(common==contrast['item_ids'] and len(common)==contrast['n'],'direct comparison cohort differs')
        _interval_check(transition_path,contrast['resampling_path'],common,
            [int(after[i]['scores'][scorer]['correct'])-int(before[i]['scores'][scorer]['correct']) for i in common],
            contrast['difference'])
    observed=defaultdict(dict)
    for row in samples:
        key=(row['item_id'],row['step'])
        _require(row['draw_index'] not in observed[key], 'duplicate sampled draw')
        observed[key][row['draw_index']]=row
    ids=sorted({key[0] for key in observed})
    _require(len(ids)==sampled['n'] and len(samples)==sampled['response_count']==len(ids)*16,
             'sampled response denominator mismatch')
    for item in ids:
        for step in [0,64]:_require(set(observed[item,step])==set(range(8)), 'eight sampled draws required')
    for scorer,metric in sampled['metrics'].items():
        derived=[]
        for item in ids:
            derived.append(dict(item_id=item,**{f'correct_{step}':sum(bool(r['scores'][scorer]['correct'])
                for r in observed[item,step].values()) for step in [0,64]}))
        _require(derived==metric['per_item'], 'sampled per-item counts changed')
        gains=[r['item_id'] for r in derived if r['correct_0']==0 and r['correct_64']>0]
        losses=[r['item_id'] for r in derived if r['correct_0']>0 and r['correct_64']==0]
        both=[r for r in derived if r['correct_0']>0 and r['correct_64']>0]
        _require(gains==metric['coverage_gain_ids'] and losses==metric['coverage_loss_ids'],
                 'sampled coverage-change IDs disagree with per-item counts')
        saved_both=metric['already_observed_solved']
        _require([r['item_id'] for r in both]==saved_both['item_ids'], 'already-solved subset IDs disagree')
        change=float(np.mean([r['correct_64']-r['correct_0'] for r in both])) if both else None
        _require((change is None and saved_both['mean_count_change'] is None) or
                 (change is not None and _close(change,saved_both['mean_count_change'])),
                 'already-solved mean count change disagrees')
        matrix=[[sum(r['correct_0']==a and r['correct_64']==b for r in derived) for b in range(9)] for a in range(9)]
        _require(matrix==metric['count_transition'], 'sampled count transition mismatch')
        _interval_check(sampling_path,metric['resampling_path'],ids,
                        [(r['correct_64']-r['correct_0'])/8 for r in derived],metric['mean_draw_reward']['difference'])
        _interval_check(sampling_path,metric['resampling_path'],ids,
                        [int(r['correct_64']>0)-int(r['correct_0']>0) for r in derived],metric['observed_coverage']['difference'])
        for step,key in [(0,'before'),(64,'after')]:
            _require(_close(sum(r[f'correct_{step}'] for r in derived)/(8*len(ids)),metric['mean_draw_reward'][key]),
                     'sampled reward reconciliation failed')
            _require(_close(sum(r[f'correct_{step}']>0 for r in derived)/len(ids),metric['observed_coverage'][key]),
                     'sampled coverage reconciliation failed')
    lengths={}
    for method in ['sft','grpo']:
        selected=[p for p in pairs if p['method']==method]
        lengths[method]={str(step):dict(n=len(selected),mean=float(np.mean([p['tokens_'+str(step)] for p in selected])),
                        median=float(np.median([p['tokens_'+str(step)] for p in selected]))) for step in [0,64]}
    return groups,lengths


def _audit(audit_path, assisted, transitions_path, pairs, root):
    if audit_path is None:
        _require(assisted is None,'an assisted report requires an audit config')
        return dict(state='not-supplied',complete_human_audit=False),{},None
    config=json.loads(audit_path.read_text());directory=_inside(root,config['run_path'])
    seal=_seal_files(directory,'packet.complete.json',['reviewer/packet.json','private/mapping.json','config.json'])
    _require(file_hash(audit_path)==seal['files']['config.json'], 'audit plan differs from sealed config')
    _require(config['transitions_path']==str(transitions_path.relative_to(root)) and
             config['transitions_seal_sha256']==file_hash(transitions_path/'complete.json'), 'audit belongs to another transition report')
    from pilot_eval.behaviour_audit import _draft, _frozen, _reveal_result, _check_seal, REVEAL_FILES
    draft,ids,_=_draft(config,directory)
    counts=dict(Counter(r['status'] for r in draft['labels']))
    state=dict(state='awaiting-review',complete_human_audit=False,packet_responses=len(ids),journal_label_counts=counts,
               reviewer_count=1,unconfirmed_automatic_proposals_used=0)
    inputs={str(audit_path.relative_to(root)):file_hash(audit_path),
            str((directory/'packet.complete.json').relative_to(root)):file_hash(directory/'packet.complete.json')}
    for f in sorted((directory/'drafts').glob('*')) if (directory/'drafts').exists() else []:
        if f.is_file():inputs[str(f.relative_to(root))]=file_hash(f)
    reveal=None
    if (directory/'labels/frozen.complete.json').exists():
        frozen=_frozen(config,directory)
        state['state']='labels-frozen'
        for f in (directory/'labels').glob('*'):
            if f.is_file():inputs[str(f.relative_to(root))]=file_hash(f)
        if (directory/'reveal/complete.json').exists():
            _check_seal(directory,'reveal/complete.json',REVEAL_FILES)
            reveal=json.loads((directory/'reveal/results.json').read_text())
            _require(reveal==_reveal_result(config,directory,frozen), 'revealed labels/results differ from frozen evidence')
            state.update(state='complete-selected-human-audit',complete_human_audit=True,
                         labelled_unique_responses=len(reveal['responses']),explanation_review=reveal['explanation_review'])
            for name in ['reveal/complete.json',*REVEAL_FILES]:
                f=directory/name
                if f.exists():inputs[str(f.relative_to(root))]=file_hash(f)
    correction=None
    if assisted is not None:
        _require(not state['complete_human_audit'],'select either original full audit or assisted correction report')
        receipt=_seal_files(assisted,'status.json',['results.json','config.json','report.md','report.html'])
        correction=json.loads((assisted/'results.json').read_text())
        provenance=json.loads((assisted/'config.json').read_text());docs=[];assistance=None
        for raw,digest in provenance['sources'].items():
            file=Path(raw)
            if not file.exists() and '/artifacts/' in raw:
                file=_inside(root,raw.split('/artifacts/',1)[1])
            _require(file.is_file() and file_hash(file)==digest,'assisted source changed or missing: '+str(file))
            if file.is_relative_to(root):inputs[str(file.relative_to(root))]=digest
            if file.name.startswith('confirmations-'):docs.append(json.loads(file.read_text()))
            elif file.name=='results.json' and 'assistance' in file.parts:
                assistance=json.loads(file.read_text())
                proposal_dir=file.parent
                proposal_receipt=_seal_files(proposal_dir,'status.json',['results.json','config.json','review.html'])
                proposal_config=json.loads((proposal_dir/'config.json').read_text())
                for source,expected in proposal_config['sources'].items():
                    original=Path(source)
                    if not original.exists() and '/artifacts/' in source:
                        original=_inside(root,source.split('/artifacts/',1)[1])
                    _require(original.is_file() and file_hash(original)==expected,
                             'assisted proposal source changed: '+str(original))
                    if original.is_relative_to(root):inputs[str(original.relative_to(root))]=expected
                for filename in [*proposal_receipt['files'],'status.json']:
                    original=proposal_dir/filename
                    inputs[str(original.relative_to(root))]=file_hash(original)
        _require(assistance is not None and assistance['packet_sha256']==config['packet_sha256'],
                 'assisted proposal packet mismatch')
        from pilot_eval.behaviour_assisted_report import summarize
        mapping=json.loads((directory/'private/mapping.json').read_text())
        _require(correction==summarize(pairs,mapping,assistance,docs), 'assisted summary fails case-level reconciliation')
        _require(correction['unconfirmed_proposals_used']==0,'automatic proposals cannot count as human corrections')
        state.update(state='assisted-targeted-review; original-full-audit-not-complete',
                     human_classification_counts=correction['human_classification_counts'],
                     confirmed_numeric_answers=correction['unique_confirmed_numeric'],
                     unconfirmed_numeric_proposals=sum(r['human_label']['status']=='numeric' for r in assistance['rows'])-correction['unique_confirmed_numeric'])
        for name in [*receipt['files'],'status.json']:
            file=assisted/name;inputs[str(file.relative_to(root))]=file_hash(file)
    if reveal is not None:
        mapping=json.loads((directory/'private/mapping.json').read_text())
        assistance=dict(rows=[dict(response_id=r['response_id'],index=i,human_label=r['label']) for i,r in enumerate(reveal['responses'],1)])
        from pilot_eval.behaviour_assisted_report import summarize
        correction=summarize(pairs,mapping,assistance,[])
        correction['original_audit_frozen']=True
        correction['scope']='confirmed numeric overlay from completed selected-answer audit; not whole-cohort human accuracy'
    return state,inputs,correction


def _summary(scoring, transitions, sampled, lengths, audit, corrections, links):
    return dict(schema_version=1,execution_kind='cpu-saved-output-only',
        evidence_state='automatic-results-ready; '+('selected-human-audit-complete' if audit['complete_human_audit'] else 'human-audit-not-complete'),
        evidence_kind='real-data' if scoring['real_artifact_execution'] else 'fixture-data',
        greedy=scoring['groups'],source_inventory=scoring['source_inventory'],source_comparison=scoring['comparison'],
        transitions=transitions['within_method'],direct_comparison=transitions['direct_comparison'],
        original_endpoint_totals=transitions['original_endpoint_totals'],excluded_from_within_method=transitions['excluded_from_within_method'],
        sft_descriptive=transitions['sft_descriptive'],response_lengths=lengths,
        sampling=sampled['metrics'],sampling_n=sampled['n'],sampled_responses=sampled['response_count'],
        audit=audit,confirmed_case_overlay=corrections,links=links,
        methods=dict(greedy_bootstrap=transitions['bootstrap'],sampling_bootstrap=sampled['bootstrap'],
                     draws_per_question=sampled['draws_per_question'],new_inference=False),
        limitations=['Scorer-defined answer outcomes do not establish reasoning validity or newly acquired capability.',
                     'A targeted single-reviewer audit is not exact overall human-corrected accuracy or an inter-reviewer agreement estimate.',
                     'Assisted confirmations were informed by triage; unconfirmed automatic proposals never change reported corrections.',
                     'Confirmed-case overlays retain original scores outside the confirmed cases and have no revised confidence intervals.',
                     'Sampled reward and coverage remain scored automatically; the greedy audit does not correct the sampled responses.',
                     'Intervals condition on saved outputs and these trained checkpoints, not training seeds or fresh generations.',
                     'Original cohort denominators and common-ID comparisons are distinct; representation or reasoning causality is not measured here.'],
        next_decisions=['Forced read-off is optional: identify unresolved responses and approve a measured inference budget before running it.',
                        'A reasoning-quality audit is a separate extension if the question is valid reasoning rather than identifiable final answers.',
                        'Before broader claims, assess remaining extraction uncertainty and validate on additional training seeds or held-out tasks.'])


def _csv(result):
    stream=io.StringIO();writer=csv.writer(stream)
    writer.writerow(['method','step','scorer','correct','wrong','invalid','n','accuracy','evidence'])
    for key,group in result['greedy'].items():
        method,step=key.split(':')
        for scorer in SCORERS:
            s=group[scorer];writer.writerow([method,step,scorer,s['correct'],s['wrong'],s['invalid'],group['total'],s['accuracy'],'original automatic'])
    if result['confirmed_case_overlay']:
        for method,scenarios in result['confirmed_case_overlay']['methods'].items():
            s=scenarios['confirmed_corrections_only']
            for step,key in [(0,'before_correct'),(64,'after_correct')]:
                writer.writerow([method,step,'confirmed-case-overlay',s[key],'','',s['n'],s[key]/s['n'],'partial confirmed corrections only'])
    return stream.getvalue().encode()


def build_findings(path,transition_path,sampling_path,root,name,*,audit_path=None,assisted_report=None,max_new_units=None):
    if max_new_units is not None:_require(max_new_units>=1,'max-new-units must be positive')
    root=Path(root).resolve();path=Path(path).resolve();name=_name(name)
    for file in [path,transition_path,sampling_path,audit_path,assisted_report]:
        if file is not None:_require(Path(file).resolve().is_relative_to(root),'all findings inputs must be inside artifact root')
    transition_path=Path(transition_path).resolve();sampling_path=Path(sampling_path).resolve()
    audit_path=Path(audit_path).resolve() if audit_path else None
    assisted_report=Path(assisted_report).resolve() if assisted_report else None
    scoring=verify_behaviour(path,root);parent=json.loads(path.read_text());parent_dir=_inside(root,parent['run_path'])
    transitions=_verify_files(transition_path);sampled=verify_sampling(sampling_path)
    for directory in [transition_path,sampling_path]:
        cfg=json.loads((directory/'config.json').read_text())
        _require(cfg['parent_path']==str(path.relative_to(root)) and cfg['parent_sha256']==file_hash(path) and
                 cfg['parent_seal_sha256']==file_hash(parent_dir/'complete.json'),'report belongs to another scored parent')
    records=_jsonl(parent_dir/'results/responses.jsonl');pairs=_jsonl(transition_path/'results/paired-items.jsonl')
    samples=_jsonl(sampling_path/'results/responses.jsonl')
    _,lengths=_reconcile(scoring,records,transitions,pairs,sampled,samples,transition_path,sampling_path)
    audit,extra,corrections=_audit(audit_path,assisted_report,transition_path,pairs,root)
    from pilot_eval import behaviour_findings_render
    files=[path,parent_dir/'complete.json',transition_path/'config.json',transition_path/'complete.json',
           sampling_path/'config.json',sampling_path/'complete.json']
    inputs={str(f.relative_to(root)):file_hash(f) for f in files};inputs.update(extra)
    # Mutable journal additions change audit state. Record the explicit inventory,
    # including absence, so a later review requires a new combined-report identity.
    config=dict(protocol='pilot4-combined-findings-v1',name=name,inputs=inputs,audit=audit,
                implementation={p.name:file_hash(p) for p in [Path(__file__),Path(behaviour_findings_render.__file__)]},
                scores=str(path.relative_to(root)),transitions=str(transition_path.relative_to(root)),
                sampling=str(sampling_path.relative_to(root)),audit_config=str(audit_path.relative_to(root)) if audit_path else None,
                assisted_report=str(assisted_report.relative_to(root)) if assisted_report else None)
    directory=root/'reports'/name
    links=dict(scores=str(parent_dir.relative_to(root)),transitions=str(transition_path.relative_to(root)),
               sampling=str(sampling_path.relative_to(root)),audit_config=config['audit_config'],assisted_report=config['assisted_report'])
    result=_summary(scoring,transitions,sampled,lengths,audit,corrections,links)
    expected_outputs={
        'synthesis':['results/results.json','results/summary.csv','results/paired-items.jsonl','results/sampled-counts.jsonl'],
        'plots':['plots/greedy-outcomes.svg','plots/answer-transitions.svg','plots/sampling-reliability.svg']+
                (['plots/confirmed-corrections.svg'] if corrections else []),
        'presentation':['results/report.md','report.html']}
    required_complete=[f for group in expected_outputs.values() for f in group]
    with run_lock(directory):
        _save_frozen(directory/'config.json',config)
        _save_frozen(root/'plans'/name/'behaviour.findings.json',config)
        _save_frozen(directory/'meta/run_manifest.json',dict(stage='combined-behaviour-findings',**config))
        if (directory/'complete.json').exists():
            seal=_seal_files(directory,'complete.json',required_complete)
            _require(seal['config_sha256']==file_hash(directory/'config.json'),'combined report config changed')
            return dict(state='completed',evidence_state=result['evidence_state'],report=str(directory/'report.html'),reused=True)
        tasks=['synthesis','plots','presentation'];new=0
        for index,task in enumerate(tasks):
            marker=directory/'units'/(task+'.complete.json')
            if marker.exists():
                _seal_files(directory,str(marker.relative_to(directory)),expected_outputs[task])
                continue
            if max_new_units is not None and new>=max_new_units:
                _write_state(directory,'incomplete',index,len(tasks))
                return dict(state='incomplete',completed_units=index)
            print(f'Combined findings: {task} ({index+1}/{len(tasks)})',flush=True)
            if task=='synthesis':
                _save_frozen(directory/'results/results.json',result)
                _save_bytes(directory/'results/summary.csv',_csv(result))
                _save_bytes(directory/'results/paired-items.jsonl',(transition_path/'results/paired-items.jsonl').read_bytes())
                _save_bytes(directory/'results/sampled-counts.jsonl',(sampling_path/'results/per-item.jsonl').read_bytes())
                outputs=['results/results.json','results/summary.csv','results/paired-items.jsonl','results/sampled-counts.jsonl']
            elif task=='plots':outputs=behaviour_findings_render.plots(result,directory)
            else:
                _save_bytes(directory/'results/report.md',behaviour_findings_render.markdown(result).encode())
                _save_bytes(directory/'report.html',behaviour_findings_render.page(result,directory,root).encode())
                outputs=['results/report.md','report.html']
            _require(set(outputs)==set(expected_outputs[task]),'combined unit inventory differs')
            _save_frozen(marker,dict(files={f:file_hash(directory/f) for f in outputs}))
            new+=1;_write_state(directory,'running',index+1,len(tasks))
            _save_frozen(directory/'checkpoints'/f'{index+1}.json',dict(completed_unit=task,unit_receipt=str(marker.relative_to(directory))))
        inventory={str(p.relative_to(directory)):file_hash(p) for folder in ['results','plots','units','checkpoints']
                   for p in (directory/folder).glob('*') if p.is_file()}
        inventory['report.html']=file_hash(directory/'report.html')
        _save_frozen(directory/'complete.json',dict(config_sha256=file_hash(directory/'config.json'),files=inventory))
        _write_state(directory,'completed',3,3)
        return dict(state='completed',evidence_state=result['evidence_state'],report=str(directory/'report.html'),reused=False)
