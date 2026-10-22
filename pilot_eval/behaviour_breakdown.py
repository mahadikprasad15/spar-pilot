"""CPU-only, immutable import and harmonized scoring of saved endpoint outputs."""
import hashlib
import json
import re
from pathlib import Path

from pilot_eval.recovery import _inside, _name
from pilot_eval.rescoring import _save_bytes
from pilot_eval.run import _write_state
from pilot_eval.scoring import _NUMBER, _numeric_value, score_gsm8k, score_gsm8k_flexible_v3
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _save_frozen


def _implementation():
    from pilot_eval import scoring
    return {p.name:file_hash(p) for p in [Path(__file__),Path(scoring.__file__)]}


def _jsonl(path):
    value=path.read_text()
    if value.lstrip().startswith('['):
        rows=json.loads(value)
    else:
        rows=[json.loads(line) for line in value.splitlines() if line.strip()]
    if not isinstance(rows,list) or not all(isinstance(r,dict) for r in rows):
        raise ValueError('source responses must be records')
    return rows


def _resolve(base,value):
    p=Path(value)
    return (base/p).resolve() if not p.is_absolute() else p.resolve()


def _normalize(record,method,step,source_index,source_row):
    item=record.get('id',record.get('item_id',record.get('group_id')))
    text=record.get('generated_text',record.get('text'))
    if (not isinstance(item,str) or not isinstance(text,str)
        or not isinstance(record.get('prompt'),str) or not isinstance(record.get('gold'),str)
        or type(record.get('token_count')) is not int or record['token_count']<0
        or record.get('stop_reason') not in ['eos','cap','other']):
        raise ValueError('invalid saved response fields')
    capped=record['stop_reason']=='cap'
    if 'capped' in record and record['capped'] is not capped:
        raise ValueError('conflicting cap metadata')
    return dict(method=method,step=step,item_id=item,prompt=record['prompt'],gold=record['gold'],
        generated_text=text,token_count=record['token_count'],stop_reason=record['stop_reason'],
        capped=capped,source_index=source_index,source_row=source_row,source_record=record)


def _source_rows(entry,config,path,index):
    rows=_jsonl(path);normalized=[];baselines={}
    for line,row in enumerate(rows):
        if entry['format']=='grpo-behaviour':
            if row.get('mode')!='greedy' or row.get('step') not in [0,64]:continue
            if row.get('draw_index',0)!=0:raise ValueError('greedy draw must be zero')
            normalized.append(_normalize(row,entry['method'],row['step'],index,line))
        elif entry['format']=='paired-trajectory':
            step=row.get('step')
            if step not in [0,8,16,32,64]:raise ValueError('unexpected paired checkpoint')
            baseline=row['baseline'];item=row.get('id',row.get('item_id'))
            if baseline.get('id',baseline.get('item_id'))!=item:
                raise ValueError('paired item identity mismatch')
            if item in baselines and baselines[item]!=baseline:
                raise ValueError('repeated baseline changed')
            baselines[item]=baseline
            if step in [0,64]:
                adapted=_normalize(row['adapted'],entry['method'],step,index,line)
                if adapted['item_id']!=item:raise ValueError('paired item identity mismatch')
                if step==0 and row['adapted']!=baseline:raise ValueError('zero pair differs from baseline')
                normalized.append(adapted)
        else:raise ValueError('unsupported source format')
    expected=entry.get('expected_ids') or [r['id'] for r in config.get('items',[])] or config.get('cohort_manifest',{}).get('item_ids')
    if not expected or len(set(expected))!=len(expected):
        raise ValueError('independent expected question IDs required from config or selection')
    steps={r['step'] for r in normalized}
    for step in steps:
        ids=[r['item_id'] for r in normalized if r['step']==step]
        if len(ids)!=len(set(ids)) or set(ids)!=set(expected):
            raise ValueError('duplicate or incomplete endpoint cohort')
    return normalized


def prepare_behaviour(selection_path,output_root,name):
    root=Path(output_root).resolve();name=_name(name);selection_path=Path(selection_path).resolve()
    selection=json.loads(selection_path.read_text());sources=selection.get('sources',[])
    if selection.get('schema_version')!=1 or not sources:
        raise ValueError('selection requires schema_version 1 and explicit sources')
    directory=root/'runs/pilot-4/behaviour-breakdown'/name
    config_path=root/'plans'/name/'behaviour.analysis.json'
    records=[];snapshot={};source_meta=[]
    for index,entry in enumerate(sources):
        if entry.get('method') not in ['sft','grpo']:raise ValueError('source method must be sft or grpo')
        response=_resolve(selection_path.parent,entry['responses']);source_config=_resolve(selection_path.parent,entry['config'])
        cfg=json.loads(source_config.read_text())
        records.extend(_source_rows(entry,cfg,response,index))
        files=[]
        for role,path in [('responses',response),('config',source_config)]:
            relative=f'inputs/source-{index:03d}/{role}'+('.jsonl' if role=='responses' else '.json')
            content=path.read_bytes();snapshot[relative]=content
            files.append(dict(role=role,original=str(path),snapshot=relative,sha256=hashlib.sha256(content).hexdigest()))
        source_meta.append(dict(method=entry['method'],format=entry['format'],files=files,
            metadata={k:cfg.get(k) for k in ['model','model_revision','tokenizer_revision','prompt_template','decoding','policies','runtime']},
            provenance='verified exported bytes; original checkpoint/training integrity not independently established'))
    keys=[(r['method'],r['step'],r['item_id']) for r in records]
    if len(keys)!=len(set(keys)):raise ValueError('duplicate endpoint response identity')
    if {(r['method'],r['step']) for r in records}!={('sft',0),('sft',64),('grpo',0),('grpo',64)}:
        raise ValueError('both methods require greedy endpoints 0 and 64')
    by_id={}
    for row in records:
        value=(row['prompt'],row['gold'])
        if row['item_id'] in by_id and by_id[row['item_id']]!=value:
            raise ValueError('question/prompt/gold mismatch for shared item')
        by_id[row['item_id']]=value
    config=dict(schema_version=1,protocol='pilot4-behaviour-breakdown-v1',name=name,
        run_path=str(directory.relative_to(root)),selection_sha256=file_hash(selection_path),
        sources=source_meta,implementation=_implementation(),
        source_evidence_kind=selection.get('evidence_kind','unverified-export'))
    with run_lock(directory):
        _save_frozen(config_path,config);_save_frozen(directory/'config.json',config)
        for relative,content in snapshot.items():_save_bytes(directory/relative,content)
        _save_bytes(directory/'inputs/responses.jsonl', ''.join(json.dumps(r,sort_keys=True)+'\n' for r in records).encode())
        _save_frozen(directory/'inputs/complete.json',dict(files={relative:file_hash(directory/relative)
            for relative in [*snapshot,'inputs/responses.jsonl']}))
        _save_frozen(directory/'meta/run_manifest.json',dict(config_path=str(config_path.relative_to(root)),
            mode='saved-output-cpu',implementation=config['implementation'],model_calls=0))
        if not (directory/'complete.json').exists():_write_state(directory,'prepared',0,4)
    return config_path


def _load(path,root):
    root=Path(root).resolve();path=Path(path).resolve()
    if not path.is_relative_to(root):raise ValueError('config escapes artifact root')
    config=json.loads(path.read_text());directory=_inside(root,config['run_path'])
    if config['implementation']!=_implementation():raise ValueError('scoring implementation changed; use a new analysis name')
    if json.loads((directory/'config.json').read_text())!=config:raise ValueError('analysis config changed')
    for source in config['sources']:
        for file in source['files']:
            if file_hash(Path(file['original']))!=file['sha256']:raise ValueError('original source changed')
    seal=json.loads((directory/'inputs/complete.json').read_text())
    for relative,digest in seal['files'].items():
        if file_hash(_inside(directory,relative))!=digest:raise ValueError('source snapshot changed')
    return config,directory,_jsonl(directory/'inputs/responses.jsonl')


def _last_number(text,gold,capped):
    matches=list(re.finditer(_NUMBER,text))
    result=dict(extracted=None,span=None,value=None,status='invalid',correct=False,invalid_reason='no_numeric_answer')
    if not matches:return result
    match=matches[-1];result.update(extracted=match.group(),span=list(match.span()))
    try:value=_numeric_value(match.group())
    except (ValueError,ZeroDivisionError):return {**result,'invalid_reason':'malformed_number'}
    result['value']=str(value)
    if capped:return {**result,'invalid_reason':'capped_response'}
    return {**result,'status':'valid','correct':value==_numeric_value(gold.rsplit('####',1)[-1].strip()),'invalid_reason':None}


def _scored(row):
    text=row['generated_text'];gold=row['gold'];capped=row['capped']
    return {**row,'scores':dict(strict=score_gsm8k(text,gold,capped)['strict'],
        flexible_v3=score_gsm8k_flexible_v3(text,gold,capped),last_number=_last_number(text,gold,capped))}


def _summary(records,config):
    groups={}
    for method in ['sft','grpo']:
        for step in [0,64]:
            rows=[r for r in records if r['method']==method and r['step']==step];entry=dict(total=len(rows),
                cap_count=sum(r['capped'] for r in rows),mean_tokens=sum(r['token_count'] for r in rows)/len(rows))
            for scorer in ['strict','flexible_v3','last_number']:
                valid=sum(r['scores'][scorer]['status']=='valid' for r in rows)
                correct=sum(r['scores'][scorer]['correct'] for r in rows)
                entry[scorer]=dict(correct=correct,wrong=valid-correct,invalid=len(rows)-valid,
                    accuracy=correct/len(rows),extractable_fraction=valid/len(rows),
                    accuracy_among_extractable=correct/valid if valid else None)
            groups[f'{method}:{step}']=entry
    ids={m:{r['item_id'] for r in records if r['method']==m} for m in ['sft','grpo']}
    return dict(groups=groups,comparison=dict(common_ids=sorted(ids['sft']&ids['grpo']),
        original_ids={m:sorted(v) for m,v in ids.items()}),execution_kind='cpu-saved-output-only',
        real_artifact_execution=config['source_evidence_kind']=='real-data',
        evidence_boundary='Verified exported bytes and harmonized scores; no independent training verification or human audit.')


def score_behaviour(path,root,max_new_units=None):
    config,directory,records=_load(path,root)
    with run_lock(directory):
        if (directory/'complete.json').exists():return verify_behaviour(path,root)
        count=0;scored=[]
        for method in ['sft','grpo']:
            for step in [0,64]:
                target=directory/'units'/f'{method}-{step}.json'
                rows=[r for r in records if r['method']==method and r['step']==step]
                if target.exists():values=json.loads(target.read_text())
                else:
                    if max_new_units is not None and count>=max_new_units:
                        _write_state(directory,'incomplete',len(scored),len(records));return dict(state='incomplete')
                    values=[_scored(r) for r in rows];_save_frozen(target,values);count+=1
                scored.extend(values)
                print(f'Behaviour scoring: {method}:{step}; {len(scored)}/{len(records)} responses',flush=True)
        result=_summary(scored,config)
        _save_frozen(directory/'results/results.json',result)
        _save_bytes(directory/'results/responses.jsonl',''.join(json.dumps(r,sort_keys=True)+'\n' for r in scored).encode())
        lines=['# Saved endpoint scoring','',result['evidence_boundary'],'',
            '| Method / step | N | Strict | Flexible-v3 | Last number | Capped |','|---|---:|---:|---:|---:|---:|']
        for group,row in result['groups'].items():
            lines.append(f"| {group} | {row['total']} | {row['strict']['accuracy']:.3f} | {row['flexible_v3']['accuracy']:.3f} | {row['last_number']['accuracy']:.3f} | {row['cap_count']} |")
        _save_bytes(directory/'results/report.md',('\n'.join(lines)+'\n').encode())
        files=[p for folder in ['units','results'] for p in (directory/folder).glob('*') if p.is_file()]
        _save_frozen(directory/'complete.json',dict(files={str(p.relative_to(directory)):file_hash(p) for p in files}))
        _write_state(directory,'completed',len(scored),len(records))
        return result


def verify_behaviour(path,root):
    config,directory,_=_load(path,root)
    seal=json.loads((directory/'complete.json').read_text())
    for relative,digest in seal['files'].items():
        if file_hash(_inside(directory,relative))!=digest:raise ValueError('scored output changed')
    return json.loads((directory/'results/results.json').read_text())
