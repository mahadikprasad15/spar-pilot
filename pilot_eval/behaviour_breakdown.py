"""CPU-only, immutable import and harmonized scoring of saved endpoint outputs."""
import hashlib
import json
import re
from pathlib import Path

from pilot_eval.recovery import _inside, _name
from pilot_eval.rescoring import _save_bytes
from pilot_eval.run import _write_state
from pilot_eval.scoring import (_NUMBER, _numeric_value, score_gsm8k,
    score_gsm8k_flexible_v2, score_gsm8k_flexible_v3)
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
    aliases=[record[k] for k in ['id','item_id','group_id'] if k in record]
    if any(value!=item for value in aliases):raise ValueError('conflicting question ID fields')
    if 'capped' in record and record['capped'] is not capped:
        raise ValueError('conflicting cap metadata')
    if 'token_ids' in record and (not isinstance(record['token_ids'],list)
        or any(type(t) is not int or t<0 for t in record['token_ids'])
        or len(record['token_ids'])!=record['token_count']):
        raise ValueError('saved token-count metadata mismatch')
    legacy=score_gsm8k(text,record['gold'],capped)
    v3=score_gsm8k_flexible_v3(text,record['gold'],capped)
    expected={'strict':legacy['strict'],'flexible':legacy['flexible'],
        'flexible_v2':score_gsm8k_flexible_v2(text,record['gold'],capped),'flexible_v3':v3}
    checks=[]
    stored=record.get('score',{})
    if not isinstance(stored,dict):raise ValueError('invalid stored score object')
    for key,value in stored.items():
        if key not in expected:
            checks.append(dict(location='score.'+key,state='unrecognized'));continue
        if not isinstance(value,dict) or any(value.get(k)!=v for k,v in expected[key].items()):
            raise ValueError(f'stored score mismatch: {item}/score.{key}')
        checks.append(dict(location='score.'+key,state='matches'))
    for key,target in [('strict',legacy['strict']),('flexible',v3),('flexible_v3',v3)]:
        if key not in record:continue
        value=record[key]
        if not isinstance(value,dict) or any(value.get(k)!=v for k,v in target.items()):
            raise ValueError(f'stored score mismatch: {item}/{key}')
        checks.append(dict(location=key,state='matches'))
    if 'reward' in record and record['reward']!=int(v3['correct']):raise ValueError('saved reward mismatch')
    return dict(method=method,step=step,item_id=item,prompt=record['prompt'],gold=record['gold'],
        generated_text=text,token_count=record['token_count'],stop_reason=record['stop_reason'],
        capped=capped,source_index=source_index,source_row=source_row,source_record=record,
        historical_score_checks=checks)


def _source_rows(entry,config,path,index):
    rows=_jsonl(path);normalized=[];baselines={};baseline_refs={}
    if config.get('decoding',{}).get('do_sample') is True:
        raise ValueError('saved greedy source has sampled decoding config')
    if any(p.get('mode')=='greedy' and p.get('step') in [0,64] and p.get('do_sample') is True
           for p in config.get('policies',[])):
        raise ValueError('saved greedy source has sampled policy')
    for line,row in enumerate(rows):
        if entry['format'] in ['grpo-behaviour','grpo-paired']:
            if row.get('mode')!='greedy' or row.get('step') not in [0,64]:continue
            if entry['format']=='grpo-paired':
                draws=row.get('draws')
                if not isinstance(draws,list) or len(draws)!=1:raise ValueError('greedy pair requires one draw')
                outer=row;row=draws[0]
                if row.get('item_id',row.get('group_id'))!=outer.get('item_id') or row.get('step')!=outer['step'] or row.get('mode')!='greedy':
                    raise ValueError('paired draw identity mismatch')
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
            baseline_refs.setdefault(item,[]).append(line)
            adapted=_normalize(row['adapted'],entry['method'],step,index,line)
            if adapted['item_id']!=item:raise ValueError('paired item identity mismatch')
            if (adapted['prompt'],adapted['gold'])!=(baseline['prompt'],baseline['gold']):
                raise ValueError('paired prompt/gold mismatch')
            if step==0 and row['adapted']!=baseline:raise ValueError('zero pair differs from baseline')
            if step==64:
                normalized.append(adapted)
        elif entry['format']=='evaluation':
            if entry.get('step') not in [0,64]:raise ValueError('native evaluation requires an explicit endpoint')
            if config.get('decoding',{}).get('do_sample') is True:raise ValueError('evaluation source must be greedy')
            normalized.append(_normalize(row,entry['method'],entry['step'],index,line))
        else:raise ValueError('unsupported source format')
    for item,baseline in baselines.items():
        normalized.append({**_normalize(baseline,entry['method'],0,index,baseline_refs[item][0]),
            'source_references':baseline_refs[item]})
    configured=[r['id'] for r in config.get('items',[])] or config.get('cohort_manifest',{}).get('item_ids')
    expected=entry.get('expected_ids') or configured
    if configured and entry.get('expected_ids') and set(configured)!=set(entry['expected_ids']):
        raise ValueError('selected IDs differ from source config cohort')
    if not expected or len(set(expected))!=len(expected):
        raise ValueError('independent expected question IDs required from config or selection')
    steps={r['step'] for r in normalized}
    for step in steps:
        ids=[r['item_id'] for r in normalized if r['step']==step]
        if len(ids)!=len(set(ids)) or set(ids)!=set(expected):
            raise ValueError('duplicate or incomplete endpoint cohort')
    configured_items={r['id']:r for r in config.get('items',[])}
    for row in normalized:
        item=configured_items.get(row['item_id'])
        if item and any(key in item and item[key]!=row[key] for key in ['prompt','gold']):
            raise ValueError('saved response differs from configured prompt/gold')
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
            metadata={k:cfg.get(k) for k in ['model','model_revision','tokenizer_revision','prompt_template','chat_template_sha256','decoding','policies','runtime','dtype','attention_implementation']},
            provenance='verified exported bytes; original checkpoint/training integrity not independently established'))
    keys=[(r['method'],r['step'],r['item_id']) for r in records]
    if len(keys)!=len(set(keys)):raise ValueError('duplicate endpoint response identity')
    if {(r['method'],r['step']) for r in records}!={('sft',0),('sft',64),('grpo',0),('grpo',64)}:
        raise ValueError('both methods require greedy endpoints 0 and 64')
    by_id={}
    for row in records:
        value=(row['prompt'],row['gold'])
        key=(row['method'],row['item_id'])
        if key in by_id and by_id[key]!=value:
            raise ValueError('question/prompt/gold mismatch for shared item')
        by_id[key]=value
    config=dict(schema_version=1,protocol='pilot4-behaviour-breakdown-v1',name=name,
        run_path=str(directory.relative_to(root)),selection_sha256=file_hash(selection_path),
        sources=source_meta,implementation=_implementation(),
        source_evidence_kind=selection.get('evidence_kind','unverified-export'))
    snapshot['inputs/responses.jsonl']=''.join(json.dumps(r,sort_keys=True)+'\n' for r in records).encode()
    config['input_files']={relative:hashlib.sha256(content).hexdigest() for relative,content in snapshot.items()}
    with run_lock(directory):
        _save_frozen(config_path,config);_save_frozen(directory/'config.json',config)
        for relative,content in snapshot.items():_save_bytes(directory/relative,content)
        _save_frozen(directory/'inputs/complete.json',dict(files=config['input_files']))
        _save_frozen(directory/'meta/run_manifest.json',dict(config_path=str(config_path.relative_to(root)),
            mode='saved-output-cpu',implementation=config['implementation'],model_calls=0))
        if not (directory/'complete.json').exists():_write_state(directory,'prepared',0,len(records))
    return config_path


def _load(path,root):
    root=Path(root).resolve();path=Path(path).resolve()
    if not path.is_relative_to(root):raise ValueError('config escapes artifact root')
    config=json.loads(path.read_text());directory=_inside(root,config['run_path'])
    if config['implementation']!=_implementation():raise ValueError('scoring implementation changed; use a new analysis name')
    if json.loads((directory/'config.json').read_text())!=config:raise ValueError('analysis config changed')
    for source in config['sources']:
        for file in source['files']:
            original=Path(file['original'])
            if original.exists() and file_hash(original)!=file['sha256']:raise ValueError('original source changed')
    seal=json.loads((directory/'inputs/complete.json').read_text())
    if seal['files']!=config['input_files']:raise ValueError('source snapshot inventory changed')
    for relative,digest in seal['files'].items():
        if file_hash(_inside(directory,relative))!=digest:raise ValueError('source snapshot changed')
    return config,directory,_jsonl(directory/'inputs/responses.jsonl')


def _extract_last_number(text,capped):
    # Consume unsupported compound expressions too, so 1e3 or 1,00 cannot
    # accidentally become a different valid final answer such as 3 or 00.
    candidates=r'(?<![\w.])[+-]?(?:\d|\.\d)[\d,./eE+\-]*(?:\s*/\s*[+-]?\d+)?'
    matches=list(re.finditer(candidates,text))
    result=dict(extracted=None,span=None,value=None,status='invalid',correct=False,invalid_reason='no_numeric_answer')
    if not matches:return result
    match=matches[-1];literal=match.group();end=match.end()
    if literal.endswith(('.',',')):
        literal=literal[:-1];end-=1
    result.update(extracted=literal,span=[match.start(),end])
    if not re.fullmatch(_NUMBER,literal):return {**result,'invalid_reason':'unsupported_number'}
    try:value=_numeric_value(literal)
    except (ValueError,ZeroDivisionError):return {**result,'invalid_reason':'malformed_number'}
    result['value']=str(value)
    if capped:return {**result,'invalid_reason':'capped_response'}
    return {**result,'status':'valid','invalid_reason':None}


def _scored(row):
    text=row['generated_text'];gold=row['gold'];capped=row['capped']
    last=_extract_last_number(text,capped)
    if last['status']=='valid':last['correct']=_numeric_value(last['value'])==_numeric_value(gold.rsplit('####',1)[-1].strip())
    return {**row,'scores':dict(strict=score_gsm8k(text,gold,capped)['strict'],
        flexible_v3=score_gsm8k_flexible_v3(text,gold,capped),last_number=last)}


def _declared_settings(row,config):
    meta=config['sources'][row['source_index']]['metadata']
    contract=meta.get('prompt_template');contract_fields=contract if isinstance(contract,dict) else {}
    decoder=meta.get('decoding') or contract_fields.get('decoding') or {}
    policies=[p for p in meta.get('policies') or [] if p.get('mode')=='greedy' and p.get('step')==row['step']]
    if len(policies)==1:decoder=policies[0]
    value={k:meta.get(k) for k in ['model','model_revision','tokenizer_revision']}
    value.update(prompt_template=contract_fields.get('prompt_template',contract),
        do_sample=decoder.get('do_sample'),max_new_tokens=decoder.get('max_new_tokens',decoder.get('completion_limit')),
        num_beams=decoder.get('num_beams'),chat_template_sha256=meta.get('chat_template_sha256') or contract_fields.get('chat_template_sha256'))
    complete=(all(value[k] is not None for k in ['model','model_revision','tokenizer_revision','prompt_template'])
        and value['do_sample'] is False and type(value['max_new_tokens']) is int and value['max_new_tokens']>0
        and type(value['num_beams']) is int and value['num_beams']>0)
    return value,complete


def _comparison(records,config):
    groups={(m,s):{r['item_id']:r for r in records if r['method']==m and r['step']==s}
        for m in ['sft','grpo'] for s in [0,64]}
    ids={m:set(groups[m,0])&set(groups[m,64]) for m in ['sft','grpo']}
    common=sorted(ids['sft']&ids['grpo'])
    conflicts=[i for i in common if len({(groups[m,s][i]['prompt'],groups[m,s][i]['gold'])
        for m in ['sft','grpo'] for s in [0,64]})!=1]
    settings={key:_declared_settings(next(iter(rows.values())),config) for key,rows in groups.items()}
    setting_values={k:v[0] for k,v in settings.items()}
    complete=all(v[1] for v in settings.values())
    equal=setting_values['sft',64]==setting_values['grpo',64]
    same_ids=all(set(rows)==set(groups['sft',0]) for rows in groups.values())
    contents_equal=(set(groups['sft',0])==set(groups['grpo',0]) and all(
        all(groups['sft',0][i][key]==groups['grpo',0][i][key]
            for key in ['prompt','gold','generated_text','token_count','stop_reason'])
        and all(groups['sft',0][i]['source_record'].get(key)==groups['grpo',0][i]['source_record'].get(key)
            for key in ['token_ids','generated_token_ids']
            if key in groups['sft',0][i]['source_record'] and key in groups['grpo',0][i]['source_record'])
        for i in groups['sft',0]))
    baseline_settings=(settings['sft',0][1] and settings['grpo',0][1]
        and setting_values['sft',0]==setting_values['grpo',0])
    return dict(common_ids=common,original_ids={m:sorted(set(groups[m,0])|set(groups[m,64])) for m in ids},
        cohorts_identical=same_ids,prompt_gold_conflicts=conflicts,
        declared_settings={f'{m}:{s}':value for (m,s),value in setting_values.items()},
        declared_settings_complete=complete,endpoint_settings_equal=equal,
        paired_comparison_eligible=bool(common) and not conflicts and complete and equal,
        within_method_cohorts_equal={m:set(groups[m,0])==set(groups[m,64]) for m in ids},
        step0_identity=dict(response_content_equal=contents_equal,declared_settings_equal=baseline_settings,
            shared_reference_eligible=contents_equal and baseline_settings,
            evidence_boundary='Equality of exported content and declared settings; no independent weight/runtime verification.'))


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
    inventory=[dict(source_index=i,method=source['method'],format=source['format'],
        selected_records=sum(r['source_index']==i for r in records),metadata=source['metadata'],
        provenance=source['provenance']) for i,source in enumerate(config['sources'])]
    return dict(groups=groups,comparison=_comparison(records,config),source_inventory=inventory,execution_kind='cpu-saved-output-only',
        real_artifact_execution=config['source_evidence_kind']=='real-data',
        evidence_boundary='Verified exported bytes and harmonized scores; no independent training verification or human audit.')


def score_behaviour(path,root,max_new_units=None):
    config,directory,records=_load(path,root)
    if max_new_units is not None and max_new_units<1:raise ValueError('max-new-units must be positive')
    config_digest=file_hash(Path(path))
    with run_lock(directory):
        if (directory/'complete.json').exists():return verify_behaviour(path,root)
        count=0;scored=[]
        for method in ['sft','grpo']:
            for step in [0,64]:
                target=directory/'units'/f'{method}-{step}.json'
                marker=target.with_suffix('.complete.json')
                rows=[r for r in records if r['method']==method and r['step']==step]
                if marker.exists():
                    seal=json.loads(marker.read_text())
                    if seal!=dict(sha256=file_hash(target),config_sha256=config_digest):
                        _write_state(directory,'failed',len(scored),len(records),'scored unit changed')
                        raise ValueError('scored unit changed')
                    values=json.loads(target.read_text())
                else:
                    if max_new_units is not None and count>=max_new_units:
                        _write_state(directory,'incomplete',len(scored),len(records));return dict(state='incomplete')
                    values=[_scored(r) for r in rows];_save_frozen(target,values);count+=1
                    _save_frozen(marker,dict(sha256=file_hash(target),config_sha256=config_digest))
                scored.extend(values)
                _write_state(directory,'running',len(scored),len(records))
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
        _save_frozen(directory/'complete.json',dict(config_sha256=config_digest,
            files={str(p.relative_to(directory)):file_hash(p) for p in files}))
        _write_state(directory,'completed',len(scored),len(records))
        return result


def verify_behaviour(path,root):
    config,directory,_=_load(path,root)
    seal=json.loads((directory/'complete.json').read_text())
    required={'results/'+n for n in ['results.json','responses.jsonl','report.md']}
    required|={f'units/{method}-{step}{suffix}' for method in ['sft','grpo'] for step in [0,64] for suffix in ['.json','.complete.json']}
    if seal.get('config_sha256')!=file_hash(Path(path)) or set(seal.get('files',{}))!=required:
        raise ValueError('scored output inventory/config changed')
    for relative,digest in seal['files'].items():
        if file_hash(_inside(directory,relative))!=digest:raise ValueError('scored output changed')
    return json.loads((directory/'results/results.json').read_text())
