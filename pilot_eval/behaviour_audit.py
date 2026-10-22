"""Source-bound single-reviewer audit; answer identification precedes reveal."""
import hashlib
import json
import random
import secrets
import os
import re
from pathlib import Path

from pilot_eval.behaviour_breakdown import _declared_settings, _jsonl
from pilot_eval.behaviour_transitions import report_transitions
from pilot_eval.recovery import _inside, _name
from pilot_eval.rescoring import _save_bytes
from pilot_eval.run import _write_state
from pilot_eval.scoring import _NUMBER, _numeric_value
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _save_frozen
from pilot_eval.behaviour_audit_ui import review_page, explanation_page


LIMITS = ('Targeted single-reviewer audit; no whole-cohort human-corrected accuracy, '
          'reasoning-validity rate or inter-reviewer agreement is estimated. Blinding '
          'is partial: writing style and previously seen results can suggest the source. '
          'Reviewer exports omit hidden fields; separate local files are not access control.')


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2)+'\n').encode()


def _parent(transitions, root):
    transitions=Path(transitions).resolve()
    if not transitions.is_relative_to(root) or not (transitions/'complete.json').is_file():
        raise ValueError('a completed transition report inside the artifact root is required')
    stage=json.loads((transitions/'config.json').read_text())
    parent_path=_inside(root,stage['parent_path'])
    report_transitions(parent_path,root,transitions.name)
    scored_config=json.loads(parent_path.read_text())
    rows=_jsonl(transitions/'results/paired-items.jsonl')
    return scored_config,rows


def _value(score):
    if score['status']!='valid':
        return None
    return _numeric_value(score['extracted'])


def _select(rows):
    selections={}
    for method in ['sft','grpo']:
        ordered=sorted((r for r in rows if r['method']==method),key=lambda r:r['item_id'])
        reasons={}
        for row in ordered:
            why=[]
            if row['bins']['flexible_v3'][0]!=row['bins']['flexible_v3'][1]:
                why.append('flexible_bin_change')
            if 'U' in row['bins']['flexible_v3']:
                why.append('flexible_invalid_endpoint')
            for endpoint in ['before','after']:
                scores=row[endpoint]['scores'];left=scores['flexible_v3'];right=scores['last_number']
                if (left['status']=='valid')!=(right['status']=='valid'):
                    why.append('validity_disagreement')
                elif left['status']=='valid' and _value(left)!=_value(right):
                    why.append('numeric_disagreement')
            if why:
                reasons[row['item_id']]=sorted(set(why))
        pool=[r['item_id'] for r in ordered if r['item_id'] not in reasons]
        controls=sorted(random.Random(42).sample(pool,min(20,len(pool))))
        mandatory=sorted(reasons)
        for item in controls:
            reasons[item]=['random_control']
        selections[method]=dict(mandatory_ids=mandatory,control_pool_ids=pool,control_ids=controls,
            selected_ids=sorted(reasons),reasons=reasons,control_seed=42,
            control_generator='Python random.Random; separate generator per method; sorted pool')
    return selections


def _same_response(left,right,parent):
    fields=['item_id','prompt','gold','generated_text','token_count','stop_reason']
    if any(left[k]!=right[k] for k in fields):
        return False
    a,complete_a=_declared_settings(left,parent);b,complete_b=_declared_settings(right,parent)
    if not (complete_a and complete_b and a==b):
        return False
    for key in ['token_ids','generated_token_ids']:
        if key in left['source_record'] and key in right['source_record']:
            if left['source_record'][key]!=right['source_record'][key]:
                return False
    return True


def _packet(rows,selection,parent):
    mapping=[]
    for pair in rows:
        if pair['item_id'] not in selection[pair['method']]['selected_ids']:
            continue
        for endpoint in ['before','after']:
            row=pair[endpoint]
            group=next((g for g in mapping if all(_same_response(row,r['record'],parent) for r in g['references'])),None)
            if group is None:
                group=dict(response_id='r_'+secrets.token_hex(12),references=[])
                mapping.append(group)
            group['references'].append(dict(method=pair['method'],step=row['step'],item_id=pair['item_id'],
                reasons=selection[pair['method']]['reasons'][pair['item_id']],record=row))
    packet=[dict(response_id=g['response_id'],question=g['references'][0]['record']['prompt'],
                 response=g['references'][0]['record']['generated_text']) for g in mapping]
    secrets.SystemRandom().shuffle(packet)
    return packet,mapping


def _implementation():
    from pilot_eval import behaviour_audit_ui
    return {p.name:file_hash(p) for p in [Path(__file__),Path(behaviour_audit_ui.__file__)]}


def _seal(directory,marker,files):
    _save_frozen(directory/marker,dict(files={relative:file_hash(directory/relative) for relative in files}))


def _check_seal(directory,marker,required):
    seal=json.loads((directory/marker).read_text())
    if set(seal.get('files',{}))!=set(required):
        raise ValueError('audit receipt inventory changed')
    for relative,digest in seal['files'].items():
        if file_hash(_inside(directory,relative))!=digest:
            raise ValueError('audit artifact changed: '+relative)


PACKET_FILES=['config.json','reviewer/packet.json','reviewer/review.html','private/selection.json','private/mapping.json']
REVEAL_FILES=['reveal/results.json','reveal/report.md','reveal/review.html']


def _revision_files(revision):
    return [f'revisions/{revision}/'+n for n in ['labels.json','results.json','report.md','review.html']]


def _load(path,root):
    root=Path(root).resolve();path=Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('audit config escapes artifact root')
    config=json.loads(path.read_text());directory=_inside(root,config['run_path'])
    if config['implementation']!=_implementation():
        raise ValueError('audit implementation changed; create a new audit identity')
    if config!=json.loads((directory/'config.json').read_text()):
        raise ValueError('audit plan differs from saved config')
    parent=_inside(root,config['transitions_path'])
    if file_hash(parent/'complete.json')!=config['transitions_seal_sha256']:
        raise ValueError('audit parent receipt changed')
    _parent(parent,root)
    _check_seal(directory,'packet.complete.json',PACKET_FILES)
    return config,directory


def prepare_audit(transitions,root,name):
    root=Path(root).resolve();name=_name(name);transitions=Path(transitions).resolve()
    parent,rows=_parent(transitions,root)
    path=root/'plans'/name/'behaviour.audit.json'
    directory=root/'runs/pilot-4/behaviour-audit'/name
    with run_lock(directory):
        if path.exists():
            config,saved=_load(path,root)
            if config['transitions_path']!=str(transitions.relative_to(root)):
                raise ValueError('audit identity already binds another parent')
            return path
        selection=_select(rows);packet,mapping=_packet(rows,selection,parent)
        content=_json_bytes(packet)
        config=dict(schema_version=1,protocol='pilot4-blinded-answer-audit-v1',name=name,
            run_path=str(directory.relative_to(root)),transitions_path=str(transitions.relative_to(root)),
            transitions_seal_sha256=file_hash(transitions/'complete.json'),
            implementation=_implementation(),packet_sha256=hashlib.sha256(content).hexdigest(),
            source_evidence_kind=parent['source_evidence_kind'],
            limitations=LIMITS)
        _save_frozen(directory/'config.json',config)
        _save_bytes(directory/'reviewer/packet.json',content)
        _save_bytes(directory/'reviewer/review.html',review_page(packet,config['packet_sha256']))
        _save_frozen(directory/'private/selection.json',selection)
        _save_frozen(directory/'private/mapping.json',mapping)
        _seal(directory,'packet.complete.json',PACKET_FILES)
        _save_frozen(directory/'meta/run_manifest.json',dict(mode='single-reviewer-cpu',model_calls=0,**config))
        _save_frozen(path,config)
        _write_state(directory,'awaiting-review',0,len(packet))
        return path


def _atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_bytes(_json_bytes(value));os.replace(temporary,path)


def _labels(values,ids,*,complete=False):
    if not isinstance(values,list):
        raise ValueError('labels must be a list')
    result=[];seen=set()
    for row in values:
        if not isinstance(row,dict) or set(row)-{'response_id','status','answer','evidence','note'}:
            raise ValueError('unknown or malformed intended-answer fields')
        item=row.get('response_id');status=row.get('status')
        if not isinstance(item,str) or item not in ids or item in seen:
            raise ValueError('unknown or duplicate response ID')
        seen.add(item)
        if status not in ['numeric','absent','ambiguous','pending']:
            raise ValueError('unsupported intended-answer status')
        answer=row.get('answer')
        if status=='numeric':
            if not isinstance(answer,str) or not re.fullmatch(_NUMBER,answer.strip()):
                raise ValueError('numeric labels require one supported numeric transcription')
            try:_numeric_value(answer.strip())
            except (ValueError,ZeroDivisionError):raise ValueError('undefined numeric intended answer')
            answer=answer.strip()
        elif answer not in [None,'']:
            raise ValueError('nonnumeric labels cannot carry a numeric answer')
        else:answer=None
        if any(not isinstance(row.get(k,''),str) for k in ['evidence','note']):
            raise ValueError('label evidence and notes must be text')
        result.append(dict(response_id=item,status=status,answer=answer,
                           evidence=row.get('evidence',''),note=row.get('note','')))
    if complete and (seen!=set(ids) or any(r['status']=='pending' for r in result)):
        raise ValueError('all selected responses require nonpending labels before freeze')
    return result


def _draft(config,directory):
    packet=json.loads((directory/'reviewer/packet.json').read_text());ids=[r['response_id'] for r in packet]
    value=dict(packet_sha256=config['packet_sha256'],reviewer=None,revision=0,
        labels=[dict(response_id=i,status='pending',answer=None,evidence='',note='') for i in ids])
    previous=None
    markers=sorted((directory/'drafts').glob('*.complete.json')) if (directory/'drafts').exists() else []
    for expected,marker in enumerate(markers,1):
        stem=f'drafts/{expected:06d}.json'
        if marker.name!=f'{expected:06d}.complete.json':
            raise ValueError('audit draft revision sequence changed')
        _check_seal(directory,str(marker.relative_to(directory)),[stem])
        row=json.loads((directory/stem).read_text())
        if (row.get('revision')!=expected or row.get('previous_sha256')!=previous
            or row.get('packet_sha256')!=config['packet_sha256']
            or (value['reviewer'] is not None and row.get('reviewer')!=value['reviewer'])):
            raise ValueError('audit draft identity/chain changed')
        _labels(row['labels'],ids)
        if {r['response_id'] for r in row['labels']}!=set(ids):
            raise ValueError('audit draft is not a complete packet snapshot')
        value=row;previous=file_hash(directory/stem)
    return value,ids,previous


def _export_draft(directory,value):
    exported={k:value[k] for k in ['packet_sha256','reviewer','revision','labels']}
    exported['base_revision']=value['revision']
    _atomic_json(directory/'reviewer/draft.json',exported)


def save_labels(path,root,labels_path):
    config,directory=_load(path,root)
    patch=json.loads(Path(labels_path).read_text())
    if not isinstance(patch,dict) or set(patch)-{'packet_sha256','reviewer','base_revision','revision','labels'}:
        raise ValueError('malformed audit label envelope')
    if patch.get('packet_sha256')!=config['packet_sha256']:
        raise ValueError('labels bind a different reviewer packet')
    reviewer=patch.get('reviewer');base=patch.get('base_revision')
    if not isinstance(reviewer,str) or not reviewer.strip() or type(base) is not int or base<0:
        raise ValueError('reviewer and explicit base_revision required')
    if 'revision' in patch and patch['revision']!=base:
        raise ValueError('export revision differs from base_revision')
    update_digest=hashlib.sha256(_json_bytes(patch)).hexdigest()
    with run_lock(directory):
        if (directory/'labels/frozen.json').exists():
            raise ValueError('labels are frozen; use an explicit exposed revision after reveal')
        current,ids,previous=_draft(config,directory)
        updates=_labels(patch.get('labels'),ids)
        if current.get('update_sha256')==update_digest:
            _export_draft(directory,current)
            return dict(revision=current['revision'],reused=True)
        if base!=current['revision']:
            raise ValueError('stale audit draft; import the current draft before saving')
        if current['reviewer'] is not None and reviewer!=current['reviewer']:
            raise ValueError('single-reviewer identity changed')
        merged={r['response_id']:r for r in current['labels']}
        merged.update({r['response_id']:r for r in updates})
        next_value=dict(packet_sha256=config['packet_sha256'],reviewer=reviewer,revision=base+1,
            previous_sha256=previous,update_sha256=update_digest,labels=[merged[i] for i in ids])
        relative=f'drafts/{base+1:06d}.json'
        _save_frozen(directory/relative,next_value)
        _seal(directory,f'drafts/{base+1:06d}.complete.json',[relative])
        _export_draft(directory,next_value)
        completed=sum(r['status']!='pending' for r in next_value['labels'])
        _write_state(directory,'awaiting-review',completed,len(ids))
        return dict(revision=base+1,completed=completed,total=len(ids))


def audit_status(path,root):
    config,directory=_load(path,root)
    with run_lock(directory):
        draft,ids,_=_draft(config,directory)
        _export_draft(directory,draft)
        counts={s:sum(r['status']==s for r in draft['labels']) for s in ['numeric','absent','ambiguous','pending']}
        selection=json.loads((directory/'private/selection.json').read_text())
        state='awaiting-review'
        if (directory/'labels/frozen.complete.json').exists():
            _frozen(config,directory)
            state='labels-frozen'
        if (directory/'reveal/complete.json').exists():
            _check_seal(directory,'reveal/complete.json',REVEAL_FILES)
            state='completed'
        return dict(state=state,revision=draft['revision'],unique_responses=len(ids),label_counts=counts,
            pair_counts={m:dict(mandatory=len(s['mandatory_ids']),controls=len(s['control_ids']),selected=len(s['selected_ids']))
                         for m,s in selection.items()},limitations=LIMITS,
            reviewer_page=str(directory/'reviewer/review.html'),draft_path=str(directory/'reviewer/draft.json'))


def freeze_labels(path,root):
    config,directory=_load(path,root)
    with run_lock(directory):
        if (directory/'labels/frozen.complete.json').exists():
            frozen=_frozen(config,directory)
            return dict(state='completed' if (directory/'reveal/complete.json').exists() else 'labels-frozen',
                        unique_responses=len(frozen['labels']))
        draft,ids,digest=_draft(config,directory)
        labels=_labels(draft['labels'],ids,complete=True)
        if draft['reviewer'] is None:
            raise ValueError('no reviewer labels have been saved')
        frozen=dict(packet_sha256=config['packet_sha256'],reviewer=draft['reviewer'],
            draft_revision=draft['revision'],draft_sha256=digest,labels=labels,label_scope='before-reveal')
        _save_frozen(directory/'labels/frozen.json',frozen)
        _seal(directory,'labels/frozen.complete.json',['labels/frozen.json'])
        _write_state(directory,'labels-frozen',len(ids),len(ids))
        return dict(state='labels-frozen',unique_responses=len(ids))


def _frozen(config,directory):
    _check_seal(directory,'labels/frozen.complete.json',['labels/frozen.json'])
    frozen=json.loads((directory/'labels/frozen.json').read_text())
    draft,ids,digest=_draft(config,directory)
    _labels(frozen['labels'],ids,complete=True)
    if (frozen.get('packet_sha256')!=config['packet_sha256'] or frozen.get('draft_sha256')!=digest
        or frozen.get('draft_revision')!=draft['revision'] or frozen.get('reviewer')!=draft['reviewer']
        or frozen['labels']!=draft['labels'] or frozen.get('label_scope')!='before-reveal'):
        raise ValueError('frozen labels differ from the bound draft or packet')
    return frozen


def _reveal_result(config,directory,frozen,*,exposed=False):
    labels={r['response_id']:r for r in frozen['labels']}
    mapping=json.loads((directory/'private/mapping.json').read_text())
    output=[]
    for group in mapping:
        label=labels[group['response_id']];references=[]
        for reference in group['references']:
            record=reference['record'];numeric=label['status']=='numeric'
            intended=_numeric_value(label['answer']) if numeric else None
            gold=_numeric_value(record['gold'].rsplit('####',1)[-1].strip())
            comparisons={}
            for scorer,score in record['scores'].items():
                extracted=_value(score)
                relationship=('reviewer_'+label['status'] if not numeric else
                    'missed_intended_answer' if score['status']!='valid' else
                    'agreement' if extracted==intended else 'wrong_number_extraction')
                comparisons[scorer]=dict(relationship=relationship,scorer=score,
                    correctness_disagrees=(bool(score['correct'])!=(intended==gold)) if numeric else None)
            references.append(dict(method=reference['method'],step=reference['step'],item_id=reference['item_id'],
                inclusion_reasons=reference['reasons'],gold=record['gold'],capped=record['capped'],
                intended_answer_correct=(intended==gold) if numeric else None,comparisons=comparisons))
        record=group['references'][0]['record']
        output.append(dict(response_id=group['response_id'],question=record['prompt'],response=record['generated_text'],
            label=label,references=references))
    return dict(schema_version=1,review_scope='selected responses only',reviewer_count=1,
        reviewer=frozen['reviewer'],labelled_unique_responses=len(output),post_reveal_revision=exposed,
        packet_sha256=config['packet_sha256'],responses=output,limitations=LIMITS,
        source_evidence_kind=config['source_evidence_kind'],explanation_review='pending')


def _revealed_markdown(result):
    lines=['# Selected intended-answer audit','',LIMITS,'',
        f"Labelled unique responses: {result['labelled_unique_responses']}; one reviewer.",
        f"Post-reveal correction: {result['post_reveal_revision']}.",
        'Explanatory categories remain separate reviewer annotations; automatic discrepancies are not causal explanations.','']
    for row in result['responses']:
        lines += ['## '+row['response_id'],'',f"Intended-answer label: {row['label']['status']} / {row['label']['answer']}",'',
                  '| Method | Step | Question ID | Intended answer matches gold | Strict | Flexible-v3 | Last number |',
                  '|---|---:|---|---|---|---|---|']
        for ref in row['references']:
            lines.append(f"| {ref['method']} | {ref['step']} | {ref['item_id']} | {ref['intended_answer_correct']} | "+
                ' | '.join(ref['comparisons'][s]['relationship'] for s in ['strict','flexible_v3','last_number'])+' |')
        lines += ['', 'Question:', '', '    '+row['question'].replace('\n','\n    '),'',
                  'Response:', '', '    '+row['response'].replace('\n','\n    '),'']
    return '\n'.join(lines).encode()


def reveal_audit(path,root):
    config,directory=_load(path,root)
    with run_lock(directory):
        frozen=_frozen(config,directory)
        ids={r['response_id'] for r in frozen['labels']}
        result=_reveal_result(config,directory,frozen)
        _save_frozen(directory/'reveal/results.json',result)
        _save_bytes(directory/'reveal/report.md',_revealed_markdown(result))
        _save_bytes(directory/'reveal/review.html',explanation_page(result,file_hash(directory/'reveal/results.json'),CATEGORIES))
        _seal(directory,'reveal/complete.json',REVEAL_FILES)
        _write_state(directory,'completed',len(ids),len(ids))
        return dict(state='completed',labelled_unique_responses=len(ids),explanation_review='pending')


def revise_labels(path,root,labels_path,revision,reason):
    config,directory=_load(path,root);revision=_name(revision)
    if not isinstance(reason,str) or not reason.strip():
        raise ValueError('post-reveal correction requires a reason')
    with run_lock(directory):
        frozen=_frozen(config,directory)
        _check_seal(directory,'reveal/complete.json',REVEAL_FILES)
        patch=json.loads(Path(labels_path).read_text())
        if (not isinstance(patch,dict) or set(patch)-{'packet_sha256','reviewer','base_revision','revision','labels'}
            or patch.get('packet_sha256')!=config['packet_sha256'] or patch.get('reviewer')!=frozen['reviewer']
            or patch.get('base_revision')!=frozen['draft_revision']):
            raise ValueError('correction must bind the original frozen packet, reviewer and draft revision')
        ids=[r['response_id'] for r in frozen['labels']]
        updates=_labels(patch.get('labels'),ids)
        if not updates:
            raise ValueError('correction requires explicit answer labels')
        merged={r['response_id']:r for r in frozen['labels']}
        merged.update({r['response_id']:r for r in updates})
        labels=_labels([merged[i] for i in ids],ids,complete=True)
        corrected=dict(packet_sha256=config['packet_sha256'],reviewer=frozen['reviewer'],labels=labels,
            original_frozen_sha256=file_hash(directory/'labels/frozen.json'),
            original_reveal_sha256=file_hash(directory/'reveal/results.json'),
            revision=revision,reason=reason.strip(),label_scope='post-reveal-exposed')
        folder=directory/'revisions'/revision
        result=_reveal_result(config,directory,corrected,exposed=True)
        result.update(revision=revision,revision_reason=reason.strip())
        _save_frozen(folder/'labels.json',corrected)
        _save_frozen(folder/'results.json',result)
        _save_bytes(folder/'report.md',_revealed_markdown(result))
        _save_bytes(folder/'review.html',explanation_page(result,file_hash(folder/'results.json'),CATEGORIES))
        _seal(directory,f'revisions/{revision}/complete.json',
              _revision_files(revision))
        return dict(state='completed',revision=revision,post_reveal_revision=True,
                    original_frozen_labels_changed=False)


CATEGORIES = ('intermediate_number','missed_format','conflicting_or_ambiguous',
              'cap_or_truncation','other','uncertain')


def _revealed_source(directory,source_revision=None):
    if source_revision is None:
        _check_seal(directory,'reveal/complete.json',REVEAL_FILES)
        path=directory/'reveal/results.json'
    else:
        source_revision=_name(source_revision)
        files=_revision_files(source_revision)
        _check_seal(directory,f'revisions/{source_revision}/complete.json',files)
        path=directory/'revisions'/source_revision/'results.json'
    return json.loads(path.read_text()),file_hash(path)


def explain_audit(path,root,annotations_path,revision,*,source_revision=None):
    config,directory=_load(path,root);revision=_name(revision)
    with run_lock(directory):
        _frozen(config,directory)
        result,source_sha=_revealed_source(directory,source_revision)
        payload=json.loads(Path(annotations_path).read_text())
        if (not isinstance(payload,dict) or set(payload)-{'packet_sha256','reviewer','reveal_sha256','annotations'}
            or payload.get('packet_sha256')!=config['packet_sha256'] or payload.get('reveal_sha256')!=source_sha
            or payload.get('reviewer')!=result['reviewer']):
            raise ValueError('explanations must bind the revealed result, packet and reviewer')
        entries=payload.get('annotations')
        if not isinstance(entries,list):
            raise ValueError('explanation annotations must be a list')
        ids={r['response_id'] for r in result['responses']};seen=set();clean=[]
        for entry in entries:
            if (not isinstance(entry,dict) or set(entry)-{'response_id','scorer','category','evidence','note'}
                or entry.get('response_id') not in ids or entry.get('scorer') not in ['strict','flexible_v3','last_number']
                or entry.get('category') not in CATEGORIES
                or any(not isinstance(entry.get(k,''),str) for k in ['evidence','note'])):
                raise ValueError('invalid explanation ID, scorer, category or evidence')
            key=(entry['response_id'],entry['scorer'])
            if key in seen:
                raise ValueError('duplicate response/scorer explanation')
            seen.add(key)
            clean.append({**entry,'evidence':entry.get('evidence',''),'note':entry.get('note','')})
        clean.sort(key=lambda r:(r['response_id'],r['scorer']))
        saved=dict(packet_sha256=config['packet_sha256'],reveal_sha256=source_sha,
            source_revision=source_revision,reviewer=payload['reviewer'],annotations=clean)
        summary=dict(review_scope='selected response/scorer pairs only',reviewer_count=1,
            annotated_response_scorer_pairs=len(clean),possible_response_scorer_pairs=3*len(ids),
            explanation_review='completed' if len(clean)==3*len(ids) else 'partial',
            category_counts={c:sum(r['category']==c for r in clean) for c in CATEGORIES},
            source_revision=source_revision,post_reveal_revision=result['post_reveal_revision'],limitations=LIMITS)
        folder=directory/'explanations'/revision
        _save_frozen(folder/'annotations.json',saved);_save_frozen(folder/'results.json',summary)
        _seal(directory,f'explanations/{revision}/complete.json',
              [f'explanations/{revision}/'+n for n in ['annotations.json','results.json']])
        return summary
