"""Source-bound single-reviewer audit; answer identification precedes reveal."""
import hashlib
import html
import json
import random
import secrets
from pathlib import Path

from pilot_eval.behaviour_breakdown import _declared_settings, _jsonl
from pilot_eval.behaviour_transitions import report_transitions
from pilot_eval.recovery import _inside, _name
from pilot_eval.rescoring import _save_bytes
from pilot_eval.run import _write_state
from pilot_eval.scoring import _numeric_value
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _save_frozen


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


def _review_page(packet):
    cards='\n'.join('<article><h2>'+html.escape(row['response_id'])+'</h2><h3>Question</h3><pre>'+
        html.escape(row['question'])+'</pre><h3>Response</h3><pre>'+html.escape(row['response'])+'</pre></article>' for row in packet)
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>Answer review</title><style>body{font:16px/1.6 system-ui;max-width:900px;margin:40px auto;padding:20px;color:#263b40;background:#f8f8f3}article{padding:24px;background:white;border:1px solid #dbe2dc;margin:24px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}h2{font:13px monospace}</style>'
        '<h1>Intended-answer review</h1><p>Identify the response’s intended final answer. Mark absent or ambiguous when appropriate. '
        'This is an answer-identification task, not a judgement of reasoning validity.</p>'+cards+'</html>').encode()


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


def _load(path,root):
    root=Path(root).resolve();path=Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('audit config escapes artifact root')
    config=json.loads(path.read_text());directory=_inside(root,config['run_path'])
    if config['implementation_sha256']!=file_hash(Path(__file__)):
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
            implementation_sha256=file_hash(Path(__file__)),packet_sha256=hashlib.sha256(content).hexdigest(),
            limitations=LIMITS)
        _save_frozen(directory/'config.json',config)
        _save_bytes(directory/'reviewer/packet.json',content)
        _save_bytes(directory/'reviewer/review.html',_review_page(packet))
        _save_frozen(directory/'private/selection.json',selection)
        _save_frozen(directory/'private/mapping.json',mapping)
        _seal(directory,'packet.complete.json',PACKET_FILES)
        _save_frozen(directory/'meta/run_manifest.json',dict(mode='single-reviewer-cpu',model_calls=0,**config))
        _save_frozen(path,config)
        _write_state(directory,'awaiting-review',0,len(packet))
        return path
