"""CPU-only eligibility review; automatic generation products stay immutable.

Draft revisions are separate source-bound snapshots. Freezing publishes reviewed
coverage, not a training authorization. Agent suggestions never become labels
without a separately confirmed human import.
"""
import json
import re
from pathlib import Path

from pilot_eval import sampling_sft as source
from pilot_eval import sampling_sft_generation as generation
from pilot_eval.run import _write_state
from pilot_eval.sampling_sft_sources import inside,verify_parents
from pilot_eval.scoring import _NUMBER,_numeric_value
from pilot_eval.sft import safe_name
from pilot_eval.training import file_hash,run_lock
from pilot_eval.workflow import _hash


STATUSES=['pending','numeric','absent','ambiguous']


def implementation():
    return {name:file_hash(Path(__file__).with_name(name)) for name in
            ['sampling_sft_review.py','sampling_sft_review_ui.py','scoring.py']}


def selected(row):
    return row['end_state']=='finished' and row['automatic_flexible_v3']['status']=='invalid'


def prepare(generation_path,output_root,name):
    root,parent,directory=generation.load(generation_path,output_root)
    generation.verify(generation_path,root);safe_name(name)
    parents={str(inside(root,generation_path).relative_to(root)):file_hash(inside(root,generation_path))}
    for k in generation.ARMS:
        for filename in ['complete.json','attempts.jsonl','dispositions.jsonl','results.json']:
            path=directory/f'k{k}'/filename;parents[str(path.relative_to(root))]=file_hash(path)
    packet=[];problems={p['problem_id']:p for p in parent['problems']}
    for k in generation.ARMS:
        for row in generation.read_lines(directory/f'k{k}/attempts.jsonl'):
            if selected(row):packet.append(dict(response_id=row['attempt_id'],attempt_sha256=_hash(row),
                k=k,problem_id=row['problem_id'],source_id=row['source_id'],retry_index=row['retry_index'],
                question=problems[row['problem_id']]['user_message'],response=row['text'],
                reason='finished-automatic-extraction-failure'))
    folder=root/json.loads((root/parent['source_plan']).read_text())['run_path']/'eligibility'/name
    config=dict(protocol='sampling-sft-eligibility-v1',name=name,generation_plan=str(inside(root,generation_path).relative_to(root)),
        source_plan=parent['source_plan'],parents=parents,packet_sha256=_hash(packet),implementation=implementation(),
        run_path=str(folder.relative_to(root)),selection_rule='every finished attempt with flexible-v3 status invalid',
        training_allowed=False)
    path=root/'plans'/name/'sampling-sft.review.json'
    with run_lock(folder):
        source.frozen(path,config)
        source.frozen(folder/'meta/run_manifest.json',dict(config_sha256=_hash(config)))
        source.frozen(folder/'reviewer/packet.json',packet)
        from pilot_eval.sampling_sft_review_ui import page
        target=folder/'reviewer/review.html';html=page(packet,config['packet_sha256'])
        if target.exists() and target.read_text()!=html:raise ValueError('Review page changed.')
        if not target.exists():target.write_text(html)
        source.frozen(folder/'reviewer/page-seal.json',dict(sha256=file_hash(target)))
        if not (folder/'meta/status.json').exists():_write_state(folder,'review-pending',0,len(packet))
    return path


def load(path,output_root):
    root=Path(output_root).resolve();path=inside(root,path)
    config=json.loads(path.read_text());folder=inside(root,config['run_path'])
    generation.verify(root/config['generation_plan'],root)
    if (config.get('protocol')!='sampling-sft-eligibility-v1' or config['implementation']!=implementation()
            or json.loads((folder/'meta/run_manifest.json').read_text())!={'config_sha256':_hash(config)}
            or _hash(json.loads((folder/'reviewer/packet.json').read_text()))!=config['packet_sha256']
            or json.loads((folder/'reviewer/page-seal.json').read_text())!={'sha256':file_hash(folder/'reviewer/review.html')}):
        raise ValueError('Review plan/page/packet/code identity changed.')
    verify_parents(root,config['parents'])
    return root,config,folder


def validate_labels(labels,packet):
    ids={p['response_id']:p for p in packet}
    if not isinstance(labels,list):raise ValueError('Labels must be a list.')
    result=[];seen=set()
    for row in labels:
        if not isinstance(row,dict) or set(row)!={'response_id','status','answer','evidence','note'}:
            raise ValueError('Unexpected label schema; response text and tokens cannot be edited.')
        identity=row['response_id'];status=row['status']
        if identity not in ids or identity in seen or status not in STATUSES:raise ValueError('Unknown/duplicate response or invalid label status.')
        seen.add(identity)
        if not all(isinstance(row[k],str) for k in ['evidence','note']):raise ValueError('Evidence and notes must be text.')
        answer=row['answer']
        if status=='numeric':
            if not isinstance(answer,str) or re.fullmatch(_NUMBER,answer.strip()) is None:
                raise ValueError('Transcribe one supported number; correctness is computed separately.')
            try:_numeric_value(answer.strip())
            except (ValueError,ZeroDivisionError):raise ValueError('Numeric transcription is undefined.')
            if not row['evidence'].strip() or row['evidence'] not in ids[identity]['response']:
                raise ValueError('Numeric labels need an exact supporting excerpt from this response.')
            answer=answer.strip()
        elif answer not in [None,'']:raise ValueError('Only numeric labels can carry an answer.')
        else:answer=None
        result.append(dict(response_id=identity,status=status,answer=answer,evidence=row['evidence'],note=row['note']))
    if seen!=set(ids):raise ValueError('Draft must contain every packet row, including pending entries.')
    by_id={r['response_id']:r for r in result}
    return [by_id[p['response_id']] for p in packet]


def current(config,folder,origin):
    if origin not in ['human','agent']:raise ValueError('Provenance must be human or agent.')
    packet=json.loads((folder/'reviewer/packet.json').read_text())
    result=dict(packet_sha256=config['packet_sha256'],base_revision=0,reviewer='',
        labels=[dict(response_id=p['response_id'],status='pending',answer=None,evidence='',note='') for p in packet])
    previous=None
    markers=sorted((folder/'drafts'/origin).glob('*.complete.json'))
    for index,marker in enumerate(markers,1):
        if marker.name!=f'{index:06d}.complete.json':raise ValueError('Draft history is not contiguous.')
        path=marker.with_name(f'{index:06d}.json');seal=json.loads(marker.read_text())
        if seal!={'sha256':file_hash(path)}:raise ValueError('Draft revision changed.')
        value=json.loads(path.read_text())
        if (value['previous_sha256']!=previous or value['revision']!=index or value['origin']!=origin
                or value['packet_sha256']!=config['packet_sha256'] or value['config_sha256']!=_hash(config)
                or value['confirmed_human']!=(origin=='human')):raise ValueError('Label revision/provenance mismatch.')
        if not isinstance(value['reviewer'],str) or not value['reviewer'].strip():raise ValueError('Missing reviewer provenance.')
        labels=validate_labels(value['labels'],packet)
        result=dict(packet_sha256=config['packet_sha256'],base_revision=index,reviewer=value['reviewer'],labels=labels)
        previous=file_hash(path)
    return result,previous


def export(path,output_root,destination,origin='human'):
    _,config,folder=load(path,output_root)
    with run_lock(folder):value,_=current(config,folder,origin)
    destination=Path(destination).resolve()
    if destination.is_relative_to(folder):raise ValueError('Download outside the immutable review-product tree.')
    if destination.exists() and json.loads(destination.read_text())!=value:
        raise ValueError('Destination contains a different draft; preserve it and choose a new filename.')
    source.save(destination,value)
    return dict(path=str(destination),origin=origin,revision=value['base_revision'],entries=len(value['labels']))


def import_labels(path,output_root,labels_path,origin,confirm_human=False):
    _,config,folder=load(path,output_root)
    value=json.loads(Path(labels_path).read_text())
    with run_lock(folder):
        if (folder/'complete.json').exists():raise ValueError('Reviewed coverage is frozen; use a separately named review for corrections.')
        old,previous=current(config,folder,origin)
        if not isinstance(value,dict) or set(value)!={'packet_sha256','base_revision','reviewer','labels'}:
            raise ValueError('Import the source-bound exported draft, not an edited source packet.')
        if (value['packet_sha256']!=config['packet_sha256'] or type(value['base_revision']) is not int
                or value['base_revision']!=old['base_revision']):raise ValueError('Stale or wrong-source draft; import the latest saved revision first.')
        if origin=='human' and not confirm_human:raise ValueError('Human decisions require explicit --confirm-human; agent suggestions are separate.')
        if origin=='agent' and confirm_human:raise ValueError('An agent suggestion cannot be confirmed as human.')
        if not isinstance(value['reviewer'],str) or not value['reviewer'].strip():raise ValueError('Record reviewer identity.')
        packet=json.loads((folder/'reviewer/packet.json').read_text())
        labels=validate_labels(value['labels'],packet)
        revision=old['base_revision']+1
        target=folder/'drafts'/origin/f'{revision:06d}.json'
        row=dict(packet_sha256=config['packet_sha256'],config_sha256=_hash(config),revision=revision,
            previous_sha256=previous,origin=origin,confirmed_human=origin=='human',reviewer=value['reviewer'],labels=labels)
        source.frozen(target,row);source.frozen(target.with_suffix('.complete.json'),dict(sha256=file_hash(target)))
        human,_=current(config,folder,'human')
        count=sum(r['status']!='pending' for r in human['labels'])
        _write_state(folder,'review-partial' if count<len(packet) else 'reviewed',count,len(packet))
    return dict(origin=origin,saved_revision=revision,human_reviewed=count,required=len(packet),training_allowed=False)


def human_answer_correct(attempt,label,problem):
    if not selected(attempt):raise ValueError('Overrides apply only to finished extraction failures, never valid wrong or unfinished text.')
    if label['status']=='pending':raise ValueError('Required human review is unfinished.')
    if label['status'] not in ['numeric','absent','ambiguous']:raise ValueError('Unknown human decision.')
    if label['status']!='numeric':return False
    if not label['evidence'].strip() or label['evidence'] not in attempt['text']:
        raise ValueError('Human transcription needs response evidence; gold cannot supply missing text.')
    return _numeric_value(label['answer'])==_numeric_value(problem['gold'].rsplit('####',1)[-1].strip())


def eligibility_override(attempt,label,problem):
    correct=human_answer_correct(attempt,label,problem)
    ids=attempt['response_ids'];eos=problem['gold_end_token_id']
    return correct and bool(ids) and ids[-1]==eos and ids.count(eos)==1


def coverage_arm(k,problems,batches,attempts,overrides):
    by_pid={p['problem_id']:[] for p in problems};source_ids={p['source_id'] for p in problems}
    order=[i for batch in batches for i in batch]
    if (len(problems)!=512 or len(by_pid)!=512 or len(source_ids)!=512 or len(batches)!=64 or any(len(b)!=8 for b in batches)
            or len(order)!=512 or set(order)!=source_ids):
        raise ValueError('Coverage must preserve the512 gold problem identities and64 eight-slot batches.')
    for row in attempts:
        if row['problem_id'] not in by_pid:raise ValueError('Unknown attempt problem.')
        by_pid[row['problem_id']].append(row)
    dispositions=[]
    for problem in problems:
        rows=by_pid[problem['problem_id']]
        if not rows:raise ValueError('Missing problem attempts.')
        chosen=next((row for row in rows if row['eligible'] or overrides.get(row['attempt_id'],False)),None)
        dispositions.append(dict(problem_id=problem['problem_id'],source_id=problem['source_id'],
            disposition='covered' if chosen else 'uncovered',selected_attempt=chosen['attempt_id'] if chosen else None,
            eligibility_origin=('automatic' if chosen['eligible'] else 'confirmed-human') if chosen else None,
            reason=None if chosen else 'no-eligible-finished-attempt-after-review'))
    lookup={d['source_id']:d for d in dispositions}
    marked=[]
    for index,batch in enumerate(batches):
        slots=[dict(source_id=i,problem_id=lookup[i]['problem_id'],selected_attempt=lookup[i]['selected_attempt'],
            missing=lookup[i]['selected_attempt'] is None,reason=lookup[i]['reason']) for i in batch]
        marked.append(dict(step=index+1,slots=slots,nonmissing_count=sum(not s['missing'] for s in slots)))
    uncovered=[d['source_id'] for d in dispositions if d['selected_attempt'] is None]
    n=len(uncovered)
    return dict(k=k,covered_count=512-n,uncovered_count=n,uncovered_ids=uncovered,
        coverage_decision='passes-coverage-only' if n==0 else 'conditional-on-T16-and-training-approval' if n<=5 else 'blocked-more-than-five-uncovered',
        t16_required=0<n<=5,required_short_batch_sizes=[3,4,5,6,7] if 0<n<=5 else [],
        dispositions=dispositions,batches=marked,attempts=attempts,training_allowed=False)


def build_coverage(root,config,folder,human,previous):
    packet=json.loads((folder/'reviewer/packet.json').read_text())
    _,plan,directory=generation.load(root/config['generation_plan'],root)
    audit=source.verify(root/plan['source_plan'],root)['audit']
    labels={r['response_id']:r for r in human['labels']};problems={p['problem_id']:p for p in plan['problems']}
    arms={};decisions=[]
    for k in generation.ARMS:
        attempts=generation.read_lines(directory/f'k{k}/attempts.jsonl');overrides={}
        for attempt in attempts:
            if attempt['attempt_id'] in labels:
                label=labels[attempt['attempt_id']]
                eligible=eligibility_override(attempt,label,problems[attempt['problem_id']])
                overrides[attempt['attempt_id']]=eligible
                decisions.append(dict(attempt_id=attempt['attempt_id'],label=label,human_correct=human_answer_correct(attempt,label,problems[attempt['problem_id']]),
                    training_eligible=eligible,
                    origin='confirmed-human',reviewer=human['reviewer'],revision=human['base_revision'],
                    original_automatic=attempt['automatic_flexible_v3']))
        arms[str(k)]=coverage_arm(k,plan['problems'],audit['training']['batches'],attempts,overrides)
    result=dict(protocol='sampling-sft-reviewed-coverage-v1',human_review_complete=True,
        human_revision=human['base_revision'],human_revision_sha256=previous,
        reviewer=human['reviewer'],required_reviews=len(packet),decisions=decisions,arms=arms,
        **cohort_summary(arms),coverage_blocked=any(a['uncovered_count']>5 for a in arms.values()),
        training_allowed=False,scope='reviewed coverage only; T16, data gate, budget and explicit training approval remain separate')
    return result


def freeze(path,output_root):
    root,config,folder=load(path,output_root)
    with run_lock(folder):
        if (folder/'complete.json').exists():return verify(path,root)
        human,previous=current(config,folder,'human')
        packet=json.loads((folder/'reviewer/packet.json').read_text())
        if any(r['status']=='pending' for r in human['labels']):raise ValueError('Finish every required human review before freezing coverage.')
        result=build_coverage(root,config,folder,human,previous)
        source.frozen(folder/'results/coverage.json',result)
        files={'results/coverage.json':file_hash(folder/'results/coverage.json')}
        for k,arm in result['arms'].items():
            source.frozen(folder/f'results/k{k}/batches.json',arm['batches'])
            source.frozen(folder/f'results/k{k}/dispositions.json',arm['dispositions'])
            for name in ['batches.json','dispositions.json']:
                files[f'results/k{k}/{name}']=file_hash(folder/f'results/k{k}/{name}')
        _write_state(folder,'coverage-frozen',len(packet),len(packet))
        source.frozen(folder/'complete.json',dict(config_sha256=_hash(config),human_revision_sha256=previous,
            files=files))
    return result


def verify(path,output_root):
    root,config,folder=load(path,output_root)
    human,previous=current(config,folder,'human')
    seal=json.loads((folder/'complete.json').read_text())
    expected_files=['results/coverage.json']+[f'results/k{k}/{name}' for k in generation.ARMS for name in ['batches.json','dispositions.json']]
    if (seal!={'config_sha256':_hash(config),'human_revision_sha256':previous,
               'files':{name:file_hash(folder/name) for name in expected_files}}
            or any(r['status']=='pending' for r in human['labels'])):
        raise ValueError('Reviewed coverage/labels changed or required review is incomplete.')
    result=json.loads((folder/'results/coverage.json').read_text())
    if result!=build_coverage(root,config,folder,human,previous):
        raise ValueError('Coverage no longer reconciles with immutable attempts and confirmed labels.')
    for k,arm in result['arms'].items():
        if (json.loads((folder/f'results/k{k}/batches.json').read_text())!=arm['batches']
                or json.loads((folder/f'results/k{k}/dispositions.json').read_text())!=arm['dispositions']):
            raise ValueError('Marked slots or dispositions do not reconcile.')
    return result


def status(path,output_root):
    _,config,folder=load(path,output_root)
    human,_=current(config,folder,'human');agent,_=current(config,folder,'agent')
    result=dict(required=len(human['labels']),human_reviewed=sum(r['status']!='pending' for r in human['labels']),
        human_revision=human['base_revision'],agent_revision=agent['base_revision'],training_allowed=False,
        page=str(folder/'reviewer/review.html'),frozen=(folder/'complete.json').exists())
    if result['frozen']:result['coverage']=verify(path,output_root)
    return result


def console(value):
    if 'arms' not in value and 'coverage' not in value:return value
    large=['common_covered_ids','union_covered_ids','union_uncovered_ids','intersection_uncovered_ids']
    result={k:v for k,v in value.items() if k not in ['arms','coverage','decisions',*large]}
    result.update({key.removesuffix('_ids')+'_count':len(value[key]) for key in large if key in value})
    if 'arms' in value:result['arms']={k:{field:arm[field] for field in ['covered_count','uncovered_count','coverage_decision','t16_required']} for k,arm in value['arms'].items()}
    if 'coverage' in value:result['coverage']=console(value['coverage'])
    return result


def cohort_summary(arms):
    covered={k:{d['source_id'] for d in a['dispositions'] if d['disposition']=='covered'} for k,a in arms.items()}
    missing={k:set(a['uncovered_ids']) for k,a in arms.items()}
    pairs={a+'-'+b:dict(symmetric_difference_ids=sorted(missing[a]^missing[b]),
        count=len(missing[a]^missing[b]),flagged=len(missing[a]^missing[b])>5) for a,b in [('0','2'),('0','5'),('2','5')]}
    return dict(common_covered_ids=sorted(set.intersection(*covered.values())),union_covered_ids=sorted(set.union(*covered.values())),
        union_uncovered_ids=sorted(set.union(*missing.values())),intersection_uncovered_ids=sorted(set.intersection(*missing.values())),
        pairwise_uncovered_differences=pairs)
