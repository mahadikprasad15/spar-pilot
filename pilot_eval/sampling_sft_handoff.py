"""CPU-only, source-bound snapshots of Sampling SFT execution evidence.

No collection, label repair, training, cloud action or source migration is run.
Existing validators may refresh disposable status projections; those projections
are excluded from the scientific inventory. A snapshot never changes in place.
"""
import json
import re
import subprocess
from pathlib import Path

from pilot_eval import sampling_sft as source
from pilot_eval import sampling_sft_generation as generation
from pilot_eval import sampling_sft_review as review
from pilot_eval import sampling_sft_gate as gate
from pilot_eval import sampling_sft_training_data as data
from pilot_eval import sampling_sft_training as training
from pilot_eval import sampling_sft_jobs as jobs
from pilot_eval import sampling_sft_exports as exports
from pilot_eval import sampling_sft_budget as budget
from pilot_eval.sampling_sft_sources import inside,verify_parents
from pilot_eval.sft import safe_name
from pilot_eval.training import file_hash,run_lock
from pilot_eval.workflow import _hash
from pilot_eval.run import _write_state

DEFERRED=['held-out evaluation','internal activations / KL / geometry','Stage0 research',
    'temperature-1 512x8 baseline and NLL offset','k2 training','k10 sampling',
    'k5+5 polish','self-correct SFT','extra seeds']
KEYS={'source_plan','generation_plan','review_plan','gate_plan','training_data_plan',
      'training_selection','training_plans','jobs','software_evidence'}


def implementation():return {Path(__file__).name:file_hash(Path(__file__))}


def release_files(repo):
    return sorted({str(p.relative_to(repo)) for p in (repo/'pilot_eval').glob('sampling_sft*.py')}|
        {f'pilot_eval/{n}.py' for n in ['sft','sft_backend','training','workflow','protocol','run','scoring',
                                      'stage0_sources','trace_readout_assets']})


def check_release(commit):
    if re.fullmatch('[0-9a-f]{40}',commit or '') is None:raise ValueError('Use an exact40-character source release commit.')
    repo=Path(__file__).resolve().parents[1];files={}
    for name in release_files(repo):
        process=subprocess.run(['git','show',f'{commit}:{name}'],cwd=repo,capture_output=True,check=False)
        if process.returncode or process.stdout!=(repo/name).read_bytes():
            raise ValueError('Pinned sampling/training source differs or is absent: '+name)
        files[name]=file_hash(repo/name)
    return dict(release_commit=commit,source_files=files,source_sha256=_hash(files),
        scope='local Git source verification; no checkout, network, install or publication')


def normalize(value,root):
    if not isinstance(value,dict) or set(value)-KEYS or not isinstance(value.get('source_plan'),str) or not value['source_plan'].strip():
        raise ValueError('Select explicit source/stage plans; unknown selection fields are not accepted.')
    def relative(p):
        if not isinstance(p,str) or not p.strip():raise ValueError('Artifact selections must be nonempty path strings.')
        return str(inside(root,p).relative_to(root))
    out={k:None for k in ['generation_plan','review_plan','gate_plan','training_data_plan','training_selection']}
    for key in ['source_plan',*out]:
        if value.get(key) is not None:out[key]=relative(value[key])
    plans=value.get('training_plans',{})
    if not isinstance(plans,dict) or set(plans)-{'0','5'}:raise ValueError('Handoff training plans can select k0/k5 only.')
    out['training_plans']={k:relative(p) for k,p in plans.items()}
    for key in ['jobs','software_evidence']:
        rows=value.get(key,[])
        if not isinstance(rows,list):raise ValueError('Receipt/evidence selections must be lists.')
        if key=='software_evidence':out[key]=[relative(p) for p in rows]
        else:
            out[key]=[]
            for row in rows:
                if not isinstance(row,dict) or set(row)-{'plan','network_receipt','download_receipt'} or 'plan' not in row:
                    raise ValueError('Select job plan and optional explicit receipt paths.')
                out[key].append({k:relative(p) for k,p in row.items() if p is not None})
    return out


def prepare(selection_path,output_root,name,release_commit=None):
    root=Path(output_root).resolve();selected=normalize(json.loads(Path(selection_path).read_text()),root)
    _,parent,_=source.load(root/selected['source_plan'],root);safe_name(name)
    folder=root/parent['run_path']/'handoffs'/name
    release=check_release(release_commit) if release_commit else None
    config=dict(protocol='sampling-sft-handoff-v1',name=name,selection=selected,run_path=str(folder.relative_to(root)),
        implementation=implementation(),release=release,training_authorized=False,
        parents={selected['source_plan']:file_hash(root/selected['source_plan'])})
    # Paths selected later are explicit and immutable once they exist. Missing
    # scientific products are reported at build, not manufactured by prepare.
    for key in ['generation_plan','review_plan','gate_plan','training_data_plan','training_selection']:
        if selected.get(key):config['parents'][selected[key]]=file_hash(root/selected[key])
    for path in [*selected['training_plans'].values(),*selected['software_evidence']]:config['parents'][path]=file_hash(root/path)
    for row in selected['jobs']:
        for path in row.values():config['parents'][path]=file_hash(root/path)
    path=root/'plans'/name/'sampling-sft.handoff.json'
    with run_lock(folder):
        source.frozen(path,config);source.frozen(folder/'inputs/selection.json',selected)
        source.frozen(folder/'meta/run_manifest.json',dict(config_sha256=_hash(config)))
    return path


def load(path,output_root):
    root=Path(output_root).resolve();path=inside(root,path);config=json.loads(path.read_text());folder=inside(root,config['run_path'])
    if (config.get('protocol')!='sampling-sft-handoff-v1' or config['implementation']!=implementation()
            or json.loads((folder/'meta/run_manifest.json').read_text())!={'config_sha256':_hash(config)}
            or json.loads((folder/'inputs/selection.json').read_text())!=config['selection']):
        raise ValueError('Handoff plan/source/manifest changed; use its pinned implementation.')
    verify_parents(root,config['parents'])
    if config['release']:
        repo=Path(__file__).resolve().parents[1]
        for name,digest in config['release']['source_files'].items():
            if file_hash(inside(repo,name))!=digest:raise ValueError('Use the pinned sampling/training source release.')
    return root,config,folder


def completion(stages):
    return all(stages.get(k) is True for k in ['source_audited','real_qualified','generation_complete',
        'generation_real','review_complete','gate_complete','gate_real','t15_t16_ready',
        'training_complete','training_real','lifecycle_complete','budget_settled'])


def frozen_files(folder,seal_name='complete.json'):
    seal=json.loads((folder/seal_name).read_text())
    for name,digest in seal.get('files',{}).items():
        if file_hash(inside(folder,name))!=digest:raise ValueError('Selected completed evidence changed: '+name)
    return seal


def receipt_status(manifest,row,root,role):
    value=json.loads(inside(root,row).read_text());payload={k:v for k,v in value.items() if k!='sha256'}
    if (value.get('sha256')!=_hash(payload) or value.get('role')!=role
            or value.get('manifest_sha256')!=manifest['sha256'] or value.get('job_sha256')!=manifest['job_sha256']
            or value.get('simulation')!=manifest['simulation'] or value.get('verified_files')!=len(manifest['files'])
            or not value.get('host') or not value.get('copy_root')):
        raise ValueError('Export receipt is stale, corrupt or bound to another job.')
    if role=='downloaded' and not value['simulation'] and value['host']==manifest['worker_host']:
        raise ValueError('Real downloaded-copy evidence must come from an external client.')
    return value


def job_evidence(path,root,overrides):
    _,job,folder=jobs._load(path,root)
    state=json.loads((folder/'state.json').read_text());payload={k:v for k,v in state.items() if k!='sha256'}
    expected=''.join(json.dumps(e,sort_keys=True)+'\n' for e in state['events'])
    if (state['sha256']!=_hash(payload) or state['job_sha256']!=_hash(job)
            or (folder/'events.jsonl').read_text()!=expected):raise ValueError('Job control state/log changed.')
    launched=(folder/'launch.json').exists();work=jobs._work(folder,root,job)
    real=job['request']['mode']=='runpod';receipts=[];confirmed=False
    if work:
        for role,key,candidates in [('network','network_receipt',['receipts/network.json','network-receipt.json']),
                                    ('downloaded','download_receipt',['receipts/downloaded.json'])]:
            selected=overrides.get(key)
            if selected is None:
                selected=next((str((folder/n).relative_to(root)) for n in candidates if (folder/n).exists()),None)
            if selected:receipts.append(receipt_status(work,selected,root,role))
        if len(receipts)==2 and (receipts[0]['host'],receipts[0]['copy_root'])==(receipts[1]['host'],receipts[1]['copy_root']):
            raise ValueError('Network/downloaded receipts name the same copy.')
    proof=folder/'shutdown-receipt.json'
    if proof.exists():
        stop=json.loads(proof.read_text());body={k:v for k,v in stop.items() if k!='sha256'}
        launch=json.loads((folder/'launch.json').read_text())
        if (stop.get('sha256')!=_hash(body) or stop.get('job_sha256')!=_hash(job)
                or stop.get('request_sha256')!=job['request_sha256'] or stop.get('pod_id')!=launch['pod_id']
                or stop.get('provider_status')!='EXITED' or stop.get('worker_host')!=launch['host']
                or stop.get('controller_host')==stop.get('worker_host')
                or not any(e['kind']=='stop-confirmed' for e in state['events'])):
            raise ValueError('External stopped-Pod evidence is inconsistent.')
        confirmed=True
    return dict(plan=str(inside(root,path).relative_to(root)),mode=job['request']['mode'],stage=job['request']['stage'],
        launched=launched,work_complete=work is not None,workload_success=work['workload_success'] if work else None,
        export_receipts=len(receipts),stop_requested=any(e['kind']=='stop-requested' for e in state['events']),
        stop_confirmed=confirmed,closed=(not real or not launched or (len(receipts)==2 and confirmed)),
        scope='retained client hash receipts and external provider observation; no live cloud request')


def collect(root,config):
    selected=config['selection'];paths=set(config['parents']);absent=set();folders=set();expected_jobs=set()
    stages={k:False for k in ['source_audited','real_qualified','generation_complete','generation_real','review_complete',
        'gate_complete','gate_real','t15_t16_ready','training_complete','training_real','lifecycle_complete','budget_settled']}
    details={};pending=[]

    def add_parents(mapping):
        for name,digest in mapping.items():
            (absent if digest is None else paths).add(name)

    def add_plan(path):
        path=inside(root,path);paths.add(str(path.relative_to(root)));cfg=json.loads(path.read_text())
        if cfg.get('source_plan') and cfg['source_plan']!=selected['source_plan']:
            raise ValueError('Selected stages belong to different source plans.')
        if 'run_path' in cfg:folders.add(inside(root,cfg['run_path']))
        add_parents(cfg.get('parents',{}));return cfg

    _,base,base_folder=source.load(root/selected['source_plan'],root)
    add_parents(base['parents'])
    if (base_folder/'writer.json').exists():raise ValueError('Source audit writer is active or unrecovered.')
    # Only source-audit-owned files, not the entire experiment subtree.
    for name in ['meta/run_manifest.json','meta/software.json','meta/status.json','checkpoints/progress.json','logs/run.log','inputs/selection.json','complete.json','results/source-audit.json']:
        if (base_folder/name).exists():paths.add(str((base_folder/name).relative_to(root)))
    audit=source.status(root/selected['source_plan'],root)['audit']
    if audit:
        add_parents(audit['parents']);stages['source_audited']=audit['source_ready'];details['source']=audit['blockers']
    else:pending.append('source audit')
    print('Handoff: source audit evidence checked.',flush=True)
    gen=None;coverage=None;gate_cfg=None
    if selected['generation_plan']:
        gen=add_plan(root/selected['generation_plan']);_,gen,gf=generation.load(root/selected['generation_plan'],root)
        rows=[generation.verify_arm(root,gen,k,gf/f'k{k}') for k in generation.ARMS]
        if (gf/'complete.json').exists():
            seal=frozen_files(gf)
            if (seal['config_sha256']!=_hash(gen) or seal['arms']!={str(k):file_hash(gf/f'k{k}/complete.json') for k in generation.ARMS}
                    or seal['result_sha256']!=file_hash(gf/'results/results.json')):raise ValueError('Generation overall seal changed.')
            stages['generation_complete']=all(r['state']=='completed' for r in rows)
            stages['generation_real']=stages['generation_complete'] and all(r['execution_kind']=='real-cuda' for r in rows)
        details['generation']=rows
        parent=add_plan(root/gen['pilot_plan']);add_plan(root/parent['instrument_plan'])
        if parent.get('selection'):paths.add(parent['selection'])
        if stages['generation_real']:
            # These proofs must already exist: qualification may verify them,
            # but a handoff must never manufacture missing qualification receipts.
            pf=inside(root,parent['run_path']);inst=json.loads((root/parent['instrument_plan']).read_text());idir=root/inst['run_path']
            needed=[pf/'replay.json',pf/'inputs/runtime-review.json',idir/'runtime-selection.json']
            if not parent.get('selection') or any(not p.exists() for p in needed):raise ValueError('Real generation lacks existing qualification proofs.')
            saved=json.loads((root/parent['selection']).read_text())
            for p in saved['review']['candidates']+[saved['review']['score_config'],saved['review']['oracle_config']]:
                cfg=add_plan(root/p)
            scfg=json.loads((root/saved['review']['score_config']).read_text())
            if not (root/scfg['run_path']/'score-receipt.json').exists():raise ValueError('Missing existing T3 score receipt.')
            generation.qualification(root,gen);stages['real_qualified']=True
    print('Handoff: generation/qualification evidence checked or not selected.',flush=True)
    if selected['review_plan']:
        if not stages['generation_complete']:raise ValueError('Selected review requires a sealed complete generation product.')
        rc=add_plan(root/selected['review_plan'])
        if rc['generation_plan']!=selected['generation_plan']:raise ValueError('Review/generation selection mismatch.')
        rf=root/rc['run_path']
        if (rf/'complete.json').exists():coverage=review.verify(root/selected['review_plan'],root);stages['review_complete']=True
        else:details['review']=review.status(root/selected['review_plan'],root)
    if selected['gate_plan']:
        if not stages['review_complete']:raise ValueError('Selected gate requires frozen reviewed coverage.')
        gate_cfg=add_plan(root/selected['gate_plan'])
        if gate_cfg['review_plan']!=selected['review_plan']:raise ValueError('Gate/review selection mismatch.')
        gfolder=root/gate_cfg['run_path']
        if (gfolder/'complete.json').exists():
            report=gate.verify(root/selected['gate_plan'],root);stages['gate_complete']=True;stages['gate_real']=report['execution_kind']=='real-cuda'
            details['gate']={k:report[k] for k in ['own_cohort','common_cohort','common_problem_count','coverage','baseline','relative_sharpening']}
    print('Handoff: review and gate evidence checked or pending.',flush=True)
    chosen=[0,5]
    if selected['training_selection']:
        if not stages['gate_complete']:raise ValueError('Training selection needs its completed gate report.')
        receipt,_=training.selection(root,root/selected['training_selection'],dict(review_plan=selected['review_plan']))
        chosen=receipt['arms'];details['selected_training_arms']=chosen
    if set(map(int,selected['training_plans']))-set(chosen):raise ValueError('A supplied training arm was not selected after the gate.')
    if selected['training_data_plan']:
        dc=add_plan(root/selected['training_data_plan'])
        if dc['review_plan']!=selected['review_plan']:raise ValueError('Training-data/review selection mismatch.')
        df=root/dc['run_path']
        if (df/'export.complete.json').exists():
            data.verify(root/selected['training_data_plan'],root)
            ready=data.status(root/selected['training_data_plan'],root)['readiness']
            stages['t15_t16_ready']=bool(ready and ready['t15_passed'] and ready['t16_passed'] and ready['training_ready'])
            details['readiness']=ready
    trained={}
    for k,path in selected['training_plans'].items():
        tc=add_plan(root/path)
        if tc['k']!=int(k) or tc['data_plan']!=selected['training_data_plan'] or tc['selection']!=selected['training_selection']:
            raise ValueError('Training arm/data/selection mismatch.')
        folder=root/tc['run_path']
        trained[k]=training.verify(root/path,root) if (folder/'complete.json').exists() else training.status(root/path,root)
    stages['training_complete']=bool(chosen) and all(str(k) in trained and trained[str(k)].get('successful_steps')==64 for k in chosen)
    stages['training_real']=stages['training_complete'] and all(trained[str(k)]['execution_kind']=='real-cuda' for k in chosen)
    details['training']=trained
    print('Handoff: selected training products checked.',flush=True)
    software=[];repo=Path(__file__).resolve().parents[1]
    for p in selected['software_evidence']:
        value=json.loads((root/p).read_text())
        if value.get('state')!='software-validated' or not value.get('source_files'):raise ValueError('Expected recorded software-validation evidence.')
        for name,h in value['source_files'].items():
            if file_hash(inside(repo,name))!=h:raise ValueError('Software receipt describes a different source version.')
        software.append(dict(path=p,ticket=value.get('ticket'),tests=value.get('tests'),scope='recorded CPU software checks; not real GPU qualification'))
    stages['software_tested']=bool(software);details['software']=software
    job_paths={str(p.relative_to(root)) for p in (base_folder/'jobs').glob('*/job.plan.json')}
    overrides={row['plan']:row for row in selected['jobs']};job_paths.update(overrides)
    for folder in list(folders):
        for p in folder.rglob('*.jsonl'):
            if p.name not in ['events.jsonl']:continue
            for row in generation.read_lines(p):
                if row.get('job'):expected_jobs.add(row['job'])
        started=folder/'started.json'
        if started.exists() and json.loads(started.read_text()).get('job'):expected_jobs.add(json.loads(started.read_text())['job'])
    if expected_jobs-job_paths:raise ValueError('Missing lifecycle plans referenced by scientific work.')
    evidence=[]
    for p in sorted(job_paths):
        _,job,jf=jobs._load(root/p,root)
        if job['source_plan']!=selected['source_plan']:raise ValueError('Job belongs to another experiment.')
        paths.add(p);folders.add(jf);evidence.append(job_evidence(root/p,root,overrides.get(p,{})))
        for key,value in overrides.get(p,{}).items():
            if key!='plan':paths.add(value)
    details['jobs']=evidence
    real_jobs=[v for v in evidence if v['mode']=='runpod' and v['launched']]
    stages['lifecycle_complete']=bool(real_jobs) and all(v['closed'] for v in real_jobs)
    ledger_path=budget.folder(root)/'budget.json'
    if ledger_path.exists():
        ledger=budget.status(root);details['budget']=ledger
        paths.update(str(p.relative_to(root)) for p in budget.folder(root).iterdir() if p.is_file() and p.name!='.lock')
        stages['budget_settled']=not ledger['cap_exceeded'] and not ledger['unreconciled_gpu_jobs'] and all(
            j['state']=='settled' for j in ledger['jobs'].values() if j['kind']=='gpu' and j['launch_recorded'])
    else:details['budget']=None
    print('Handoff: budget, export and stop evidence checked; indexing files.',flush=True)
    for folder in folders:
        if list(folder.rglob('writer.json')):raise ValueError('Selected upstream writer is active or unrecovered; do not snapshot it.')
        for p in folder.rglob('*'):
            if (not p.is_file() or p.name in ['.lock','status.json','progress.json','run.log']):continue
            paths.add(str(inside(root,p).relative_to(root)))
    for key,value in stages.items():
        if key!='software_tested' and not value:pending.append(key)
    if any(inside(root,p).exists() for p in absent):raise ValueError('A previously absent source observation appeared; verify the new source state.')
    return stages,details,sorted(paths),sorted(set(pending)),sorted(absent)


def render(report):
    lines=['# Sampling SFT handoff','',f"Status: **{report['state']}**. This is a saved evidence snapshot, not a live monitor.",'',
        'Data-recipe comparison; no causal claim about on-policyness. No typical-NLL baseline/offset is measured.','',
        '| Evidence | Status |','|---|---|']
    lines.extend(f'| {k} | {"verified" if v else "pending / not established"} |' for k,v in report['stages'].items())
    lines+=['','## Selected training outputs','']
    for k,row in report['details']['training'].items():
        lines.append(f'- k{k}: {row.get("execution_kind","pending")}; steps={row.get("successful_steps","pending")}; exposures={row.get("example_exposures","pending")}.')
    lines+=['','## Missing evidence','']+[f'- {p}' for p in report['pending']]
    lines+=['','## Deferred; not completed','']+[f'- {p}' for p in DEFERRED]
    lines+=['','## Operation','',
        'Verify this snapshot on CPU using its saved handoff plan. If upstream evidence progresses, create a new named snapshot; do not overwrite this one.',
        'Missing selected checkpoints or changed source bytes are errors, not successful partial completion.',
        'Copy receipts are retained client hash confirmations, not authentication or a fresh cloud query. A self-stop request alone is not shutdown confirmation.',
        'No training, paid launch, Pod termination or volume deletion is authorized by this report. Deletion still needs explicit approval.','']
    return '\n'.join(lines)


def build(path,output_root):
    root,config,folder=load(path,output_root)
    if (folder/'complete.json').exists():return verify(path,root)
    with run_lock(folder):
        print('Handoff: verifying selected evidence; no model or cloud work.',flush=True)
        stages,details,paths,pending,absent=collect(root,config)
        paths.extend([str(inside(root,path).relative_to(root)),str((folder/'meta/run_manifest.json').relative_to(root)),str((folder/'inputs/selection.json').relative_to(root))])
        inventory=exports.inventory(root,paths)
        report=dict(protocol='sampling-sft-handoff-report-v1',state='completed' if completion(stages) else 'partial',
            stages=stages,details=details,pending=pending,deferred=DEFERRED,
            config_sha256=_hash(config),inventory_files=len(inventory),inventory_absent=len(absent),inventory_bytes=sum(v['bytes'] for v in inventory.values()),
            training_authorized=False,scope='verified saved artifacts; CPU fixtures never establish real experiment completion')
        source.frozen(folder/'results/inventory.json',dict(files=inventory,absent_paths=absent,config_sha256=_hash(config)))
        source.frozen(folder/'results/report.json',report)
        text=render(report);target=folder/'report.md'
        if target.exists() and target.read_text()!=text:raise ValueError('Existing handoff text differs.')
        target.write_text(text)
        source.frozen(folder/'complete.json',dict(config_sha256=_hash(config),files={n:file_hash(folder/n) for n in
            ['results/inventory.json','results/report.json','report.md']}))
        _write_state(folder,'snapshot-verified-'+report['state'],1,1)
    return verify(path,root)


def verify(path,output_root):
    root,config,folder=load(path,output_root);seal=frozen_files(folder)
    if seal['config_sha256']!=_hash(config) or set(seal['files'])!={'results/inventory.json','results/report.json','report.md'}:
        raise ValueError('Handoff product seal changed.')
    inventory=json.loads((folder/'results/inventory.json').read_text());report=json.loads((folder/'results/report.json').read_text())
    if any(inside(root,p).exists() for p in inventory['absent_paths']):raise ValueError('Handoff absent-parent observations changed.')
    if (inventory['config_sha256']!=_hash(config) or report['config_sha256']!=_hash(config)
            or exports.inventory(root,inventory['files'])!=inventory['files']):raise ValueError('Handoff parent inventory changed; preserve snapshot and inspect sources.')
    if (report['state']!=('completed' if completion(report['stages']) else 'partial') or report['deferred']!=DEFERRED
            or report['inventory_absent']!=len(inventory['absent_paths']) or report['inventory_files']!=len(inventory['files']) or report['inventory_bytes']!=sum(v['bytes'] for v in inventory['files'].values())
            or (folder/'report.md').read_text()!=render(report)):
        raise ValueError('Handoff report/raw/status reconciliation failed.')
    return report


def status(path,output_root):
    _,_,folder=load(path,output_root)
    return verify(path,output_root) if (folder/'complete.json').exists() else dict(state='prepared',training_authorized=False)


def console(value):
    return {k:value[k] for k in ['state','stages','pending','inventory_files','inventory_bytes','training_authorized'] if k in value}
