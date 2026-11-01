"""Resume-first approved L4 scoring, three-arm training, verified HF export and optional own-Pod stop.

Run:    python -m pilot_eval.sampling_sft_overnight --root ... (see main)
Status: python -m pilot_eval.sampling_sft_overnight status --root ... --name ...  (read-only)
"""
import argparse
import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pilot_eval import sampling_sft as source, sampling_sft_gate as gate
from pilot_eval import sampling_sft_jobs as jobs, sampling_sft_pipeline as pipeline
from pilot_eval import sampling_sft_review as review, sampling_sft_training as training
from pilot_eval.sampling_sft_provider import RunPodProvider
from pilot_eval.sampling_sft_sources import inside
from pilot_eval.training import file_hash,run_lock
from pilot_eval.workflow import _hash

LIMITS={'0':34,'2':35,'5':40}
DIFFERENCES={'0-2':49,'0-5':54,'2-5':49}
FLAGS=sorted([f'k{k}:omissions-require-T16' for k in LIMITS]+['uncovered-set-difference:'+p for p in DIFFERENCES])
SKIP_NAMES={'.lock','writer.json'}
APPROVAL_NOTES=('User pre-authorized all three arms (k0/k2/k5) with automatic-verdict omissions 34/35/40 and '
    'pairwise uncovered-set differences 49/54/49, and waived the cost caps, before this report existed. '
    'The user has not reviewed these gate metrics; eligibility verdicts are automatic, not human-confirmed. '
    'review_80_percent records that explicit cost waiver, not a line-item budget review.')


def now():return datetime.now(timezone.utc).isoformat()


def validate_coverage(report):
    c=report['coverage'];ids=c['uncovered_ids'];pairs=c['pairwise_flags']
    if (c['blocked'] or set(ids)!=set(LIMITS) or any(len(ids[k])!=n or len(set(ids[k]))!=n for k,n in LIMITS.items())
            or set(pairs)!=set(DIFFERENCES) or any(not pairs[k]['flagged'] or pairs[k]['count']!=n or
                len(set(ids[k.split('-')[0]]) ^ set(ids[k.split('-')[1]]))!=n for k,n in DIFFERENCES.items())
            or c['t16_required']!={k:True for k in LIMITS}):
        raise ValueError('Coverage differs from the explicitly accepted three-arm cohort.')


def record(folder,stage,**extra):
    """Human-readable current stage plus an append-only fsynced history."""
    value=dict(state=stage,updated_at=now(),host=platform.node(),pid=os.getpid(),**extra)
    value.setdefault('external_copy_verified',False)
    source.save(folder/'status.json',value)
    with (folder/'events.jsonl').open('a') as stream:
        stream.write(json.dumps(value,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
    print('Overnight stage:',stage,flush=True)


def orchestrate(folder,ops):
    """Completed stages have durable pointers; failed stages are never marked complete."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    record(folder,'preflight');ops.preflight()
    score=folder/'score.complete.json'
    if not score.exists():
        record(folder,'gate-scoring')
        ops.score();source.frozen(score,{'state':'verified'})
    selection_file=folder/'selection.complete.json'
    if not selection_file.exists():
        record(folder,'gate-report-and-selection')
        selection=ops.report_selection();source.frozen(selection_file,{'selection':selection})
    selection=json.loads(selection_file.read_text())['selection']
    trained=folder/'training.complete.json'
    if not trained.exists():
        record(folder,'training');source.frozen(trained,{'inventory':ops.training(selection)})
    inventory=json.loads(trained.read_text())['inventory']
    exported=folder/'export.complete.json'
    if not exported.exists():
        record(folder,'hf-export');source.frozen(exported,ops.export(inventory))
    receipt=json.loads(exported.read_text())
    source.frozen(folder/'complete.json',{'state':'exported-verified','inventory':inventory,'external_copy_verified':True,
        'hf_repo':receipt['repo'],'hf_revision':receipt['revision'],'manifest_sha256':receipt['manifest_sha256']})
    record(folder,'exported-verified',external_copy_verified=True,hf_revision=receipt['revision'])
    if ops.stop_requested:
        record(folder,'pod-stop-requested',external_copy_verified=True)
        ops.stop()
        record(folder,'pod-stop-command-returned',external_copy_verified=True)
    return inventory


def check_interpreter(executable):
    """A child must resolve to this same environment and package set, not a symlink target."""
    probe=('import sys,json,importlib.metadata as m;print(json.dumps(dict(prefix=sys.prefix,'
           'versions={n:m.version(n) for n in ["torch","transformers","peft","trl"]})))')
    child=subprocess.run([executable,'-c',probe],capture_output=True,text=True,timeout=120)
    if child.returncode!=0:raise ValueError('Job interpreter cannot import the pinned packages: '+executable)
    observed=json.loads(child.stdout.strip().splitlines()[-1])
    import importlib.metadata as metadata
    expected=dict(prefix=sys.prefix,versions={n:metadata.version(n) for n in observed['versions']})
    if observed!=expected:raise ValueError(f'Job interpreter environment differs from this wrapper: {executable}')


def tree(root,folder,exclude=()):
    names=[]
    for path in sorted(Path(folder).rglob('*')):
        if path.is_symlink():raise ValueError('Export refuses symlink: '+str(path))
        if (not path.is_file() or path.name in SKIP_NAMES or path.name.endswith('.tmp')
                or any(path.is_relative_to(e) for e in exclude)):continue
        names.append(str(path.relative_to(root)))
    return names


def verify_download(record_value,manifest,manifest_path,root):
    """Fresh HF download into an empty cache; every archived file must match the sealed manifest."""
    from huggingface_hub import hf_hub_download
    from pilot_eval.sampling_sft_hf import require_space
    with tempfile.TemporaryDirectory(prefix='hf-verify-') as temp:
        require_space(temp,2*record_value['archive_bytes'])
        archive=Path(hf_hub_download(record_value['repo'],record_value['archive'],repo_type='dataset',
            revision=record_value['revision'],cache_dir=temp,force_download=True))
        if file_hash(archive)!=record_value['archive_sha256']:raise ValueError('Fresh HF download archive hash mismatch.')
        expected={n:v['sha256'] for n,v in manifest['files'].items()}
        expected[str(manifest_path.relative_to(root))]=file_hash(manifest_path)
        seen={}
        with tarfile.open(archive,'r:gz') as stream:
            for member in stream:
                if not member.isfile() or member.name in seen:raise ValueError('Unexpected/duplicate HF archive member.')
                digest=hashlib.sha256();handle=stream.extractfile(member)
                for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
                seen[member.name]=digest.hexdigest()
        if seen!=expected:raise ValueError('Fresh HF download differs from the sealed manifest.')
    return dict(verified_files=len(seen),archive_sha256=record_value['archive_sha256'],verified_at=now(),
        scope='fresh download from HF into an empty cache on the worker Pod; per-file SHA256 matches; not an external host')


def probe_hf_write_read(api,repo,state):
    """Persist a tiny unique private-repo object and verify a fresh revision-pinned read."""
    from huggingface_hub import hf_hub_download
    token=uuid.uuid4().hex;payload=('sampling-sft-preflight:'+token+'\n').encode()
    target=Path(state)/('hf-preflight-'+token+'.json')
    name='preflight/sampling-sft-'+token+'.txt'
    commit=api.upload_file(repo_id=repo,repo_type='dataset',path_in_repo=name,path_or_fileobj=payload,
        commit_message='Verify Sampling SFT export permissions')
    revision=commit.oid
    if not isinstance(revision,str) or len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('HF preflight upload returned no immutable commit.')
    with tempfile.TemporaryDirectory(prefix='hf-preflight-') as temp:
        path=hf_hub_download(repo,name,repo_type='dataset',revision=revision,cache_dir=temp,force_download=True)
        if Path(path).read_bytes()!=payload:raise ValueError('HF preflight fresh download differs from upload.')
    result=dict(repo=repo,path=name,revision=revision,sha256=hashlib.sha256(payload).hexdigest(),verified_at=now())
    source.frozen(target,result)
    return result


class Operations:
    def __init__(self,config):
        self.cfg=config;self.root=Path(config['root']).resolve();self.gate=inside(self.root,config['gate']);self.job=inside(self.root,config['score_job']);self.data=inside(self.root,config['data'])
        self.stop_requested=bool(config.get('stop_pod'))

    def preflight(self):
        for name,digest in self.cfg['input_sha256'].items():
            if file_hash(inside(self.root,name))!=digest:raise ValueError('Overnight input changed: '+name)
        for name,digest in self.cfg['code_sha256'].items():
            if file_hash(Path(__file__).with_name(name))!=digest:raise ValueError('Overnight implementation changed: '+name)
        subprocess.run(['runpodctl','--version'],check=True,capture_output=True,timeout=30)
        provider=RunPodProvider();pod=provider.own_pod_id();observed=provider.inspect(pod)
        if observed.get('id')!=pod or observed.get('storageVolumeId',observed.get('networkVolumeId'))!=self.cfg['volume_id']:
            raise ValueError('Pod or persistent volume identity mismatch.')
        if observed.get('storageMountPath') and not self.root.is_relative_to(Path(observed['storageMountPath'])):
            raise ValueError('Artifact root is not on the verified persistent volume.')
        from pilot_eval import sampling_sft_budget as budget
        if budget.number(observed.get('costPerHr'),True)>budget.number(self.cfg['rate'],True):raise ValueError('Live rate exceeds approved estimate.')
        _,g,_=gate.load(self.gate,self.root);_,j,_=jobs._load(self.job,self.root)
        pipeline.validate_gate_job(self.root,self.gate,self.job,j)
        if j['request']['network_volume_id']!=self.cfg['volume_id']:raise ValueError('Gate job volume mismatch.')
        # Both the frozen score command and future training commands must run this pinned environment.
        for executable in {j['request']['command'][0],sys.executable}:check_interpreter(executable)
        if g['protocol']!='sampling-sft-gate-l4-v1':raise ValueError('Expected independently bound L4 oracle.')
        from pilot_eval.sampling_sft_l4_oracle import capture
        if capture()!=g['oracle_runtime']:raise ValueError('Current L4 runtime does not match the gate.')
        rp=json.loads((self.root/g['review_plan']).read_text())
        if rp.get('coverage_policy')!=review.AUTO_POLICY or file_hash(self.root/rp['generation_plan'])!=review.AUTO_POLICY['generation_sha256']:
            raise ValueError('Not the specifically accepted generation and automatic coverage policy.')
        pipeline.training_ready(self.data,self.root)
        data=json.loads(self.data.read_text())
        if data['review_plan']!=g['review_plan']:raise ValueError('Gate and training use different reviews.')
        cov=review.verify(self.root/g['review_plan'],self.root)
        validate_coverage({'coverage':{'blocked':cov['coverage_blocked'],
            'uncovered_ids':{k:cov['arms'][k]['uncovered_ids'] for k in LIMITS},
            'pairwise_flags':cov['pairwise_uncovered_differences'],
            't16_required':{k:cov['arms'][k]['t16_required'] for k in LIMITS}}})
        export=json.loads((inside(self.root,data['run_path'])/'results/export.json').read_text())
        if any(export['arms'][k]['missing']!=n or export['arms'][k]['examples']!=512-n for k,n in LIMITS.items()):
            raise ValueError('Training exports do not match accepted coverage counts.')
        # The training backend loads by Hub ID: verify the pinned cache is present before any scoring spend.
        from huggingface_hub import try_to_load_from_cache
        from pilot_eval import sampling_sft as sampling
        audit=sampling.verify(self.root/g['source_plan'],self.root)['audit'];identity=audit['training']['identity']
        model_root=inside(self.root,g['settings']['local_model'])
        for name in ('config.json','generation_config.json','merges.txt','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.json'):
            cached=try_to_load_from_cache(identity['model'],name,revision=identity['model_revision'])
            if not isinstance(cached,str) or not Path(cached).is_file() or file_hash(cached)!=file_hash(model_root/name):
                raise ValueError('Pinned HF training cache missing or mismatched: '+name)
        # Fail before any GPU work if the final export destination is unusable.
        from pilot_eval import sampling_sft_hf as hf
        client=hf.Hub();hf.private(client,self.cfg['hf_repo']);client.api.whoami()
        # Known current data size is a lower bound only, not a forecast of final score/checkpoint size.
        data_folder=inside(self.root,data['run_path'])
        known_bytes=sum(p.stat().st_size for p in data_folder.rglob('*') if p.is_file() and not p.is_symlink())
        hf.require_space(self.root,known_bytes)
        hf.require_space(tempfile.gettempdir(),2*known_bytes)
        probe_hf_write_read(client.api,self.cfg['hf_repo'],inside(self.root,self.cfg['state_dir']))

    def score(self):
        print('Starting or verifying independent L4 gate scoring; child log: '+str(self.job.parent/'logs/child.console.log'),flush=True)
        pipeline.gate_score(self.gate,self.root,self.job,self.cfg['reviewer'])

    def report_selection(self):
        print('Verifying report and accepted coverage.',flush=True)
        pipeline.gate_report(self.gate,self.root)
        report=gate.verify(self.gate,self.root)
        if report['execution_kind']!='real-cuda':raise ValueError('Real CUDA gate evidence required.')
        validate_coverage(report)
        c=self.cfg;forecast={'arms':[0,2,5],'per_arm':{k:{'rate_usd_per_hour':c['rate'],'estimated_hours':c['hours'][k],
            'setup_hours':c['setup_hours']} for k in LIMITS},'notes':'User approved all three arms and stated cost waiver for this run.'}
        target=inside(self.root,c['state_dir'])/'forecast.json';source.frozen(target,forecast)
        request=gate.request(self.gate,self.root,target)
        if request['required_acknowledgments']!=FLAGS or request['coverage_blocked']:raise ValueError('Unexpected gate flags; stop for review.')
        approval={'request_path':request['request_path'],'request_sha256':request['request_sha256'],
            'report_sha256':request['report_sha256'],'budget_sha256':request['budget_snapshot']['sha256'],
            'reviewer':c['reviewer'],'notes':APPROVAL_NOTES,
            'gate_approved':True,'acknowledged_coverage_flags':FLAGS,'review_80_percent':True,'waive_budget_limits':True}
        approval_path=inside(self.root,c['state_dir'])/'gate-approval.json';source.frozen(approval_path,approval)
        return gate.approve(self.gate,self.root,approval_path)['selection_path']

    def training(self,selection):
        print('Verifying selection and running k0, k2, k5 training sequentially.',flush=True)
        c=self.cfg;training.selection(self.root,self.root/selection,json.loads(self.data.read_text()))
        entries=self.training_entries(selection)
        for e in json.loads(entries.read_text()):
            print(f'k{e["arm"]} child log: '+str(self.root/Path(e['job']).parent/'logs/child.console.log'),flush=True)
        return pipeline.training(self.data,self.root,entries)['inventory']

    def training_entries(self,selection):
        c=self.cfg
        return pipeline.prepare_training_jobs(self.data,self.root/selection,self.root,c['name']+'-train',
            c['volume_id'],c['rate'],c['reviewer'],c['hours'],c['setup_hours'])

    def export(self,inventory):
        """One sealed bundle: gate tree, three training trees, jobs, ledger and this wrapper's decisions."""
        from pilot_eval import sampling_sft_exports as exports, sampling_sft_hf as hf
        root=self.root;state=inside(root,self.cfg['state_dir']);folder=state/'export'
        manifest_path=folder/'export-manifest.json';receipt=folder/'hf-upload.json'
        if not receipt.exists():
            _,_,gate_folder=gate.load(self.gate,root)
            data=json.loads(self.data.read_text())
            entries=json.loads((inside(root,data['run_path'])/'pipeline'/(self.cfg['name']+'-train')/'training-jobs.json').read_text())
            names=set(tree(root,gate_folder)+tree(root,self.job.parent)+tree(root,root/'runs/sampling-sft/budget'))
            names|=set(tree(root,state,exclude=(folder,state/'status.json',state/'events.jsonl')))
            names|={str(inside(root,inventory).relative_to(root)),str(self.data.relative_to(root))}
            for e in entries:
                _,plan,run_folder=training.load(root/e['config'],root)
                names|=set(tree(root,run_folder)+tree(root,(root/e['job']).parent))|{e['config']}
            manifest=dict(schema_version=1,job_sha256=_hash(self.cfg),files=exports.inventory(root,sorted(names)),
                simulation=False,worker_host=platform.node(),pod_id=os.environ.get('RUNPOD_POD_ID'),workload_success=True,
                missing_expected_outputs=[],scope='overnight gate+training export; sealed after verified training')
            manifest['sha256']=_hash(manifest);source.save(manifest_path,manifest)
            print(f'Uploading {len(manifest["files"])} sealed files to private HF repo {self.cfg["hf_repo"]}.',flush=True)
        manifest=exports.verify_manifest(manifest_path,root)
        uploaded=hf.upload(manifest_path,root,repo=self.cfg['hf_repo'])
        print('HF upload readback passed; verifying a fresh download file by file.',flush=True)
        check=verify_download(uploaded,manifest,manifest_path,root)
        source.save(folder/'hf-download-verification.json',check)
        return dict(repo=uploaded['repo'],revision=uploaded['revision'],archive=uploaded['archive'],
            archive_sha256=uploaded['archive_sha256'],manifest_sha256=manifest['sha256'],verification=check)

    def stop(self):
        provider=RunPodProvider();provider.stop(provider.own_pod_id())


def status(root,name):
    """Read-only summary for a human after reconnecting; never takes locks or writes."""
    root=Path(root).resolve();state=root/'runs/sampling-sft/overnight'/name
    def read(path):
        try:return json.loads(Path(path).read_text())
        except (OSError,ValueError):return None
    def tail(path,lines=4):
        try:return Path(path).read_text(errors='replace').splitlines()[-lines:]
        except OSError:return []
    out=dict(overnight=read(state/'status.json'),complete=read(state/'complete.json'),jobs={})
    config=read(state/'config.json') or {}
    job_dirs=[]
    if config.get('score_job'):job_dirs.append(('gate-score',(root/config['score_job']).parent))
    data=read(root/config['data']) if config.get('data') else None
    if data:
        entries=read(root/data['run_path']/'pipeline'/(name+'-train')/'training-jobs.json') or []
        job_dirs+=[(f'k{e["arm"]}',(root/e['job']).parent) for e in entries]
        for e in entries:
            plan=read(root/e['config']) or {}
            if plan.get('run_path'):out['jobs'].setdefault(f'k{e["arm"]}',{})['arm_status']=read(root/plan['run_path']/'meta/status.json')
    for label,directory in job_dirs:
        events=(read(directory/'state.json') or {}).get('events',[])
        execution=read(directory/'execution.json')
        out['jobs'].setdefault(label,{}).update(phase=events[-1]['kind'] if events else 'prepared',
            launched=(directory/'launch.json').exists(),success=execution.get('success') if execution else None,
            error=execution.get('error') if execution else None,log_tail=tail(directory/'logs/child.console.log'))
    out['events_tail']=tail(state/'events.jsonl',3)
    return out


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv[:1]==['status']:
        p=argparse.ArgumentParser(description='Read-only overnight status');p.add_argument('--root',required=True);p.add_argument('--name',required=True)
        a=p.parse_args(argv[1:]);print(json.dumps(status(a.root,a.name),indent=2));return 0
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','gate','score-job','data','name','volume-id','rate','reviewer','hours-k0','hours-k2','hours-k5','setup-hours'):
        p.add_argument('--'+n,required=True)
    from pilot_eval.sampling_sft_hf import DEFAULT_REPO
    p.add_argument('--hf-repo',default=DEFAULT_REPO)
    p.add_argument('--stop-pod',action='store_true',help='Stop this Pod only after verified HF export.')
    a=p.parse_args(argv);root=Path(a.root).resolve()
    from pilot_eval.sft import safe_name
    safe_name(a.name)
    state_dir='runs/sampling-sft/overnight/'+a.name
    cfg={'root':str(root),'gate':str(inside(root,a.gate).relative_to(root)),'score_job':str(inside(root,a.score_job).relative_to(root)),
        'data':str(inside(root,a.data).relative_to(root)),'name':a.name,'state_dir':state_dir,'volume_id':a.volume_id,
        'rate':a.rate,'reviewer':a.reviewer,'hours':{'0':a.hours_k0,'2':a.hours_k2,'5':a.hours_k5},'setup_hours':a.setup_hours,
        'hf_repo':a.hf_repo,'stop_pod':a.stop_pod}
    cfg['input_sha256']={cfg[k]:file_hash(root/cfg[k]) for k in ('gate','score_job','data')}
    names=['sampling_sft_overnight.py','sampling_sft_training.py','sampling_sft_training_backend.py','sampling_sft_pipeline.py','sampling_sft_gate.py','sampling_sft_l4_oracle.py']
    cfg['code_sha256']={name:file_hash(Path(__file__).with_name(name)) for name in names}
    folder=inside(root,state_dir)
    import signal
    def terminated(*_):raise SystemExit(143) # Record a tmux kill or container shutdown, not a silent stale stage.
    signal.signal(signal.SIGTERM,terminated)
    with contextlib.ExitStack() as stack:
        try:stack.enter_context(run_lock(folder))
        except RuntimeError:
            # Never overwrite the live wrapper's status: that would make a healthy run look dead.
            print('Another overnight wrapper holds this run; leaving its status untouched. Use the status command.',file=sys.stderr)
            return 2
        stage=(json.loads((folder/'status.json').read_text()).get('state') if (folder/'status.json').exists() else None)
        try:
            source.frozen(folder/'config.json',cfg)
            result=orchestrate(folder,Operations(cfg))
        except BaseException as exc:
            stage=json.loads((folder/'status.json').read_text()).get('state') if (folder/'status.json').exists() else stage
            if ((folder/'complete.json').exists() and stage=='pod-stop-requested'
                    and isinstance(exc,SystemExit) and exc.code==143):return 0 # SIGTERM during our own requested shutdown.
            (folder/'failure-traceback.txt').write_text(traceback.format_exc())
            stop_failed=(folder/'complete.json').exists() and stage=='pod-stop-requested'
            record(folder,'pod-stop-failed' if stop_failed else 'stopped',failed_stage=stage,error_type=type(exc).__name__,error=str(exc),
                external_copy_verified=bool(stop_failed),
                traceback=str(folder/'failure-traceback.txt'),note='Completed work is preserved; no relaunch or Pod stop attempted.')
            print('Overnight workflow stopped:',type(exc).__name__,exc,file=sys.stderr)
            return 130 if isinstance(exc,KeyboardInterrupt) else 1
    print(json.dumps({'state':'exported-verified','inventory':result,'external_copy_verified':True},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
