"""Unattended L4 Stage 0 for the three sampling-SFT adapters, export-verified, optional own-Pod stop.

Stages: preflight -> plans -> batch agreement -> behaviour -> fixed-context KL/writes
-> CPU geometry -> sealed HF export + fresh-download check -> optional Pod stop.
Each measurement runs as a child process with a wall-clock limit and its own log;
the wrapper then re-verifies its sealed result on CPU. Every collector resumes from
sealed units, so rerunning this command after any stop continues where it left off.

Run:    python -m pilot_eval.sampling_stage0_overnight --root ... (see main)
Status: python -m pilot_eval.sampling_stage0_overnight status --root ... --name ...  (read-only)
"""
import argparse
import contextlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from pilot_eval import sampling_sft as source
from pilot_eval.sampling_sft_overnight import now, probe_hf_write_read, record, tree, verify_download
from pilot_eval.stage0_sources import inside
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _hash

STAGES = ('agreement', 'behaviour', 'drift', 'dataset', 'geometry')
MODULES = {s:f'pilot_eval.sampling_stage0_{m}' for s,m in
           [('agreement','agreement'),('behaviour','evaluation'),('drift','drift'),('dataset','dataset'),('geometry','geometry')]}
CODE = ['sampling_stage0_overnight.py','sampling_stage0_agreement.py','sampling_stage0_evaluation.py',
        'sampling_stage0_drift.py','sampling_stage0_dataset.py','sampling_stage0_geometry.py','sampling_stage0_direct_backend.py',
        'stage0_evaluation_backend.py','stage0_batching_backend.py','stage0_drift_backend.py','stage0_distance_backend.py']
OPTIONAL=('geometry',) # CPU-only; its failure must not strand the GPU results on the Pod.
SCOPE=dict(stages=list(STAGES),
    not_included=['historical Base/Gold/GRPO behaviour join','held-out gold NLL/entropy under each adapter',
                  'initial pushes','activation energy','interventions/edited adapters'],
    meaning='Completion covers only the declared stages for k0/k2/k5, not the whole Stage 0 programme.')


def persistent_mount(root,mounts,provider_path):
    """The artifact root's actual mount from /proc/mounts; never the container disk, never unknown."""
    root=Path(root);best=None
    for line in mounts.splitlines():
        fields=line.split()
        if len(fields)<3:continue
        point=Path(fields[1].replace('\\040',' '))
        if root.is_relative_to(point) and (best is None or len(point.parts)>len(best[0].parts)):best=(point,fields[2])
    if best is None or best[0]==Path('/') or best[1]=='overlay':
        raise ValueError(f'Artifact root {root} is on the container disk, which a Pod stop discards; use the persistent volume.')
    if provider_path and best[0]!=Path(provider_path):
        raise ValueError(f'Artifact root mount {best[0]} differs from the provider volume mount {provider_path}.')
    if not provider_path and (best[0]!=Path('/workspace') or not best[1].startswith(('fuse','nfs','ceph'))):
        raise ValueError(f'Provider did not report a mount path and {best[0]} ({best[1]}) is not the RunPod network volume '
                         'at /workspace; refusing a Pod stop that could discard outputs.')
    return dict(mount=str(best[0]),fstype=best[1])


def export_status(missing):
    """Manifest fields that state exactly which declared stages finished and which failed."""
    done=[s for s in STAGES if s not in missing]
    scope='sampling Stage 0 stages verified on CPU before sealing: '+', '.join(done)
    if missing:scope+='; failed (not included): '+', '.join(missing)
    return dict(workload_success=not missing,missing_expected_outputs=list(missing),scope=scope)


def same_environment(state,runtime):
    """Every GPU stage of one run measures in one frozen hardware/software environment."""
    from pilot_eval.sampling_stage0_evaluation import environment
    path=Path(state)/'environment.json';value=environment(runtime)
    if path.exists() and json.loads(path.read_text())!=value:
        raise ValueError('GPU/driver/package environment changed since earlier stages of this run; '
                         'start a new run name rather than mixing environments.')
    if not path.exists():source.save(path,value)


def orchestrate(folder,ops):
    """Completed stages have durable pointers; a failed stage is never marked complete."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    record(folder,'preflight');ops.preflight()
    plans=folder/'plans.complete.json'
    if not plans.exists():
        record(folder,'plans');source.frozen(plans,ops.plans())
    missing=[]
    for stage in STAGES:
        marker=folder/f'{stage}.complete.json'
        # A marker proves one earlier verification; re-check the sealed product before trusting it.
        if marker.exists():ops.reverify(stage,json.loads(marker.read_text()))
        else:
            record(folder,stage)
            try:source.frozen(marker,ops.stage(stage))
            except Exception as exc:
                if stage not in OPTIONAL:raise
                missing.append(stage);failure=exc
    if missing:
        # Get the verified GPU products off the Pod anyway; never stop the Pod on a partial run.
        record(folder,'hf-export-partial',missing_stages=missing)
        receipt=ops.export(partial=True,missing=missing)
        source.save(folder/'partial-export.json',dict(state='exported-partial',missing_stages=missing,receipt=receipt))
        record(folder,'exported-partial',external_copy_verified=True,missing_stages=missing,hf_revision=receipt['revision'])
        raise failure
    exported=folder/'export.complete.json'
    if exported.exists():ops.reverify_export(json.loads(exported.read_text()))
    else:
        record(folder,'hf-export');source.frozen(exported,ops.export())
    receipt=json.loads(exported.read_text())
    source.frozen(folder/'complete.json',{'state':'exported-verified','external_copy_verified':True,
        'hf_repo':receipt['repo'],'hf_revision':receipt['revision'],'manifest_sha256':receipt['manifest_sha256'],
        'scope':SCOPE,'results':{s:json.loads((folder/f'{s}.complete.json').read_text()) for s in STAGES}})
    record(folder,'exported-verified',external_copy_verified=True,hf_revision=receipt['revision'])
    if ops.stop_requested:
        record(folder,'pod-stop-requested',external_copy_verified=True)
        ops.stop()
        record(folder,'pod-stop-command-returned',external_copy_verified=True)
    return receipt


def sources_manifest(root,training_jobs):
    """Gold config, local model and the three completed arms, read from the training wrapper's record."""
    entries=json.loads(inside(root,training_jobs).read_text())
    if sorted(e['arm'] for e in entries)!=[0,2,5]:raise ValueError('Expected exactly the k0/k2/k5 training jobs.')
    plans={f'k{e["arm"]}':e['config'] for e in entries}
    configs=[json.loads(inside(root,p).read_text()) for p in plans.values()]
    gold={c['gold_config'] for c in configs};data={c['data_plan'] for c in configs}
    if len(gold)!=1 or len(data)!=1:raise ValueError('Arms disagree on Gold config or training data.')
    dc=json.loads(inside(root,data.pop()).read_text())
    gen=json.loads(inside(root,dc['generation_plan']).read_text())
    return dict(gold_config=gold.pop(),weights_dir=gen['settings']['local_model'],training_plans=plans)


def run_child(stage,plan,cfg,log,*,runner=subprocess.run):
    """One child per attempt; a retry resumes sealed units. An integrity failure is never retried."""
    root=Path(cfg['root']);command=[sys.executable,'-m',MODULES[stage],'run','--config',str(plan),'--output-root',str(root)]
    log.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(1+cfg['retries']):
        print(f'{stage}: attempt {attempt+1}; child log {log}',flush=True)
        with log.open('a') as stream:
            stream.write(f'\n=== {stage} attempt {attempt+1} {now()} ===\n');stream.flush()
            try:
                code=runner(command,stdout=stream,stderr=subprocess.STDOUT,timeout=float(cfg['hours'][stage])*3600,
                            env={**os.environ,'HF_HUB_OFFLINE':'1','PYTHONUNBUFFERED':'1'}).returncode
            except subprocess.TimeoutExpired:
                code='timeout'
        if code==0:return
        run_path=inside(root,json.loads(inside(root,plan).read_text())['run_path'])
        if (run_path/'meta/integrity-failure.json').exists() or code=='timeout':break
    tail=log.read_text(errors='replace').splitlines()[-15:]
    raise RuntimeError(f'{stage} child failed ({code}); sealed units are kept. Last log lines:\n'+'\n'.join(tail))


class Operations:
    def __init__(self,cfg):
        self.cfg=cfg;self.root=Path(cfg['root']).resolve();self.state=inside(self.root,cfg['state_dir'])
        self.stop_requested=bool(cfg.get('stop_pod'))

    def path(self,key):return inside(self.root,self.cfg[key])

    def preflight(self):
        for name,digest in self.cfg['input_sha256'].items():
            if file_hash(inside(self.root,name))!=digest:raise ValueError('Overnight input changed: '+name)
        for name,digest in self.cfg['code_sha256'].items():
            if file_hash(Path(__file__).with_name(name))!=digest:raise ValueError('Overnight implementation changed: '+name)
        from pilot_eval.stage0_qualification_backend import QualificationDependencies
        runtime=QualificationDependencies().runtime() # pinned packages, one CUDA GPU, driver, FP32 settings
        if self.cfg['gpu'] not in runtime['device']:raise ValueError(f"Expected {self.cfg['gpu']}, found {runtime['device']}.")
        same_environment(self.state,runtime)
        if self.cfg.get('volume_id'):
            from pilot_eval.sampling_sft_provider import RunPodProvider
            subprocess.run(['runpodctl','--version'],check=True,capture_output=True,timeout=30)
            provider=RunPodProvider();pod=provider.own_pod_id();observed=provider.inspect(pod)
            if observed.get('id')!=pod or observed.get('storageVolumeId',observed.get('networkVolumeId'))!=self.cfg['volume_id']:
                raise ValueError('Pod or persistent volume identity mismatch.')
            self.mount=persistent_mount(self.root,Path('/proc/mounts').read_text(),observed.get('storageMountPath'))
        from pilot_eval import sampling_sft_hf as hf
        client=hf.Hub();hf.private(client,self.cfg['hf_repo']);client.api.whoami()
        hf.require_space(self.root,2*1024**3);hf.require_space(tempfile.gettempdir(),2*1024**3)
        probe_hf_write_read(client.api,self.cfg['hf_repo'],self.state)

    def plans(self):
        from pilot_eval import sampling_stage0_agreement as agreement
        manifest=self.state/'sources.json'
        source.frozen(manifest,sources_manifest(self.root,self.cfg['training_jobs']))
        plan=agreement.prepare(manifest,self.root,self.cfg['name'],gpu=self.cfg['gpu'])
        return dict(manifest=str(manifest.relative_to(self.root)),agreement=str(plan.relative_to(self.root)))

    def _plan(self,stage):
        from pilot_eval import sampling_stage0_evaluation as evaluation, sampling_stage0_drift as drift
        from pilot_eval import sampling_stage0_geometry as geometry
        plans=json.loads((self.state/'plans.complete.json').read_text());name=self.cfg['name']
        if stage=='agreement':return self.root/plans['agreement']
        if stage=='behaviour':
            return evaluation.prepare(self.root/plans['manifest'],self.root,name,gpu=self.cfg['gpu'],
                                      agreement_plan=self.root/plans['agreement'])
        behaviour=self.root/'plans'/name/'sampling-stage0.evaluation.json'
        if stage=='drift':return drift.prepare(behaviour,self.path('inputs'),self.root,name)
        if stage=='dataset':
            from pilot_eval import sampling_stage0_dataset as dataset
            return dataset.prepare(behaviour,self.root,name)
        return geometry.prepare(behaviour,self.root,name)

    def stage(self,stage):
        import importlib
        plan=self._plan(stage);module=importlib.import_module(MODULES[stage])
        run_child(stage,plan,self.cfg,self.state/'logs'/f'{stage}.console.log')
        result=module.verify(plan,self.root) # CPU re-verification of the sealed product
        brief={k:result[k] for k in ['batch_sizes','arms','variants','controls','paired_differences','trajectory','endpoint_overlap','cross_arm'] if k in result}
        return dict(plan=str(plan.relative_to(self.root)),plan_sha256=file_hash(plan),summary=brief)

    def export(self,partial=False,missing=()):
        """One sealed bundle: the Stage 0 plans and run trees plus this wrapper's decisions. A partial
        export gets its own folder so the final full export is always built fresh."""
        import uuid
        from pilot_eval import sampling_sft_exports as exports, sampling_sft_hf as hf
        root=self.root;folder=self.state/('export-partial-'+uuid.uuid4().hex[:8] if partial else 'export')
        manifest_path=folder/'export-manifest.json'
        if not (folder/'hf-upload.json').exists():
            names=set(tree(root,root/'plans'/self.cfg['name']))|set(tree(root,root/'runs/stage0'/self.cfg['name']))
            internal=[p for p in self.state.iterdir() if p.name.startswith('export')]+[self.state/'status.json',self.state/'events.jsonl']
            names|=set(tree(root,self.state,exclude=internal))
            manifest=dict(schema_version=1,job_sha256=_hash(self.cfg),files=exports.inventory(root,sorted(names)),
                simulation=False,worker_host=platform.node(),pod_id=os.environ.get('RUNPOD_POD_ID'),**export_status(missing))
            manifest['sha256']=_hash(manifest);source.save(manifest_path,manifest)
            print(f'Uploading {len(manifest["files"])} sealed files to private HF repo {self.cfg["hf_repo"]}.',flush=True)
        manifest=exports.verify_manifest(manifest_path,root)
        uploaded=hf.upload(manifest_path,root,repo=self.cfg['hf_repo'])
        check=verify_download(uploaded,manifest,manifest_path,root)
        source.save(folder/'hf-download-verification.json',check)
        return dict(repo=uploaded['repo'],revision=uploaded['revision'],archive=uploaded['archive'],
            archive_sha256=uploaded['archive_sha256'],manifest_sha256=manifest['sha256'],verification=check)

    def reverify(self,stage,value):
        import importlib
        plan=inside(self.root,value['plan'])
        if file_hash(plan)!=value['plan_sha256']:raise ValueError(f'{stage} plan changed after completion.')
        result=importlib.import_module(MODULES[stage]).verify(plan,self.root)
        brief={k:result[k] for k in value['summary']}
        if json.loads(json.dumps(brief,default=str))!=value['summary']:raise ValueError(f'{stage} evidence changed after completion.')

    def reverify_export(self,receipt):
        from pilot_eval import sampling_sft_exports as exports
        folder=self.state/'export';manifest=exports.verify_manifest(folder/'export-manifest.json',self.root)
        upload=json.loads((folder/'hf-upload.json').read_text())
        check=json.loads((folder/'hf-download-verification.json').read_text())
        if (manifest['sha256']!=receipt['manifest_sha256'] or upload['revision']!=receipt['revision']
                or check.get('archive_sha256')!=upload['archive_sha256']
                or check.get('verified_files')!=len(manifest['files'])+1):
            raise ValueError('Export evidence changed after verification.')

    def stop(self):
        from pilot_eval.sampling_sft_provider import RunPodProvider
        provider=RunPodProvider();provider.stop(provider.own_pod_id())


def status(root,name):
    """Read-only summary after reconnecting; never takes locks or writes."""
    root=Path(root).resolve();state=root/'runs/stage0/overnight'/name
    def read(path):
        try:return json.loads(Path(path).read_text())
        except (OSError,ValueError):return None
    def tail(path,lines=4):
        try:return Path(path).read_text(errors='replace').splitlines()[-lines:]
        except OSError:return []
    folders={'agreement':'batch-agreement','behaviour':'sampling-behaviour','drift':'fixed-context','dataset':'dataset','geometry':'geometry'}
    out=dict(overnight=read(state/'status.json'),complete=read(state/'complete.json'),stages={})
    for stage,folder in folders.items():
        out['stages'][stage]=dict(done=(state/f'{stage}.complete.json').exists(),
            progress=read(root/'runs/stage0'/name/folder/'meta/status.json'),log_tail=tail(state/'logs'/f'{stage}.console.log'))
    out['events_tail']=tail(state/'events.jsonl',3)
    return out


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv[:1]==['status']:
        p=argparse.ArgumentParser(description='Read-only overnight status');p.add_argument('--root',required=True);p.add_argument('--name',required=True)
        a=p.parse_args(argv[1:]);print(json.dumps(status(a.root,a.name),indent=2,default=str));return 0
    from pilot_eval.sampling_sft_hf import DEFAULT_REPO
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','training-jobs','inputs','name'):p.add_argument('--'+n,required=True)
    p.add_argument('--volume-id',help='Verify this Pod mounts the persistent volume (required with --stop-pod).')
    p.add_argument('--gpu',default='L4');p.add_argument('--hf-repo',default=DEFAULT_REPO)
    p.add_argument('--retries',type=int,default=1,help='Child relaunches per stage after a crash (resumes sealed units).')
    for stage,hours in [('agreement',2),('behaviour',4),('drift',3),('dataset',2),('geometry',2)]:
        p.add_argument(f'--hours-{stage}',default=str(hours),help='Child wall-clock limit in hours.')
    p.add_argument('--stop-pod',action='store_true',help='Stop this Pod only after verified HF export.')
    a=p.parse_args(argv);root=Path(a.root).resolve()
    from pilot_eval.sft import safe_name
    safe_name(a.name)
    if a.stop_pod and not a.volume_id:p.error('--stop-pod requires --volume-id so outputs are proven persistent first.')
    if a.retries<0:p.error('--retries must be nonnegative.')
    state_dir='runs/stage0/overnight/'+a.name
    cfg={'root':str(root),'training_jobs':str(inside(root,a.training_jobs).relative_to(root)),
        'inputs':str(inside(root,a.inputs).relative_to(root)),'name':a.name,'state_dir':state_dir,'volume_id':a.volume_id,
        'gpu':a.gpu,'hf_repo':a.hf_repo,'stop_pod':a.stop_pod,'retries':a.retries,
        'hours':{s:getattr(a,'hours_'+s) for s in STAGES}}
    cfg['input_sha256']={cfg[k]:file_hash(root/cfg[k]) for k in ('training_jobs','inputs')}
    cfg['code_sha256']={n:file_hash(Path(__file__).with_name(n)) for n in CODE}
    folder=inside(root,state_dir)
    import signal
    def terminated(*_):raise SystemExit(143) # Record a tmux kill or container shutdown, not a silent stale stage.
    signal.signal(signal.SIGTERM,terminated)
    with contextlib.ExitStack() as stack:
        try:stack.enter_context(run_lock(folder))
        except RuntimeError:
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
                external_copy_verified=bool(stop_failed),traceback=str(folder/'failure-traceback.txt'),
                note='Sealed units are preserved; rerun the same command to resume. No Pod stop attempted.')
            print('Overnight workflow stopped:',type(exc).__name__,exc,file=sys.stderr)
            return 130 if isinstance(exc,KeyboardInterrupt) else 1
    print(json.dumps({'state':'exported-verified','hf_revision':result['revision'],'external_copy_verified':True},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
