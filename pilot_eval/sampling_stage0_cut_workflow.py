"""Standalone principal-cut build, paired singleton evaluation and private export."""
import json
from pathlib import Path


def export(directory, root, *, repo, client=None):
    """Upload only a verified complete run and verify the exact remote archive bytes."""
    import hashlib
    import tarfile
    import tempfile
    from pilot_eval.sampling_sft_hf import Hub, private
    from pilot_eval.training import file_hash
    from pilot_eval.run import _write_json
    directory=Path(directory).resolve(); root=Path(root).resolve()
    if not directory.is_relative_to(root): raise ValueError('Run outside artifact root.')
    if not (directory/'complete.json').is_file(): raise ValueError('Cannot export incomplete run.')
    verify(directory)
    client=client or Hub();private(client,repo)
    files=sorted(p for p in directory.rglob('*') if p.is_file() and not p.is_symlink()
                 and p.name != 'hf-upload.json' and not p.name.endswith('.tmp'))
    manifest={str(p.relative_to(root)):file_hash(p) for p in files}
    identity=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    receipt=directory/'meta/hf-upload.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['manifest_sha256']!=identity or saved['repo']!=repo: raise ValueError('Export identity changed.')
        if file_hash(client.fetch(repo,saved['archive'],saved['revision']))!=saved['archive_sha256']:
            raise ValueError('Remote archive readback changed.')
        return saved
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        archive=Path(temporary)/'bundle.tar.gz'
        with tarfile.open(archive,'w:gz') as stream:
            for path in files: stream.add(path,arcname=str(path.relative_to(root)),recursive=False)
        if any(file_hash(p)!=manifest[str(p.relative_to(root))] for p in files):
            raise ValueError('Local artifact changed during export.')
        index=Path(temporary)/'index.json';index.write_text(json.dumps(dict(files=manifest,manifest_sha256=identity),sort_keys=True))
        remote=f'exports/cuts-{identity}'
        archive_name=remote+'/bundle.tar.gz'; index_name=remote+'/index.json'
        archive_hash=file_hash(archive); index_hash=file_hash(index)
        revision=client.publish(repo,{archive_name:archive,index_name:index})
        if not isinstance(revision,str) or len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision):
            raise ValueError('HF export lacks exact commit revision.')
        if file_hash(client.fetch(repo,archive_name,revision))!=archive_hash or file_hash(client.fetch(repo,index_name,revision))!=index_hash:
            raise ValueError('HF readback verification failed.')
        result=dict(repo=repo,revision=revision,archive=archive_name,archive_sha256=archive_hash,
                    index=index_name,index_sha256=index_hash,manifest_sha256=identity)
        _write_json(receipt,result)
        return result


def verify(directory):
    """Verify complete aggregate and every response unit without loading a model."""
    import hashlib
    from pilot_eval.training import file_hash
    from pilot_eval.workflow import _hash
    from pilot_eval.stage0_evaluation_records import OUTPUT_FIELDS, records
    directory=Path(directory)
    if not (directory/'complete.json').is_file(): raise ValueError('Run is not complete.')
    config=json.loads((directory/'config.json').read_text())
    items=json.loads((directory/'inputs/items.json').read_text())
    inputs=json.loads((directory/'inputs/tokenized-prompts.json').read_text())
    if [r['id'] for r in inputs] != [r['id'] for r in items]: raise ValueError('Input identity changed.')
    if any(t.get('prompt_sha256')!=_hash(i['prompt']) or t.get('attention_mask')!=[1]*len(t['input_ids']) for t,i in zip(inputs,items)):
        raise ValueError('Prompt identity changed.')
    rh=file_hash(directory/'meta/runtime.json'); ih=file_hash(directory/'inputs/tokenized-prompts.json'); items_hash=file_hash(directory/'inputs/items.json'); rows=[]
    for variant in config['variant_order']:
        for index,item in enumerate(items):
            key=hashlib.sha256((variant+'\0'+str(item['id'])).encode()).hexdigest()
            unit=dict(key=key,variant=variant,ids=[item['id']],seed=config['seed']+index//16)
            folder=directory/'units'/key; payload=folder/'responses.jsonl'; proof=folder/'integrity.json'
            expected=dict(config_sha256=_hash(config),runtime_sha256=rh,input_sha256=ih,items_sha256=items_hash,
                          responses_sha256=file_hash(payload),integrity_sha256=file_hash(proof))
            if json.loads((folder/'complete.json').read_text())!=expected: raise ValueError('Response unit changed.')
            evidence=json.loads(proof.read_text())
            if any(not all(evidence[s].get(k) is True for k in ('base_unchanged','adapters_unchanged')) for s in ('before','after')):
                raise ValueError('Response integrity failed.')
            saved=[json.loads(line) for line in payload.read_text().splitlines()]
            if saved!=records(config,unit,[item],[{k:r[k] for k in OUTPUT_FIELDS} for r in saved],runtime_sha256=rh,input_sha256=ih):
                raise ValueError('Response scoring changed.')
            rows.extend(saved)
    response=directory/'results/responses.jsonl';summary=directory/'results/results.json'
    expected=dict(config_sha256=_hash(config),items_sha256=items_hash,responses_sha256=file_hash(response),results_sha256=file_hash(summary))
    if json.loads((directory/'complete.json').read_text())!=expected: raise ValueError('Completed result seal changed.')
    if [json.loads(x) for x in response.read_text().splitlines()]!=rows: raise ValueError('Aggregate changed.')
    return json.loads(summary.read_text())


IMPLEMENTATION = ('sampling_stage0_cut_workflow.py','sampling_stage0_cut_backend.py','sampling_stage0_cuts.py',
                  'stage0_edit_math.py','stage0_spectral_math.py','stage0_edits.py','stage0_weights.py',
                  'stage0_evaluation_backend.py','stage0_evaluation_records.py','stage0_sources.py','scoring.py')


def implementation():
    from pilot_eval.training import file_hash
    return {name:file_hash(Path(__file__).with_name(name)) for name in IMPLEMENTATION}


def prepare(geometry_path, root, name):
    from pilot_eval import sampling_stage0_geometry as geometry
    from pilot_eval.sft import safe_name
    from pilot_eval.stage0_sources import inside
    from pilot_eval.training import file_hash
    from pilot_eval.workflow import _save_frozen
    root=Path(root).resolve();safe_name(name);path=inside(root,geometry_path)
    _,source,_=geometry._load(path,root)
    if source['execution_kind'] != 'cpu-local-pinned-weights': raise ValueError('Real pinned base geometry required.')
    plan=dict(protocol='sampling-principal-cuts-v1',name=name,geometry_path=str(path.relative_to(root)),
              geometry_sha256=file_hash(path),run_path=f'runs/stage0/{name}/principal-cuts',
              implementation=implementation(),gpu='A100')
    output=root/'plans'/name/'sampling-stage0.cuts.json';_save_frozen(output,plan)
    return output


def load_plan(path,root):
    from pilot_eval import sampling_stage0_geometry as geometry
    from pilot_eval.stage0_sources import inside
    from pilot_eval.training import file_hash
    root=Path(root).resolve();path=inside(root,path);plan=json.loads(path.read_text())
    if plan.get('protocol')!='sampling-principal-cuts-v1' or plan['implementation']!=implementation():
        raise ValueError('Cut implementation or protocol changed.')
    geometry_path=inside(root,plan['geometry_path'])
    if file_hash(geometry_path)!=plan['geometry_sha256']:raise ValueError('Geometry plan changed.')
    _,source,_=geometry._load(geometry_path,root)
    return plan,source


def build(path,root,*,limit=None):
    from pilot_eval.stage0_sources import factors,inside
    from pilot_eval.stage0_weights import LocalWeights
    from pilot_eval.sampling_stage0_cuts import build_modules,assemble
    from pilot_eval.training import file_hash,run_lock
    from pilot_eval.workflow import _hash,_save_frozen
    root=Path(root).resolve();plan,source=load_plan(path,root)
    directory=inside(root,plan['run_path']); modules=list(source['modules'])
    weights=LocalWeights()
    with run_lock(directory):
        inventory=weights.inventory(inside(root,source['weights_dir']),dict(identity=source['model_identity']))
        actual={m:dict(path=str(Path(e['path']).resolve().relative_to(root)),key=e['key']) for m,e in inventory['modules'].items()}
        if actual!=source['modules'] or inventory['base_sha256']!=source['base_sha256']:
            raise ValueError('Base source inventory changed.')
        paths={arm:inside(root,v['adapter_path']) for arm,v in source['variants'].items()}
        originals={arm:factors(path) for arm,path in paths.items()}
        identity=_hash(plan)
        result=build_modules(directory/'build',modules,originals,
            lambda m:weights.matrix({**actual[m],'path':str(inside(root,actual[m]['path']))}),identity=identity,limit=limit)
        if result['state']!='complete':return result
        variants=assemble(directory/'build',root,paths,originals,modules,identity=identity)
        adapter_files={name:{n:file_hash(inside(root,v['adapter_path'])/n) for n in
                       ('adapter_config.json','adapter_model.safetensors','complete.json')}
                       for name,v in variants.items()}
        behaviour=json.loads(inside(root,source['behaviour_plan']).read_text())
        config={key:behaviour[key] for key in ('model_identity','weights_dir','model_snapshot','prompt_contract','decoding','cohort_path','cohort_sha256')}
        config.update(protocol='sampling-principal-cut-evaluation-v1',run_path=plan['run_path'],
                      geometry_plan=plan['geometry_path'],geometry_sha256=plan['geometry_sha256'],
                      plan_sha256=_hash(plan),implementation=plan['implementation'],variants=variants,
                      variant_order=list(variants),adapter_files=adapter_files,scorer='gsm8k-flexible-v3',seed=42,
                      execution_kind='real-cuda',gpu=plan['gpu'],batch_size=1)
        _save_frozen(directory/'evaluation-plan.json',config)
        return dict(state='complete',variants=len(variants),modules=len(modules),evaluation_plan=str(directory/'evaluation-plan.json'))


def run(path,root,*,limit=None,deadline_seconds=86400):
    from pilot_eval.sampling_stage0_evaluation import environment
    from pilot_eval.sampling_stage0_cut_backend import CutEngine
    from pilot_eval.sampling_stage0_cuts import collect
    from pilot_eval.stage0_qualification_backend import QualificationDependencies,deadline_guard
    from pilot_eval.stage0_sources import inside
    from pilot_eval.training import file_hash,run_lock
    from pilot_eval.workflow import _hash
    root=Path(root).resolve();plan,source=load_plan(path,root);directory=inside(root,plan['run_path'])
    config=json.loads((directory/'evaluation-plan.json').read_text())
    if config['plan_sha256']!=_hash(plan) or config['implementation']!=implementation(): raise ValueError('Evaluation plan changed.')
    for variant,files in config['adapter_files'].items():
        folder=inside(root,config['variants'][variant]['adapter_path'])
        if any(file_hash(folder/n)!=digest for n,digest in files.items()):raise ValueError('Cut adapter changed.')
    items=json.loads(inside(root,config['cohort_path']).read_text())
    if _hash(items)!=config['cohort_sha256']:raise ValueError('Cohort changed.')
    if deadline_seconds<=0:raise ValueError('Positive deadline required.')
    deps=QualificationDependencies();runtime=deps.runtime()
    if runtime['execution_kind']!='real-cuda' or plan['gpu'] not in runtime['device']:
        raise ValueError('This run requires the configured A100 CUDA GPU.')
    deps.preflight(config,root)
    with run_lock(directory),deadline_guard(deadline_seconds,True):
        engine=CutEngine(config,root)
        try:return collect(config,items,directory,environment(runtime),engine,limit=limit)
        finally:engine.close()


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--geometry',required=True);p.add_argument('--output-root',required=True);p.add_argument('--name',required=True)
    for command in ('build','run'):
        p=sub.add_parser(command);p.add_argument('--plan',required=True);p.add_argument('--output-root',required=True);p.add_argument('--max-new-units',type=int)
        if command=='run':p.add_argument('--deadline-seconds',type=int,default=86400)
    p=sub.add_parser('verify');p.add_argument('--run-dir',required=True)
    p=sub.add_parser('export');p.add_argument('--run-dir',required=True);p.add_argument('--output-root',required=True);p.add_argument('--repo',required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare':result=str(prepare(args.geometry,args.output_root,args.name))
    elif args.command=='build':result=build(args.plan,args.output_root,limit=args.max_new_units)
    elif args.command=='run':result=run(args.plan,args.output_root,limit=args.max_new_units,deadline_seconds=args.deadline_seconds)
    elif args.command=='verify':result=verify(args.run_dir)
    else:result=export(args.run_dir,args.output_root,repo=args.repo)
    print(json.dumps(result,indent=2,sort_keys=True))
    return result


if __name__=='__main__': main()
