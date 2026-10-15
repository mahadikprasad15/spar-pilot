"""Bounded Pilot 5 instrument diagnostics, separate from numerical qualification."""
import json
import time
from pathlib import Path
import numpy as np
from pilot_eval.trace_corpus import load_corpus,_inside
from pilot_eval.trace_capture import summarize_pairs,retention_evidence,CaptureFailure
from pilot_eval.training import file_hash,run_lock
from pilot_eval.workflow import _hash,_save_frozen
from pilot_eval.run import _write_json,_write_state
from pilot_eval.sft import safe_name
from pilot_eval.activation_workflow import _save_arrays

CONDITIONS=['base','sft-zero','grpo-zero']
SITES=['embedding-output']+[f'block-{i}-output' for i in range(28)]


def _implementation():
    folder=Path(__file__).parent
    return {name:file_hash(folder/name) for name in ['trace_capture.py','trace_capture_workflow.py','trace_capture_backend.py']}


def _source(path,root,arm):
    """Verify sealed preparation and only the adapter payloads used here.

    This does not re-audit unrelated training optimizer states or checkpoints.
    """
    path=_inside(root,path);config=json.loads(path.read_text())
    if (config.get('protocol_version')!=('pilot3-write-v1' if arm=='sft' else 'pilot4-writes-v1')
            or (arm=='grpo' and config.get('source_contract',{}).get('kind')!='grpo')):
        raise ValueError('Wrong learned source kind for bounded zero check.')
    marker=path.parent/'prepare-complete.json';seal=json.loads(marker.read_text())
    if seal['config_sha256']!=_hash(config) or str(path.relative_to(root)) not in seal['files']:
        raise ValueError('Prepared source seal mismatch.')
    parents={str(path.relative_to(root)):file_hash(path),str(marker.relative_to(root)):file_hash(marker)}
    for relative,expected in seal['files'].items():
        p=_inside(root,relative)
        if file_hash(p)!=expected:raise ValueError('Prepared source payload hash mismatch.')
        parents[relative]=expected
    zero=config['source_evidence']['checkpoints']['0'];checkpoint=_inside(root,zero['path'])
    cp=checkpoint/'complete.json';saved=json.loads(cp.read_text())
    if saved['step']!=0 or file_hash(cp)!=zero['complete_sha256']:
        raise ValueError('Step-zero checkpoint seal mismatch.')
    for name in ['adapter_config.json','adapter_model.safetensors']:
        expected=saved['files'][name];p=checkpoint/name
        if file_hash(p)!=expected:raise ValueError('Step-zero adapter hash mismatch.')
        parents[str(p.relative_to(root))]=expected
    if parents[str((checkpoint/'adapter_model.safetensors').relative_to(root))]!=zero['adapter_sha256']:
        raise ValueError('Step-zero source identity mismatch.')
    parents[str(cp.relative_to(root))]=file_hash(cp)
    adapter=json.loads((checkpoint/'adapter_config.json').read_text())
    if adapter.get('r')!=1 or adapter.get('lora_alpha')!=1 or adapter.get('lora_dropout')!=0:
        raise ValueError('Unexpected zero-adapter configuration.')
    return config,parents


def prepare_diagnostic(corpus_path,sft_path,grpo_path,output_root,name):
    safe_name(name);root=Path(output_root).resolve()
    path=root/'plans'/name/'traces.capture-diagnostic.json'
    requested=[str(_inside(root,p).relative_to(root)) for p in [corpus_path,sft_path,grpo_path]]
    with run_lock(path.parent):
        if path.exists():
            config=_load(path,root)
            if config['source_paths']!=requested:raise ValueError('Diagnostic source selection changed.')
            return path
        corpus,rows=load_corpus(_inside(root,corpus_path),root)
        sft,parents=_source(sft_path,root,'sft');grpo,more=_source(grpo_path,root,'grpo');parents.update(more)
        for key in ['model','model_revision','tokenizer_revision']:
            if corpus[key]!=sft[key] or corpus[key]!=grpo[key]:raise ValueError('Diagnostic model/tokenizer identity mismatch.')
        if sft['source_evidence']['base_sha256']!=grpo['source_evidence']['base_sha256']:
            raise ValueError('Paired sources have different frozen bases.')
        chosen=[]
        for stratum in ['legacy-150','new-50']:
            candidates=[r for r in rows if r['stratum']==stratum]
            if len(candidates)<2:raise ValueError('Diagnostic requires two documents per stratum.')
            chosen.extend(candidates[:2])
        parents[requested[0]]=file_hash(_inside(root,corpus_path))
        for relative in [corpus['items_path']]:parents[relative]=file_hash(_inside(root,relative))
        config=dict(schema_version=1,mode='bounded-zero-diagnostic',name=name,
            run_path=f'runs/pilot-5/{name}/capture-diagnostic',source_paths=requested,parents=parents,
            implementation=_implementation(),examples=chosen,conditions=CONDITIONS,sites=SITES,
            selection='first two in frozen order per stratum; approved four-document scope',
            model=corpus['model'],model_revision=corpus['model_revision'],tokenizer_revision=corpus['tokenizer_revision'],
            adapters={arm:c['source_evidence']['checkpoints']['0']['path'] for arm,c in [('sft',sft),('grpo',grpo)]},
            expected_base_hash=sft['source_evidence']['base_sha256'],seed=42,content_tokens=128,batch_size=1,
            capture_precision='float32',accumulation='float64',numerical_gates='pending measured-evidence review',
            approved_scope='Prasad: four documents, base and both step-zero adapters; diagnostics before GPU gates',
            workload=dict(sequence_forwards=12,model_calls=12,sites_per_forward=29,base_moment_count=508),
            verification_scope='sealed input/prepared parents and used adapter-zero files; no full trainer-state audit')
        _save_frozen(path,config)
        directory=root/config['run_path']
        _save_frozen(directory/'meta/run_manifest.json',dict(config_path=str(path.relative_to(root)),config_sha256=_hash(config)))
        _write_state(directory,'prepared',0,3)
        print('Prepared diagnostic: 4 documents × 3 conditions = 12 singleton model calls; 29 sites per call.',flush=True)
        return path


def _load(path,root):
    path=_inside(root,path);config=json.loads(path.read_text())
    if config['mode']!='bounded-zero-diagnostic' or config['implementation']!=_implementation():
        raise ValueError('Diagnostic implementation changed; preserve the old run and prepare a new name.')
    manifest=json.loads((_inside(root,config['run_path'])/'meta/run_manifest.json').read_text())
    if manifest!=dict(config_path=str(path.relative_to(root)),config_sha256=_hash(config)):
        raise ValueError('Diagnostic plan differs from its saved manifest.')
    for relative,expected in config['parents'].items():
        if file_hash(_inside(root,relative))!=expected:raise ValueError('Diagnostic input parent hash mismatch.')
    return config


def _unit(directory,condition,identity):
    folder=directory/'units'/condition;marker=folder/'complete.json'
    if not marker.exists():return None
    payload=json.loads(marker.read_text())
    if payload['identity']!=identity or set(payload['files'])!={'arrays.npz','evidence.json'}:
        raise ValueError('Diagnostic unit identity mismatch.')
    for name,expected in payload['files'].items():
        if file_hash(folder/name)!=expected:raise ValueError('Diagnostic unit hash mismatch.')
    with np.load(folder/'arrays.npz',allow_pickle=False) as saved:
        arrays={k:saved[k] for k in saved.files}
    return arrays,json.loads((folder/'evidence.json').read_text())


def verify_diagnostic(path,output_root):
    root=Path(output_root).resolve();config=_load(path,root);directory=_inside(root,config['run_path'])
    marker=json.loads((directory/'complete.json').read_text())
    if marker['config_sha256']!=_hash(config):raise ValueError('Diagnostic completion identity mismatch.')
    for relative,expected in marker['files'].items():
        if file_hash(_inside(directory,relative))!=expected:raise ValueError('Diagnostic completion hash mismatch.')
    runtime=json.loads((directory/'meta/runtime.json').read_text())
    for condition in CONDITIONS:
        if _unit(directory,condition,dict(config_sha256=_hash(config),runtime=runtime,condition=condition)) is None:
            raise ValueError('Diagnostic unit is missing.')
    return json.loads((directory/'results/results.json').read_text())


def run_diagnostic(path,output_root,*,dependencies=None):
    root=Path(output_root).resolve();config=_load(path,root);directory=_inside(root,config['run_path'])
    with run_lock(directory):
        if (directory/'complete.json').exists():return verify_diagnostic(path,root)
        from pilot_eval.trace_capture_backend import CaptureDependencies
        deps=dependencies or CaptureDependencies();engine=None;completed=0;condition='verify-runtime'
        try:
            runtime=deps.runtime();_save_frozen(directory/'meta/runtime.json',runtime)
            existing={}
            for condition in CONDITIONS:
                identity=dict(config_sha256=_hash(config),runtime=runtime,condition=condition)
                existing[condition]=_unit(directory,condition,identity)
            completed=sum(v is not None for v in existing.values())
            print(f'Diagnostic: {completed}/3 sealed conditions; missing calls only; no qualification claim.',flush=True)
            if completed<3:
                engine=deps.engine(config,root)
                if engine.integrity()!=config['expected_base_hash']:raise ValueError('Frozen base identity mismatch before workload.')
            base=existing['base'][0] if existing['base'] else None;checks={}
            for condition in CONDITIONS:
                if existing[condition]:
                    checks[condition]=existing[condition][1];continue
                _write_state(directory,'running',completed,3)
                started=time.monotonic();captured=engine.capture(config['examples'],condition)
                if len(captured)!=4:raise ValueError('Diagnostic capture omitted examples.')
                arrays={};comparison=[];conversion=[]
                for i,(row,values) in enumerate(zip(config['examples'],captured)):
                    if set(values)!=set(SITES):raise ValueError('Missing named diagnostic capture site.')
                    for site in SITES:
                        value=np.asarray(values[site])
                        if value.dtype!=np.float32 or value.shape[:2]!=(1,len(row['input_ids'])) or not np.isfinite(value).all():
                            raise ValueError(f'Invalid FP32 activation: example {row["id"]}, site {site}')
                        key=f'{i}.{site}';b=value if condition=='base' else base[key]
                        delta=value.astype(np.float64)-b.astype(np.float64)
                        if b.shape!=value.shape:raise ValueError('Base/adapted shape mismatch.')
                        arrays[key]=value if condition=='base' else delta
                        coord=np.unravel_index(int(np.argmax(np.abs(delta))),delta.shape)
                        comparison.append(dict(example_id=row['id'],site=site,max_absolute_error=float(np.abs(delta).max()),
                            coordinate=[int(x) for x in coord],base_value=float(b[coord]),adapted_value=float(value[coord]),allowance=0.,gate='exact-zero'))
                        if condition=='base':conversion.append(dict(example_id=row['id'],site=site,**retention_evidence(value,storage_dtype='float16')))
                if engine.integrity()!=config['expected_base_hash']:raise ValueError('Frozen base identity changed during workload.')
                evidence=dict(condition=condition,base_unchanged=True,seconds=time.monotonic()-started,
                    exact_zero=all(r['max_absolute_error']==0 for r in comparison),comparisons=comparison,
                    retention=conversion,qualified=False)
                if condition=='base':
                    evidence['numeric_lens']=getattr(engine,'numeric_lens_evidence',None)
                    if runtime['execution_kind']=='real-cuda' and not evidence['numeric_lens']:
                        raise ValueError('Real diagnostic is missing numeric lens evidence.')
                folder=directory/'units'/condition
                _save_arrays(folder/'arrays.npz',arrays);_write_json(folder/'evidence.json',evidence)
                if condition!='base' and not evidence['exact_zero']:
                    raise CaptureFailure('Step-zero activation changed; exact coordinates saved in unit evidence.',
                        dict(worst=max(comparison,key=lambda r:r['max_absolute_error']),evidence_path=str((folder/'evidence.json').relative_to(root))))
                identity=dict(config_sha256=_hash(config),runtime=runtime,condition=condition)
                _write_json(folder/'complete.json',dict(identity=identity,files={name:file_hash(folder/name) for name in ['arrays.npz','evidence.json']}))
                checks[condition]=evidence
                if condition=='base':base=arrays
                completed+=1;print(f'Diagnostic: {completed}/3 sealed; {condition}; {evidence["seconds"]:.1f}s.',flush=True)
            if engine:engine.close();engine=None
            # Verify pooling and moments from the actual saved base arrays. Nonzero
            # storage/ranking/bootstrap sensitivity is a later declared workload.
            batches=[]
            for i,row in enumerate(config['examples']):
                values={s:base[f'{i}.{s}'] for s in SITES}
                batches.append(dict(example_ids=[row['id']],content_positions=[row['content_positions']],base=values,adapted=values))
            summary=summarize_pairs(batches,sites=SITES)
            _save_arrays(directory/'results/base-moments.npz',{k:summary['base_moments'][k] for k in ['sum','outer_sum']})
            result=dict(state='awaiting-review',diagnostic_complete=True,qualified=False,
                execution_kind=runtime['execution_kind'],sequence_forwards=12,model_calls=12,sites=29,
                base_moment_count=summary['base_moments']['count'],
                numeric_lens=checks['base'].get('numeric_lens'),
                checks={k:{field:v[field] for field in ['exact_zero','base_unchanged','seconds']} for k,v in checks.items()},
                limits=['Four-document zero check only; no nonzero-adapter sensitivity claim.',
                        'GPU numerical acceptance and nonzero FP16/ranking/bootstrap checks remain pending.'])
            _write_json(directory/'results/results.json',result)
            files=['meta/run_manifest.json','meta/runtime.json','results/results.json','results/base-moments.npz']
            for c in CONDITIONS:files.extend(f'units/{c}/{n}' for n in ['complete.json','evidence.json','arrays.npz'])
            _write_json(directory/'complete.json',dict(config_sha256=_hash(config),files={p:file_hash(directory/p) for p in files}))
            _write_state(directory,'awaiting-review',3,3)
            return result
        except BaseException as exc:
            _write_state(directory,'failed',completed,3,str(exc))
            errors=directory/'logs/errors.jsonl';errors.parent.mkdir(parents=True,exist_ok=True)
            with errors.open('a') as f:f.write(json.dumps(dict(condition=condition,error=str(exc),type=type(exc).__name__,diagnostics=getattr(exc,'diagnostics',{})))+'\n')
            raise
        finally:
            if engine:engine.close()
