"""CPU weight geometry of the three sampling-SFT arms across checkpoints 0-64.

Per adapted base matrix W0: the original Stage 0 FP64 SVD and its gates
(stage0_spectral_math.decompose), then, for every arm and saved step, the
fraction of the rank-one input factor a and output factor b inside the top
singular subspaces holding 30/50/70% of W0's squared-singular-value energy,
with the same k/d chance and 100-Gaussian controls. Rows share the original
principal-overlap fields so they join with its SFT/GRPO rows. Also: update
size ||dW||_F = ||a|| ||b|| (rank one, alpha one), cosine of each step's dW
with the endpoint's, and endpoint cross-arm dW / a / b cosines. Bases are not
retained; the decomposition is deterministic from the hashed base weights.
Geometry is descriptive, not causal.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from pilot_eval.run import _write_json, _write_state
from pilot_eval.sft import safe_name
from pilot_eval.stage0_sources import factors as read_factors, inside
from pilot_eval.stage0_spectral_math import LEVELS, chance_band, decompose, projection
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _hash, _save_frozen
from pilot_eval import sampling_stage0_evaluation as evaluation

ARMS = evaluation.ARMS
STEPS = (0, 8, 16, 32, 64)
STAGE0_BASE_SHA256 = '08d084a9ae96f21a2304339fbb8b1d00ae985210e94b481efd361d96a8a89289'
IMPLEMENTATION = ['sampling_stage0_geometry.py','sampling_stage0_evaluation.py','stage0_spectral_math.py',
                  'stage0_weights.py','stage0_sources.py','trace_readout_assets.py']
FIELDS = ['module_id','arm','step','energy_level','k','pi_out','pi_in','chance_out','chance_in']


def module_id(module):
    parts=module.split('.')
    return f'L{int(parts[4]):02d}.{parts[-1]}'


def checkpoints(root,behaviour):
    """Every saved step of every arm, hashed from its sealed training checkpoint."""
    result={}
    for arm in ARMS:
        folder=inside(root,behaviour['variants'][arm]['adapter_path']).parent
        for step in STEPS:
            cp=folder/f'checkpoint-{step}'
            seal=json.loads((cp/'complete.json').read_text())
            if seal.get('step')!=step or any(file_hash(cp/n)!=seal['files'][n] for n in ['adapter_config.json','adapter_model.safetensors']):
                raise ValueError(f'Training checkpoint payload changed: {arm}/{step}')
            result[f'{arm}:{step}']=dict(arm=arm,step=step,path=str(cp.relative_to(root)),
                adapter_sha256=file_hash(cp/'adapter_model.safetensors'),complete_sha256=file_hash(cp/'complete.json'))
    zero={v['adapter_sha256'] for k,v in result.items() if k.endswith(':0')}
    if len(zero)!=1 or zero!={behaviour['zero_adapter']['adapter_sha256']}:
        raise ValueError('Arms do not share the recorded checkpoint-0 adapter.')
    return result


def prepare(behaviour_plan,output_root,name,*,weights=None):
    from pilot_eval.stage0_weights import LocalWeights
    root=Path(output_root).resolve();safe_name(name)
    plan=inside(root,behaviour_plan);behaviour=json.loads(plan.read_text())
    if behaviour.get('protocol')!='sampling-stage0-evaluation-v2':raise ValueError('Expected the sampling behaviour plan.')
    evaluation.check_sources(behaviour,root)
    identity=behaviour['model_identity']
    inventory=(weights or LocalWeights()).inventory(inside(root,behaviour['weights_dir']),dict(identity=identity))
    modules={m:dict(path=str(Path(e['path']).resolve().relative_to(root)),key=e['key']) for m,e in inventory['modules'].items()}
    config=dict(protocol='sampling-stage0-geometry-v1',name=name,run_path=f'runs/stage0/{name}/geometry',
        behaviour_plan=str(plan.relative_to(root)),behaviour_sha256=file_hash(plan),
        parents=behaviour['parents'],training_plans=behaviour['training_plans'],cohort_path=behaviour['cohort_path'],
        cohort_sha256=behaviour['cohort_sha256'],weights_dir=behaviour['weights_dir'],model_snapshot=behaviour['model_snapshot'],
        variants=behaviour['variants'],adapter_files=behaviour['adapter_files'],model_identity=identity,
        base_sha256=inventory['base_sha256'],same_base_as_stage0=inventory['base_sha256']==STAGE0_BASE_SHA256,
        checkpoints=checkpoints(root,behaviour),modules=modules,levels=list(LEVELS),steps=list(STEPS),
        implementation={n:file_hash(Path(__file__).with_name(n)) for n in IMPLEMENTATION},
        execution_kind=inventory['execution_kind'],retained_bases=False)
    path=root/'plans'/name/'sampling-stage0.geometry.json';directory=inside(root,config['run_path'])
    _save_frozen(path,config);_save_frozen(directory/'config.json',config)
    if not (directory/'meta/status.json').exists():_write_state(directory,'prepared',0,len(modules))
    return path


def _load(path,root):
    root=Path(root).resolve();config=json.loads(inside(root,path).read_text())
    if config.get('protocol')!='sampling-stage0-geometry-v1':raise ValueError('Unexpected geometry plan.')
    evaluation.check_sources(config,root)
    if file_hash(inside(root,config['behaviour_plan']))!=config['behaviour_sha256']:raise ValueError('Behaviour plan changed.')
    for cp in config['checkpoints'].values():
        folder=inside(root,cp['path'])
        if file_hash(folder/'adapter_model.safetensors')!=cp['adapter_sha256'] or file_hash(folder/'complete.json')!=cp['complete_sha256']:
            raise ValueError(f'Checkpoint changed: {cp["path"]}')
    directory=inside(root,config['run_path'])
    if json.loads((directory/'config.json').read_text())!=config:raise ValueError('Geometry run config changed.')
    return root,config,directory


def _sealed(config,directory,module):
    folder=directory/'units'/module_id(module)
    if not (folder/'complete.json').exists():return None
    if json.loads((folder/'complete.json').read_text())!=dict(config_sha256=_hash(config),module=module,sha256=file_hash(folder/'result.json')):
        raise ValueError(f'Geometry unit changed: {module}')
    return json.loads((folder/'result.json').read_text())


def _cosine(x,y):
    x=np.asarray(x,dtype=np.float64).reshape(-1);y=np.asarray(y,dtype=np.float64).reshape(-1)
    n=float(np.linalg.norm(x)*np.linalg.norm(y))
    return None if n==0 else float(x@y/n)


def _delta_cosine(f,g):
    """Frobenius cosine of rank-one updates b a^T: (a.a')(b.b')/(|a||a'||b||b'|)."""
    a,b,c,d=(np.asarray(x,dtype=np.float64).reshape(-1) for x in (*f,*g))
    n=float(np.linalg.norm(a)*np.linalg.norm(b)*np.linalg.norm(c)*np.linalg.norm(d))
    return None if n==0 else float((a@c)*(b@d)/n)


def unit(module,matrix,adapters):
    """Pure CPU measurement for one adapted base matrix across every arm/step."""
    arrays,evidence=decompose(matrix);rows=[]
    for level in LEVELS:
        k=evidence[f'k{level}'];u=arrays['u'][:,:k];v=arrays['v'][:,:k]
        gaussian_out,gaussian_in=chance_band(u),chance_band(v)
        for key,adapter in adapters.items():
            arm,step=key.split(':');a,b=adapter[module]
            if matrix.shape!=(b.shape[0],a.shape[1]):raise ValueError('Base weight and adapter dimensions differ.')
            rows.append(dict(module=module,module_id=module_id(module),arm=arm,step=int(step),energy_level=level,k=k,
                pi_out=projection(b,u),pi_in=projection(a,v),boundary_gap=evidence['gaps'][str(level)],
                undefined_out_reason='zero factor' if not np.any(b) else None,
                undefined_in_reason='zero factor' if not np.any(a) else None,
                chance_out=k/u.shape[0],chance_in=k/v.shape[0],gaussian_out=gaussian_out,gaussian_in=gaussian_in))
    norms=[];cross=[]
    for key,adapter in adapters.items():
        arm,step=key.split(':');a,b=adapter[module]
        norms.append(dict(module=module,module_id=module_id(module),arm=arm,step=int(step),
            a_norm=float(np.linalg.norm(a.astype(float))),b_norm=float(np.linalg.norm(b.astype(float))),
            delta_frobenius=float(np.linalg.norm(a.astype(float))*np.linalg.norm(b.astype(float))),
            delta_cosine_to_endpoint=_delta_cosine(adapter[module],adapters[f'{arm}:64'][module])))
    for i,x in enumerate(ARMS):
        for y in ARMS[i+1:]:
            f,g=adapters[f'{x}:64'][module],adapters[f'{y}:64'][module]
            cross.append(dict(module=module,module_id=module_id(module),left=x,right=y,step=64,
                delta_cosine=_delta_cosine(f,g),a_cosine=_cosine(f[0],g[0]),b_cosine=_cosine(f[1],g[1])))
    return dict(module=module,spectrum={k:v for k,v in evidence.items()},projections=rows,norms=norms,cross_arm=cross)


def summarize(config,values):
    projections=[r for v in values for r in v['projections']];norms=[r for v in values for r in v['norms']]
    cross=[r for v in values for r in v['cross_arm']];trajectory=[]
    for arm in ARMS:
        for step in STEPS:
            selected=[r for r in norms if r['arm']==arm and r['step']==step]
            trajectory.append(dict(arm=arm,step=step,total_delta_frobenius=float(np.sqrt(sum(r['delta_frobenius']**2 for r in selected))),
                sum_delta_frobenius=float(sum(r['delta_frobenius'] for r in selected)),modules=len(selected)))
    endpoint=[]
    for arm in ARMS:
        for level in LEVELS:
            rows=[r for r in projections if r['arm']==arm and r['step']==64 and r['energy_level']==level]
            out=[r['pi_out'] for r in rows if r['pi_out'] is not None];inn=[r['pi_in'] for r in rows if r['pi_in'] is not None]
            endpoint.append(dict(arm=arm,energy_level=level,modules=len(rows),mean_pi_out=float(np.mean(out)),
                mean_pi_in=float(np.mean(inn)),mean_chance_out=float(np.mean([r['chance_out'] for r in rows])),
                mean_chance_in=float(np.mean([r['chance_in'] for r in rows]))))
    pairs=[]
    for i,x in enumerate(ARMS):
        for y in ARMS[i+1:]:
            rows=[r for r in cross if r['left']==x and r['right']==y and r['delta_cosine'] is not None]
            pairs.append(dict(left=x,right=y,modules=len(rows),mean_delta_cosine=float(np.mean([r['delta_cosine'] for r in rows]))))
    return dict(state='complete',modules=len(values),rows=len(projections),trajectory=trajectory,endpoint_overlap=endpoint,
        cross_arm=pairs,base_sha256=config['base_sha256'],same_base_as_stage0=config['same_base_as_stage0'],
        execution_kind=config['execution_kind'],gaussian_draws=100,
        limits=['one training seed per arm','projections are descriptive geometry, not causal evidence',
                'arms trained on different covered subsets of the same problems'])


def run(path,output_root,*,weights=None,load_factors=read_factors,max_new_units=None):
    from pilot_eval.stage0_weights import LocalWeights
    root,config,directory=_load(path,output_root);deps=weights or LocalWeights()
    with run_lock(directory):
        if (directory/'complete.json').exists():return verify(path,root)
        adapters={key:load_factors(inside(root,cp['path'])) for key,cp in config['checkpoints'].items()}
        values=[];created=0
        try:
            for module,entry in config['modules'].items():
                value=_sealed(config,directory,module)
                if value is None:
                    if max_new_units is not None and created>=max_new_units:
                        _write_state(directory,'incomplete',len(values),len(config['modules']))
                        return dict(state='incomplete',completed=len(values))
                    begin=time.monotonic()
                    value=unit(module,deps.matrix({**entry,'path':str(inside(root,entry['path']))}),adapters)
                    value['seconds']=time.monotonic()-begin
                    folder=directory/'units'/module_id(module);_write_json(folder/'result.json',value)
                    _write_json(folder/'complete.json',dict(config_sha256=_hash(config),module=module,sha256=file_hash(folder/'result.json')))
                    created+=1
                    print(f'Sampling geometry: {len(values)+1}/{len(config["modules"])} {module_id(module)} {value["seconds"]:.1f}s',flush=True)
                values.append(value);_write_state(directory,'running',len(values),len(config['modules']))
            return _finish(config,directory,values)
        except BaseException as exc:
            _write_state(directory,'failed',len(values),len(config['modules']),error=str(exc))
            raise


def _finish(config,directory,values):
    results=directory/'results';results.mkdir(parents=True,exist_ok=True)
    rows=[r for v in values for r in v['projections']]
    (results/'projections.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    with (results/'projections.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    (results/'norms.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for v in values for r in v['norms']))
    (results/'cross-arm.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for v in values for r in v['cross_arm']))
    summary=summarize(config,values);_write_json(results/'summary.json',summary)
    files=['config.json']+[f'results/{n}' for n in ['projections.jsonl','projections.csv','norms.jsonl','cross-arm.jsonl','summary.json']]
    files+=[f'units/{module_id(m)}/complete.json' for m in config['modules']]
    _write_json(directory/'complete.json',dict(config_sha256=_hash(config),files={f:file_hash(directory/f) for f in files}))
    _write_state(directory,'complete',len(values),len(values))
    return summary


def verify(path,output_root):
    root,config,directory=_load(path,output_root)
    seal=json.loads((directory/'complete.json').read_text())
    if seal['config_sha256']!=_hash(config):raise ValueError('Geometry completion identity changed.')
    for name,digest in seal['files'].items():
        if file_hash(directory/name)!=digest:raise ValueError(f'Geometry evidence changed: {name}')
    values=[_sealed(config,directory,m) for m in config['modules']]
    if any(v is None for v in values):raise ValueError('Geometry evidence is incomplete.')
    summary=json.loads((directory/'results/summary.json').read_text())
    if summary!=json.loads(json.dumps(summarize(config,values))):raise ValueError('Geometry summary differs from sealed units.')
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--behaviour-plan',required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--name',required=True)
    for command in ['run','verify']:
        p=sub.add_parser(command);p.add_argument('--config',required=True);p.add_argument('--output-root',required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare':result=str(prepare(args.behaviour_plan,args.output_root,args.name))
    elif args.command=='run':result=run(args.config,args.output_root)
    else:result=verify(args.config,args.output_root)
    print(json.dumps(result,indent=2,sort_keys=True,default=str))


if __name__=='__main__':main()
