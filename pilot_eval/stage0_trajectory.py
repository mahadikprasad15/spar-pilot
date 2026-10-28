"""CPU checkpoint inventory, geometry and verified saved-product import."""
import json
from pathlib import Path

import numpy as np

from pilot_eval.run import _write_json
from pilot_eval.sft import safe_name,read_artifact
from pilot_eval.stage0_sources import inside,factors,checkpoint,audit_source
from pilot_eval.stage0_spectral import _load as spectral_load,_verify_cache
from pilot_eval.stage0_spectral_math import projection,LEVELS
from pilot_eval.stage0_drift_math import positions
from pilot_eval.stage0_evaluation import _jsonl,_recover_owner
from pilot_eval.stage0_workflow import _writer,_state,_verify_parents,budget_status
from pilot_eval.stage0_trajectory_sources import read_product
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash,_save_frozen


def implementation():
    folder=Path(__file__).parent
    return {n:file_hash(folder/n) for n in ['stage0_trajectory.py','stage0_trajectory_sources.py','stage0_trajectory_math.py','stage0_trajectory_report.py','stage0_trajectory_collect.py','stage0_drift_backend.py','stage0_drift_math.py','stage0_evaluation_backend.py']}


def prepare(spectral_path,inputs_path,selection_path,output_root,name):
    root=Path(output_root).resolve();safe_name(name);sp,ip,sel=(inside(root,p) for p in [spectral_path,inputs_path,selection_path])
    spectral,audit=spectral_load(sp,root);_verify_cache(spectral,inside(root,spectral['run_path']))
    selected=json.loads(sel.read_text())
    if set(selected)-{'products','random','evaluation'} or not isinstance(selected.get('products',[]),list):raise ValueError('Select products, optional random source and optional saved evaluation.')
    real=spectral['inventory']['execution_kind']!='synthetic-cpu-fixture'
    if real and ip!=inside(root,audit['sources']['sft']['path']):raise ValueError('Use the original audited fixed-input product.')
    prepared=json.loads(ip.read_text());rows=read_artifact(root,prepared['items_path']);positions(rows)
    if _hash(rows)!=prepared['items_sha256'] or len(rows)!=300 or len({r['id'] for r in rows})!=300 or {c:sum(r['corpus']==c for r in rows) for c in ['gsm8k','fineweb']}!={'gsm8k':150,'fineweb':150}:raise ValueError('Expected original150 GSM8K and150 FineWeb sequences.')
    model=audit['sources']['sft']['identity']
    if any(prepared.get(k)!=model[k] for k in ['model','model_revision','tokenizer_revision','dtype']):raise ValueError('Trajectory corpus/base identity differs.')
    parents={str(p.relative_to(root)):file_hash(p) for p in [sp,ip,sel,inside(root,prepared['items_path'])]};states=[]
    for arm in ['sft','grpo']:
        original=json.loads(inside(root,audit['sources'][arm]['path']).read_text())
        for step in [0,8,16,32,64]:
            evidence=original['source_evidence'].get('checkpoints',{}).get(str(step))
            result=checkpoint(root,evidence,str(step),parents)
            state=dict(alias=f'{arm}-{step}',arm=arm,step=step,**result)
            if result['state']=='verified-adapter-payload':
                state['adapter_sha256']=file_hash(inside(root,result['path'])/'adapter_model.safetensors');state['sum_factor_norm_products']=float(sum(result['update_norms'].values()));state['total_frobenius_size']=float(np.linalg.norm(list(result['update_norms'].values())))
            states.append(state)
    if selected.get('random'):
        rp=inside(root,selected['random'])
        if rp.exists():
            random=audit_source(root,'random',rp);parents.update(random['parents'])
            if random.get('identity')!=model:raise ValueError('Random trajectory control base differs.')
            result=random['checkpoints']['64'];state=dict(alias='random-64',arm='random',step=64,**result)
            if result['state']=='verified-adapter-payload':state.update(adapter_sha256=file_hash(inside(root,result['path'])/'adapter_model.safetensors'),sum_factor_norm_products=float(sum(result['update_norms'].values())),total_frobenius_size=float(np.linalg.norm(list(result['update_norms'].values()))))
            states.append(state)
        else:states.append(dict(alias='random-64',arm='random',step=64,state='missing',path=str(rp.relative_to(root))))
    else:states.append(dict(alias='random-64',arm='random',step=64,state='not-selected'))
    products=[]
    for index,entry in enumerate(selected.get('products',[])):
        if set(entry)!={'kind','config'} or entry['kind'] not in ['stage0-drift','pilot4-tokens','pilot3-writes','trajectory-collection']:raise ValueError('Select supported saved-workflow configs, not report summaries.')
        source=inside(root,entry['config']);product=dict(key=f'product-{index:03d}',**entry,state='missing')
        if source.exists():
            source_config=json.loads(source.read_text());directory=inside(root,source_config['run_path'])
            if (directory/'complete.json').exists():
                product['state']='ready';parents[str(source.relative_to(root))]=file_hash(source);parents[str((directory/'complete.json').relative_to(root))]=file_hash(directory/'complete.json')
            else:product['state']='incomplete'
        products.append(product)
    evaluation=selected.get('evaluation');evaluation_state='not-selected'
    if evaluation:
        if isinstance(evaluation,str):evaluation=dict(kind='stage0-evaluation',config=evaluation)
        if set(evaluation)!={'kind','config'} or evaluation['kind'] not in ['stage0-evaluation','behaviour']:raise ValueError('Select a verified stage0-evaluation or behaviour workflow for saved lengths.')
        ep=inside(root,evaluation['config'])
        if ep.exists():
            ev=json.loads(ep.read_text());ed=inside(root,ev['run_path']);evaluation_state='ready' if (ed/'complete.json').exists() else 'incomplete'
            if evaluation_state=='ready':
                parents[str(ep.relative_to(root))]=file_hash(ep);parents[str((ed/'complete.json').relative_to(root))]=file_hash(ed/'complete.json')
        else:evaluation_state='missing'
    units=[dict(key='geometry-'+s['alias'],kind='geometry',alias=s['alias']) for s in states if s['state']=='verified-adapter-payload']+[
        dict(key=p['key'],kind='product',product=p) for p in products if p['state']=='ready']
    directory=root/f'runs/stage0/{name}/trajectory';path=root/'plans'/name/'stage0.trajectory.json'
    config=dict(protocol='stage0-trajectory-v1',schema_version=1,name=name,run_path=str(directory.relative_to(root)),spectral_path=str(sp.relative_to(root)),inputs_path=str(ip.relative_to(root)),
        selection_path=str(sel.relative_to(root)),parents=parents,implementation=implementation(),numpy_version=np.__version__,states=states,products=products,
        evaluation=evaluation,evaluation_state=evaluation_state,model_identity=model,weights_dir=spectral['weights_dir'],modules=list(spectral['inventory']['modules']),
        execution_kind='real-cuda' if real else 'synthetic-cpu-fixture',items_sha256=_hash(rows),energy_levels=list(LEVELS),units=units,
        comparisons='measured checkpoint trajectories; incompatible or unknown sessions separated; no causal direction inference',
        raw_coefficient_policy='historical batch sensitivity remains descriptive; repeat/chunk/KL/reconstruction/integrity gates unchanged',
        approved_axes=dict(primary='sqrt(sum module norm-product squared), total Frobenius size',secondary='equal-module mean of per-module token-mean effective-write ratio',additional='sum module norm products'),
        approved_analysis=dict(primary_KL='FineWeb per-token full-vocabulary KL from Ticket06',ratio='token-mean KL / total Frobenius size squared',slope='within-run log-log fitted and adjacent slopes; near2 unclassified, no direction-sensitivity claim',zero='separate labelled linear-origin inset; no epsilon replacement',rescaling_followup='proposal only, do not implement'),
        statistics=dict(draws=10000,seed=42,percentiles=[2.5,97.5],unit='whole document/item',intervals='pointwise, not simultaneous'),budget_ceiling_units=20)
    with _writer(directory):
        _save_frozen(path,config);_save_frozen(directory/'config.json',config);_jsonl(directory/'inputs/items.jsonl',rows)
        _save_frozen(directory/'meta/run_manifest.json',dict(config_sha256=_hash(config),resources='CPU audit/geometry/import/report; optional separately budgeted GPU collection',source_states={s['alias']:s['state'] for s in states}))
        if not (directory/'meta/status.json').exists():_state(directory,'prepared',0,len(units))
    return path


def load(path,root):
    config=json.loads(inside(root,path).read_text());directory=inside(root,config['run_path'])
    if config.get('protocol')!='stage0-trajectory-v1' or config['implementation']!=implementation() or config['numpy_version']!=np.__version__:raise ValueError('Trajectory implementation/environment changed; preserve results and select a new plan.')
    if json.loads((directory/'config.json').read_text())!=config or json.loads((directory/'meta/run_manifest.json').read_text())['config_sha256']!=_hash(config):raise ValueError('Frozen trajectory identity changed.')
    _verify_parents(root,config['parents']);spectral,audit=spectral_load(inside(root,config['spectral_path']),root);_verify_cache(spectral,inside(root,spectral['run_path']))
    rows=[json.loads(x) for x in (directory/'inputs/items.jsonl').read_text().splitlines()]
    if _hash(rows)!=config['items_sha256']:raise ValueError('Trajectory fixed inputs changed.')
    return config,directory,rows


def read_unit(config,directory,unit):
    folder=directory/'units'/unit['key'];marker=folder/'complete.json'
    if not marker.exists():return None
    seal=json.loads(marker.read_text())
    if seal.get('identity')!=dict(config_sha256=_hash(config),unit_sha256=_hash(unit)) or seal.get('files')!={'result.json':file_hash(folder/'result.json')}:raise ValueError('Trajectory CPU unit changed.')
    return json.loads((folder/'result.json').read_text())


def run(path,output_root,*,max_new_units=None,progress=True):
    root=Path(output_root).resolve();config,directory,_=load(path,root)
    if max_new_units is not None and (type(max_new_units) is not int or max_new_units<0):raise ValueError('Invalid CPU unit limit.')
    with _writer(directory):
        spectral,_=spectral_load(inside(root,config['spectral_path']),root);cache=_verify_cache(spectral,inside(root,spectral['run_path']));new=0
        for unit in config['units']:
            if read_unit(config,directory,unit) is not None:continue
            if max_new_units is not None and new>=max_new_units:break
            try:
                if unit['kind']=='product':result=read_product(unit['product'],config,root)
                else:
                    state=next(s for s in config['states'] if s['alias']==unit['alias']);values=factors(inside(root,state['path']));result=[]
                    for row in cache['module_results']:
                        a,b=values[row['module']]
                        with np.load(inside(root,spectral['run_path'])/row['arrays_path'],allow_pickle=False) as basis:
                            for level in config['energy_levels']:
                                k=row[f'k{level}'];result.append(dict(alias=state['alias'],arm=state['arm'],step=state['step'],module=row['module'],energy_level=level,k=k,
                                    pi_in=projection(a,basis['v'][:,:k]),pi_out=projection(b,basis['u'][:,:k]),chance_in=k/row['d_in'],chance_out=k/row['d_out'],
                                    norm_product=float(np.linalg.norm(a.astype(float))*np.linalg.norm(b.astype(float)))))
                folder=directory/'units'/unit['key'];_write_json(folder/'result.json',result)
                _write_json(folder/'complete.json',dict(identity=dict(config_sha256=_hash(config),unit_sha256=_hash(unit)),files={'result.json':file_hash(folder/'result.json')}));new+=1
                done=sum(read_unit(config,directory,u) is not None for u in config['units']);_state(directory,'running',done,len(config['units']))
                if progress:print(f'Stage0 trajectory CPU: {done}/{len(config["units"])} units sealed; {unit["key"]}.',flush=True)
            except (Exception,KeyboardInterrupt) as exc:
                _write_json(directory/'diagnostics'/f'{unit["key"]}.json',dict(error=str(exc),unit=unit));_state(directory,'failed',sum(read_unit(config,directory,u) is not None for u in config['units']),len(config['units']),error=str(exc));raise
        done=sum(read_unit(config,directory,u) is not None for u in config['units'])
        if done==len(config['units']):_write_json(directory/'cpu.complete.json',dict(config_sha256=_hash(config),files={f'units/{u["key"]}/complete.json':file_hash(directory/'units'/u['key']/'complete.json') for u in config['units']}))
        return _state(directory,'cpu-complete' if done==len(config['units']) else 'incomplete',done,len(config['units']))


def verify_cpu(path,root):
    config,directory,_=load(path,root);seal=json.loads((directory/'cpu.complete.json').read_text())
    if seal['config_sha256']!=_hash(config):raise ValueError('CPU trajectory completion changed.')
    for p,digest in seal['files'].items():
        if file_hash(inside(directory,p))!=digest:raise ValueError('CPU trajectory completion payload changed.')
    results=[read_unit(config,directory,u) for u in config['units']]
    if any(r is None for r in results):raise ValueError('Trajectory CPU inputs incomplete.')
    for u,r in zip(config['units'],results):
        if u['kind']=='product':
            # Revalidate raw saved-workflow units, not just copied summaries.
            if read_product(u['product'],config,root)!=r:raise ValueError('Saved-product reconstruction changed.')
    return config,directory,results


def status(path,output_root):
    root=Path(output_root).resolve();config,directory,_=load(path,root)
    return dict(cpu_completed=sum(read_unit(config,directory,u) is not None for u in config['units']),cpu_total=len(config['units']),states={s['alias']:s['state'] for s in config['states']},products=config['products'],budget=budget_status(root),scientific_completion=False)


def recover_writer(path,output_root,owner_token,review_notes):
    root=Path(output_root).resolve();_,directory,_=load(path,root);return _recover_owner(directory,owner_token,review_notes)
