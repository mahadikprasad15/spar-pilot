"""Normalize verified saved products without inventing cross-session agreement."""
import json

import numpy as np

from pilot_eval.activation_prepare import VIEWS,PROJECTIONS
from pilot_eval.stage0_sources import inside,factors
from pilot_eval.stage0_trajectory_math import comparison_group
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash


def scheduling(batch_size,*,batches=None):
    value=dict(batch_size=batch_size,padding='right',position_ids='attention-cumsum-minus-one')
    if batch_size!=1:value['membership']=[b.get('example_ids',b.get('ids')) for b in batches]
    return value


def identity(config,runtime,batch_size,product,*,batches=None):
    value=dict(runtime=runtime,inputs_sha256=config['items_sha256'],model_identity=config['model_identity'],scheduling=scheduling(batch_size,batches=batches))
    return dict(**value,comparison_group=comparison_group(value,product))


def state_aliases(config,digest):
    return [s['alias'] for s in config['states'] if s['state']=='verified-adapter-payload' and s['adapter_sha256']==digest]


def drift_records(measurement,batches,rows,config,product):
    by_id={r['id']:r for r in rows};documents=[]
    for unit,result in zip(measurement['units'],batches):
        variant=unit['variant']
        digest=measurement['adapter_files'][variant]['adapter_model.safetensors']
        aliases=state_aliases(config,digest)
        if variant in aliases:aliases=[variant]
        if not aliases:continue  # edited variants are not checkpoint trajectories
        meta=result['metadata'];arrays=result['arrays']
        for ei,identifier in enumerate(meta['example_ids']):
            row=by_id[identifier]
            for vi,view in enumerate(VIEWS):
                if not any(row['masks'][view]):continue
                pos=[i for i,p in enumerate(meta['predictions']) if p['example']==ei and p['view']==view]
                writes=[dict(module=module,count=sum(row['masks'][view]),ratio_sum=float(s[2]),update_squared_sum=float(s[0]),base_squared_sum=float(s[1])) for module,s in zip(measurement['modules'],arrays['write_sums'][ei,vi])]
                for alias in aliases:documents.append(dict(alias=alias,id=identifier,corpus=row['corpus'],view=view,kl_values=arrays['kl'][pos].tolist(),writes=writes,product=product,
                    kl_positions=[meta['predictions'][p] for p in pos],write_positions=[i for i,b in enumerate(row['masks'][view]) if b]))
    return documents


def read_product(entry,config,root):
    path=inside(root,entry['config']);product=entry['key'];kind=entry['kind']
    if kind=='trajectory-collection':
        from pilot_eval.stage0_trajectory_collect import verify,load
        verify(path,root);source,directory,parent,_,_=load(path,root)
        if source['items_sha256']!=config['items_sha256'] or source['model_identity']!=config['model_identity']:raise ValueError('Saved trajectory collection cohort/base differs.')
        documents=[]
        for line in (directory/'results/per-document.jsonl').read_text().splitlines():
            r=json.loads(line);digest=source['adapter_files'][r['alias']]['adapter_model.safetensors']
            for alias in state_aliases(config,digest):documents.append(dict(r,alias=alias,product=product))
        return dict(documents=documents,identity=source['comparison_identity'],completion_path=str((directory/'complete.json').relative_to(root)),completion_sha256=file_hash(directory/'complete.json'),coefficient_policy=source['coefficient_policy'])
    if kind=='stage0-drift':
        from pilot_eval.stage0_drift import _load,_verify,_saved,_rows
        source,directory=_load(path,root);_verify(source,directory,root);rows=_rows(source,root)
        if source['items_sha256']!=config['items_sha256'] or source['model_identity']!=config['model_identity']:raise ValueError('Saved drift cohort/base differs from trajectory sources.')
        result=drift_records(source,_saved(source,directory,root),rows,config,product)
        return dict(documents=result,identity=identity(config,source['runtime'],source['batch_size'],product),
            completion_path=str((directory/'complete.json').relative_to(root)),completion_sha256=file_hash(directory/'complete.json'),
            coefficient_policy='no production raw coefficients; strict diagnostic repeat/chunk/reconstruction unchanged')
    if kind=='pilot4-tokens':
        from pilot_eval.token_measurement_workflow import verify_tokens,_execution,_units,_read_unit,COMPARISON_POLICY
        verify_tokens(path,root);source,products=_execution(path,root);documents=[]
        for arm,prepared,_batch,step,rows,folder,unit in _units(source,products,root):
            if prepared['items_sha256']!=config['items_sha256'] or prepared['source_evidence']['base_sha256']!=config['model_identity']['base_sha256'] or any(prepared[k]!=config['model_identity'][k] for k in ['model','model_revision','tokenizer_revision','dtype']):raise ValueError('Historical KL corpus/base differs.')
            digest=prepared['source_evidence']['checkpoints'][str(step)]['adapter_sha256'];aliases=state_aliases(config,digest)
            result=_read_unit(folder,unit,rows,step,source['settings']);meta=result['metadata']
            for ei,row in enumerate(rows):
                for view in VIEWS:
                    if not any(row['masks'][view]):continue
                    pos=[i for i,p in enumerate(meta['predictions']) if p['example']==ei and p['view']==view]
                    for alias in aliases:documents.append(dict(alias=alias,id=row['id'],corpus=row['corpus'],view=view,kl_values=result['arrays']['kl'][pos].tolist(),writes=None,product=product,kl_positions=[meta['predictions'][p] for p in pos],write_positions=None))
        first=products['sft'];runtime=json.loads((root/source['run_path']/'meta/runtime.json').read_text())
        return dict(documents=documents,identity=identity(config,runtime,first[0]['batch_size'],product,batches=source['batches']),
            completion_path=str((root/source['run_path']/'complete.json').relative_to(root)),completion_sha256=file_hash(root/source['run_path']/'complete.json'),
            coefficient_policy=COMPARISON_POLICY,raw_coefficient_use='not regraded or reinterpreted; historical descriptive sensitivity policy retained')
    if kind=='pilot3-writes':
        from pilot_eval.activation_measurement import load_execution,verify_measurement,iter_measurement_batches
        verify_measurement(path,root);source,prepared,rows=load_execution(path,root)
        if prepared['items_sha256']!=config['items_sha256'] or prepared['source_evidence']['base_sha256']!=config['model_identity']['base_sha256'] or any(prepared[k]!=config['model_identity'][k] for k in ['model','model_revision','tokenizer_revision','dtype']):raise ValueError('Historical writes corpus/base differs.')
        documents=[];by_id={r['id']:r for r in rows}
        for step in source['checkpoint_steps']:
            aliases=state_aliases(config,prepared['source_evidence']['checkpoints'][str(step)]['adapter_sha256'])
            for batch in iter_measurement_batches(path,root,step=step):
                arrays=batch['arrays']
                for ei,identifier in enumerate(batch['example_ids']):
                    row=by_id[identifier]
                    for vi,view in enumerate(VIEWS):
                        if not any(row['masks'][view]):continue
                        writes=[]
                        for layer in range(28):
                            for pi,projection in enumerate(PROJECTIONS):
                                module=f'base_model.model.model.layers.{layer}.{"self_attn" if projection in PROJECTIONS[:4] else "mlp"}.{projection}'
                                if module not in config['modules']:continue
                                writes.append(dict(module=module,count=int(arrays['module_defined_count'][ei,vi,layer,pi]),ratio_sum=float(arrays['module_ratio_sum'][ei,vi,layer,pi]),update_squared_sum=None,base_squared_sum=None))
                        for alias in aliases:documents.append(dict(alias=alias,id=identifier,corpus=row['corpus'],view=view,kl_values=None,writes=writes,product=product,kl_positions=None,write_positions=[i for i,b in enumerate(row['masks'][view]) if b]))
        directory=root/source['run_path']
        return dict(documents=documents,identity=identity(config,source['runtime'],source['batch_size'],product,batches=source['batches']),
            completion_path=str((directory/'complete.json').relative_to(root)),completion_sha256=file_hash(directory/'complete.json'),
            coefficient_policy='direct effective-write ratios only; no raw coefficient gate changes',rms_available=False)
    raise ValueError('Unsupported trajectory product kind: '+kind)


def merge_products(products):
    """Only identical proven sessions/scheduling may supply complementary fields."""
    records={}
    for product in products:
        group=product['identity']['comparison_group']
        for row in product['documents']:
            key=(group,row['alias'],row['id'],row['view']);old=records.get(key)
            if old is None:
                records[key]=dict(row,comparison_group=group,products=[row['product']]);continue
            for field in ['kl_values','writes','kl_positions','write_positions']:
                if old[field] is None:old[field]=row[field]
                elif row[field] is not None and old[field]!=row[field]:raise ValueError('Conflicting same-runtime checkpoint evidence; keep products separate for diagnosis.')
            old['products'].append(row['product'])
    return list(records.values())
