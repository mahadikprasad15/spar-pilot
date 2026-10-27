"""Frozen base-input corpus and independently checked adapter decompositions."""
import json
from pathlib import Path

import numpy as np

from pilot_eval.activation_profile import AGREEMENT
from pilot_eval.run import _write_json
from pilot_eval.sft import safe_name,read_artifact
from pilot_eval.stage0_energy_math import parallel_factors,projection_check
from pilot_eval.stage0_spectral_math import projection
from pilot_eval.stage0_sources import inside,factors,audit_source
from pilot_eval.stage0_spectral import _load as spectral_load,_verify_cache
from pilot_eval.stage0_qualification import _load as qualification_load,_verify as qualification_verify
from pilot_eval.stage0_workflow import _writer,_state,_verify_parents
from pilot_eval.stage0_drift_math import positions
from pilot_eval.stage0_evaluation import _jsonl
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash,_save_frozen


def implementation():
    folder=Path(__file__).parent
    return {n:file_hash(folder/n) for n in ['stage0_energy.py','stage0_energy_plan.py','stage0_energy_math.py','stage0_energy_backend.py','stage0_energy_report.py','stage0_spectral_math.py','stage0_evaluation_backend.py']}


def prepare(spectral_path,qualification_path,inputs_path,random_path,output_root,name,*,batches_per_unit=8):
    root=Path(output_root).resolve();safe_name(name)
    sp,qp,ip,rp=(inside(root,p) for p in [spectral_path,qualification_path,inputs_path,random_path])
    path=root/'plans'/name/'stage0.energy.json';directory=root/f'runs/stage0/{name}/input-energy'
    if path.exists():
        config,_,_=load(path,root)
        if [config[k] for k in ['spectral_path','qualification_path','inputs_path','random_path']]!=[str(p.relative_to(root)) for p in [sp,qp,ip,rp]] or config['batches_per_unit']!=batches_per_unit:raise ValueError('Energy selection changed; select a new name.')
        return path
    if type(batches_per_unit) is not int or not 1<=batches_per_unit<=150:raise ValueError('Positive save-group size up to150 required.')
    spectral,audit=spectral_load(sp,root);cache=_verify_cache(spectral,inside(root,spectral['run_path']))
    qualification,qd=qualification_load(qp,root);diagnostic=qualification_verify(qualification,qd)
    if not diagnostic['eligible_for_review']:raise ValueError('A successful reviewed Ticket04 diagnostic is required.')
    receipts=[]
    for receipt in sorted((qd/'reviews').glob('*/qualification.json')):
        value=json.loads(receipt.read_text());review=value.get('review',{})
        if value.get('qualified') is True and value.get('config_sha256')==_hash(qualification) and receipt.parent.name==_hash(review) and review.get('diagnostic_sha256')==file_hash(qd/'complete.json') and review.get('absolute_gate')==diagnostic['proposed_absolute_gate']:receipts.append(receipt)
    if not receipts:raise ValueError('Record the Ticket04 numerical review before preparing energy collection.')
    for key in ['model','model_revision','base_sha256']:
        if qualification['model_identity'][key]!=audit['sources']['sft']['identity'][key]:raise ValueError('Qualification and spectral base identity differ.')
    if spectral['inventory']['execution_kind'] not in ['synthetic-cpu-fixture','cpu-local-pinned-weights']:raise ValueError('Unknown weight-inventory execution kind.')
    real=spectral['inventory']['execution_kind']!='synthetic-cpu-fixture'
    if (diagnostic['execution_kind']=='real-cuda')!=real:raise ValueError('Real weight sources require actual GPU qualification; fixture receipts do not qualify them.')
    if real and ip!=inside(root,audit['sources']['sft']['path']):raise ValueError('Use the audited original SFT fixed-input product.')
    prepared=json.loads(ip.read_text());rows=read_artifact(root,prepared['items_path'])
    if _hash(rows)!=prepared['items_sha256'] or len(rows)!=300 or len({r['id'] for r in rows})!=300 or {c:sum(r['corpus']==c for r in rows) for c in ['gsm8k','fineweb']}!={'gsm8k':150,'fineweb':150}:raise ValueError('Expected the original verified 150 GSM8K and150 FineWeb sequences.')
    positions(rows)
    if any(prepared.get(k)!=qualification['model_identity'].get(k) for k in ['model','model_revision','tokenizer_revision','dtype']):raise ValueError('Fixed-input model/tokenizer identity differs.')
    random=audit_source(root,'random',rp)
    if random.get('state')!='audited' or random['checkpoints']['64']['state']!='verified-adapter-payload':raise ValueError('Restore the saved audited random-64 control; do not generate a replacement.')
    if random['identity']!=audit['sources']['grpo']['identity']:raise ValueError('Random control base identity differs.')
    labels=['sft-64','grpo-64','sft-0','grpo-0','random-64']
    source_paths={label:(random if label.startswith('random') else audit['sources'][label.split('-')[0]])['checkpoints'][label.split('-')[1]]['path'] for label in labels}
    values={label:factors(inside(root,p)) for label,p in source_paths.items()};modules=list(spectral['inventory']['modules'])
    parents={str(p.relative_to(root)):file_hash(p) for p in [sp,qp,ip,rp,receipts[0],qd/'complete.json',inside(root,prepared['items_path'])]}
    parents.update(random['parents']);units=[]
    for corpus in ['gsm8k','fineweb']:
        ids=[r['id'] for r in rows if r['corpus']==corpus]
        for start in range(0,len(ids),batches_per_unit):units.append(dict(key=f'{corpus}-{start//batches_per_unit:03d}',corpus=corpus,ids=ids[start:start+batches_per_unit]))
    factor_files={};projection_checks=[];geometry=[];levels=[30,50,70]
    with _writer(directory):
        try:
            for index,module in enumerate(modules):
                print(f'Stage0 energy prepare: {index}/{len(modules)} module projections checked; {module}.',flush=True)
                row=next(r for r in cache['module_results'] if r['module']==module)
                with np.load(inside(root,spectral['run_path'])/row['arrays_path'],allow_pickle=False) as basis:
                    a=np.stack([values[label][module][0].reshape(-1) for label in labels]).astype(np.float64)
                    retained_v=basis['v'].astype(np.float64);gram=retained_v.T@retained_v
                    parallels=[]
                    for level in levels:
                        k=row[f'k{level}'];v=retained_v[:,:k];par=parallel_factors(a,v);parallels.append(par)
                        for fi,label in enumerate(labels):
                            pi_in=projection(a[fi],v);check=projection_check(a[fi],par[fi],v,pi_in,AGREEMENT,gram=gram[:k,:k])
                            projection_checks.append(dict(module=module,factor=label,energy_level=level,k=k,**check))
                            geometry.append(dict(module=module,factor=label,energy_level=level,k=k,pi_in=pi_in,
                                pi_out=projection(values[label][module][1],basis['u'][:,:k]),chance_in=k/row['d_in'],chance_out=k/row['d_out']))
                folder=directory/'inputs/factors';folder.mkdir(parents=True,exist_ok=True);payload=folder/f'{index:03d}.npz'
                np.savez(payload,a=a,parallel=np.stack(parallels,axis=1))
                factor_files[str(index)]=dict(path=str(payload.relative_to(directory)),sha256=file_hash(payload))
        except Exception as exc:
            _write_json(directory/'diagnostics/projection-failure.json',dict(error=str(exc),module=module,checks_completed=projection_checks));raise
        config=dict(protocol='stage0-input-energy-v1',schema_version=1,name=name,run_path=str(directory.relative_to(root)),
            spectral_path=str(sp.relative_to(root)),qualification_path=str(qp.relative_to(root)),inputs_path=str(ip.relative_to(root)),random_path=str(rp.relative_to(root)),
            parents=parents,implementation=implementation(),numpy_version=np.__version__,source_paths=source_paths,
            weights_dir=spectral['weights_dir'],model_identity=qualification['model_identity'],qualification_runtime=diagnostic['runtime'],
            execution_kind='real-cuda' if real else 'synthetic-cpu-fixture',variants={'sft':dict(adapter_path=source_paths['sft-64'])},variant_order=['sft'],
            decoding=qualification['decoding'],prompt_contract=qualification.get('evaluation_config',{}).get('prompt_template',{}),seed=42,
            factor_labels=labels,modules=modules,energy_levels=levels,agreement=dict(AGREEMENT),batches_per_unit=batches_per_unit,batch_size=1,
            units=units,profile_keys=[next(u['key'] for u in units if u['corpus']==c) for c in ['gsm8k','fineweb']],
            factor_files=factor_files,items_sha256=_hash(rows),geometry_sha256=_hash(geometry),projection_checks_sha256=_hash(projection_checks),
            input_scope='untuned base module inputs; all factors share each forward; unshifted saved question/solution/user masks',
            total='independent squared a dot x; not reconstructed from components',
            projection_reference='Ticket02 projection(a, V_k), same module/checkpoint/cutoff; checked against norm of V_k V_k^T a',
            headline='parallel / (parallel + rest)',components='parallel/total, rest/total, twice_cross/total; undefined if total is zero',
            budget_ceiling_units=20,numerical_gate_approval='user 2026-10-27 corrected response: atol1e-5 rtol1e-5 on reconstruction and normalized projection checks')
        _save_frozen(path,config);_save_frozen(directory/'config.json',config);_jsonl(directory/'inputs/items.jsonl',rows)
        _save_frozen(directory/'inputs/geometry.json',geometry);_save_frozen(directory/'checks/projections.json',projection_checks)
        _save_frozen(directory/'meta/run_manifest.json',dict(config_sha256=_hash(config),main_sequence_forwards=300,extra_model_forwards=0,resources='one FP32 base forward per sequence on GPU; CPU preparation/report'))
        _state(directory,'prepared',0,len(units))
        print(f'Stage0 energy prepare: {len(modules)}/{len(modules)} checked; 300 base forwards planned, all factors shared.',flush=True)
    return path


def load(path,root):
    config=json.loads(inside(root,path).read_text());directory=inside(root,config['run_path'])
    if config.get('protocol')!='stage0-input-energy-v1' or config['implementation']!=implementation() or config['numpy_version']!=np.__version__ or config['agreement']!=AGREEMENT:raise ValueError('Energy implementation/settings changed; preserve evidence and select a new plan.')
    if json.loads((directory/'config.json').read_text())!=config or json.loads((directory/'meta/run_manifest.json').read_text())['config_sha256']!=_hash(config):raise ValueError('Frozen energy plan changed.')
    _verify_parents(root,config['parents']);spectral,_=spectral_load(inside(root,config['spectral_path']),root);_verify_cache(spectral,inside(root,spectral['run_path']))
    qualification,qd=qualification_load(inside(root,config['qualification_path']),root);qualification_verify(qualification,qd)
    for entry in config['factor_files'].values():
        if file_hash(inside(directory,entry['path']))!=entry['sha256']:raise ValueError('Frozen projected factors changed.')
    rows=[json.loads(x) for x in (directory/'inputs/items.jsonl').read_text().splitlines()]
    if _hash(rows)!=config['items_sha256'] or _hash(json.loads((directory/'inputs/geometry.json').read_text()))!=config['geometry_sha256'] or _hash(json.loads((directory/'checks/projections.json').read_text()))!=config['projection_checks_sha256']:raise ValueError('Energy corpus/geometry/projection checks changed.')
    return config,directory,rows
