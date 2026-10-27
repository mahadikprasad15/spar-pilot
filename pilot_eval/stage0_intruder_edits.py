"""CPU conditional full-matrix catalogues from verified intruder geometry."""
import csv
import html
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from pilot_eval.run import _write_json
from pilot_eval.sft import safe_name
from pilot_eval.stage0_edits import _load as load_edits, _verify as verify_edits
from pilot_eval.stage0_intruders import _load as load_geometry, _verify as verify_geometry, _adapters, _key
from pilot_eval.stage0_intruder_edit_math import MODES, module_selection, edit_matrix
from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_workflow import _writer, _state, _now
from pilot_eval.stage0_weights import LocalWeights
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _hash, _save_frozen


def _implementation():
    folder=Path(__file__).parent
    return {n:file_hash(folder/n) for n in ['stage0_intruder_edits.py','stage0_intruder_edit_math.py','stage0_merged_backend.py']}


def _parents(edits_path,geometry_path,root):
    edits,spectral,audit,_=load_edits(edits_path,root)
    catalogue=verify_edits(edits,inside(root,edits['run_path']),root)
    gconfig,gdir,gspectral,cache=load_geometry(geometry_path,root)
    geometry=verify_geometry(gconfig,gdir,gspectral,cache,root,_adapters(gconfig,root))
    if (edits['spectral_sha256']!=gconfig['spectral_sha256'] or edits['modules']!=gconfig['modules']
            or edits['execution_kind']!=gconfig['execution_kind']):
        raise ValueError('Principal edits and intruder geometry must use the identical base spectral source/inventory.')
    references={v:catalogue['variants'][v] for v in ['V0a','V0b']}
    for v,arm in [('V0a','sft'),('V0b','grpo')]:
        if file_hash(inside(root,references[v]['adapter_path'])/'adapter_model.safetensors')!=file_hash(inside(root,gconfig['sources'][arm]['path'])/'adapter_model.safetensors'):
            raise ValueError('Unedited reference payload differs from the measured intruder source.')
    return edits,spectral,audit,gconfig,gdir,geometry,references,cache


def _selections(geometry):
    return [dict(module=r['module'],arm=r['arm'],geometry_arrays=r['arrays_path'],**module_selection(r))
            for r in geometry['module_results']]


def prepare(edits_path,geometry_path,output_root,name):
    root=Path(output_root).resolve();safe_name(name);ep=inside(root,edits_path);gp=inside(root,geometry_path)
    edits,spectral,audit,gconfig,gdir,geometry,references,cache=_parents(ep,gp,root)
    selections=_selections(geometry)
    path=root/'plans'/name/'stage0.intruder-edits.json';directory=root/f'runs/stage0/{name}/intruder-edits'
    # Dimensions are bound by the verified base cache, not inferred from a filename.
    dimensions={r['module']:(r['d_out'],r['d_in']) for r in cache['module_results']}
    upper=sum(dimensions[r['module']][0]*dimensions[r['module']][1]*4*6 for r in selections if r['flagged_indices'])
    config=dict(schema_version=1,protocol='stage0-intruder-edits-v1',name=name,run_path=str(directory.relative_to(root)),
        edits_path=str(ep.relative_to(root)),edits_sha256=file_hash(ep),geometry_path=str(gp.relative_to(root)),geometry_sha256=file_hash(gp),
        geometry_complete_sha256=file_hash(gdir/'complete.json'),geometry_run_path=gconfig['run_path'],
        principal_complete_sha256=file_hash(inside(root,edits['run_path'])/'complete.json'),
        sources=gconfig['sources'],references=references,selections=selections,dimensions=dimensions,
        execution_kind=gconfig['execution_kind'],model_identity=audit['sources']['sft']['identity'],weights_dir=spectral['weights_dir'],
        representation='mixed-full-matrices',halving_factor=.5,ratio_range=[.8,1.25],rematch_sensitivity=False,
        matching='minimum total absolute singular difference; lexicographic control indices; exact FP64-value rational objective',
        control_top=40,detection_top=20,primary_threshold=.5,primary_level=50,subspace_floor=40,
        matrix_dtype='float32',construction_dtype='float64',reconstruction_gate=1e-6,
        numpy_version=np.__version__,implementation=_implementation(),
        storage_upper_bound_bytes=upper,storage_note='six full matrices per eligible module/arm before identical-selection reuse; NPY headers extra',
        gpu_qualified=False,measurement_complete=False,
        next_stage='Un-halved mixed-loader qualification and bounded evaluation are Ticket 10; CPU catalogue is not GPU approval.')
    with _writer(directory):
        _save_frozen(path,config);_save_frozen(directory/'config.json',config)
        _save_frozen(directory/'meta/run_manifest.json',dict(config_sha256=_hash(config),total_units=len(selections),
            resources='CPU matrix construction; no model or GPU',storage_upper_bound_bytes=upper,recorded_at=_now()))
        if not (directory/'meta/status.json').exists():_state(directory,'prepared',0,len(selections))
    return path


def _load(path,root):
    config=json.loads(inside(root,path).read_text());directory=inside(root,config['run_path'])
    if (config.get('protocol')!='stage0-intruder-edits-v1' or config.get('implementation')!=_implementation()
            or config.get('numpy_version')!=np.__version__ or config.get('halving_factor')!=.5
            or config.get('ratio_range')!=[.8,1.25] or config.get('rematch_sensitivity') is not False
            or config.get('control_top')!=40 or config.get('detection_top')!=20
            or config.get('primary_threshold')!=.5 or config.get('subspace_floor')!=40
            or config.get('reconstruction_gate')!=1e-6):
        raise ValueError('Conditional edit code/settings changed; retain evidence and use a new name.')
    if (json.loads((directory/'config.json').read_text())!=config
            or json.loads((directory/'meta/run_manifest.json').read_text())['config_sha256']!=_hash(config)):
        raise ValueError('Conditional catalogue frozen plan changed.')
    ep,gp=inside(root,config['edits_path']),inside(root,config['geometry_path'])
    if file_hash(ep)!=config['edits_sha256'] or file_hash(gp)!=config['geometry_sha256']:
        raise ValueError('Conditional catalogue parent plan changed.')
    edits,spectral,audit,gconfig,gdir,geometry,references,cache=_parents(ep,gp,root)
    if (file_hash(gdir/'complete.json')!=config['geometry_complete_sha256']
            or file_hash(inside(root,edits['run_path'])/'complete.json')!=config['principal_complete_sha256']
            or config['selections']!=_selections(geometry) or config['references']!=references
            or config['sources']!=gconfig['sources']):
        raise ValueError('Conditional edit sources, eligibility or matching changed.')
    return config,directory,spectral,gconfig


def _choices(selection):
    if not selection['flagged_indices']:return {}
    return dict(unhalved=[],**{mode:selection['selections'][mode] for mode in MODES[1:] if selection['selections'][mode]})


def _unit(config,directory,selection):
    folder=directory/'units'/_key(selection['module'],selection['arm']);marker=folder/'complete.json'
    if not marker.exists():return None
    seal=json.loads(marker.read_text())
    if seal.get('identity')!=dict(config_sha256=_hash(config),selection=selection):raise ValueError('Conditional module identity changed.')
    if 'result.json' not in seal.get('files',{}):raise ValueError('Conditional unit omits its result.')
    for p,digest in seal['files'].items():
        if file_hash(inside(folder,p))!=digest:raise ValueError('Conditional matrix or module payload changed.')
    result=json.loads((folder/'result.json').read_text())
    if result['selection']!=selection or set(result['matrices'])!=set(_choices(selection)):
        raise ValueError('Conditional module matrices/masks differ from selection.')
    expected_files={'result.json'}|{m['file'] for m in result['matrices'].values()}
    if set(seal['files'])!=expected_files:raise ValueError('Conditional module payload inventory differs.')
    shape=tuple(config['dimensions'][selection['module']])
    for mode,row in result['matrices'].items():
        path=inside(folder,row['file']);matrix=np.load(path,mmap_mode='r',allow_pickle=False)
        if matrix.shape!=shape or matrix.dtype!=np.float32 or not np.isfinite(matrix).all():
            raise ValueError('Full edited matrix shape/precision/finiteness mismatch.')
        proof=row['construction'];relative=proof['serialized_reconstruction_relative_error']
        if (proof['selected_indices']!=_choices(selection)[mode]
                or proof['halving_factor']!=.5 or (relative is not None and not 0<=relative<1e-6)
                or (relative is None and (not proof['zero_target'] or proof['serialized_reconstruction_absolute_error']!=0))):
            raise ValueError('Conditional matrix reconstruction/selection evidence differs.')
        if row['sha256']!=file_hash(path):raise ValueError('Conditional matrix descriptor hash changed.')
    return result


def _build_module(config,directory,selection,matrix,factors,root):
    folder=directory/'units'/_key(selection['module'],selection['arm']);folder.mkdir(parents=True,exist_ok=True)
    begin=time.monotonic();choices=_choices(selection);matrices={};seen={}
    if choices:
        a,b=factors;merged=np.asarray(matrix,dtype=float)+b.astype(float)@a.astype(float)
        with np.load(inside(root,config['geometry_run_path'])/selection['geometry_arrays'],allow_pickle=False) as arrays:
            sigma,u,v=arrays['new_sigma'],arrays['new_u'],arrays['new_v']
        count=min(40,len(sigma));denominator=float(np.linalg.norm(merged))
        residual=float(np.linalg.norm(merged@v[:,:count].astype(float)-u[:,:count].astype(float)*sigma[:count]))
        relative=None if denominator==0 else residual/denominator
        if (denominator==0 and residual!=0) or (denominator>0 and not relative<1e-6):
            raise ValueError('Saved singular components do not reconstruct the merged operator within the inherited gate.')
        for mode,indices in choices.items():
            key=tuple(indices)
            if key in seen:
                file,proof=seen[key]
            else:
                value,proof=edit_matrix(merged,u,v,sigma,indices)
                file=f'matrices/{mode}.npy';target=folder/file;target.parent.mkdir(parents=True,exist_ok=True)
                with target.with_suffix('.tmp').open('wb') as stream:np.save(stream,value,allow_pickle=False)
                target.with_suffix('.tmp').replace(target)
                seen[key]=(file,proof)
            matrices[mode]=dict(file=file,sha256=file_hash(folder/file),shape=list(merged.shape),dtype='float32',
                construction=proof,source_singular_component_relative_residual=relative)
    result=dict(selection=selection,matrices=matrices,status='built' if choices else 'skipped-no-eligible-flags',
        original_sources_modified=False,seconds=time.monotonic()-begin,
        retained_bytes=sum((folder/p).stat().st_size for p in {m['file'] for m in matrices.values()}))
    _write_json(folder/'result.json',result)
    paths={'result.json'}|{m['file'] for m in matrices.values()}
    _write_json(folder/'complete.json',dict(identity=dict(config_sha256=_hash(config),selection=selection),
        files={p:file_hash(folder/p) for p in paths}))
    return result


def _variant_id(arm,mode):
    if mode=='control':return f'V6c-{arm}-matched'
    if mode=='ratio-control':return f'V6c-{arm}-ratio'
    return f'V6-{arm}-{mode}'


def _catalogue(config,directory,units,root):
    variants={};arms={};matching=[]
    for arm in ['sft','grpo']:
        selected=[u for u in units if u['selection']['arm']==arm]
        eligible=[u for u in selected if u['selection']['flagged_indices']]
        arms[arm]=dict(status='applicable' if eligible else 'not-applicable',eligible_modules=len(eligible),
            skipped_modules=[dict(module=u['selection']['module'],reason=u['selection']['skipped_full_reason']) for u in selected if not u['selection']['flagged_indices']],
            matched_module_exclusions=[dict(module=u['selection']['module'],reason=u['selection']['excluded_matched_reason']) for u in eligible if u['selection']['excluded_matched_reason']])
        matching.extend(u['selection'] for u in eligible)
        if not eligible:continue
        reference='V0a' if arm=='sft' else 'V0b'
        for mode in MODES:
            matrices={}
            for unit in eligible:
                if mode not in unit['matrices']:continue
                selection=unit['selection'];payload=unit['matrices'][mode]
                path=directory/'units'/_key(selection['module'],arm)/payload['file']
                matrices[selection['module']]=dict(path=str(path.relative_to(root)),sha256=payload['sha256'],
                    shape=payload['shape'],dtype=payload['dtype'],selected_indices=payload['construction']['selected_indices'])
            name=_variant_id(arm,mode)
            state='ready-for-qualification' if matrices else ('not-applicable-no-retained-pairs' if 'ratio' in mode else 'not-applicable-no-matched-modules')
            variants[name]=dict(variant=name,source_arm=arm,source_reference=reference,mode=mode,
                operation='unhalved-merged-loader-control' if mode=='unhalved' else 'halve-selected-singular-values',
                adapter_path=config['references'][reference]['adapter_path'],representation='mixed-full-matrices',
                merged_modules=matrices,adapter_disable_modules=sorted(matrices),modules_edited=len(matrices),
                status=state,gpu_qualified=False,measurement_complete=False,source_geometry_sha256=config['geometry_complete_sha256'])
    return dict(schema_version=1,state='complete' if variants else 'not-applicable',catalogue_complete=True,
        execution_kind=config['execution_kind'],real_model_execution=False,real_weight_construction=config['execution_kind']!='synthetic-cpu-fixture',
        arms=arms,variants=variants,references=config['references'],matching=matching,
        model_identity=config['model_identity'],weights_dir=config['weights_dir'],
        representation='mixed-full-matrices',gpu_qualified=False,measurement_complete=False,
        limits=['Conditional CPU operators only; no model outputs or behavioral findings.',
                'Full V6 is separate from matched/sensitivity variants; sensitivity uses the original pairs without rematching.',
                'Edited modules disable the adapter, untouched modules retain the original source adapter.',
                'Unhalved merged matrices require bounded same-runtime qualification before evaluation.'])


def _report(config,directory,result):
    pairs=[]
    for row in result['matching']:
        for pair in row['matching']['pairs']:pairs.append(dict(module=row['module'],arm=row['arm'],**pair))
    target=directory/'results/matching.csv';target.parent.mkdir(parents=True,exist_ok=True)
    fields=['module','arm','flagged_index','control_index','sigma_flagged','sigma_control','absolute_difference','sigma_ratio','ratio_status','ratio_retained']
    with target.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(pairs)
    escape=lambda x:html.escape(str(x))
    rows=''.join('<tr>'+''.join('<td>'+escape(v)+'</td>' for v in [k,d['source_arm'],d['mode'],d['modules_edited'],d['status']])+'</tr>' for k,d in result['variants'].items())
    synthetic='<aside>Synthetic software fixture; not a scientific result or model qualification.</aside>' if config['execution_kind']=='synthetic-cpu-fixture' else ''
    (directory/'report.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Conditional intruder edits</title>
<style>body{max-width:1100px;margin:40px auto;padding:0 24px;background:#f7f8f4;color:#263e44;font:16px/1.65 system-ui}h1{font:38px Georgia}h2{font:26px Georgia}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:10px;border-bottom:1px solid #d8dfd9;text-align:left}th,aside{background:#eaf0e9}aside{padding:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}a{color:#36758a}</style>
<h1>Conditional merged-weight catalogue</h1>'''+synthetic+'<p>State: <strong>'+escape(result['state'])+'</strong>. This CPU stage builds matrices; GPU qualification and evaluation remain pending.</p>'+
        '<p>Full V6 halves every eligible flagged singular component. Matched V6/V6c exclude modules short of distinct non-flagged top-40 controls. Ratio variants retain only original pairs in [0.8,1.25], without rematching. A missing eligible arm is explicitly not applicable.</p>'+
        '<table><tr><th>Variant</th><th>Arm</th><th>Scope</th><th>Merged modules</th><th>Status</th></tr>'+rows+'</table>'+
        '<h2>Module exclusions and applicability</h2><pre>'+escape(json.dumps(result['arms'],indent=2))+'</pre>'+
        '<h2>Loading contract</h2><p>Load full saved matrices with the source adapter disabled only in the listed modules. All other modules retain that adapter. Do not merge the adapter into an already edited matrix. The unhalved matrices are a separate qualification control, not an edited experiment.</p>'+
        '<p><a href="results/catalogue.json">Complete catalogue, masks and edit selections</a> · <a href="results/matching.csv">Matched-pair ratios</a> · <a href="config.json">Frozen construction plan</a>. Per-module matrix reconstruction and before/after singular-coordinate checks are saved under units/.</p>'+
        '<h2>Limits</h2><ul>'+''.join('<li>'+escape(v)+'</li>' for v in result['limits'])+'</ul></html>')
    (directory/'results/catalogue.md').write_text('# Conditional intruder catalogue\n\nState: '+result['state']+
        '\n\nCPU construction only; qualification/evaluation pending.\n\n'+'\n'.join('- '+v for v in result['limits'])+'\n')


def run(path,output_root,*,dependencies=None,max_new_units=None,progress=True):
    root=Path(output_root).resolve();config,directory,spectral,gconfig=_load(path,root)
    if max_new_units is not None and (type(max_new_units) is not int or max_new_units<0):raise ValueError('Invalid unit limit.')
    deps=dependencies or LocalWeights();units=[];new=0;adapters=None
    with _writer(directory):
        if (directory/'complete.json').exists():return _verify(config,directory,root)
        try:
            for selection in config['selections']:
                unit=_unit(config,directory,selection)
                if unit is None:
                    if max_new_units is not None and new>=max_new_units:return _state(directory,'incomplete',len(units),len(config['selections']))
                    begin=time.monotonic();matrix=None;pair=None
                    if selection['flagged_indices']:
                        if adapters is None:adapters=_adapters(gconfig,root)
                        entry=spectral['inventory']['modules'][selection['module']]
                        matrix=deps.matrix({**entry,'path':str(inside(root,entry['path']))})
                        pair=adapters[selection['arm']][selection['module']]
                    unit=_build_module(config,directory,selection,matrix,pair,root);new+=1
                    if progress:print(f'Stage 0 conditional edits: {len(units)+1}/{len(config["selections"])} sealed; {_key(selection["module"],selection["arm"])}; {unit["status"]}; {time.monotonic()-begin:.2f}s.',flush=True)
                units.append(unit);_state(directory,'running',len(units),len(config['selections']))
            _load(path,root) # source hash recheck after construction, before publication
            result=_catalogue(config,directory,units,root)
            _write_json(directory/'results/catalogue.json',result);_report(config,directory,result)
            paths=[directory/'config.json',directory/'meta/run_manifest.json',directory/'report.html']
            paths += [p for sub in ['units','results'] for p in (directory/sub).rglob('*') if p.is_file()]
            _write_json(directory/'complete.json',dict(config_sha256=_hash(config),files={str(p.relative_to(directory)):file_hash(p) for p in paths}))
            _state(directory,result['state'],len(units),len(units));return result
        except (Exception,KeyboardInterrupt) as exc:
            _state(directory,'failed',len(units),len(config['selections']),error=str(exc))
            log=directory/'logs/errors.jsonl';log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a') as stream:stream.write(json.dumps(dict(error=str(exc),type=type(exc).__name__))+'\n')
            raise



def _source_matrix(spectral,module,root):
    entry=spectral['inventory']['modules'][module]
    entry={**entry,'path':str(inside(root,entry['path']))}
    if spectral['inventory']['execution_kind']=='synthetic-cpu-fixture':
        with np.load(entry['path'],allow_pickle=False) as saved:return saved[entry['key']]
    return LocalWeights().matrix(entry)


def _reconstruct_unit(config,directory,selection,spectral,gconfig,adapters,root):
    if not selection['flagged_indices']:return
    folder=directory/'units'/_key(selection['module'],selection['arm'])
    saved=_unit(config,directory,selection)
    w=_source_matrix(spectral,selection['module'],root)
    a,b=adapters[selection['arm']][selection['module']]
    merged=w.astype(float)+b.astype(float)@a.astype(float)
    with np.load(inside(root,config['geometry_run_path'])/selection['geometry_arrays'],allow_pickle=False) as source:
        sigma,u,v=source['new_sigma'],source['new_u'],source['new_v']
    checked=set()
    for mode,indices in _choices(selection).items():
        descriptor=saved['matrices'][mode]
        if descriptor['file'] in checked:continue
        checked.add(descriptor['file'])
        expected,proof=edit_matrix(merged,u,v,sigma,indices)
        actual=np.load(folder/descriptor['file'],allow_pickle=False)
        if not np.array_equal(actual,expected) or descriptor['construction']!=proof:
            raise ValueError('Serialized merged matrix does not reconstruct the intended source operator.')


def _verify(config,directory,root):
    marker=json.loads((directory/'complete.json').read_text())
    required={'config.json','meta/run_manifest.json','report.html','results/catalogue.json','results/catalogue.md','results/matching.csv'}
    if marker.get('config_sha256')!=_hash(config) or not required<=marker.get('files',{}).keys():raise ValueError('Conditional catalogue completion identity differs.')
    for p,digest in marker['files'].items():
        if file_hash(inside(directory,p))!=digest:raise ValueError('Conditional catalogue payload changed: '+p)
    units=[_unit(config,directory,s) for s in config['selections']]
    if any(u is None for u in units):raise ValueError('Conditional catalogue module is incomplete.')
    _,_,spectral,gconfig=_load(directory/'config.json',root)
    adapters=None
    for selection in config['selections']:
        if selection['flagged_indices']:
            if adapters is None:adapters=_adapters(gconfig,root)
            _reconstruct_unit(config,directory,selection,spectral,gconfig,adapters,root)
    result=_catalogue(config,directory,units,root)
    if result!=json.loads((directory/'results/catalogue.json').read_text()):raise ValueError('Conditional catalogue masks or matching do not reconcile.')
    return result


def verify(path,output_root):
    root=Path(output_root).resolve();config,directory,_,_=_load(path,root)
    return _verify(config,directory,root)


def status(path,output_root):
    root=Path(output_root).resolve();config,directory,_,_=_load(path,root)
    units=[_unit(config,directory,s) for s in config['selections']]
    result=_verify(config,directory,root) if (directory/'complete.json').exists() else None
    return dict(catalogue_complete=result is not None,completed_units=len(units) if result else sum(u is not None for u in units),
        total_units=len(units),state=result['state'] if result else 'incomplete',
        report=str(directory/'report.html'),gpu_qualified=False)


def recover_writer(path,output_root,owner_token,review_notes):
    root=Path(output_root).resolve();_,directory,_,_=_load(path,root)
    if not owner_token or not review_notes.strip():raise ValueError('Explicit owner token and recovery notes required.')
    with run_lock(directory):
        marker=directory/'writer.json';owner=json.loads(marker.read_text())
        if owner.get('token')!=owner_token:raise ValueError('Writer token mismatch.')
        if owner['host']==platform.node():
            try:os.kill(owner['pid'],0)
            except ProcessLookupError:pass
            else:raise ValueError('Recorded writer is still active.')
        log=directory/'logs/ownership-recovery.jsonl';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as stream:
            stream.write(json.dumps(dict(owner=owner,notes=review_notes,updated_at=_now()))+'\n')
            stream.flush();os.fsync(stream.fileno())
        marker.unlink()
    return dict(state='ownership-released',scope='writer only; sealed catalogue matrices retained')
