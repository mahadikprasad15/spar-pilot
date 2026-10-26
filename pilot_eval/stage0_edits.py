"""Resumable principal edit catalogue; CPU factor construction, not inference."""
import json
from pathlib import Path
import shutil

import numpy as np

from pilot_eval.run import _write_json
from pilot_eval.sft import safe_name
from pilot_eval.stage0_edit_math import edit_module
from pilot_eval.stage0_sources import factors,inside
from pilot_eval.stage0_spectral import _load as load_spectral,_module_id,_verify_cache
from pilot_eval.stage0_workflow import _software,_state,_writer
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash,_save_frozen

VARIANTS={'V0a':('sft','reference'),'V0b':('grpo','reference'),
          'V1':('sft','principal-output-removal'),'V2':('sft','principal-input-removal'),
          **{f'V3-{i}':('sft','random-output-removal') for i in range(3)},
          'V4':('sft','V1-per-module-size-control'),
          'V4-input':('sft','V2-per-module-size-control'),'V5':('grpo','principal-output-removal')}


def _implementation():
    folder=Path(__file__).parent
    return {name:file_hash(folder/name) for name in ['stage0_edits.py','stage0_edit_math.py']}


def prepare(spectral_path,output_root,name):
    root=Path(output_root).resolve();safe_name(name)
    spectral_path=inside(root,spectral_path)
    spectral,audit=load_spectral(spectral_path,root)
    cache=_verify_cache(spectral,inside(root,spectral['run_path']))
    config=dict(schema_version=1,protocol='stage0-principal-edits-v1',name=name,
        run_path=f'runs/stage0/{name}/edits',spectral_path=str(spectral_path.relative_to(root)),
        spectral_sha256=file_hash(spectral_path),spectral_complete_sha256=file_hash(root/spectral['run_path']/'complete.json'),
        implementation=_implementation(),variants={key:dict(arm=arm,operation=op) for key,(arm,op) in VARIANTS.items()},
        modules=[row['module'] for row in cache['module_results']],energy_level=50,
        random_seeds=[1000,1001,1002],random_generator='numpy-PCG64; Gaussian QR; reset declared draw seed per module',
        execution_kind=cache['execution_kind'],partial_module_fixture=cache['execution_kind']=='synthetic-cpu-fixture',
        projection_residual_gate=1e-6,magnitude_control_relative_gate=1e-6)
    path=root/'plans'/name/'stage0.edits.json'
    with _writer(path.parent):
        _save_frozen(path,config)
        directory=root/config['run_path']
        if not (directory/'meta/run_manifest.json').exists():
            _save_frozen(directory/'meta/run_manifest.json',dict(config_sha256=_hash(config),software=_software()))
            _state(directory,'prepared',0,len(config['modules']))
    return path


def _load(path,root):
    config=json.loads(inside(root,path).read_text())
    if config.get('protocol')!='stage0-principal-edits-v1' or config['implementation']!=_implementation():
        raise ValueError('Edit implementation changed; preserve product and prepare a new name.')
    directory=inside(root,config['run_path'])
    if json.loads((directory/'meta/run_manifest.json').read_text())['config_sha256']!=_hash(config):
        raise ValueError('Edit plan changed after preparation.')
    spectral_path=inside(root,config['spectral_path'])
    if file_hash(spectral_path)!=config['spectral_sha256']:raise ValueError('Spectral source plan changed.')
    spectral,audit=load_spectral(spectral_path,root)
    if file_hash(root/spectral['run_path']/'complete.json')!=config['spectral_complete_sha256']:
        raise ValueError('Spectral completion identity changed.')
    cache=_verify_cache(spectral,inside(root,spectral['run_path']))
    if config['modules']!=[row['module'] for row in cache['module_results']]:
        raise ValueError('Edit/cache module inventory mismatch.')
    return config,spectral,audit,cache


def _unit(directory,config,module):
    folder=directory/'units'/_module_id(module)
    marker=folder/'complete.json'
    if not marker.exists():return None
    saved=json.loads(marker.read_text())
    if saved.get('identity')!=dict(config_sha256=_hash(config),module=module) or set(saved.get('files',{}))!={'factors.npz','checks.json','random-bases.npz'}:
        raise ValueError('Edit module receipt identity mismatch.')
    for name,digest in saved['files'].items():
        if file_hash(folder/name)!=digest:raise ValueError('Edited module payload changed.')
    return json.loads((folder/'checks.json').read_text())


def _save_tensors(path,arrays):
    """Standard FP32 safetensor encoding; no Torch or optional writer dependency."""
    rows,chunks,offset={},[],0
    for key,value in sorted(arrays.items()):
        data=np.ascontiguousarray(value,dtype='<f4').tobytes()
        rows[key]=dict(dtype='F32',shape=list(value.shape),data_offsets=[offset,offset+len(data)])
        chunks.append(data);offset+=len(data)
    header=json.dumps(rows,separators=(',',':')).encode()
    header+=b' '*((-len(header))%8)
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        stream.write(len(header).to_bytes(8,'little'));stream.write(header)
        for chunk in chunks:stream.write(chunk)
    temporary.replace(path)


def _assemble(directory,root,config,audit,original):
    variants={}
    for variant,selection in config['variants'].items():
        arm=selection['arm'];folder=directory/'adapters'/variant;folder.mkdir(parents=True,exist_ok=True)
        source=inside(root,audit['sources'][arm]['checkpoints']['64']['path'])
        shutil.copyfile(source/'adapter_config.json',folder/'adapter_config.json')
        arrays={key+suffix:factor.copy() for key,pair in original[arm].items()
                for suffix,factor in zip(['.lora_A.weight','.lora_B.weight'],pair)}
        if selection['operation']=='reference':
            shutil.copyfile(source/'adapter_model.safetensors',folder/'adapter_model.safetensors')
        else:
            for module in config['modules']:
                with np.load(directory/'units'/_module_id(module)/'factors.npz',allow_pickle=False) as saved:
                    arrays[module+'.lora_A.weight']=saved[variant+'/a']
                    arrays[module+'.lora_B.weight']=saved[variant+'/b']
            _save_tensors(folder/'adapter_model.safetensors',arrays)
        # Verify the actual encoded PEFT factors, not just pre-write arrays.
        actual=factors(folder)
        for module,(a,b) in actual.items():
            if not np.array_equal(a,arrays[module+'.lora_A.weight']) or not np.array_equal(b,arrays[module+'.lora_B.weight']):
                raise ValueError('Serialized edit differs from constructed factors.')
        evidence=dict(variant=variant,source_arm=arm,source_path=str(source.relative_to(root)),
                      operation=selection['operation'],energy_level=50,modules_edited=0 if selection['operation']=='reference' else len(config['modules']),
                      execution_kind=config['execution_kind'],random_seed=(1000+int(variant[-1])) if variant.startswith('V3-') else None,
                      scaling=1.,representation='rank-one-PEFT',gpu_qualified=False)
        _write_json(folder/'edit.json',evidence)
        files={name:file_hash(folder/name) for name in ['adapter_config.json','adapter_model.safetensors','edit.json']}
        _write_json(folder/'complete.json',dict(identity=dict(config_sha256=_hash(config),variant=variant),files=files))
        variants[variant]=dict(adapter_path=str(folder.relative_to(root)),**evidence)
    return variants


def run(path,output_root,*,max_new_units=None):
    root=Path(output_root).resolve();config,spectral,audit,cache=_load(path,root)
    directory=inside(root,config['run_path'])
    if max_new_units is not None and max_new_units<0:raise ValueError('Invalid new-unit limit.')
    with _writer(directory):
        if (directory/'complete.json').exists():return _verify(config,directory,root)
        completed=[];created=0
        try:
            original={arm:factors(inside(root,audit['sources'][arm]['checkpoints']['64']['path'])) for arm in ['sft','grpo']}
            for row in cache['module_results']:
                module=row['module'];check=_unit(directory,config,module)
                if check is None:
                    if max_new_units is not None and created>=max_new_units:
                        _state(directory,'incomplete',len(completed),len(config['modules']))
                        return dict(state='incomplete',completed=len(completed),total=len(config['modules']))
                    with np.load(root/spectral['run_path']/row['arrays_path'],allow_pickle=False) as basis:
                        k=row['k50'];u,v=basis['u'][:,:k],basis['v'][:,:k]
                    edits,checks,random_bases=edit_module(*original['sft'][module],u,v)
                    grpo,grpo_checks,_=edit_module(*original['grpo'][module],u,v)
                    edits['V5']=grpo['V1'];checks['V5']=grpo_checks['V1']
                    folder=directory/'units'/_module_id(module);folder.mkdir(parents=True,exist_ok=True)
                    temp=folder/'factors.tmp.npz'
                    np.savez(temp,**{key+'/'+factor:value for key,pair in edits.items() for factor,value in zip(['a','b'],pair)})
                    temp.replace(folder/'factors.npz')
                    temp=folder/'random-bases.tmp.npz';np.savez(temp,**random_bases);temp.replace(folder/'random-bases.npz')
                    check=dict(module=module,k=k,variants=checks,
                               sources={arm:str(audit['sources'][arm]['checkpoints']['64']['path']) for arm in ['sft','grpo']})
                    _write_json(folder/'checks.json',check)
                    _write_json(folder/'complete.json',dict(identity=dict(config_sha256=_hash(config),module=module),
                        files={name:file_hash(folder/name) for name in ['factors.npz','checks.json','random-bases.npz']}))
                    created+=1
                completed.append(check)
                _state(directory,'running',len(completed),len(config['modules']))
                print(f'Stage 0 edits: {len(completed)}/{len(config["modules"])} sealed; {_module_id(module)}.',flush=True)
            variants=_assemble(directory,root,config,audit,original)
            result=dict(schema_version=1,state='complete',variants=variants,module_count=len(completed),
                        gpu_qualified=False,measurement_complete=False,execution_kind=config['execution_kind'],
                        projection_report_revealed=False)
            _write_json(directory/'results/catalogue.json',result)
            (directory/'results/catalogue.md').write_text('# Stage 0 principal intervention catalogue\n\n'
                'CPU construction complete; GPU qualification and evaluation remain pending.\n\n'
                '| Variant | Source | Operation |\n|---|---|---|\n'+
                '\n'.join(f'| {key} | {row["source_arm"]} | {row["operation"]} |' for key,row in variants.items())+'\n')
            paths=[directory/'results/catalogue.json',directory/'results/catalogue.md']
            paths += [directory/'adapters'/variant/'complete.json' for variant in variants]
            _write_json(directory/'complete.json',dict(config_sha256=_hash(config),modules=config['modules'],
                files={str(p.relative_to(directory)):file_hash(p) for p in paths}))
            _state(directory,'complete',len(completed),len(completed))
            return result
        except (ValueError,OSError,RuntimeError) as exc:
            _state(directory,'failed',len(completed),len(config['modules']),error=str(exc))
            with (directory/'logs/errors.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(error=str(exc),type=type(exc).__name__))+'\n')
            raise


def _verify(config,directory,root):
    saved=json.loads((directory/'complete.json').read_text())
    required={'results/catalogue.json','results/catalogue.md'}|{f'adapters/{key}/complete.json' for key in config['variants']}
    if saved.get('config_sha256')!=_hash(config) or saved.get('modules')!=config['modules'] or set(saved.get('files',{}))!=required:
        raise ValueError('Edit catalogue completion identity mismatch.')
    for relative,digest in saved['files'].items():
        if file_hash(inside(directory,relative))!=digest:raise ValueError('Edit catalogue payload changed.')
    for module in config['modules']:
        if _unit(directory,config,module) is None:raise ValueError('Missing edit module unit.')
    result=json.loads((directory/'results/catalogue.json').read_text())
    if set(result['variants'])!=set(config['variants']):raise ValueError('Incomplete variant catalogue.')
    for variant,row in result['variants'].items():
        expected=directory/'adapters'/variant
        if inside(root,row['adapter_path'])!=expected:raise ValueError('Adapter path differs from catalogue identity.')
        seal=json.loads((expected/'complete.json').read_text())
        if seal.get('identity')!=dict(config_sha256=_hash(config),variant=variant) or set(seal.get('files',{}))!={'adapter_config.json','adapter_model.safetensors','edit.json'}:
            raise ValueError('Adapter completion identity mismatch.')
        for relative,digest in seal['files'].items():
            if file_hash(expected/relative)!=digest:raise ValueError('Edited adapter payload changed.')
    return result


def verify(path,output_root):
    root=Path(output_root).resolve();config,_,_,_=_load(path,root)
    return _verify(config,inside(root,config['run_path']),root)
