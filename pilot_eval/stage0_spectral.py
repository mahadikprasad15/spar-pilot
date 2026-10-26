"""Resumable CPU spectral cache; trained-factor projections require explicit reveal."""
import csv
import json
from pathlib import Path
import time

import numpy as np

from pilot_eval.stage0_sources import factors, inside
from pilot_eval.stage0_spectral_math import LEVELS, chance_band, decompose, projection
from pilot_eval.stage0_weights import LocalWeights
from pilot_eval.stage0_workflow import _software, _state, _verify_parents, _writer, status
from pilot_eval.sft import safe_name
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash, _save_frozen
from pilot_eval.run import _write_json


def _implementation():
    folder=Path(__file__).parent
    return {name:file_hash(folder/name) for name in
            ['stage0_spectral.py','stage0_spectral_math.py','stage0_weights.py','trace_readout_assets.py']}


def _module_id(module):
    parts=module.split('.')
    return f'L{int(parts[4]):02d}.{parts[-1]}'


def prepare(audit_path, weights_dir, output_root, name, *, dependencies=None):
    root=Path(output_root).resolve();safe_name(name)
    audit_path=inside(root,audit_path)
    audit=status(audit_path,root)['audit']
    if not audit or not audit['primary_sources_ready']:
        raise ValueError('Verify both endpoint sources before preparing the spectral cache.')
    path=root/'plans'/name/'stage0.spectral.json'
    with _writer(path.parent):
        if path.exists():
            config=_load(path,root)
            if config['audit_path']!=str(audit_path.relative_to(root)) or config['weights_dir']!=str(inside(root,weights_dir).relative_to(root)):
                raise ValueError('Spectral selection changed; use a new name.')
            return path
        source=audit['sources']['sft']
        inventory=(dependencies or LocalWeights()).inventory(inside(root,weights_dir),source)
        for key in ['model','model_revision','base_sha256']:
            if inventory[key]!=source['identity'][key]:
                raise ValueError('Weight inventory differs from audited base identity.')
        allowed=set(factors(inside(root,source['checkpoints']['64']['path'])))
        if (not inventory['modules'] or not set(inventory['modules'])<=allowed
                or (inventory['execution_kind']!='synthetic-cpu-fixture' and set(inventory['modules'])!=allowed)):
            raise ValueError('Unexpected base module inventory.')
        parents={str(inside(root,p).relative_to(root)):digest for p,digest in inventory['parents'].items()}
        _verify_parents(root,parents)
        parents[str(audit_path.relative_to(root))]=file_hash(audit_path)
        inventory={**inventory,'parents':parents,'modules':{module:{**entry,
            'path':str(inside(root,entry['path']).relative_to(root))} for module,entry in inventory['modules'].items()}}
        config=dict(schema_version=1,protocol='stage0-spectral-v1',name=name,
            run_path=f'runs/stage0/{name}/spectral',audit_path=str(audit_path.relative_to(root)),
            weights_dir=str(inside(root,weights_dir).relative_to(root)),inventory=inventory,
            implementation=_implementation(),parents=parents,energy_levels=list(LEVELS),
            decomposition_dtype='float64',retained_basis_dtype='float32',intruder_floor=40,
            reconstruction_gate=1e-6,orthogonality_gate=1e-6,seed=42)
        _save_frozen(path,config)
        directory=root/config['run_path']
        _save_frozen(directory/'meta/run_manifest.json',dict(config_sha256=_hash(config),software=_software()))
        _state(directory,'prepared',0,len(inventory['modules']))
    return path


def _load(path,root):
    path=inside(root,path);config=json.loads(path.read_text())
    if config.get('protocol')!='stage0-spectral-v1' or config['implementation']!=_implementation():
        raise ValueError('Spectral implementation changed; preserve old cache and use a new name.')
    directory=inside(root,config['run_path'])
    manifest=json.loads((directory/'meta/run_manifest.json').read_text())
    if manifest['config_sha256']!=_hash(config):
        raise ValueError('Spectral frozen plan mismatch.')
    _verify_parents(root,config['parents'])
    if not status(inside(root,config['audit_path']),root)['audit']:
        raise ValueError('Source audit is no longer complete.')
    return config


def _unit(directory,config,module):
    folder=directory/'units'/_module_id(module)
    marker=folder/'complete.json'
    if not marker.exists():return None
    seal=json.loads(marker.read_text())
    if seal.get('identity')!=dict(config_sha256=_hash(config),module=module) or set(seal.get('files',{}))!={'basis.npz','result.json'}:
        raise ValueError('Spectral unit identity mismatch.')
    for name,digest in seal['files'].items():
        if file_hash(folder/name)!=digest:raise ValueError('Spectral unit payload changed.')
    return json.loads((folder/'result.json').read_text())


def run(path,output_root,*,dependencies=None,max_new_units=None):
    root=Path(output_root).resolve();config=_load(path,root);directory=inside(root,config['run_path'])
    if max_new_units is not None and max_new_units<0:raise ValueError('Invalid maximum new units.')
    deps=dependencies or LocalWeights()
    with _writer(directory):
        if (directory/'complete.json').exists():return verify(path,root)
        rows=[];created=0
        try:
            audit=status(inside(root,config['audit_path']),root)['audit']
            adapter=factors(inside(root,audit['sources']['sft']['checkpoints']['64']['path']))
            for module,entry in config['inventory']['modules'].items():
                row=_unit(directory,config,module)
                if row is None:
                    if max_new_units is not None and created>=max_new_units:
                        _state(directory,'incomplete',len(rows),len(config['inventory']['modules']))
                        return dict(state='incomplete',completed=len(rows),trained_overlap_revealed=False)
                    start=time.monotonic()
                    print(f'Stage 0 SVD: {len(rows)}/{len(config["inventory"]["modules"])} sealed; decomposing {_module_id(module)}.',flush=True)
                    matrix=deps.matrix({**entry,'path':str(inside(root,entry['path']))})
                    a,b=adapter[module]
                    if matrix.shape!=(b.shape[0],a.shape[1]):
                        raise ValueError('Base weight and audited adapter dimensions differ.')
                    arrays,evidence=decompose(matrix)
                    folder=directory/'units'/_module_id(module);folder.mkdir(parents=True,exist_ok=True)
                    temporary=folder/'basis.tmp.npz'
                    np.savez(temporary,**arrays);temporary.replace(folder/'basis.npz')
                    row=dict(module=module,**evidence,seconds=time.monotonic()-start,
                        arrays_path=str((folder/'basis.npz').relative_to(directory)),
                        retained_bytes=(folder/'basis.npz').stat().st_size)
                    _write_json(folder/'result.json',row)
                    _write_json(folder/'complete.json',dict(identity=dict(config_sha256=_hash(config),module=module),
                        files={name:file_hash(folder/name) for name in ['basis.npz','result.json']}))
                    created+=1
                rows.append(row)
                _state(directory,'running',len(rows),len(config['inventory']['modules']))
                print(f'Stage 0 SVD: {len(rows)}/{len(config["inventory"]["modules"])} sealed; {_module_id(module)}; {row["seconds"]:.2f}s; {row["retained_bytes"]} bytes.',flush=True)
            result=dict(schema_version=1,state='complete',modules=len(rows),module_results=rows,
                        trained_overlap_revealed=False,execution_kind=config['inventory']['execution_kind'],
                        retained_bytes=sum(row['retained_bytes'] for row in rows))
            _write_json(directory/'results/cache.json',result)
            _write_json(directory/'complete.json',dict(config_sha256=_hash(config),
                files={'results/cache.json':file_hash(directory/'results/cache.json')},modules=list(config['inventory']['modules'])))
            _state(directory,'complete',len(rows),len(rows))
            return result
        except (ValueError,OSError,RuntimeError,np.linalg.LinAlgError) as exc:
            _state(directory,'failed',len(rows),len(config['inventory']['modules']),error=str(exc))
            with (directory/'logs/errors.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(type=type(exc).__name__,error=str(exc)))+'\n')
            raise ValueError(str(exc)) from exc


def verify(path,output_root):
    root=Path(output_root).resolve();config=_load(path,root);directory=inside(root,config['run_path'])
    seal=json.loads((directory/'complete.json').read_text())
    expected=dict(config_sha256=_hash(config),files={'results/cache.json':file_hash(directory/'results/cache.json')},
                  modules=list(config['inventory']['modules']))
    if seal!=expected:raise ValueError('Spectral completion mismatch.')
    rows=[_unit(directory,config,module) for module in config['inventory']['modules']]
    result=json.loads((directory/'results/cache.json').read_text())
    if any(row is None for row in rows) or result['module_results']!=rows or result['modules']!=len(rows):
        raise ValueError('Spectral summary differs from sealed modules.')
    return result


def report(path,output_root,name,review_path):
    root=Path(output_root).resolve();config=_load(path,root);cache=verify(path,root);safe_name(name)
    review=json.loads(Path(review_path).read_text())
    if (review.get('reveal_trained_overlaps') is not True
            or review.get('prediction_status') not in ['exploratory','frozen-before-reveal']
            or not str(review.get('notes','')).strip()):
        raise ValueError('Explicit trained-overlap reveal review is required.')
    parents={str(inside(root,path).relative_to(root)):file_hash(inside(root,path))}
    if review['prediction_status']=='frozen-before-reveal':
        prediction=inside(root,review.get('prediction_path',''))
        if not prediction.is_file() or not json.loads(prediction.read_text()).get('predictions'):
            raise ValueError('Frozen conditional predictions are missing.')
        parents[str(prediction.relative_to(root))]=file_hash(prediction)
    directory=root/'reports'/name
    with _writer(directory):
        identity=dict(config_sha256=_hash(config),review=review,parents=parents)
        _save_frozen(directory/'exposure.json',identity)
        if (directory/'complete.json').exists():
            seal=json.loads((directory/'complete.json').read_text())
            if seal['identity']!=identity:raise ValueError('Projection report identity changed.')
            for relative,digest in seal['files'].items():
                if file_hash(directory/relative)!=digest:raise ValueError('Projection report payload changed.')
            return json.loads((directory/'results/summary.json').read_text())
        try:
            audit=status(inside(root,config['audit_path']),root)['audit']
            selected={}
            for arm,source in audit['sources'].items():
                for step,cp in source.get('checkpoints',{}).items():
                    if cp.get('state')=='verified-adapter-payload':
                        selected[(arm,int(step))]=factors(inside(root,cp['path']))
            rows=[]
            cache_dir=inside(root,config['run_path'])
            for evidence in cache['module_results']:
                module=evidence['module']
                with np.load(cache_dir/evidence['arrays_path'],allow_pickle=False) as arrays:
                    for level in LEVELS:
                        k=evidence[f'k{level}'];u=arrays['u'][:,:k];v=arrays['v'][:,:k]
                        random_out,random_in=chance_band(u),chance_band(v)
                        for (arm,step),adapter in selected.items():
                            a,b=adapter[module]
                            rows.append(dict(module=module,module_id=_module_id(module),arm=arm,step=step,
                                energy_level=level,k=k,pi_out=projection(b,u),pi_in=projection(a,v),
                                undefined_out_reason='zero factor' if not np.any(b) else None,
                                undefined_in_reason='zero factor' if not np.any(a) else None,
                                chance_out=k/u.shape[0],chance_in=k/v.shape[0],
                                gaussian_out=random_out,gaussian_in=random_in))
            results=directory/'results';results.mkdir(parents=True,exist_ok=True)
            (results/'projections.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            fields=['module_id','arm','step','energy_level','k','pi_out','pi_in','chance_out','chance_in']
            with (results/'projections.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
            from pilot_eval.stage0_spectral_plot import draw_overlap
            draw_overlap(rows,directory/'plots/principal-overlap.svg')
            summary=dict(state='complete',rows=len(rows),modules=cache['modules'],prediction_status=review['prediction_status'],
                         execution_kind=config['inventory']['execution_kind'],gaussian_draws=100,
                         limits=['one saved training seed per arm','random adapter is one reference, not a null distribution'])
            _write_json(results/'summary.json',summary)
            _write_json(directory/'complete.json',dict(identity=identity,files={str(p.relative_to(directory)):file_hash(p)
                for p in [results/'projections.jsonl',results/'projections.csv',results/'summary.json',directory/'plots/principal-overlap.svg']}))
            _state(directory,'complete',len(cache['module_results']),len(cache['module_results']))
            return summary
        except (ValueError,OSError,RuntimeError,ImportError) as exc:
            _state(directory,'failed',0,cache['modules'],error=str(exc))
            with (directory/'logs/errors.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(type=type(exc).__name__,error=str(exc)))+'\n')
            raise
