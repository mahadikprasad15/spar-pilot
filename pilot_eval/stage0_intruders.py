"""Resumable CPU endpoint geometry; eligibility evidence for conditional edits."""
import csv
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from pilot_eval.run import _write_json
from pilot_eval.sft import safe_name
from pilot_eval.stage0_intruder_math import LEVELS,THRESHOLDS,diagnostics,measure
from pilot_eval.stage0_sources import factors,inside
from pilot_eval.stage0_spectral import _load as load_spectral,_verify_cache,_module_id
from pilot_eval.stage0_weights import LocalWeights
from pilot_eval.stage0_workflow import _writer,_state,_now
from pilot_eval.training import file_hash,run_lock
from pilot_eval.workflow import _hash,_save_frozen


def _implementation():
    folder=Path(__file__).parent
    return {n:file_hash(folder/n) for n in ['stage0_intruders.py','stage0_intruder_math.py','stage0_intruder_render.py']}


def prepare(spectral_path,output_root,name):
    root=Path(output_root).resolve();safe_name(name);spectral_path=inside(root,spectral_path)
    spectral,audit=load_spectral(spectral_path,root)
    cache_dir=inside(root,spectral['run_path']);cache=_verify_cache(spectral,cache_dir)
    sources={arm:audit['sources'][arm]['checkpoints']['64'] for arm in ['sft','grpo']}
    if any(s['state']!='verified-adapter-payload' for s in sources.values()):
        raise ValueError('Both verified step-64 endpoint adapters are required.')
    path=root/'plans'/name/'stage0.intruders.json';directory=root/f'runs/stage0/{name}/intruders'
    config=dict(schema_version=1,protocol='stage0-intruder-geometry-v1',name=name,
        run_path=str(directory.relative_to(root)),spectral_path=str(spectral_path.relative_to(root)),
        spectral_sha256=file_hash(spectral_path),spectral_complete_sha256=file_hash(cache_dir/'complete.json'),
        sources=sources,modules=[r['module'] for r in cache['module_results']],arms=['sft','grpo'],
        execution_kind=cache['execution_kind'],levels=list(LEVELS),thresholds=list(THRESHOLDS),
        detection_top=20,control_top=40,subspace_floor=40,primary_level=50,primary_threshold=.5,
        chance='sqrt(m/d_out); thresholds at or below chance are undecidable',
        rotation='equal base-defined k left/right subspaces; principal angles in radians and degrees',
        gap_convention='sigma_k - sigma_(k+1); no-next undefined; positive-over-zero infinite; zero-over-zero undefined',
        small_update_skip=False,decomposition_dtype='float64',retained_dtype='float32',
        reconstruction_gate=1e-6,orthogonality_gate=1e-6,numpy_version=np.__version__,implementation=_implementation(),
        future_edits='Eligibility only; Ticket 09 constructs conditional V6/V6c. No model inference or behavior claim.')
    with _writer(directory):
        _save_frozen(path,config);_save_frozen(directory/'config.json',config)
        _save_frozen(directory/'meta/run_manifest.json',dict(config_sha256=_hash(config),resources='CPU only; exact SVD of each updated module',
            total_units=2*len(config['modules']),model_calls=0,recorded_at=_now()))
        if not (directory/'meta/status.json').exists():_state(directory,'prepared',0,2*len(config['modules']))
    return path


def _load(path,root):
    config=json.loads(inside(root,path).read_text())
    if (config.get('protocol')!='stage0-intruder-geometry-v1' or config.get('implementation')!=_implementation()
            or config.get('numpy_version')!=np.__version__ or config.get('levels')!=list(LEVELS)
            or config.get('thresholds')!=list(THRESHOLDS) or config.get('subspace_floor')!=40
            or config.get('detection_top')!=20 or config.get('control_top')!=40
            or config.get('primary_level')!=50 or config.get('primary_threshold')!=.5
            or config.get('small_update_skip') is not False or config.get('reconstruction_gate')!=1e-6
            or config.get('orthogonality_gate')!=1e-6 or config.get('arms')!=['sft','grpo']):
        raise ValueError('Geometry implementation/runtime changed; retain evidence and use a new name.')
    directory=inside(root,config['run_path'])
    if (json.loads((directory/'config.json').read_text())!=config
            or json.loads((directory/'meta/run_manifest.json').read_text())['config_sha256']!=_hash(config)):
        raise ValueError('Frozen intruder configuration changed.')
    spectral_path=inside(root,config['spectral_path'])
    if file_hash(spectral_path)!=config['spectral_sha256']:raise ValueError('Base spectral plan changed.')
    spectral,audit=load_spectral(spectral_path,root);cache_dir=inside(root,spectral['run_path'])
    if file_hash(cache_dir/'complete.json')!=config['spectral_complete_sha256']:raise ValueError('Spectral completion changed.')
    cache=_verify_cache(spectral,cache_dir)
    if (config['sources']!={arm:audit['sources'][arm]['checkpoints']['64'] for arm in ['sft','grpo']}
            or config['modules']!=[r['module'] for r in cache['module_results']]
            or config['execution_kind']!=cache['execution_kind']):
        raise ValueError('Geometry source/inventory changed.')
    return config,directory,spectral,cache


def _units(config):
    return [(m,arm) for m in config['modules'] for arm in config['arms']]


def _key(module,arm):return _module_id(module)+'-'+arm


def _basis(spectral,cache,module,root):
    evidence=next(r for r in cache['module_results'] if r['module']==module)
    path=inside(root,spectral['run_path'])/evidence['arrays_path']
    with np.load(path,allow_pickle=False) as arrays:
        old={k:arrays[k] for k in ['sigma','u','v']}
    return old,evidence


def _adapters(config,root):
    return {arm:factors(inside(root,source['path'])) for arm,source in config['sources'].items()}


def _read_unit(config,directory,spectral,cache,module,arm,adapters,root):
    folder=directory/'units'/_key(module,arm);marker=folder/'complete.json'
    if not marker.exists():return None
    seal=json.loads(marker.read_text())
    if (seal.get('identity')!=dict(config_sha256=_hash(config),module=module,arm=arm)
            or set(seal.get('files',{}))!={'geometry.npz','result.json'}):
        raise ValueError('Geometry unit mapping or identity changed.')
    for p,digest in seal['files'].items():
        if file_hash(folder/p)!=digest:raise ValueError('Geometry unit payload changed.')
    row=json.loads((folder/'result.json').read_text())
    with np.load(folder/'geometry.npz',allow_pickle=False) as saved:
        if set(saved.files)!={'new_sigma','new_u','new_v'}:raise ValueError('Unexpected retained geometry arrays.')
        arrays={k:saved[k] for k in saved.files}
    if (arrays['new_sigma'].dtype!=np.float64 or arrays['new_u'].dtype!=np.float32 or arrays['new_v'].dtype!=np.float32
            or any(not np.isfinite(a).all() for a in arrays.values())):
        raise ValueError('Retained SVD precision/finiteness differs.')
    old,ranks=_basis(spectral,cache,module,root);a,b=adapters[arm][module]
    expected=diagnostics(a,b,old,arrays,ranks,zero_update=not np.any(a) or not np.any(b))
    if any(row.get(k)!=v for k,v in expected.items()) or row.get('module')!=module or row.get('arm')!=arm:
        raise ValueError('Intruder/rotation summary does not reconstruct from saved singular geometry.')
    proof=row['decomposition']
    if (not 0<=proof['full_reconstruction_relative_error']<1e-6
            or not 0<=proof['full_orthogonality_error']<=1e-6
            or not 0<=proof['retained_orthogonality_error']<=1e-6
            or proof['retained_columns']!=arrays['new_u'].shape[1]
            or arrays['new_u'].shape[1]!=arrays['new_v'].shape[1]):
        raise ValueError('Updated SVD reconstruction/orthogonality evidence failed.')
    return row


def _summary(config,rows):
    counts={arm:{state:sum(r['arm']==arm and r['primary']['status']==state for r in rows)
                 for state in ['eligible-intruders','no-intruders','undecidable-at-or-below-chance']} for arm in config['arms']}
    return dict(schema_version=1,state='complete',geometry_complete=True,unit_count=len(rows),module_count=len(config['modules']),
        arms=config['arms'],execution_kind=config['execution_kind'],real_model_execution=False,
        real_weight_geometry=config['execution_kind']!='synthetic-cpu-fixture',small_update_skip=False,
        eligibility=counts,module_results=rows,
        limits=['Geometry of one saved training seed per arm; no behavior or causal intervention is measured.',
                'Undecidable at/below chance is separate from no intruders.',
                'Tied spectral boundaries and individual tied singular vectors are not uniquely identified.',
                'Chance sqrt(m/d_out) is an isotropic RMS reference, not a statistical significance test.',
                'No rotation-under-one-degree prediction; small updates are never skipped.'])


def _tables(directory,rows):
    candidates=[];levels=[]
    for row in rows:
        for c in row['candidates']:
            candidates.append(dict(module=row['module'],arm=row['arm'],index=c['index'],sigma=c['sigma'],
                gap_before=c['gap_before'],gap_after=c['gap_after'],numerically_degenerate=c['numerically_degenerate'],
                detection_candidate=c['detection_candidate'],primary_control_candidate=c['primary_control_candidate'],
                **{f'overlap_norm_{v}':c['overlap_norm'][v] for v in ['30','50','70']},
                **{f'chance_norm_{v}':c['chance_norm'][v] for v in ['30','50','70']}))
        for level,data in row['levels'].items():
            levels.append(dict(module=row['module'],arm=row['arm'],energy_level=int(level),k=data['k'],m=data['m'],
                chance_norm=data['chance_norm'],base_gap=data['base_boundary']['gap'],updated_gap=data['updated_boundary']['gap'],
                update_operator_norm=row['update_operator_norm'],update_over_sigma=data['update_over_sigma']['value'],
                update_over_sigma_status=data['update_over_sigma']['status'],update_over_gap=data['update_over_gap']['value'],
                update_over_gap_status=data['update_over_gap']['status'],
                max_left_degrees=data['rotation']['max_left_degrees'],max_right_degrees=data['rotation']['max_right_degrees'],
                primary_status=data['threshold_sweep']['0.5']['status']))
    for name,values in [('candidates.csv',candidates),('geometry.csv',levels)]:
        path=directory/'results'/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)


def run(path,output_root,*,dependencies=None,max_new_units=None,progress=True):
    root=Path(output_root).resolve();config,directory,spectral,cache=_load(path,root)
    if max_new_units is not None and (type(max_new_units) is not int or max_new_units<0):raise ValueError('Invalid unit limit.')
    deps=dependencies or LocalWeights();adapters=_adapters(config,root);rows=[];new=0
    with _writer(directory):
        if (directory/'complete.json').exists():return _verify(config,directory,spectral,cache,root,adapters)
        try:
            for module in config['modules']:
                matrix=None
                for arm in config['arms']:
                    row=_read_unit(config,directory,spectral,cache,module,arm,adapters,root)
                    if row is None:
                        if max_new_units is not None and new>=max_new_units:
                            return _state(directory,'incomplete',len(rows),len(_units(config)))
                        begin=time.monotonic()
                        if progress:print(f'Stage 0 geometry: {len(rows)}/{len(_units(config))} sealed; {_module_id(module)}; {arm}; exact CPU SVD.',flush=True)
                        if matrix is None:
                            entry=spectral['inventory']['modules'][module]
                            matrix=deps.matrix({**entry,'path':str(inside(root,entry['path']))})
                        old,ranks=_basis(spectral,cache,module,root);a,b=adapters[arm][module]
                        arrays,evidence=measure(matrix,a,b,old,ranks)
                        folder=directory/'units'/_key(module,arm);folder.mkdir(parents=True,exist_ok=True)
                        with (folder/'geometry.npz.tmp').open('wb') as stream:np.savez(stream,**arrays)
                        (folder/'geometry.npz.tmp').replace(folder/'geometry.npz')
                        row=dict(module=module,module_id=_module_id(module),arm=arm,**evidence,
                            seconds=time.monotonic()-begin,retained_bytes=(folder/'geometry.npz').stat().st_size,arrays_path=f'units/{_key(module,arm)}/geometry.npz')
                        _write_json(folder/'result.json',row)
                        _write_json(folder/'complete.json',dict(identity=dict(config_sha256=_hash(config),module=module,arm=arm),
                            files={p:file_hash(folder/p) for p in ['geometry.npz','result.json']}))
                        new+=1
                        if progress:print(f'Stage 0 geometry: sealed {_key(module,arm)}; {row["seconds"]:.2f}s; {(folder/"geometry.npz").stat().st_size} bytes.',flush=True)
                    rows.append(row);_state(directory,'running',len(rows),len(_units(config)))
            # Revalidate source hashes at publication, after the long CPU work.
            _load(path,root)
            result=_summary(config,rows)
            _write_json(directory/'results/results.json',result);_tables(directory,rows)
            from pilot_eval.stage0_intruder_render import render
            render(config,result,directory)
            paths=[directory/'config.json',directory/'meta/run_manifest.json',directory/'report.html']
            paths += [p for sub in ['units','results','plots'] for p in (directory/sub).rglob('*') if p.is_file()]
            _write_json(directory/'complete.json',dict(config_sha256=_hash(config),files={str(p.relative_to(directory)):file_hash(p) for p in paths}))
            _state(directory,'complete',len(rows),len(_units(config)));return result
        except (Exception,KeyboardInterrupt) as exc:
            _state(directory,'failed',len(rows),len(_units(config)),error=str(exc))
            log=directory/'logs/errors.jsonl';log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a') as stream:stream.write(json.dumps(dict(error=str(exc),type=type(exc).__name__,module=module if 'module' in locals() else None,arm=arm if 'arm' in locals() else None))+'\n')
            raise


def _verify(config,directory,spectral,cache,root,adapters):
    seal=json.loads((directory/'complete.json').read_text())
    required={'config.json','meta/run_manifest.json','report.html','results/results.json','results/candidates.csv','results/geometry.csv',
              'plots/rotation.svg','plots/rotation.png','plots/overlap.svg','plots/overlap.png','plots/inputs.json'}
    required|={f'units/{_key(m,a)}/{p}' for m,a in _units(config) for p in ['geometry.npz','result.json','complete.json']}
    if seal.get('config_sha256')!=_hash(config) or not required<=seal.get('files',{}).keys():
        raise ValueError('Geometry completion identity/inventory mismatch.')
    for p,digest in seal['files'].items():
        if file_hash(inside(directory,p))!=digest:raise ValueError('Geometry completed payload changed: '+p)
    rows=[_read_unit(config,directory,spectral,cache,m,a,adapters,root) for m,a in _units(config)]
    if any(r is None for r in rows):raise ValueError('Geometry has missing module/arm units.')
    expected=_summary(config,rows)
    if json.loads((directory/'results/results.json').read_text())!=expected:raise ValueError('Geometry aggregate disagrees with sealed modules.')
    return expected


def verify(path,output_root):
    root=Path(output_root).resolve();config,directory,spectral,cache=_load(path,root)
    return _verify(config,directory,spectral,cache,root,_adapters(config,root))


def status(path,output_root):
    root=Path(output_root).resolve();config,directory,spectral,cache=_load(path,root);adapters=_adapters(config,root)
    rows=[_read_unit(config,directory,spectral,cache,m,a,adapters,root) for m,a in _units(config)]
    complete=(directory/'complete.json').exists()
    if complete:_verify(config,directory,spectral,cache,root,adapters)
    owner=directory/'writer.json'
    return dict(geometry_complete=complete,completed_units=len(rows) if complete else sum(r is not None for r in rows),total_units=len(rows),
        execution_kind=config['execution_kind'],report=str(directory/'report.html'),
        writer=json.loads(owner.read_text()) if owner.exists() else None)


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
    return dict(state='ownership-released',scope='writer only; sealed geometry retained')
