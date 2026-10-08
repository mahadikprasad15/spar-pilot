"""CPU-only signed direction comparison of verified learned/control reports."""
import csv
import json
from pathlib import Path
import numpy as np
from pilot_eval.activation_measurement import load_execution, verify_measurement, _compatible_runtime
from pilot_eval.grpo_prepare import _within
from pilot_eval.run import _write_json, _write_state
from pilot_eval.sft import safe_name
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _save_frozen, _hash


def _report(path, root):
    directory=_within(root,path)
    marker=json.loads((directory/'complete.json').read_text())
    identity=json.loads((directory/'config.json').read_text())
    if marker['identity_sha256']!=_hash(identity): raise ValueError('activation report identity changed')
    required={'config.json','results/results.json','results/mean-vectors.npz'}
    if not required.issubset(marker['files']): raise ValueError('activation report lacks required evidence')
    for relative,digest in marker['files'].items():
        if file_hash(_within(directory,directory/relative))!=digest: raise ValueError('activation report payload changed')
    execution=identity['execution'];config_path=root/'plans'/execution['run_id']/'activation.execution.json'
    actual,prepared,rows=load_execution(config_path,root)
    if actual!=execution or prepared!=identity['prepared']: raise ValueError('activation report source changed')
    verify_measurement(config_path,root)
    if file_hash(root/execution['run_path']/'complete.json')!=identity['measurement_complete_sha256']:
        raise ValueError('activation report measurement seal changed')
    result=json.loads((directory/'results/results.json').read_text())
    if not result['report_complete'] or result['provenance']!=identity: raise ValueError('activation report is incomplete')
    with np.load(directory/'results/mean-vectors.npz',allow_pickle=False) as arrays:
        vectors={k:arrays[k].copy() for k in arrays.files}
    if any(not np.isfinite(v).all() for v in vectors.values()): raise ValueError('nonfinite activation report vector')
    return result,execution,prepared,rows,vectors


def report_grpo_writes(grpo_report,control_report,sft_report,output_root,name):
    root=Path(output_root).resolve();name=safe_name(name)
    sources={label:_within(root,path) for label,path in
             [('grpo',grpo_report),('random',control_report),('sft',sft_report)]}
    products={label:_report(path,root) for label,path in sources.items()}
    _,reference,prepared,rows,_=products['grpo']
    if prepared.get('source_contract',{}).get('kind')!='grpo': raise ValueError('GRPO comparison needs a GRPO source')
    random=products['random'][2]
    if (random.get('source_contract',{}).get('kind')!='random-rank1-control'
            or root/random['source_contract']['parent_prepared_path']!=root/reference['prepared_path']
            or products['sft'][2]['protocol_version']!='pilot3-write-v1'):
        raise ValueError('comparison source/control lineage mismatch')
    for label,(_,execution,config,items,_) in products.items():
        if (items!=rows or config['items_sha256']!=prepared['items_sha256']
                or config['source_evidence']['base_sha256']!=prepared['source_evidence']['base_sha256']
                or any(config[k]!=prepared[k] for k in ['model','model_revision','tokenizer_revision','dtype','views'])
                or not _compatible_runtime(execution['runtime'],reference['runtime'])
                or execution['batches']!=reference['batches'] or execution['batch_size']!=reference['batch_size']):
            raise ValueError('write comparison requires matching inputs, base, numerical runtime and batch membership')
    identity=dict(protocol='pilot4-write-comparison-v1',sources={label:dict(path=str(path.relative_to(root)),
                      complete_sha256=file_hash(path/'complete.json')) for label,path in sources.items()},
                  implementation_sha256=file_hash(Path(__file__)),endpoint=64,
                  weightings=['token','example'],random_realizations=1,inputs_sha256=prepared['items_sha256'])
    directory=root/'reports'/name
    with run_lock(directory):
        _save_frozen(directory/'config.json',identity)
        if (directory/'complete.json').exists():
            marker=json.loads((directory/'complete.json').read_text())
            if marker['identity_sha256']!=_hash(identity): raise ValueError('write comparison identity changed')
            for relative,digest in marker['files'].items():
                if file_hash(_within(directory,directory/relative))!=digest: raise ValueError('write comparison payload changed')
            return json.loads((directory/'results/results.json').read_text())
        _write_state(directory,'running',0,3)
        try:
            magnitudes={label:product[0]['measurements'] for label,product in products.items()}
            block={label:{(r['view'],r['layer'],r['weighting']):r for r in product[0]['measurements']
                          if r['kind']=='block' and r['step']==64} for label,product in products.items()}
            cosines=[];vectors={}
            for first,second in [('random','grpo'),('random','sft'),('sft','grpo')]:
                for view in ['question','solution','user']:
                    for weighting in ['token','example']:
                        for layer in range(28):
                            key=f'64-{view}-{layer}-block-{weighting}-mean_delta'
                            a,b=products[first][4][key],products[second][4][key]
                            if a.shape!=b.shape or a.ndim!=1: raise ValueError('mean-vector shape mismatch')
                            vectors[first+'-'+key]=a;vectors[second+'-'+key]=b
                            norms=[float(np.linalg.norm(v)) for v in [a,b]]
                            cells=[block[label][view,layer,weighting] for label in [first,second]]
                            reason=('zero_vector' if 0. in norms else
                                    'no_calibrated_resolution' if any(r['direction_resolved'] is None for r in cells) else
                                    'below_resolution' if not all(r['direction_resolved'] for r in cells) else None)
                            cosine=None if reason else float(np.clip(np.dot(a,b)/(norms[0]*norms[1]),-1.,1.))
                            cosines.append(dict(first=first,second=second,view=view,layer=layer,weighting=weighting,
                                                cosine=cosine,defined=reason is None,reason=reason,
                                                first_norm=norms[0],second_norm=norms[1],
                                                first_resolution=cells[0]['direction_resolution_relative'],
                                                second_resolution=cells[1]['direction_resolution_relative']))
            result=dict(report_complete=True,random_realizations=1,cosines=cosines,magnitudes=magnitudes,
                        identity=identity,defined_cosines=sum(r['defined'] for r in cosines),total_cosines=len(cosines),
                        limits=['One random realization is not a null distribution or significance threshold.',
                                'Weight-norm matching does not match activation magnitudes.',
                                'Signed cosines are descriptive; unresolved directions remain undefined.',
                                'Existing per-arm intervals condition on the fixed cohorts and one training seed.'])
            _write_json(directory/'results/results.json',result)
            from pilot_eval.activation_workflow import _save_arrays
            _save_arrays(directory/'results/mean-vectors.npz',vectors)
            with (directory/'results/cosines.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(cosines[0]));writer.writeheader();writer.writerows(cosines)
            lines=['# '+name,'','## Direction coverage','',f"Defined: {result['defined_cosines']}/{result['total_cosines']}",'',
                   '## Interpretation limits','']+['- '+s for s in result['limits']]
            lines+=['','All three views and both weightings are saved in `cosines.csv`.','',
                    'Per-arm block/module magnitudes, denominator diagnostics, coverage and intervals are in `results.json`.',
                    'Source reports retain write-depth and module plots. No training or inference ran to make this comparison.']
            (directory/'results/report.md').write_text('\n'.join(lines)+'\n')
            # Reverify sources before binding completion to their saved seals.
            for label,path in sources.items():
                _report(path,root)
                if file_hash(path/'complete.json')!=identity['sources'][label]['complete_sha256']:
                    raise ValueError('source report changed during comparison')
            files=[directory/'config.json',*list((directory/'results').iterdir())]
            _write_json(directory/'complete.json',dict(identity_sha256=_hash(identity),
                files={str(p.relative_to(directory)):file_hash(p) for p in files}))
            _write_state(directory,'completed',3,3)
            return result
        except BaseException as error:
            _write_state(directory,'failed',0,3,str(error));raise
