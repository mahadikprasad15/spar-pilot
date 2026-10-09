"""CPU-only paired-arm comparison and hash-indexed scientific handoff."""
import json
from pathlib import Path
import numpy as np
from pilot_eval.grpo_prepare import _within, load_grpo_prepared
from pilot_eval.sft import safe_name, load_sft
from pilot_eval.training import file_hash, training_directory, run_lock
from pilot_eval.run import _write_json, _write_state, run_evaluation
from pilot_eval.workflow import _hash, _save_frozen

REQUIRED = {'grpo_evaluation','sft_writes','grpo_writes','control_writes','tokens_execution'}
STEPS = [0,8,16,32,64]
LIMITS = [
    'One training seed; item bootstrap does not estimate variation across training seeds.',
    'SFT gold solutions and GRPO sampled completions differ: this is not an objective-only causal comparison.',
    'Equal optimizer steps, equal write norm and equal fixed-context KL are different comparisons.',
    'An early SFT checkpoint is a preview, not a lower-learning-rate intervention.',
    'Fixed-context KL is not unconditional generated-policy distance or proof of capability loss.',
    'One random adapter is a reference, not a null percentile, p-value or empirical noise floor.',
    'Weight-norm matching does not guarantee activation-norm matching.',
]


def _selection(path, root):
    path = _within(root,path)
    value = json.loads(path.read_text())
    if set(value)!=REQUIRED or any(not isinstance(v,str) for v in value.values()):
        raise ValueError('final report requires the five explicit source selections')
    sources={key:_within(root,entry) for key,entry in value.items()}
    if any(not entry.exists() for entry in sources.values()):raise ValueError('final handoff source is missing')
    return sources


def _sft_behaviour(path,root,expected_items):
    from pilot_eval.sft_evaluation import evaluation_config,evaluation_directory
    from pilot_eval.scoring import score_gsm8k_flexible_v3
    config,_,items=load_sft(path,root)
    if items!=expected_items:raise ValueError('SFT/GRPO evaluation contents/order differ')
    runtime=json.loads((training_directory(root,config)/'meta/runtime.json').read_text())
    cohorts={};directories=[];provenance={}
    class NoInference:
        def generate_batch(self,*args):raise ValueError('incomplete SFT evaluation; inference forbidden in report')
    for label in ['baseline',*map(str,STEPS)]:
        _,expected=evaluation_config(config,root,label)
        directory=evaluation_directory(root,expected)
        if not (directory/'results/results.json').is_file():raise ValueError('SFT behavioural source is incomplete')
        saved=json.loads((directory/'config.json').read_text())
        if saved!={**expected,'runtime':runtime}:raise ValueError('SFT evaluation config/runtime differs')
        run_evaluation(saved,items,NoInference(),root)
        records=[json.loads(line) for line in (directory/'results/responses.jsonl').read_text().splitlines()]
        if [r['id'] for r in records]!=[r['id'] for r in items]:raise ValueError('SFT response item alignment differs')
        cohorts[label]=[{**r,'flexible_v3':score_gsm8k_flexible_v3(r['generated_text'],r['gold'],r['stop_reason']=='cap')} for r in records]
        directories.append(directory)
        provenance[label]={str(p.relative_to(root)):file_hash(p) for p in
                           [directory/'config.json',directory/'results/responses.jsonl',directory/'results/results.json']}
    fields=['id','generated_text','token_count','stop_reason','score']
    if [{k:r[k] for k in fields} for r in cohorts['baseline']]!=[{k:r[k] for k in fields} for r in cohorts['0']]:
        raise ValueError('SFT zero checkpoint differs from untuned responses')
    indices=np.random.default_rng(42).integers(0,len(items),size=(10000,len(items)))
    base=np.array([r['flexible_v3']['correct'] for r in cohorts['0']],float)
    base_length=np.array([r['token_count'] for r in cohorts['0']],float)
    trajectory=[];paired=[]
    for step in STEPS:
        rows=cohorts[str(step)];accuracy=np.array([r['flexible_v3']['correct'] for r in rows],float)
        lengths=np.array([r['token_count'] for r in rows],float)
        change=accuracy-base;length_change=lengths-base_length
        trajectory.append(dict(step=step,strict_accuracy=sum(r['score']['strict']['correct'] for r in rows)/len(rows),
            flexible_accuracy=float(accuracy.mean()),mean_tokens=float(lengths.mean()),median_tokens=float(np.median(lengths)),
            cap_rate=sum(r['stop_reason']=='cap' for r in rows)/len(rows),
            flexible_invalid_rate=sum(r['flexible_v3']['status']=='invalid' for r in rows)/len(rows),
            flexible_change=dict(estimate=float(change.mean()),interval_95=np.percentile(change[indices].mean(1),[2.5,97.5]).tolist()),
            mean_token_change=dict(estimate=float(length_change.mean()),interval_95=np.percentile(length_change[indices].mean(1),[2.5,97.5]).tolist())))
        paired.extend(dict(step=step,item_id=r['id'],baseline=b,adapted=r) for b,r in zip(cohorts['0'],rows))
    return dict(trajectory=trajectory,scorer='gsm8k-flexible-v3',post_hoc_scoring=True,
                source_provenance=provenance,paired_interval=dict(draws=10000,seed=42,unit='paired item'),
                sampled={'available':False,'reason':'SFT sampled evaluation out of scope'}),paired,indices,directories,config


def _evidence(sources,root):
    from pilot_eval.grpo_evaluation import verify_evaluation
    from pilot_eval.grpo_preflight import load_frozen
    from pilot_eval.grpo_write_report import _report
    from pilot_eval.token_measurement_workflow import verify_tokens,_execution
    evaluation,records=verify_evaluation(sources['grpo_evaluation'],root)
    training=json.loads((root/evaluation['training_path']).read_text())
    frozen=load_frozen(root/training['frozen_path'],root)
    plan,*_=load_grpo_prepared(root/frozen['source_path'],root)
    writes={label:_report(sources[label+'_writes'],root) for label in ['sft','grpo','control']}
    if writes['grpo'][2].get('source_contract',{}).get('training_path')!=evaluation['training_path']:
        raise ValueError('behaviour and write checkpoints have different GRPO training sources')
    verify_tokens(sources['tokens_execution'],root)
    token,products=_execution(sources['tokens_execution'],root)
    for arm in ['sft','grpo']:
        if products[arm][0]!=writes[arm][1] or products[arm][1]!=writes[arm][2]:
            raise ValueError('KL and write execution identities differ')
    sft,paired,indices,eval_dirs,sft_config=_sft_behaviour(root/plan['sources']['sft']['path'],root,evaluation['items'])
    if writes['sft'][2]['source_config_path']!=plan['sources']['sft']['path']:
        raise ValueError('SFT behaviour and write sources differ')
    return evaluation,training,frozen,plan,writes,token,sft,paired,indices,eval_dirs,sft_config


def _inventory(root,entries):
    files={}
    for role,path in entries:
        path=_within(root,path)
        selected=[path] if path.is_file() else [p for p in path.rglob('*') if p.is_file()]
        for p in selected:
            if p.name=='.lock' or p.name.endswith('.tmp'):continue
            p=_within(root,p);relative=str(p.relative_to(root))
            row=files.setdefault(relative,dict(path=relative,sha256=file_hash(p),bytes=p.stat().st_size,roles=[]))
            if role not in row['roles']:row['roles'].append(role)
    return [files[k] for k in sorted(files)]


def _identity(sources,root):
    identity={key:dict(path=str(p.relative_to(root)),sha256=file_hash(p) if p.is_file() else file_hash(p/'complete.json'))
              for key,p in sources.items()}
    from importlib.metadata import version
    return dict(protocol='pilot4-final-handoff-v1',sources=identity,matplotlib_version=version('matplotlib'),
                implementation={name:file_hash(Path(__file__).with_name(name)) for name in
                                ['grpo_handoff.py','grpo_handoff_plots.py','grpo_write_report.py','scoring.py']},
                numpy_version=np.__version__)


def report_handoff(selection,output_root,name):
    root=Path(output_root).resolve();name=safe_name(name);sources=_selection(selection,root)
    evidence=_evidence(sources,root)
    directory=root/'reports'/name;identity=_identity(sources,root)
    with run_lock(directory):
        if (directory/'complete.json').exists():return verify_handoff(selection,root,name)
        _save_frozen(directory/'config.json',identity)
        _save_frozen(directory/'meta/run_manifest.json',dict(identity=identity,selection=str(_within(root,selection).relative_to(root))))
        _write_state(directory,'running',0,1)
        try:
            evaluation,training,frozen,plan,writes,token,sft,paired,indices,eval_dirs,sft_config=evidence
            from pilot_eval.grpo_behaviour_report import report_grpo_behaviour
            from pilot_eval.grpo_write_report import report_grpo_writes
            from pilot_eval.token_measurement_workflow import report_tokens
            grpo=report_grpo_behaviour(sources['grpo_evaluation'],root,name+'-behaviour')
            write=report_grpo_writes(sources['grpo_writes'],sources['control_writes'],sources['sft_writes'],root,name+'-writes')
            kl=report_tokens(sources['tokens_execution'],root,name+'-tokens')
            control_path=root/writes['control'][2]['source_contract']['control_path']
            construction=json.loads((control_path/'construction.json').read_text())
            from pilot_eval.grpo_random_control import _factors
            _,_,realized=_factors(control_path)
            modules=[dict(layer=a['layer'],projection=a['projection'],target_norm=a['target_norm'],
                          realized_norm=b['target_norm'],scaling=a['scaling']) for a,b in zip(construction['modules'],realized)]
            matching=dict(objective_only_causal_comparison=False,training_precision=dict(sft=sft_config['dtype'],grpo=plan['dtype']),
                completion_sources=dict(sft='gold solutions',grpo='on-policy sampled completions'),
                completions_per_step=dict(sft=8,grpo=64),training_items_sha256=plan['training_items_sha256'],
                evaluation_items_sha256=plan['evaluation_items_sha256'],fixed_inputs_sha256=plan['measurement_items_sha256'],
                sft_evaluation=dict(decoding=sft_config['source_config']['decoding'],batch_size=sft_config.get('evaluation_batch_size',1)),
                grpo_evaluation=dict(policies=evaluation['policies'],settings=evaluation['settings']),
                source_prompt_contract=plan['source_prompt_contract'],sft_optimizer=sft_config['optimizer'],grpo_optimizer=plan['optimizer'])
            result=dict(report_complete=True,provenance=identity,behaviour=dict(sft=sft,grpo=grpo),writes=write,kl=kl,
                matching=matching,checkpoint_provenance={label:product[2]['source_evidence'] for label,product in writes.items()},
                evaluation_provenance=dict(scorer=evaluation['scorer'],implementation=evaluation['implementation'],
                    checkpoints=evaluation['checkpoints'],training_sha256=evaluation['training_sha256']),
                token_execution_sha256=_hash(token),random_control=dict(realizations=1,modules=modules,construction=construction,
                    magnitudes=write['magnitudes']['random'],cosines=[r for r in write['cosines'] if r['first']=='random']),
                unavailable=dict(sft_gradient_consistency=dict(available=False,reason='not_saved'),sft_sampled=sft['sampled']),
                grpo_training_diagnostics=json.loads((root/training['run_path']/'results/results.json').read_text()),
                historical_reference=dict(accuracy_before=.640,accuracy_after=.420,mean_tokens_before=288,mean_tokens_after=77),
                limits=LIMITS)
            _write_json(directory/'results/sft-bootstrap-indices.json',indices.tolist())
            _write_json(directory/'results/sft-paired-items.json',paired)
            from pilot_eval.grpo_handoff_plots import plot_handoff
            result['plot_files']=plot_handoff(result,directory/'results')
            # Include source plans, all raw outputs, numerical calibration/profile
            # files and optimizer/RNG state. Incomplete attempts are retained as
            # diagnostics, not reclassified as accepted scientific history.
            entries=[('selection',_within(root,selection)),('sft-training',training_directory(root,sft_config)),
                     ('grpo-training',root/training['run_path']),('grpo-evaluation',root/evaluation['run_path']),
                     ('grpo-frozen',root/training['frozen_path']),('grpo-prepared',root/frozen['source_path']),
                     ('tokens',root/token['run_path']),('tokens-plan',sources['tokens_execution'].parent),
                     ('sft-plan',(root/plan['sources']['sft']['path']).parent)]
            flight_path=root/frozen['preflight_path']
            flight=json.loads(flight_path.read_text())
            baseline_path=root/flight['baseline']['config']
            baseline=json.loads(baseline_path.read_text())
            entries += [('grpo-training-plan',(root/evaluation['training_path']).parent),
                        ('grpo-evaluation-plan',sources['grpo_evaluation'].parent),
                        ('grpo-frozen-plan',(root/training['frozen_path']).parent),
                        ('grpo-prepared-plan',(root/frozen['source_path']).parent),
                        ('grpo-preparation',root/plan['run_path']),
                        ('grpo-preflight-plan',flight_path.parent),
                        ('grpo-preflight',root/flight['run_path']),
                        ('grpo-baseline-plan',baseline_path.parent),
                        ('grpo-baseline',root/baseline['run_path']),
                        ('grpo-controls',root/'runs/diagnostics/pilot-4'/frozen['controls_name']),
                        ('sft-original-evaluation-cohort',root/sft_config['source_config']['items_path'])]
            entries += [('sft-evaluation',p) for p in eval_dirs]
            for label,product in writes.items():
                execution,prepared=product[1],product[2]
                entries += [(label+'-write-report',sources[label+'_writes']),
                            (label+'-write-measurement',root/execution['run_path']),
                            (label+'-write-plan',root/'plans'/execution['run_id']),
                            (label+'-prepared-plan',(root/execution['prepared_path']).parent),
                            (label+'-preparation',root/prepared['run_path']),
                            (label+'-source-config',root/prepared['source_config_path']),
                            (label+'-fixed-inputs',root/prepared['items_path'])]
                # Execution/calibration/profile pointers bind all prerequisites.
                for key in ['profile_path','calibration_path']:
                    if isinstance(execution.get(key),str):
                        target=root/execution[key];entries.append((label+'-'+key,target))
                        if target.is_file():
                            payload=json.loads(target.read_text())
                            if isinstance(payload.get('run_path'),str):entries.append((label+'-'+key+'-run',root/payload['run_path']))
            entries += [('subreport',root/'reports'/(name+suffix)) for suffix in ['-behaviour','-writes','-tokens']]
            result['download_inventory']=_inventory(root,entries)
            _write_json(directory/'results/download-inventory.json',result['download_inventory'])
            _write_json(directory/'results/results.json',result)
            (directory/'results/report.md').write_text(_markdown(name,result))
            _evidence(sources,root)
            if identity!=_identity(sources,root):raise ValueError('handoff source changed during reporting')
            for row in result['download_inventory']:
                if file_hash(root/row['path'])!=row['sha256']:raise ValueError('download source changed during reporting')
            files=[directory/'config.json',directory/'meta/run_manifest.json',*sorted((directory/'results').iterdir())]
            _write_json(directory/'complete.json',dict(identity_sha256=_hash(identity),files={str(p.relative_to(directory)):file_hash(p) for p in files}))
            _write_state(directory,'completed',1,1)
            return result
        except BaseException as exc:
            _write_state(directory,'failed',0,1,str(exc));raise


def verify_handoff(selection,output_root,name):
    root=Path(output_root).resolve();sources=_selection(selection,root);directory=root/'reports'/safe_name(name)
    _evidence(sources,root);identity=_identity(sources,root)
    marker=json.loads((directory/'complete.json').read_text())
    required={'config.json','meta/run_manifest.json','results/results.json','results/report.md','results/download-inventory.json',
              'results/sft-bootstrap-indices.json','results/sft-paired-items.json'}
    if marker['identity_sha256']!=_hash(identity) or not required.issubset(marker['files']):raise ValueError('final handoff completion identity/inventory changed')
    for relative,digest in marker['files'].items():
        if file_hash(_within(directory,relative))!=digest:raise ValueError('final handoff payload changed')
    if json.loads((directory/'config.json').read_text())!=identity:raise ValueError('final handoff config changed')
    result=json.loads((directory/'results/results.json').read_text())
    if not result['report_complete'] or result['provenance']!=identity:raise ValueError('final handoff result identity changed')
    for plot in result['plot_files']:
        if 'results/'+plot not in marker['files']:raise ValueError('final handoff plot missing from seal')
    if json.loads((directory/'results/download-inventory.json').read_text())!=result['download_inventory']:
        raise ValueError('final handoff download inventory differs')
    for row in result['download_inventory']:
        p=_within(root,row['path'])
        if p.stat().st_size!=row['bytes'] or file_hash(p)!=row['sha256']:raise ValueError('raw download inventory source changed')
    return result


def _markdown(name,result):
    lines=['# '+name+': verified paired-arm comparison','','## Behaviour','',
           '| Arm | Mode | Step | Strict | Flexible v3 | Mean tokens | Cap rate |','|---|---|---:|---:|---:|---:|---:|']
    for r in result['behaviour']['sft']['trajectory']:
        lines.append(f"| SFT | greedy | {r['step']} | {r['strict_accuracy']:.3f} | {r['flexible_accuracy']:.3f} | {r['mean_tokens']:.2f} | {r['cap_rate']:.3f} |")
    for key,r in result['behaviour']['grpo']['summaries'].items():
        mode,step=key.split('-');lines.append(f"| GRPO | {mode} | {step} | {r['strict_accuracy']:.3f} | {r['flexible_accuracy']:.3f} | {r['mean_tokens']:.2f} | {r['cap_rate']:.3f} |")
    lines+=['','SFT flexible-v3 is recomputed uniformly from verified saved responses; this is a disclosed post-hoc scorer correction. No inference ran.',
            '','Paired changes and 95% intervals are saved in results.json. Bootstrap: 10,000 paired items, seed 42; invalid and capped items stay in the denominator.',
            '','## Matching ledger','', '```json',json.dumps(result['matching'],indent=2),'```','',
            '## Plots and formulas','',
            'Behaviour: mean correctness and generated tokens over 150 held-out items. Sampled accuracy averages eight draws per item; not pass-at-eight.',
            'Write: ||mean(adapted − untuned)|| / mean||untuned|| at each block. Direct-module ratio is mean||LoRA branch|| / mean||untuned linear output||. Both frozen weightings are shown.',
            'Direction: signed dot product / product of mean-vector norms. Unresolved/zero directions are undefined; gaps are not zeros.',
            'KL: sum_v p(v) log[p(v)/q(v)], tuned || untuned, in nats, on fixed prefixes assigned to next-target views. Token and equal-example means differ.',
            '','Worked examples: vectors (3,4) and (4,−3) each have norm 5 but cosine 0. Two tokens at KL 0.1 and eight at KL 0.5 give token mean 0.42, equal-example mean 0.30.',
            '','Write intervals are the existing per-arm paired-example bootstrap; KL/cosine curves have no new empirical noise or seed intervals. Behaviour change panels use paired item intervals, not independent draws.',
            '','## Unavailable diagnostics','',json.dumps(result['unavailable'],indent=2),
            '','## Random control','', '196 per-module target/realized norms, activation magnitudes and signed cosines are saved in results.json. One realization cannot estimate a null percentile.',
            '','## Limits','']+['- '+s for s in result['limits']]
    lines+=['','Historical context: accuracy 0.640→0.420, mean length 288→77. These are not reproduction gates.',
            '','## Recovery and downloads','',
            'Training resumes at sealed checkpoints; up to 32 optimizer steps may be redone. Incomplete attempts remain diagnostics and are excluded from accepted history. Evaluation and measurements reuse their sealed batches.',
            'After all scientific processes exit successfully and verification passes, disconnect the GPU. Reporting/downloads are CPU-only.',
            'download-inventory.json lists root-relative paths, roles, byte counts and SHA256 hashes. Preserve both source artifacts and this final report directory. Do not overwrite old runs.','']
    return '\n'.join(lines)


def export_handoff(selection,output_root,name,archive_name):
    """Stream a verified inventory and report into an immutable Drive tar bundle."""
    import tarfile
    root=Path(output_root).resolve();result=verify_handoff(selection,root,name)
    report=root/'reports'/safe_name(name);directory=root/'exports'/safe_name(archive_name)
    report_seal=file_hash(report/'complete.json')
    identity=dict(protocol='pilot4-handoff-export-v1',report_path=str(report.relative_to(root)),report_complete_sha256=report_seal)
    with run_lock(directory):
        _save_frozen(directory/'config.json',identity)
        if (directory/'complete.json').exists():
            marker=json.loads((directory/'complete.json').read_text())
            if marker['identity']!=identity:raise ValueError('export report identity changed')
            if set(marker['files'])!={'bundle.tar','manifest.json','config.json'}:raise ValueError('incomplete export inventory')
            for relative,digest in marker['files'].items():
                if file_hash(directory/relative)!=digest:raise ValueError('export archive payload changed')
            return json.loads((directory/'manifest.json').read_text())
        _write_state(directory,'running',0,1)
        try:
            files={r['path']:r for r in result['download_inventory']}
            for relative,digest in json.loads((report/'complete.json').read_text())['files'].items():
                p=report/relative;files[str(p.relative_to(root))]=dict(path=str(p.relative_to(root)),sha256=digest,bytes=p.stat().st_size,roles=['final-report'])
            files[str((report/'complete.json').relative_to(root))]=dict(path=str((report/'complete.json').relative_to(root)),sha256=report_seal,bytes=(report/'complete.json').stat().st_size,roles=['final-report-seal'])
            pending=directory/'bundle.tar.tmp'
            with tarfile.open(pending,'w') as archive:
                for relative,row in sorted(files.items()):
                    p=_within(root,relative)
                    if file_hash(p)!=row['sha256']:raise ValueError('export source changed before archiving')
                    archive.add(p,arcname=relative,recursive=False)
                    if file_hash(p)!=row['sha256']:raise ValueError('export source changed while archiving')
            # Read archive payloads rather than assume add() copied the hash-checked bytes.
            import hashlib
            with tarfile.open(pending,'r') as archive:
                members=archive.getmembers()
                if {m.name for m in members}!=set(files) or len(members)!=len(files):raise ValueError('export member inventory mismatch')
                for member in members:
                    if not member.isfile():raise ValueError('export includes non-file member')
                    digest=hashlib.sha256()
                    with archive.extractfile(member) as stream:
                        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
                    if digest.hexdigest()!=files[member.name]['sha256']:raise ValueError('export member hash mismatch')
            if file_hash(report/'complete.json')!=report_seal:raise ValueError('report changed while exporting')
            pending.replace(directory/'bundle.tar')
            manifest=dict(identity=identity,archive_path=str((directory/'bundle.tar').relative_to(root)),
                          archive_bytes=(directory/'bundle.tar').stat().st_size,files=[files[k] for k in sorted(files)])
            _write_json(directory/'manifest.json',manifest)
            _write_json(directory/'complete.json',dict(identity=identity,files={p:file_hash(directory/p) for p in ['config.json','manifest.json','bundle.tar']}))
            _write_state(directory,'completed',1,1)
            return manifest
        except BaseException as exc:
            _write_state(directory,'failed',0,1,str(exc));raise
