"""CPU push summaries and a separate, prediction-bound endpoint reveal."""
import csv
import html
import itertools
import json
from pathlib import Path

import numpy as np

from pilot_eval.run import _write_json
from pilot_eval.stage0_evaluation import _jsonl
from pilot_eval.stage0_push_plan import load
from pilot_eval.stage0_push_math import cosine,orient_endpoint,summarize_vectors
from pilot_eval.stage0_pushes import read_unit,verify_measurement
from pilot_eval.stage0_sources import factors,inside
from pilot_eval.stage0_spectral import _load,_verify_cache
from pilot_eval.stage0_spectral_math import projection,chance_band
from pilot_eval.stage0_workflow import _writer,_state,_now
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash,_save_frozen


def _seal(directory,filename,files,config):
    _write_json(directory/filename,dict(config_sha256=_hash(config),files={n:file_hash(directory/n) for n in files}))


def _verify_seal(config,directory,filename):
    value=json.loads((directory/filename).read_text())
    if value['config_sha256']!=_hash(config):raise ValueError('Push report identity changed.')
    for name,digest in value['files'].items():
        if file_hash(inside(directory,name))!=digest:raise ValueError('Push report/prediction evidence changed.')
    return value


def report(path,output_root):
    root=Path(output_root).resolve();verify_measurement(path,root);config,directory,units=load(path,root)
    with _writer(directory):
        if (directory/'summary.complete.json').exists():
            _verify_seal(config,directory,'summary.complete.json');return json.loads((directory/'results/summary.json').read_text())
        spectral,_=_load(inside(root,config['spectral_path']),root);cache=_verify_cache(spectral,inside(root,spectral['run_path']))
        cache_rows={r['module']:r for r in cache['module_results']}
        saved={u['key']:read_unit(config,directory,u,root) for u in units};rows=[];vectors={};comparisons=[]
        for i,module in enumerate(config['modules']):
            row=cache_rows[module]
            with np.load(inside(root,spectral['run_path'])/row['arrays_path'],allow_pickle=False) as basis:principal=basis['u'][:,:row['k50']]
            chance=chance_band(principal)
            for case in config['cases']:
                means={}
                for objective in config['objectives']:
                    selected=[u for u in units if u['case']==case and u['objective']==objective]
                    stats=summarize_vectors(np.array([saved[u['key']][f'gradient_b_{i}'].reshape(-1) for u in selected]),[u['token_count'] for u in selected],[u['question_count'] for u in selected])
                    stem=f'{case}-{objective}-{i}';means[objective]=np.array(stats['mean_push'])
                    vector_stats={key:stats.pop(key) for key in ['mean_push','mean_push_full_batches','token_weighted_push']}
                    for key,value in vector_stats.items():
                        if value is not None:vectors[stem+'-'+key]=np.array(value)
                    mean=means[objective]
                    rows.append(dict(case=case,a0_sha256=config['cases'][case]['a0_sha256'],objective=objective,module=module,vector_key=stem,
                        **stats,mean_push_norm=float(np.linalg.norm(mean)),pi_out=projection(mean,principal),chance=chance,
                        mean_without_partial_pi_out=None if vector_stats['mean_push_full_batches'] is None else projection(vector_stats['mean_push_full_batches'],principal),
                        token_weighted_pi_out=projection(vector_stats['token_weighted_push'],principal),
                        cosine_mean_vs_token_weighted=cosine(mean,vector_stats['token_weighted_push']),
                        cross_objective_norms_comparable=False,questions=sum(u['question_count'] for u in selected),
                        dead_groups=sum(u['dead_groups'] for u in selected),batch_direction_cosines=[cosine(mean,-saved[u['key']][f'gradient_b_{i}'].reshape(-1)) for u in selected],
                        cosine_to_initial_b=None,initial_b_note='exact zero factor: alignment undefined'))
                for first,second in itertools.combinations(config['objectives'],2):
                    comparisons.append(dict(case=case,a0_sha256=config['cases'][case]['a0_sha256'],module=module,first=first,second=second,
                        cosine=cosine(means[first],means[second]),sign_push_cosine=cosine(np.sign(means[first]),np.sign(means[second])),
                        scope='direction only; GRPO uses all512, CE uses identical eligible batches'))
        summary=dict(config_sha256=_hash(config),real_gpu_measurement=config['execution_kind']=='real-cuda',endpoint_revealed=False,
            cohorts=config['cohorts'],initialization_audit=config['initialization_audit'],modules=rows,comparisons=comparisons,
            primary=config['primary'],secondary=config['secondary'],limits=['Initial negative gradient, not an AdamW historical update.','Zero vectors have undefined cosines/ratios.','Selection on observed correctness changes the CE cohort.','Cross-objective raw norms are not comparable.'])
        _write_json(directory/'results/summary.json',summary);_jsonl(directory/'results/modules.jsonl',rows);_jsonl(directory/'results/directions.jsonl',comparisons)
        np.savez(directory/'results/mean-pushes.npz',**vectors)
        with (directory/'results/summary.csv').open('w',newline='') as stream:
            fields=['case','objective','module','questions','full_batches','partial_batches','consistency','pi_out','mean_without_partial_pi_out','token_weighted_pi_out','mean_push_norm','cross_objective_norms_comparable']
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
        esc=lambda value:html.escape(str(value))
        table=''.join('<tr>'+''.join('<td>'+esc(r[k])+'</td>' for k in ['case','objective','module','consistency','pi_out','partial_batches'])+'</tr>' for r in rows)
        (directory/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Initial pushes</title><style>body{font:16px/1.55 system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#243c43;background:#f6f8f5}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ccd}</style><h1>Initial negative-gradient pushes</h1><p>'+esc(config['execution_kind'])+'. This is the pre-reveal summary. A separate endpoint report is created after prediction/reveal.</p><p>Gold and own-correct use the same eligible question batches. GRPO keeps all 512 questions. Compare directions within one A₀ hash; raw norms across objectives have no common scale. Consistency excludes partial batches; means with and without them are in the raw summary.</p><table><thead><tr><th>A₀</th><th>Objective</th><th>Module</th><th>Consistency</th><th>π out</th><th>Partial batches</th></tr></thead><tbody>'+table+'</tbody></table><p><a href="results/summary.json">Full JSON</a> · <a href="results/summary.csv">CSV</a> · <a href="results/mean-pushes.npz">Mean vectors</a> · <a href="results/directions.jsonl">Direction comparisons</a> · <a href="endpoints.html">Endpoint report after reveal</a></p></html>')
        _seal(directory,'summary.complete.json',['results/summary.json','results/modules.jsonl','results/directions.jsonl','results/mean-pushes.npz','results/summary.csv','report.html'],config)
        _state(directory,'summary-complete-endpoint-pending',len(units),len(units));return dict(summary_complete=True,endpoint_revealed=False,report=str(directory/'report.html'))


def _exposures(config,root):
    # Spectral exposure binds the parsed-config hash rather than the file hash.
    digest=_hash(json.loads(inside(root,config['spectral_path']).read_text()))
    candidates=list((root/'reports').glob('*/exposure.json'))+list((root/'runs/stage0').glob('*/e05_pushes/exposure.json'))
    return {str(p.relative_to(root)):file_hash(p) for p in candidates if digest in [json.loads(p.read_text()).get('config_sha256'),json.loads(p.read_text()).get('spectral_config_sha256')]}


def predict(path,output_root,review_path):
    root=Path(output_root).resolve();config,directory,_=load(path,root);_verify_seal(config,directory,'summary.complete.json')
    review=json.loads(Path(review_path).read_text())
    if set(review)!={'reviewer','notes','predictions','prior_exposure'} or type(review['prior_exposure']) is not bool or any(not str(review[k]).strip() for k in ['reviewer','notes','predictions']):raise ValueError('Explicit predictions and prior-exposure declaration required.')
    with _writer(directory):
        target=directory/'predictions.json'
        if target.exists():
            value=json.loads(target.read_text())
            if value['review']!=review:raise ValueError('Predictions immutable; retain the original declaration.')
            return value
        exposure=_exposures(config,root)
        value=dict(config_sha256=_hash(config),summary_sha256=file_hash(directory/'summary.complete.json'),review=review,
            frozen_at=_now(),prior_exposure_receipts=exposure,prediction_status='exploratory' if review['prior_exposure'] or exposure or (directory/'results/endpoints.json').exists() else 'frozen-before-reveal')
        _save_frozen(target,value);_save_frozen(directory/'predictions.complete.json',dict(sha256=file_hash(target)));return value


def reveal(path,output_root):
    root=Path(output_root).resolve();verify_measurement(path,root);config,directory,units=load(path,root)
    _verify_seal(config,directory,'summary.complete.json')
    prediction=json.loads((directory/'predictions.json').read_text())
    if json.loads((directory/'predictions.complete.json').read_text())!={'sha256':file_hash(directory/'predictions.json')} or prediction['config_sha256']!=_hash(config) or prediction['summary_sha256']!=file_hash(directory/'summary.complete.json'):raise ValueError('Prediction binding changed.')
    with _writer(directory):
        if (directory/'complete.json').exists():return verify(path,root)
        spectral,audit=_load(inside(root,config['spectral_path']),root)
        rows=[];flipped=set();undefined=set()
        with np.load(directory/'results/mean-pushes.npz',allow_pickle=False) as vectors:
            for case,entry in config['cases'].items():
                initial=factors(inside(root,config['variants'][entry['initialization_arm']]['adapter_path']))
                for arm in entry['paired_arms']:
                    endpoint=factors(inside(root,audit['sources'][arm]['checkpoints']['64']['path']))
                    for i,module in enumerate(config['modules']):
                        b,orientation=orient_endpoint(*endpoint[module],initial[module][0])
                        if orientation['flipped']:flipped.add((arm,module))
                        if not orientation['defined']:undefined.add((arm,module))
                        controls=np.random.default_rng(42).normal(size=(100,endpoint[module][1].size))
                        random_cos=[] if b is None else [cosine(v,b) for v in controls]
                        random_cos=[v for v in random_cos if v is not None]
                        for objective in config['objectives']:
                            for aggregation in ['mean_push','mean_push_full_batches','token_weighted_push']:
                                key=f'{case}-{objective}-{i}-{aggregation}'
                                if key not in vectors:continue
                                push=vectors[key];alignment=None if b is None else cosine(push,b)
                                rows.append(dict(case=case,arm=arm,module=module,objective=objective,aggregation=aggregation,
                                    cosine_push_endpoint_b=alignment,cosine_sign_push_endpoint_b=None if b is None else cosine(np.sign(push),b),
                                    orientation=orientation,gaussian_cosine_mean=float(np.mean(random_cos)) if random_cos else None,
                                    gaussian_cosine_p05=float(np.percentile(random_cos,5)) if random_cos else None,gaussian_cosine_p95=float(np.percentile(random_cos,95)) if random_cos else None,
                                    gaussian_draws=100,gaussian_seed=42))
        result=dict(config_sha256=_hash(config),prediction_status=prediction['prediction_status'],flipped_modules=len(flipped),undefined_orientation_modules=len(undefined),rows=rows,
            flipped_modules_by_arm={arm:sum(a==arm for a,m in flipped) for arm in ['sft','grpo']},
            undefined_orientation_modules_by_arm={arm:sum(a==arm for a,m in undefined) for arm in ['sft','grpo']},
            real_gpu_measurement=config['execution_kind']=='real-cuda',causal_claim=False)
        _write_json(directory/'results/endpoints.json',result);_jsonl(directory/'results/endpoint-alignments.jsonl',rows)
        with (directory/'results/endpoints.csv').open('w',newline='') as stream:
            fields=['case','arm','module','objective','aggregation','cosine_push_endpoint_b','cosine_sign_push_endpoint_b','a64_a0_cosine','flipped','gaussian_cosine_mean','gaussian_cosine_p05','gaussian_cosine_p95']
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            writer.writerows({k:r['orientation'][k] if k in ['a64_a0_cosine','flipped'] else r[k] for k in fields} for r in rows)
        esc=lambda value:html.escape(str(value))
        cells=''.join('<tr>'+''.join('<td>'+esc(r[k])+'</td>' for k in ['case','arm','objective','module','cosine_push_endpoint_b','cosine_sign_push_endpoint_b'])+'</tr>' for r in rows if r['aggregation']=='mean_push')
        (directory/'endpoints.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Push and endpoint alignment</title><style>body{font:16px/1.55 system-ui;max-width:1200px;margin:40px auto;padding:20px;background:#f6f8f5;color:#243c43}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ccd}</style><h1>Initial push and trained endpoint alignment</h1><p>'+esc(config['execution_kind'])+' · '+esc(prediction['prediction_status'])+'</p><p>Primary batch-mean pushes, including the final partial batch where present. CSV and JSON also contain full-batch-only and token-weighted sensitivity results, raw/oriented A alignment, and 100 seeded Gaussian controls. A cosine measures direction agreement; it does not reconstruct AdamW or establish a causal behavioral effect.</p><p>Flipped modules by arm: '+esc(result['flipped_modules_by_arm'])+'. Undefined orientations: '+esc(result['undefined_orientation_modules_by_arm'])+'.</p><table><thead><tr><th>A₀</th><th>Endpoint</th><th>Objective</th><th>Module</th><th>Push cosine</th><th>Sign-push cosine</th></tr></thead><tbody>'+cells+'</tbody></table><p><a href="results/endpoints.csv">All alignments CSV</a> · <a href="results/endpoints.json">Full JSON</a> · <a href="predictions.json">Frozen predictions</a> · <a href="report.html">Pre-reveal summaries</a></p></html>')
        _save_frozen(directory/'exposure.json',dict(config_sha256=_hash(config),spectral_config_sha256=_hash(spectral),prediction_sha256=file_hash(directory/'predictions.json'),revealed_at=_now()))
        _seal(directory,'complete.json',['summary.complete.json','predictions.json','predictions.complete.json','results/endpoints.json','results/endpoint-alignments.jsonl','results/endpoints.csv','endpoints.html','exposure.json'],config)
        _state(directory,'complete',len(units),len(units),prediction_status=prediction['prediction_status']);return dict(analysis_complete=True,flipped_modules=len(flipped),prediction_status=prediction['prediction_status'])


def verify(path,output_root):
    root=Path(output_root).resolve();measurement=verify_measurement(path,root);config,directory,_=load(path,root)
    _verify_seal(config,directory,'summary.complete.json');_verify_seal(config,directory,'complete.json')
    return dict(**measurement,analysis_complete=True,prediction_status=json.loads((directory/'results/endpoints.json').read_text())['prediction_status'])
