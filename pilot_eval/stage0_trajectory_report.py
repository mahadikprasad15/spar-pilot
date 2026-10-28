"""CPU trajectory snapshots; separate sessions, document CIs, no causal claims."""
import csv
import html
import json
from pathlib import Path

import numpy as np

from pilot_eval.run import _write_json
from pilot_eval.stage0_trajectory import verify_cpu,load
from pilot_eval.stage0_trajectory_math import interval,lengths,log_slopes
from pilot_eval.stage0_trajectory_sources import merge_products
from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_workflow import _writer,_state,_now
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash,_save_frozen


def _table(path,rows):
    if not rows:path.write_text('status\nunavailable\n');return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def endpoint_lengths(config,root,directory):
    if config['evaluation_state']!='ready':return dict(state=config['evaluation_state'],comparisons=[])
    selection=config['evaluation'];ep=inside(root,selection['config']);groups={}
    if selection['kind']=='stage0-evaluation':
        from pilot_eval.stage0_evaluation import _load,_verify
        ev,_,ed=_load(ep,root);_verify(ev,ed,root)
        if ev['model_identity']!=config['model_identity']:raise ValueError('Saved greedy base differs from trajectory base.')
        for line in (ed/'results/responses.jsonl').read_text().splitlines():
            r=json.loads(line)
            if r['variant'] not in ['V0a','V0b']:continue
            alias='sft-64' if r['variant']=='V0a' else 'grpo-64'
            groups.setdefault(alias,[]).append(dict(id=r['item_id'],tokens=r['token_count'],capped=r['capped'],prompt_sha256=r['prompt_sha256']))
        pairs=[('sft-64','grpo-64')]
    else:
        from pilot_eval.behaviour_breakdown import verify_behaviour,_load
        verify_behaviour(ep,root);_,ed,_=_load(ep,root)
        for line in (ed/'results/responses.jsonl').read_text().splitlines():
            r=json.loads(line);groups.setdefault(f'{r["method"]}-{r["step"]}',[]).append(dict(id=r['item_id'],tokens=r['token_count'],capped=r['capped'],prompt_sha256=_hash(r['prompt'])))
        pairs=[('sft-0','sft-64'),('grpo-0','grpo-64'),('sft-64','grpo-64')]
    comparisons=[]
    for before,after in pairs:
        if before not in groups or after not in groups:continue
        n=len(groups[before]);draws=np.random.default_rng(42).integers(0,n,size=(10000,n),dtype=np.int32)
        np.save(directory/'results'/f'length-bootstrap-{before}-{after}.npy',draws)
        value=lengths(groups[before],groups[after],draws);comparisons.append(dict(reference=before,edited=after,**value))
    return dict(state='verified-saved-output-only',comparisons=comparisons,source=str(ep.relative_to(root)),
        joins_to_KL=False,scope='separate saved-response panel; no new generation or implied matching runtime with KL',human_labels_used=False)


def report(path,output_root,name,*,collection_paths=None):
    root=Path(output_root).resolve();config,source,results=verify_cpu(path,root);from pilot_eval.sft import safe_name
    safe_name(name);directory=root/'reports'/name;collections=[inside(root,p) for p in (collection_paths or [])]
    request=dict(parent_path=str(inside(root,path).relative_to(root)),parent_sha256=file_hash(inside(root,path)),cpu_complete_sha256=file_hash(source/'cpu.complete.json'),collections=[str(p.relative_to(root)) for p in collections])
    with _writer(directory):
        if (directory/'complete.json').exists():
            if json.loads((directory/'request.json').read_text())!=request:raise ValueError('Report snapshot selection changed; choose another name.')
            return verify(path,root,name)
        geometry=[];products=[];parents={}
        for unit,value in zip(config['units'],results):
            if unit['kind']=='geometry':geometry+=value
            else:products.append(value)
        from pilot_eval import stage0_trajectory_collect as collector
        for cp in collections:
            collector.verify(cp,root);cc,cd,_,_,_=collector.load(cp,root)
            if cc['parent_path']!=request['parent_path']:raise ValueError('Selected collection belongs to another trajectory plan.')
            docs=[json.loads(s) for s in (cd/'results/per-document.jsonl').read_text().splitlines()]
            products.append(dict(documents=docs,identity=cc['comparison_identity'],coefficient_policy=cc['coefficient_policy']))
            for p in [cp,cd/'complete.json']:parents[str(p.relative_to(root))]=file_hash(p)
        documents=merge_products(products);states={s['alias']:s for s in config['states']};cells=[];module_writes=[];bootstraps={};bootstrap_ids={}
        for group,alias,corpus,view in sorted({(r['comparison_group'],r['alias'],r['corpus'],r['view']) for r in documents}):
            rows=sorted([r for r in documents if (r['comparison_group'],r['alias'],r['corpus'],r['view'])==(group,alias,corpus,view)],key=lambda r:r['id'])
            valid=[r for r in rows if r['kl_values'] is not None and len(r['kl_values'])]
            cell=dict(comparison_group=group,alias=alias,arm=states[alias]['arm'],step=states[alias]['step'],corpus=corpus,view=view,
                sum_factor_norm_products=states[alias]['sum_factor_norm_products'],kl=None,paired_write_available=False,
                total_frobenius_size=states[alias]['total_frobenius_size'],effective_write_mean=None,kl_over_size_squared=None,
                document_ids=[r['id'] for r in rows],products=sorted({p for r in rows for p in r['products']}))
            if valid:
                ids=[r['id'] for r in valid];key=_hash(ids);draws=np.random.default_rng(42).integers(0,len(ids),size=(10000,len(ids)),dtype=np.int32)
                bootstraps[key]=draws;bootstrap_ids[key]=ids
                cell['kl']=interval([sum(r['kl_values']) for r in valid],[len(r['kl_values']) for r in valid],draws);cell['bootstrap_key']=key
                size=cell['total_frobenius_size'];cell['kl_over_size_squared']=None if size==0 else cell['kl']['token_mean']/size**2
            paired=[r for r in rows if r['kl_values'] is not None and r['writes'] is not None]
            cell['paired_write_available']=len(paired)==len(rows) and len(rows)==150
            for module in config['modules']:
                values=[w for r in rows if r['writes'] is not None for w in r['writes'] if w['module']==module and w['count']>0]
                if not values:continue
                n=sum(w['count'] for w in values);ratio=sum(w['ratio_sum'] for w in values)/n
                module_writes.append(dict(comparison_group=group,alias=alias,corpus=corpus,view=view,module=module,
                    token_mean_ratio=ratio,equal_example_mean_ratio=float(np.mean([w['ratio_sum']/w['count'] for w in values])),tokens=n,documents=len(values),
                    paired_to_KL=cell['paired_write_available'],rms_ratio=None if any(w['base_squared_sum'] is None for w in values) else float(np.sqrt(sum(w['update_squared_sum'] for w in values)/sum(w['base_squared_sum'] for w in values)))))
            selected_writes=[r for r in module_writes if (r['comparison_group'],r['alias'],r['corpus'],r['view'])==(group,alias,corpus,view)]
            if len(selected_writes)==len(config['modules']):cell['effective_write_mean']=float(np.mean([r['token_mean_ratio'] for r in selected_writes]))
            cells.append(cell)
        slopes=[]
        for group,arm,corpus,view in sorted({(r['comparison_group'],r['arm'],r['corpus'],r['view']) for r in cells if r['kl'] is not None}):
            points=[dict(step=r['step'],size=r['total_frobenius_size'],kl=r['kl']['token_mean']) for r in cells if (r['comparison_group'],r['arm'],r['corpus'],r['view'])==(group,arm,corpus,view) and r['kl'] is not None]
            slopes.append(dict(comparison_group=group,arm=arm,corpus=corpus,view=view,**log_slopes(points)))
        result=dict(protocol='stage0-trajectory-report-v1',observational_only=True,causal_direction_claim=False,
            geometry=geometry,cells=cells,module_writes=module_writes,slopes=slopes,missing_states={s['alias']:s['state'] for s in config['states'] if s['state']!='verified-adapter-payload'},
            selected_products=config['products'],comparison_groups=[p['identity'] for p in products],coefficient_policies=[p['coefficient_policy'] for p in products],
            drift_complete=all(any(c['alias']==s['alias'] and c['corpus']==corpus and c['view']==view and c['kl'] is not None and c['kl']['documents']==150 and c['paired_write_available'] for c in cells) for s in config['states'] for corpus,view in [('gsm8k','question'),('gsm8k','solution'),('fineweb','user')]),
            curve_policy='measured points connected in checkpoint order; no interpolation, nearest-size matching or causal equal-size claim',
            primary_size_axis='total Frobenius sqrt(sum module norm-product squared)',primary_KL='FineWeb per-token KL from Ticket06; equal-document summaries alongside',
            effective_write_axis='equal-module mean of per-module token-mean norm ratios, separately by corpus/view; module values retained',
            size_ratio='per-token KL / total Frobenius size squared; undefined at zero size',near_two='unclassified; no direction-sensitivity inference',
            proposed_followup='Not implemented: rescale SFT64 by c=size(GRPO64)/size(SFT64), evaluate W0+c*b*a^T and compare exact-equal-size KL under a new approved plan.',
            statistics=config['statistics'],limits=['One saved training seed per arm.','Pointwise document bootstrap intervals do not estimate training-seed uncertainty.','KL is fixed-context drift, not demonstrated harm.','Separate runtime groups are never merged into a single trajectory.','Endpoint lengths are saved-response properties, not reasoning quality.'])
        (directory/'results').mkdir(parents=True,exist_ok=True);result['endpoint_lengths']=endpoint_lengths(config,root,directory)
        np.savez(directory/'results/bootstrap-indices.npz',**bootstraps);_write_json(directory/'results/bootstrap-identities.json',bootstrap_ids)
        _write_json(directory/'results/results.json',result);_table(directory/'results/geometry.csv',geometry)
        _table(directory/'results/slopes.csv',[{k:v for k,v in r.items() if k!='adjacent_slopes'} for r in slopes]);_write_json(directory/'results/slopes.json',slopes)
        _table(directory/'results/kl-size.csv',[{**{k:v for k,v in c.items() if k not in ['kl','document_ids','products']},**(c['kl'] or {})} for c in cells]);_table(directory/'results/module-writes.csv',module_writes)
        _plots(directory,result);_render(directory,result)
        spectral=json.loads(inside(root,config['spectral_path']).read_text());exposure=dict(config_sha256=_hash(spectral),trajectory_config_sha256=_hash(config),revealed_at=_now(),scope='trajectory report includes trained factor geometry')
        _save_frozen(directory/'exposure.json',exposure);_save_frozen(directory/'request.json',request)
        files=[str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p.name not in ['writer.json','complete.json']]
        _write_json(directory/'complete.json',dict(parent_sha256=_hash(config),parents=parents,files={p:file_hash(directory/p) for p in files}));_state(directory,'complete',len(cells),len(cells))
        return dict(report_complete=True,drift_complete=result['drift_complete'],observational_only=True,path=str(directory/'report.html'))


def _plots(directory,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots=directory/'plots';plots.mkdir(exist_ok=True);fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,field in zip(axes,['pi_in','pi_out']):
        for arm,color in [('sft','#477e9e'),('grpo','#b87964'),('random','#899681')]:
            rows=[r for r in result['geometry'] if r['arm']==arm and r['energy_level']==50 and r[field] is not None];steps=sorted({r['step'] for r in rows})
            ax.plot(steps,[np.mean([r[field] for r in rows if r['step']==step]) for step in steps],marker='o',color=color,label=arm)
        ax.set(xlabel='Saved optimizer step',ylabel='Equal-module mean '+field,title='Principal overlap · k50');ax.spines[['top','right']].set_visible(False);ax.legend()
    fig.suptitle('Geometry of saved adapters · descriptive single-seed trajectories');fig.tight_layout();fig.savefig(plots/'geometry.svg');plt.close(fig)
    groups=sorted({c['comparison_group'] for c in result['cells']})
    for field,filename,label in [('total_frobenius_size','kl-size.svg','Total Frobenius ‖ΔW‖'),('effective_write_mean','kl-effective-write.svg','Equal-module mean effective write'),('sum_factor_norm_products','kl-norm-sum.svg','Σ module ‖b‖‖a‖ · additional view')]:
        fig,axes=plt.subplots(max(1,len(groups)),3,figsize=(13,4*max(1,len(groups))),squeeze=False)
        for gi,group in enumerate(groups):
            for ax,(corpus,view) in zip(axes[gi],[('gsm8k','question'),('gsm8k','solution'),('fineweb','user')]):
                zero=[];positive=[]
                for arm,color in [('sft','#477e9e'),('grpo','#b87964'),('random','#899681')]:
                    rows=sorted([r for r in result['cells'] if r['comparison_group']==group and r['arm']==arm and r['corpus']==corpus and r['view']==view and r['kl'] is not None and r[field] is not None and (field!='effective_write_mean' or r['paired_write_available'])],key=lambda r:r['step'])
                    zero.extend((r[field],r['kl']['token_mean'],color) for r in rows if r['step']==0)
                    rows=[r for r in rows if r[field]>0 and r['kl']['token_mean']>0];positive+=rows
                    if rows:
                        x=np.array([r[field] for r in rows]);y=np.array([r['kl']['token_mean'] for r in rows]);bounds=np.array([r['kl']['token_interval_95'] for r in rows]).T
                        ax.errorbar(x,y,yerr=np.maximum(0,np.array([y-bounds[0],bounds[1]-y])),marker='o',color=color,label=arm,linestyle='none' if arm=='random' else '-')
                        for r in rows:ax.annotate(str(r['step']),(r[field],r['kl']['token_mean']),fontsize=7,xytext=(3,3),textcoords='offset points')
                if positive:ax.set_xscale('log');ax.set_yscale('log')
                else:ax.text(.05,.5,'No positive paired measurements',transform=ax.transAxes,fontsize=8)
                if zero:
                    inset=ax.inset_axes([.06,.67,.24,.24]);inset.axhline(0,color='#ccd',lw=.5);inset.axvline(0,color='#ccd',lw=.5)
                    for x,y,color in zero:inset.scatter([x],[y],s=16,color=color)
                    inset.set(xlim=(-.1,.1),ylim=(-.1,.1),title='step0 · linear');inset.tick_params(labelsize=6);inset.title.set_fontsize(7)
                ax.set(xlabel=label,ylabel='Token-mean KL · nats',title=f'{group[:8]} · {corpus}/{view}');ax.spines[['top','right']].set_visible(False)
                if positive:ax.legend(fontsize=8)
        if not groups:axes[0,0].text(.1,.5,'No verified drift products selected.\nGeometry is available; GPU collection is optional.');axes[0,1].axis('off');axes[0,2].axis('off')
        fig.suptitle('Separate runtime groups · log–log measured checkpoints; zero in linear inset');fig.tight_layout();fig.savefig(plots/filename);plt.close(fig)


def _render(directory,result):
    esc=lambda value:html.escape(str(value))
    fields=['alias','corpus','view','total_frobenius_size','effective_write_mean','kl_over_size_squared','kl','paired_write_available','comparison_group']
    rows=''.join('<tr>'+''.join('<td>'+esc(r[k])+'</td>' for k in fields)+'</tr>' for r in result['cells'])
    slopes=''.join('<tr>'+''.join('<td>'+esc(r[k])+'</td>' for k in ['arm','corpus','view','fitted_slope','adjacent_slopes','near_two','comparison_group'])+'</tr>' for r in result['slopes'])
    (directory/'report.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Checkpoint trajectories</title>'
        '<style>body{font:16px/1.55 system-ui;max-width:1300px;margin:40px auto;padding:24px;color:#243c43;background:#f7f8f5}img{width:100%}.scroll{overflow:auto}td,th{padding:8px;border-bottom:1px solid #ccc}table{font-size:12px;border-collapse:collapse}aside{padding:16px;background:#edf1ea}</style>'
        '<h1>Checkpoint geometry and distributional drift</h1><p>Observed saved checkpoints on one training seed per arm. Separate session groups stay separate. No interpolation, nearest-checkpoint matching or causal equal-size claim.</p>'
        '<p>Primary parameter size is total Frobenius ||ΔW|| = sqrt(Σ module ||b||²||a||²). FineWeb token-mean KL is primary, using the same full-vocabulary definition as Ticket06. Equal-document means and whole-document percentile intervals are retained. Effective write averages the original per-module token-mean norm ratios equally across modules.</p>'
        '<aside>KL/size² and fitted/adjacent log–log slopes are descriptive. “Near2” is unclassified: no direction-sensitivity or small-update-validity conclusion is made. Step0 appears in a labelled linear-origin inset because zero has no logarithm.</aside>'
        '<p>Missing states: '+esc(result['missing_states'])+'. Full paired drift coverage: '+esc(result['drift_complete'])+'.</p>'
        '<img src="plots/geometry.svg" alt="Principal overlap trajectories"><img src="plots/kl-size.svg" alt="Token-mean KL versus total Frobenius size, log-log with zero inset">'
        '<img src="plots/kl-effective-write.svg" alt="KL versus mean effective write, separated by runtime"><details><summary>Additional plain norm-sum view</summary><img src="plots/kl-norm-sum.svg" alt="Additional sum of module norm products"></details>'
        '<p><a href="results/results.json">Full results</a> · <a href="results/geometry.csv">Geometry CSV</a> · <a href="results/kl-size.csv">KL-size CSV</a> · <a href="results/module-writes.csv">Module effective writes</a> · <a href="results/slopes.json">Slopes and adjacent intervals</a></p>'
        '<h2>Observed checkpoint points</h2><div class="scroll"><table><tr>'+''.join('<th>'+esc(k)+'</th>' for k in fields)+'</tr>'+rows+'</table></div>'
        '<h2>Within-run slopes</h2><div class="scroll"><table><tr><th>Arm</th><th>Corpus</th><th>View</th><th>Fitted slope</th><th>Adjacent slopes</th><th>Near2</th><th>Runtime</th></tr>'+slopes+'</table></div>'
        '<h2>Saved-output endpoint lengths</h2><p>'+esc(result['endpoint_lengths'])+'</p>'
        '<h2>Proposed follow-up, not implemented</h2><p>'+esc(result['proposed_followup'])+'</p>')


def verify(path,output_root,name):
    root=Path(output_root).resolve();config,_,_=verify_cpu(path,root);directory=root/'reports'/name;seal=json.loads((directory/'complete.json').read_text())
    if seal['parent_sha256']!=_hash(config):raise ValueError('Trajectory report parent changed.')
    for p,digest in seal['parents'].items():
        if file_hash(inside(root,p))!=digest:raise ValueError('Trajectory collection source changed.')
    for p,digest in seal['files'].items():
        if file_hash(inside(directory,p))!=digest:raise ValueError('Trajectory report payload changed.')
    request=json.loads((directory/'request.json').read_text())
    from pilot_eval.stage0_trajectory_collect import verify as verify_collection
    for cp in request['collections']:verify_collection(inside(root,cp),root)
    return dict(report_complete=True,observational_only=True,drift_complete=json.loads((directory/'results/results.json').read_text())['drift_complete'])
