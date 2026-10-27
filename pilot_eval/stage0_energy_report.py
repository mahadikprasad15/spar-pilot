"""CPU reconstruction, signed interference tables and layer/type summaries."""
import csv
import html
import json
from pathlib import Path

import numpy as np

from pilot_eval.activation_prepare import VIEWS
from pilot_eval.run import _write_json
from pilot_eval.stage0_energy import read_unit,verify_measurement,_prediction
from pilot_eval.stage0_energy_math import ratios,FIELDS
from pilot_eval.stage0_energy_plan import load
from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_workflow import _writer,_state,_now
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash,_save_frozen


def table(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def _plots(directory,cells):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#477e9e','#ba7560','#8eb4c3','#d6aa98','#8b9983'];labels=['sft-64','grpo-64','sft-0','grpo-0','random-64']
    settings=[('gsm8k','question'),('gsm8k','solution'),('fineweb','user')]
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for ax,(corpus,view) in zip(axes,settings):
        for label,color in zip(labels,colors):
            rows=[r for r in cells if r['corpus']==corpus and r['view']==view and r['factor']==label and r['energy_level']==50 and r['principal_fraction'] is not None]
            layers=sorted({r['layer'] for r in rows})
            ax.plot(layers,[np.mean([r['principal_fraction'] for r in rows if r['layer']==layer]) for layer in layers],marker='.',label=label,color=color,linestyle='--' if label.endswith('-0') else '-')
        ax.set(title=f'{corpus} · {view}',xlabel='Layer',ylim=(0,1));ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Mean module principal fraction · k50');axes[-1].legend(fontsize=8)
    fig.suptitle('Base-input activation energy: principal / (principal + rest)');fig.tight_layout()
    plots=directory/'plots';plots.mkdir(exist_ok=True);fig.savefig(plots/'principal-fractions.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharex=True,sharey=True)
    for ax,(corpus,view) in zip(axes,settings):
        lookup={(r['module'],r['factor']):r['principal_fraction'] for r in cells if r['corpus']==corpus and r['view']==view and r['energy_level']==50}
        pairs=[(value,lookup.get((module,'grpo-64'))) for (module,label),value in lookup.items() if label=='sft-64']
        pairs=[p for p in pairs if None not in p]
        if pairs:ax.scatter(*zip(*pairs),s=18,color='#477e9e',alpha=.6)
        ax.plot([0,1],[0,1],color='#a0aaa9',linestyle='--',linewidth=1);ax.set(title=f'{corpus} · {view}',xlabel='SFT-64 fraction',xlim=(0,1),ylim=(0,1));ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('GRPO-64 fraction');fig.tight_layout();fig.savefig(plots/'sft-grpo.svg');plt.close(fig)


def report(path,output_root):
    root=Path(output_root).resolve();verify_measurement(path,root);config,directory,rows=load(path,root);by_id={r['id']:r for r in rows}
    with _writer(directory):
        if (directory/'complete.json').exists():return verify(path,root)
        totals={}
        for unit in config['units']:
            arrays=read_unit(config,directory,unit,by_id)['sums']
            for ei,identifier in enumerate(unit['ids']):
                row=by_id[identifier]
                for vi,view in enumerate(VIEWS):
                    count=sum(row['masks'][view])
                    if not count:continue
                    key=(row['corpus'],view);cell=totals.setdefault(key,dict(sums=np.zeros_like(arrays[ei,:,:,vi]),tokens=0,examples=0))
                    cell['sums']+=arrays[ei,:,:,vi];cell['tokens']+=count;cell['examples']+=1
        geometry={(r['module'],r['factor'],r['energy_level']):r for r in json.loads((directory/'inputs/geometry.json').read_text())};cells=[]
        for (corpus,view),cell in totals.items():
            for fi,factor in enumerate(config['factor_labels']):
                for li,level in enumerate(config['energy_levels']):
                    for mi,module in enumerate(config['modules']):
                        sums=cell['sums'][fi,li,mi];geo=geometry[module,factor,level]
                        cells.append(dict(corpus=corpus,view=view,factor=factor,energy_level=level,module=module,layer=int(module.split('.')[4]),projection=module.split('.')[-1],
                            tokens=cell['tokens'],examples=cell['examples'],**dict(zip(FIELDS,map(float,sums))),**ratios(sums),
                            reconstruction_residual=float(sums[0]+sums[1]+sums[2]-sums[3]),pi_in=geo['pi_in'],pi_out=geo['pi_out'],chance_in=geo['chance_in'],chance_out=geo['chance_out']))
        groups=[]
        for group_kind in ['layer','projection']:
            keys={(r['corpus'],r['view'],r['factor'],r['energy_level'],r[group_kind]) for r in cells}
            for corpus,view,factor,level,value in sorted(keys):
                selected=[r for r in cells if (r['corpus'],r['view'],r['factor'],r['energy_level'],r[group_kind])==(corpus,view,factor,level,value)]
                defined=[r['principal_fraction'] for r in selected if r['principal_fraction'] is not None]
                groups.append(dict(corpus=corpus,view=view,factor=factor,energy_level=level,group_kind=group_kind,group=value,modules=len(selected),defined_modules=len(defined),mean_module_principal_fraction=None if not defined else float(np.mean(defined))))
        result=dict(config_sha256=_hash(config),real_gpu_measurement=config['execution_kind']=='real-cuda',measurement_complete=True,
            input_scope=config['input_scope'],base_forwards=300,weighting='pooled counted tokens within corpus/view/module; layer/type summaries equal-weight defined modules',
            factor_labels=config['factor_labels'],energy_levels=config['energy_levels'],agreement=config['agreement'],prediction_status=_prediction(config,directory)['prediction_status'],
            cells=cells,groups=groups,limits=['Base-input contractions, not adapted activations or causal damage.','Normalized components can exceed one and the cross term can be negative.','Zero total gives undefined total-normalized components; zero parts gives undefined headline.','Single saved adapter seed per arm.'])
        _write_json(directory/'results/results.json',result);table(directory/'results/modules.csv',cells);table(directory/'results/groups.csv',groups)
        _plots(directory,cells)
        esc=lambda value:html.escape(str(value));fields=['corpus','view','factor','energy_level','module','principal_fraction','parallel_over_total','rest_over_total','cross_over_total']
        body=''.join('<tr>'+''.join('<td>'+esc(r[k])+'</td>' for k in fields)+'</tr>' for r in cells)
        (directory/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Base-input energy decomposition</title><style>body{font:16px/1.6 system-ui;max-width:1300px;margin:40px auto;padding:24px;background:#f7f8f5;color:#233c43}img{width:100%}.scroll{overflow:auto}table{font-size:12px;border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #ccd}</style><h1>Where adapter input activity comes from</h1><p>'+esc(config['execution_kind'])+' · '+esc(result['prediction_status'])+'</p><p>All five saved factors contract the same untuned base inputs. Headline fraction is principal/(principal+rest). Parallel/total + rest/total + cross/total reconstruct one where total is nonzero; the signed cross term identifies reinforcement or cancellation. Total is measured independently from a·x. These measurements do not establish causal damage.</p><img src="plots/principal-fractions.svg" alt="Principal activity by layer"><img src="plots/sft-grpo.svg" alt="SFT versus GRPO principal activity"><p><a href="results/results.json">Full results</a> · <a href="results/modules.csv">Module CSV</a> · <a href="results/groups.csv">Layer/type CSV</a> · <a href="checks/projections.json">Normalized projection checks</a> · <a href="predictions.json">Predictions</a></p><div class="scroll"><table><tr>'+''.join('<th>'+esc(k)+'</th>' for k in fields)+'</tr>'+body+'</table></div></html>')
        exposure=root/'reports'/config['name']/'exposure.json'
        _save_frozen(exposure,dict(config_sha256=_hash(json.loads(inside(root,config['spectral_path']).read_text())),energy_config_sha256=_hash(config),revealed_at=_now(),prediction_status=result['prediction_status']))
        files=['config.json','results/results.json','results/modules.csv','results/groups.csv','plots/principal-fractions.svg','plots/sft-grpo.svg','report.html','predictions.json','predictions.complete.json']
        _write_json(directory/'complete.json',dict(config_sha256=_hash(config),files={n:file_hash(directory/n) for n in files},exposure_path=str(exposure.relative_to(root)),exposure_sha256=file_hash(exposure)))
        _state(directory,'complete',len(config['units']),len(config['units']));return dict(analysis_complete=True,report=str(directory/'report.html'),cells=len(cells))


def verify(path,output_root):
    root=Path(output_root).resolve();measurement=verify_measurement(path,root);config,directory,_=load(path,root);seal=json.loads((directory/'complete.json').read_text())
    if seal['config_sha256']!=_hash(config):raise ValueError('Energy report completion identity changed.')
    for name,digest in seal['files'].items():
        if file_hash(inside(directory,name))!=digest:raise ValueError('Energy report evidence changed.')
    if file_hash(inside(root,seal['exposure_path']))!=seal['exposure_sha256']:raise ValueError('Energy exposure receipt changed.')
    return dict(**measurement,analysis_complete=True)
