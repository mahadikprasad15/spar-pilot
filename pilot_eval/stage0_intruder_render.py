"""CPU geometry figures and an inspectable eligibility report."""
import html
import io
import json


def render(config,result,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'sft':'#ae7565','grpo':'#4b8194'}
    inputs=[]
    for name in ['rotation','overlap']:
        with plt.rc_context({'svg.hashsalt':'stage0-intruder-geometry-v1','font.size':10}):
            fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained',squeeze=False)
            for index,arm in enumerate(config['arms']):
                ax=axes[index,0];rows=[r for r in result['module_results'] if r['arm']==arm]
                x=list(range(len(rows)));color=colors[arm]
                if name=='rotation':
                    left=[r['levels']['50']['rotation']['max_left_degrees'] for r in rows]
                    right=[r['levels']['50']['rotation']['max_right_degrees'] for r in rows]
                    ax.scatter(x,left,s=15,color=color,label='Left top-k50 subspace')
                    ax.scatter(x,right,s=15,color='#73807d',marker='x',label='Right top-k50 subspace')
                    ax.set_ylabel('Maximum principal angle (degrees)');ax.set_ylim(bottom=-.5)
                    tied=[i for i,r in enumerate(rows) if not r['levels']['50']['rotation']['unique_boundary']]
                    if tied:ax.scatter(tied,[left[i] for i in tied],s=45,facecolors='none',edgecolors='#7c6346',label='Unresolved boundary')
                else:
                    overlap=[min(r['candidates'][j]['overlap_norm']['50'] for j in range(20)) for r in rows]
                    chance=[r['levels']['50']['chance_norm'] for r in rows]
                    ax.scatter(x,overlap,s=15,color=color,label='Minimum top-20 overlap norm')
                    ax.plot(x,chance,color='#8b9385',linewidth=1.2,label='RMS chance sqrt(m/d_out)')
                    ax.axhline(.5,color='#ad8a5b',linestyle='--',linewidth=1,label='Primary threshold 0.5')
                    undecidable=[i for i,r in enumerate(rows) if r['primary']['status']=='undecidable-at-or-below-chance']
                    if undecidable:ax.scatter(undecidable,[overlap[i] for i in undecidable],marker='x',s=35,color='#545e5c',label='Undecidable')
                    ax.set_ylim(-.03,1.04);ax.set_ylabel('Projection norm onto old top-m')
                ticks=list(range(0,len(rows),max(1,len(rows)//8)))
                ax.set_xticks(ticks,[rows[i]['module_id'] for i in ticks],rotation=25,ha='right')
                ax.set_title(arm.upper(),loc='left');ax.spines[['top','right']].set_visible(False)
                ax.legend(fontsize=8,frameon=False,loc='best')
            axes[-1,0].set_xlabel('Adapted linear modules in the frozen inventory order; exact IDs in CSV')
            for suffix in ['svg','png']:
                stream=io.BytesIO()
                fig.savefig(stream,format=suffix,**({'metadata':{'Date':None}} if suffix=='svg' else {'dpi':130}))
                path=directory/'plots'/f'{name}.{suffix}';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(stream.getvalue())
            plt.close(fig)
    for row in result['module_results']:
        inputs.append(dict(module=row['module'],arm=row['arm'],levels=row['levels'],
            primary_status=row['primary']['status'],eligible_indices=row['primary']['eligible_indices'],
            top20_overlap=[c['overlap_norm']['50'] for c in row['candidates'][:20]]))
    (directory/'plots/inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
    escape=lambda x:html.escape(str(x))
    summary=[]
    for arm,counts in result['eligibility'].items():
        summary.append('<tr><td>'+arm.upper()+'</td>'+''.join('<td>'+str(counts[k])+'</td>' for k in
            ['eligible-intruders','no-intruders','undecidable-at-or-below-chance'])+'</tr>')
    rows=[]
    for row in result['module_results']:
        p=row['primary'];data=row['levels']['50'];rotation=data['rotation']
        ratio=data['update_over_gap'];ratio_text=ratio['value'] if ratio['value'] is not None else ratio['status']
        rows.append('<tr>'+''.join('<td>'+escape(v)+'</td>' for v in [row['module_id'],row['arm'],data['k'],data['m'],
            f"{data['chance_norm']:.4f}",p['status'],p['eligible_indices'],f"{rotation['max_left_degrees']:.4g}",
            f"{rotation['max_right_degrees']:.4g}",ratio_text])+ '</tr>')
    synthetic=result['execution_kind']=='synthetic-cpu-fixture'
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 0 subspace geometry</title><style>body{margin:0;background:#f7f8f4;color:#263f44;font:16px/1.65 system-ui}main{max-width:1150px;margin:auto;padding:42px 28px}h1{font:40px/1.15 Georgia}h2{font:28px Georgia;margin-top:38px}aside{padding:16px;background:#fff0d6;border-left:4px solid #b69151}img{width:100%;background:white;border:1px solid #d7ded9}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}td,th{text-align:left;padding:9px 11px;border-bottom:1px solid #d7ded9}th{background:#eaf0e9}a{color:#326f84}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}p,li{max-width:950px}</style></head><body><main>
<p>STAGE 0 · CONDITIONAL INTRUDER EXTENSION</p><h1>Subspace rotation and new top directions</h1>'''
    if synthetic:page+='<aside><strong>Synthetic software fixture.</strong> These values are validation data, not findings about the trained model.</aside>'
    page+='<p>CPU singular geometry of the two saved step-64 rank-one updates. Every adapted module is measured even when its update is small. This report establishes geometric eligibility; no adapter is edited and no model forward pass is run.</p>'
    page+='<h2>Primary eligibility</h2><p>Detection uses the new top 20 left singular directions. Overlap is the norm of projection onto the whole old top-m subspace, m=max(k50,40). An overlap strictly below 0.5 is eligible only when 0.5 is above sqrt(m/d_out). Undecidable modules are not counted as negative findings.</p>'
    page+='<table><tr><th>Arm</th><th>Modules with eligible flags</th><th>No intruders</th><th>Undecidable</th></tr>'+''.join(summary)+'</table>'
    page+='<h2>Rotation of whole subspaces</h2><p>Principal angles ignore sign flips and mixing inside each top-k subspace. Base-defined k is held fixed. Ties at the old/new boundary make a particular top-k choice nonunique; these cases are marked and retained.</p><img src="plots/rotation.svg" alt="Left and right principal-angle rotation for both arms">'
    page+='<h2>Chance-aware overlap</h2><p>The minimum overlap is only a compact view; all 20 detection directions and all 40 matching candidates are saved. Chance is an isotropic RMS reference, not a p-value. Sensitivities use floored k30/k70 and thresholds 0.3–0.7.</p><img src="plots/overlap.svg" alt="Minimum top-20 overlap and module chance reference">'
    page+='<h2>Module-level primary table</h2><div class="table"><table><tr>'+''.join('<th>'+v+'</th>' for v in
        ['Module','Arm','k50','m','Chance norm','Status','Flagged indices (zero-based)','Max left °','Max right °','Update / base gap'])+'</tr>'+''.join(rows)+'</table></div>'
    page+='<h2>Undefined cases and limits</h2><ul>'+''.join('<li>'+escape(v)+'</li>' for v in result['limits'])+'</ul>'
    page+='<p>Positive update / zero gap is infinite; zero / zero and a missing next singular value are undefined. JSON stores these as explicit statuses with null numeric fields, not a fabricated epsilon. Vector flags within a tied cluster can depend on the SVD basis; the raw flags are preserved with their degeneracy metadata.</p>'
    page+='<p>Data: <a href="results/results.json">complete results</a> · <a href="results/geometry.csv">rotation and gap CSV</a> · <a href="results/candidates.csv">all top-40 candidate overlaps/singular values/gaps</a> · <a href="plots/inputs.json">figure inputs</a> · <a href="config.json">frozen plan</a>. Retained singular arrays and seals live in <code>units/</code>.</p>'
    page+='<details><summary>Exact analysis settings</summary><pre>'+escape(json.dumps(config,indent=2))+'</pre></details></main></body></html>'
    (directory/'report.html').write_text(page)
    (directory/'results/report.md').write_text('# Stage 0 singular geometry\n\nExecution: '+result['execution_kind']+
        '\n\nCPU geometry only; no behavioral intervention. All units measured, no small-update skip.\n\n'+
        '\n'.join('- '+v for v in result['limits'])+'\n')
