"""Scientific figures and readable analyst outputs for combined findings."""
import html
import io
import os
from pathlib import Path


def _pct(value):return f'{100*value:.1f}%'
def _pp(value):return f'{100*value:+.1f} pp'
def _interval(metric):return '['+', '.join(f'{100*v:+.1f}' for v in metric['interval_95'])+'] pp'


def plots(result,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from pilot_eval.rescoring import _save_bytes
    paths=[]
    def save(fig,name):
        path='plots/'+name+'.svg';stream=io.BytesIO()
        fig.savefig(stream,format='svg',metadata={'Date':None},bbox_inches='tight')
        _save_bytes(directory/path,stream.getvalue());paths.append(path)
        png=io.BytesIO();fig.savefig(png,format='png',dpi=170,bbox_inches='tight')
        png_path='plots/'+name+'.png';_save_bytes(directory/png_path,png.getvalue());paths.append(png_path)
        plt.close(fig)
    with plt.rc_context({'svg.hashsalt':'pilot4-combined-v1','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.facecolor':'#fcfcf8','figure.facecolor':'white'}):
        fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');labels=[]
        for method in ['sft','grpo']:
            for scorer in ['strict','flexible_v3','last_number']:
                i=len(labels);labels.append(method.upper()+' · '+scorer.replace('_',' '))
                a=result['greedy'][method+':0'][scorer]['accuracy']*100;b=result['greedy'][method+':64'][scorer]['accuracy']*100
                ax.plot([a,b],[i,i],color='#c9d3cd',linewidth=3)
                ax.plot(a,i,'o',color='#87948f',label='Step 0' if i==0 else None)
                ax.plot(b,i,'o',color='#437b91' if b>=a else '#ae7869',label='Step 64' if i==0 else None)
                ax.annotate(f'{a:.1f} → {b:.1f}',(max(a,b)+1,i),va='center',fontsize=9)
        ax.set(yticks=range(6),yticklabels=labels,xlabel='Original automatic accuracy (%)',xlim=(0,100),title='The apparent outcome depends on the extraction rule')
        ax.invert_yaxis();ax.legend(frameon=False,loc='upper right');save(fig,'greedy-outcomes')
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        ceiling=max(max(row) for m in ['sft','grpo'] for row in result['transitions'][m]['flexible_v3']['counts'])
        for ax,m in zip(axes,['sft','grpo']):
            flow=result['transitions'][m]['flexible_v3'];values=np.asarray(flow['counts'])
            ax.imshow(values,cmap='Blues',vmin=0,vmax=ceiling)
            for a in range(3):
                for b in range(3):ax.text(b,a,str(values[a,b]),ha='center',va='center',color='white' if values[a,b]>ceiling*.6 else '#243b42')
            ax.set(xticks=range(3),yticks=range(3),xticklabels=['Correct','Wrong','Unextractable'],yticklabels=['Correct','Wrong','Unextractable'],
                   xlabel='Step 64',ylabel='Step 0',title=f'{m.upper()}: {flow["gross_gains"]} gains / {flow["gross_losses"]} losses')
        fig.suptitle('Original flexible-v3 transitions · paired questions');save(fig,'answer-transitions')
        fig,ax=plt.subplots(figsize=(9,4),layout='constrained');labels=[]
        for scorer in ['flexible_v3','last_number']:
            for field,label in [('mean_draw_reward','Mean draw reward'),('observed_coverage','Observed ≥1/8 coverage')]:
                i=len(labels);labels.append(scorer.replace('_',' ')+' · '+label)
                metric=result['sampling'][scorer][field]['difference'];lo,hi=[x*100 for x in metric['interval_95']];v=metric['estimate']*100
                ax.plot([lo,hi],[i,i],color='#77968c',lw=2);ax.plot(v,i,'o',color='#437b91')
        ax.axvline(0,color='#b7bfba',lw=1);ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='Step 64 − step 0 (percentage points)',title='Saved sampled outputs: reliability rises more than observed coverage')
        ax.invert_yaxis();save(fig,'sampling-reliability')
        if result['confirmed_case_overlay']:
            fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
            for i,m in enumerate(['sft','grpo']):
                row=result['confirmed_case_overlay']['methods'][m]
                for offset,key,color,label in [(-.16,'original','#a9b5ae','Original flexible-v3'),(.16,'confirmed_corrections_only','#437b91','Confirmed-case overlay')]:
                    value=row[key]['gain_pp'];ax.barh(i+offset,value,height=.28,color=color,label=label if i==0 else None)
                    ax.text(value+(.3 if value>=0 else -.3),i+offset,f'{value:+.1f}',va='center',ha='left' if value>=0 else 'right')
            ax.axvline(0,color='#b7bfba',lw=1);ax.set(yticks=[0,1],yticklabels=['SFT','GRPO'],xlabel='Step 64 − step 0 (percentage points)',title='Confirmed extraction corrections change the baseline')
            ax.legend(frameon=False,loc='lower right');ax.margins(x=.25);save(fig,'confirmed-corrections')
    return paths


def markdown(r):
    lines=['# Pilot 4: behavioural findings','',
        '**Evidence status:** '+r['evidence_state']+'.','',
        'This CPU-only follow-up analyzes saved answers. It separates formatting, scorer extraction, '
        'paired answer changes and observed sampling reliability. It does not validate reasoning chains.','',
        '## 1. Greedy accuracy depends on the scorer','',
        '| Method | Scorer | Step 0 correct / n | Step 64 correct / n | Paired change, 95% interval |',
        '|---|---|---|---|---|']
    for m in ['sft','grpo']:
        for s in ['strict','flexible_v3','last_number']:
            a=r['greedy'][m+':0'];b=r['greedy'][m+':64'];flow=r['transitions'][m][s]
            lines.append(f'| {m.upper()} | {s} | {a[s]["correct"]}/{a["total"]} | {b[s]["correct"]}/{b["total"]} | '
                         f'{_pp(flow["difference"]["estimate"])}; {_interval(flow["difference"])} |')
    lines+=['','### Earlier recorded flexible decisions','',
            'These preserve historical policies, not one shared scorer. Missing decisions are not counted as wrong.',
            '', '| Method / step | Recorded correct / available | Available / cohort | Recorded version | Changed decision under v3 |',
            '|---|---|---|---|---|']
    for key,v in r['recorded_flexible'].items():
        lines.append(f'| {key} | {v["correct"] if v["correct"] is not None else "unavailable"}/{v["available"]} | '
                     f'{v["available"]}/{v["total"]} | {", ".join(v["recorded_versions"]) or "unavailable"} | '
                     f'{v["changed_decision_under_v3"]} |')
    lines+=['','Strict compliance requires the requested final-line grammar. Flexible-v3 rejects some '
            'identifiable conclusions when contextual numbers trigger ambiguity checks. The last-number '
            'diagnostic can instead select an incidental number; none is a reasoning judge.','',
            '![Original greedy scores](../plots/greedy-outcomes.svg)','',
            '## 2. Gains and losses are different questions from the net change','',
            '| Method | Gross gains | Gross losses | Churn | Net change | Net W→C flow | Net U→C flow |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for m in ['sft','grpo']:
        f=r['transitions'][m]['flexible_v3']
        lines.append(f'| {m.upper()} | {f["gross_gains"]} | {f["gross_losses"]} | {f["churn"]} | {f["net_change"]:+d} | {f["net_wrong_flow"]:+d} | {f["net_invalid_flow"]:+d} |')
    lines+=['','C = correct, W = extracted but wrong, U = unextractable. U→C is not automatically a format fix, '
            'and W→C is not automatically improved reasoning.','',
            '![Paired transitions](../plots/answer-transitions.svg)','',
            '### Direct endpoint comparison','']
    for s,v in r['direct_comparison'].items():
        if v.get('available'):lines.append(f'- GRPO64 − SFT64, {s}: {_pp(v["difference"]["estimate"])}, '
                                          f'95% interval {_interval(v["difference"])} on {v["n"]} common question IDs.')
    lines+=['','Original endpoint denominators: '+str(r['original_endpoint_totals'])+'. '
            'Within-method exclusions are retained in results.json; common-ID restrictions do not replace original totals.','',
            '## 3. Human review and confirmed extraction corrections','',
            'Audit state: '+r['audit']['state']+'.']
    if r['confirmed_case_overlay']:
        c=r['confirmed_case_overlay'];lines+=['',
            f'{c["unique_confirmed_numeric"]} unique numeric transcriptions enter this overlay. '
            'Automatic proposals are excluded. Absent/ambiguous labels do not receive invented numeric answers.','',
            '| Method | Original correct counts 0→64 | With confirmed cases 0→64 | Revised net gain |',
            '|---|---|---|---|']
        for m,v in c['methods'].items():
            a=v['original'];b=v['confirmed_corrections_only']
            lines.append(f'| {m.upper()} | {a["before_correct"]}→{a["after_correct"]} / {a["n"]} | '
                         f'{b["before_correct"]}→{b["after_correct"]} / {b["n"]} | {b["gain"]:+d} ({b["gain_pp"]:+.1f} pp) |')
        lines+=['','These are conditional, partial correction totals. They are not exhaustive human-scored accuracy. '
                'Original bootstrap intervals do not apply to the overlay. Shared untuned responses contribute '
                'to both method references and are not independent discoveries.','',
                '![Confirmed correction overlay](../plots/confirmed-corrections.svg)']
    else:lines+=['','No confirmed correction overlay is supplied. Automatic tables do not imply completed human review.']
    lines+=['','## 4. Observed sampled reliability and coverage','',
            f'{r["sampled_responses"]} saved sampled responses: {r["sampling_n"]} questions × eight draws × two endpoints. '
            'No SFT sampled comparison is available in this saved experiment.','',
            '| Scorer | Measurement | Step 0 | Step 64 | Paired change, 95% interval |',
            '|---|---|---|---|---|']
    sampling_notes=[]
    for s,v in r['sampling'].items():
        for field,label in [('mean_draw_reward','Mean draw reward'),('observed_coverage','Observed ≥1/8 coverage')]:
            metric=v[field]
            lines.append(f'| {s} | {label} | {_pct(metric["before"])} | {_pct(metric["after"])} | '
                         f'{_pp(metric["difference"]["estimate"])}; {_interval(metric["difference"])} |')
        sampling_notes.append(f'{s}: {len(v["coverage_gain_ids"])} observed coverage gains and '
                     f'{len(v["coverage_loss_ids"])} losses; {len(v["already_observed_solved"]["item_ids"])} questions solved '
                     f'at least once at both endpoints. Mean correct-draw change on that subset: '
                     f'{v["already_observed_solved"]["mean_count_change"]:+.2f} out of eight.')
    lines+=['']+sampling_notes
    lines+=['','Higher mean draw reward with little observed coverage change is compatible with greater reliability '
            'on questions already solved in the saved draws. Eight draws cannot establish the complete support '
            'of either policy. These sampled outputs remain automatically scored; greedy audit corrections do not transfer to them.','',
            '![Sampled reliability](../plots/sampling-reliability.svg)','',
            '## 5. SFT churn and response length','',
            '| Method | Step 0 mean / median tokens | Step 64 mean / median tokens |',
            '|---|---|---|']
    for m,v in r['response_lengths'].items():
        lines.append(f'| {m.upper()} | {v["0"]["mean"]:.1f} / {v["0"]["median"]:.1f} | '
                     f'{v["64"]["mean"]:.1f} / {v["64"]["median"]:.1f} |')
    lines+=['','Gold calculator-annotation and nonempty-line strata are saved in results.json. They are descriptive '
            'groupings, not validated difficulty bins. Shorter responses do not establish the cause of SFT losses.','',
            '## 6. Interpretation and next decisions','',
            'The observations support several compatible explanations: stronger adherence to answer formatting, '
            'scorer-specific extraction failures, changes in which questions are answered correctly, and changes '
            'in observed sampling reliability. They do not establish mutually exclusive causal shares.','']
    lines+=['- '+s for s in r['next_decisions']]
    lines+=['','## Methods and limits','',
            'Paired bootstrap: 10,000 question-ID resamples, local seed 42, percentile 2.5/97.5 bounds. '
            'Both endpoints travel together; sampling preserves all eight draws within each question. '
            'The original reports retain actual bootstrap indices and ordered IDs.','']
    lines+=['- '+s for s in r['limitations']]
    lines+=['','## Evidence files','',
            '- `results.json`: reconciled summaries, original/common cohort identities, audit state and limits.',
            '- `summary.csv`: original score counts and separately marked confirmed-case totals.',
            '- `paired-items.jsonl`: paired greedy answers and scorer decisions.',
            '- `sampled-counts.jsonl`: per-question observed correct-draw counts.',
            '- `../config.json`, `../complete.json`: named source identities and immutable output hashes.','']
    return '\n'.join(lines)


def page(r,directory,root):
    esc=lambda v:html.escape(str(v))
    def table(headers,rows):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    def image(name,alt):return f'<figure><img src="plots/{name}.svg" alt="{esc(alt)}"></figure>'
    corrected=r['confirmed_case_overlay'];has=bool(corrected)
    title='Extraction changes the interpretation of the gain' if has else 'What is behind the behavioural change?'
    hero='Saved greedy answers, paired transitions and eight-draw sampling distinguish what changed from what the scorer counted.'
    if has:
        grpo=corrected['methods']['grpo'];sft=corrected['methods']['sft']
        hero=(f'Confirmed extraction corrections reduce GRPO’s measured gain from {grpo["original"]["gain_pp"]:+.1f} to '
              f'{grpo["confirmed_corrections_only"]["gain_pp"]:+.1f} percentage points. '
              f'SFT’s measured change becomes {sft["confirmed_corrections_only"]["gain_pp"]:+.1f} points. '
              'These are partial confirmed-case corrections, not exhaustive human-scored accuracy.')
    greedy=[]
    for m in ['sft','grpo']:
        for s in ['strict','flexible_v3','last_number']:
            a=r['greedy'][m+':0'];b=r['greedy'][m+':64'];f=r['transitions'][m][s]
            greedy.append([m.upper(),s,f'{a[s]["correct"]}/{a["total"]}',f'{b[s]["correct"]}/{b["total"]}',_pp(f['difference']['estimate']),_interval(f['difference'])])
    flowrows=[]
    for m in ['sft','grpo']:
        f=r['transitions'][m]['flexible_v3'];flowrows.append([m.upper(),f['gross_gains'],f['gross_losses'],f['churn'],f['net_change']])
    content='<section id="scores"><div class="eyebrow">01 · Scoring</div><h2>The scorer is part of the measurement</h2><p>Strict format compliance, flexible extraction and the last-number diagnostic produce different apparent outcomes. Last-number extraction is a diagnostic, not a human oracle. A response can give an identifiable answer while missing the exact final-line grammar; repeated question quantities can also trigger flexible-v3 ambiguity checks.</p>'+table(['Method','Scorer','Step 0','Step 64','Paired change','95% interval'],greedy)+image('greedy-outcomes','Original greedy accuracy by method, scorer and checkpoint')+'</section>'
    historical=[]
    for key,v in r['recorded_flexible'].items():
        historical.append([key,str(v['correct'])+'/'+str(v['available']) if v['correct'] is not None else 'unavailable',
                           str(v['available'])+'/'+str(v['total']),', '.join(v['recorded_versions']) or 'unavailable',v['changed_decision_under_v3']])
    content+='<details><summary>Earlier stored scores and scorer changes</summary><p>Recorded flexible decisions can use different extraction policies. They are preserved as historical evidence, not substituted for harmonized scoring. Missing values are not counted as wrong.</p>'+table(['Method / step','Recorded correct / available','Available / cohort','Recorded version','Changed under v3'],historical)+'</details>'
    content+='<section id="transitions"><div class="eyebrow">02 · Paired questions</div><h2>A net change hides gains and losses</h2>'+table(['Method','Gains','Losses','Churn','Net'],flowrows)+'<p>The cells below retain the original flexible-v3 classification. Correct, wrong and unextractable are scorer-defined states. Neither a wrong→correct nor an unextractable→correct transition establishes a reasoning improvement.</p>'+image('answer-transitions','Original flexible-v3 paired correct, wrong and unextractable transitions')+'</section>'
    content+='<section id="audit"><div class="eyebrow">03 · Human evidence</div><h2>Correcting missed answers changes the starting point</h2><p class="callout">Audit status: '+esc(r['audit']['state'])+'. This report keeps completed automatic analysis distinct from a completed full human audit.</p>'
    if has:
        counts=r['audit'].get('human_classification_counts',corrected.get('human_classification_counts',{}))
        content+='<p>Saved human classifications: '+esc(', '.join(str(v)+' '+k for k,v in counts.items()))+'. '
        content+=str(corrected['unique_confirmed_numeric'])+' numeric transcriptions are human-confirmed; '+str(r['audit'].get('unconfirmed_numeric_proposals',0))+' automatic numeric proposals remain unconfirmed and are excluded.</p>'
        rows=[]
        for m,v in corrected['methods'].items():
            a=v['original'];b=v['confirmed_corrections_only'];rows.append([m.upper(),f'{a["before_correct"]} → {a["after_correct"]} / {a["n"]}',f'{b["before_correct"]} → {b["after_correct"]} / {b["n"]}',f'{b["gain_pp"]:+.1f} pp'])
        content+=table(['Method','Original counts','Confirmed-case overlay','Revised gain'],rows)+image('confirmed-corrections','Original net gains compared with confirmed-case correction overlays')
        content+='<p>'+str(corrected['unique_confirmed_numeric'])+' unique human-confirmed numeric transcriptions contribute. Other answers retain original scorer outcomes. Unconfirmed proposals are excluded, and no revised confidence interval is claimed.</p>'
        content+='<details><summary>Inspect every correctness-changing confirmed case</summary>'+table(['Response #','Method','Step','Question ID','Intended','Gold'],[[c['response_number'],c['method'],c['step'],c['item_id'],c['intended_answer'],c['gold_answer']] for c in corrected['changes']])+'</details>'
    else:content+='<p>No confirmed-case overlay is supplied. Audit completion is not inferred from software completion.</p>'
    content+='</section><section id="sampling"><div class="eyebrow">04 · Eight observed draws</div><h2>Reliability rises more than observed coverage</h2><p>Mean draw reward averages the eight answers for every question. Observed coverage asks whether at least one of those eight was scored correct. They answer different questions.</p>'
    rows=[]
    for s,v in r['sampling'].items():
        for f,label in [('mean_draw_reward','Mean draw reward'),('observed_coverage','Observed ≥1/8 coverage')]:
            metric=v[f];rows.append([s,label,_pct(metric['before']),_pct(metric['after']),_pp(metric['difference']['estimate']),_interval(metric['difference'])])
    content+=table(['Scorer','Measurement','Step 0','Step 64','Change','95% interval'],rows)+image('sampling-reliability','Saved sampled reward and coverage differences with paired percentile intervals')+'<p class="callout">These sampled responses were not human-rescored. The greedy extraction audit does not correct them. Observed coverage is limited to eight draws and is not proof of new ability.</p></section>'
    content+='<section id="length"><div class="eyebrow">05 · SFT descriptions</div><h2>Response length is context, not a causal explanation</h2>'+table(['Method','Step 0 mean tokens','Step 64 mean tokens','Step 0 median','Step 64 median'],[[m.upper(),f'{v["0"]["mean"]:.1f}',f'{v["64"]["mean"]:.1f}',f'{v["0"]["median"]:.1f}',f'{v["64"]["median"]:.1f}'] for m,v in r['response_lengths'].items()])+'<p>Gold calculator-annotation counts and nonempty-line strata are retained in the structured results. They are not validated difficulty bins, and response shortening does not establish why questions were lost.</p></section>'
    content+='<section id="limits"><div class="eyebrow">06 · Interpretation</div><h2>Several explanations remain compatible</h2><p>The evidence separates format compliance, extraction failures, answer churn and observed sampling reliability. It does not partition the net gain into mutually exclusive causes or establish reasoning quality.</p><ul>'+''.join('<li>'+esc(x)+'</li>' for x in r['limitations'])+'</ul><h3>Decisions before further GPU work</h3><ul>'+''.join('<li>'+esc(x)+'</li>' for x in r['next_decisions'])+'</ul></section>'
    links=[]
    for label,key,tail in [('Original scoring','scores','results/report.md'),('Paired transitions and bootstrap','transitions','results/report.md'),('Sampled reliability and bootstrap','sampling','results/report.md'),('Confirmed correction cases','assisted_report','report.html')]:
        if r['links'].get(key):
            path=root/r['links'][key]/tail;url=os.path.relpath(path,directory)
            links.append('<li><a href="'+esc(url)+'">'+esc(label)+'</a></li>')
    content+='<section id="evidence"><div class="eyebrow">07 · Provenance</div><h2>Sources, settings and inspectable data</h2><p>All measurements here use saved outputs. This command loads no language model and generates no answers. Original and common-ID denominators remain distinct.</p><ul>'+''.join(links)+'</ul><p><a href="results/results.json">Structured findings</a> · <a href="results/summary.csv">Summary CSV</a> · <a href="results/report.md">Methods and full tables</a> · <a href="config.json">Bound source configuration</a></p><details><summary>Declared source model and decoding settings</summary>'
    for row in r['source_inventory']:
        content+='<h3>'+esc(row.get('method','source'))+'</h3><pre>'+esc(__import__('json').dumps(row.get('metadata',row),indent=2))+'</pre>'
    content+='</details><p class="small">Paired percentile intervals use 10,000 question resamples, seed 42, preserving endpoints and all eight sampled draws within each item. They condition on these checkpoints and saved generations.</p></section>'
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pilot 4 · Behavioural findings</title><style>
:root{--ink:#203b43;--muted:#62767a;--line:#d9e2dc;--accent:#346c78}*{box-sizing:border-box}body{margin:0;background:#f6f7f2;color:var(--ink);font:16px/1.7 system-ui,-apple-system,sans-serif}header,main,footer{max-width:1120px;margin:auto;padding:40px 28px}header{padding-top:70px}h1{font:clamp(34px,5vw,58px)/1.12 Georgia,serif;max-width:900px;margin:14px 0 24px}h2{font:30px/1.25 Georgia,serif;margin:12px 0 20px}h3{font-size:18px}p{max-width:920px}.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:12px;font-weight:700;color:var(--muted)}.lead{font-size:20px;max-width:880px}.callout{background:#e7efea;border-left:4px solid #6f9b8e;padding:18px 22px}.badge{display:inline-block;font-size:12px;border:1px solid var(--line);padding:5px 10px;border-radius:20px;margin:4px 6px 0 0}nav{border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:16px 0;display:flex;gap:20px;flex-wrap:wrap}a{color:var(--accent);text-underline-offset:3px}section{padding:42px 0;border-bottom:1px solid var(--line);scroll-margin-top:20px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;background:#fffefa;font-size:14px}td,th{text-align:left;padding:13px 15px;border-bottom:1px solid var(--line);vertical-align:top}th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}figure{margin:30px 0}img{width:100%;height:auto;background:white;border:1px solid var(--line);border-radius:6px}details{margin:22px 0;padding:18px;background:#fffefa;border:1px solid var(--line);border-radius:6px}summary{cursor:pointer;font-weight:650}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 ui-monospace}.small{font-size:13px;color:var(--muted)}footer{font-size:13px;color:var(--muted)}@media(max-width:600px){header,main,footer{padding:26px 16px}header{padding-top:42px}td,th{padding:10px}nav{gap:14px}section{padding:30px 0}h2{font-size:26px}}@media print{nav{display:none}body{background:white}header,main{padding:15px}details{display:block}section{break-inside:avoid}}</style></head><body>'''+ '<header><div class="eyebrow">SPAR · Pilot 4 behavioural follow-up</div><h1>'+esc(title)+'</h1><p class="lead">'+esc(hero)+'</p><span class="badge">CPU analysis · no new inference</span><span class="badge">'+esc(r['evidence_kind'])+'</span><span class="badge">'+esc(r['audit']['state'])+'</span><nav>'+''.join('<a href="#'+k+'">'+v+'</a>' for k,v in [('scores','Scores'),('transitions','Transitions'),('audit','Human evidence'),('sampling','Sampling'),('limits','Interpretation'),('evidence','Sources')])+'</nav></header><main>'+content+'</main><footer>Named immutable evidence snapshots · original outcomes retained · automatic proposals distinct from human confirmations</footer></body></html>'
