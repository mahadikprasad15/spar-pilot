"""CPU gate calculations and inspectable scientific figures; no model imports."""
import csv
import html
import itertools
import json
import math
import os
import re
import unicodedata

from pilot_eval import sampling_sft_review as review
from pilot_eval import sampling_sft_generation as generation
from pilot_eval import sampling_sft_gate_math as stats
from pilot_eval import sampling_sft_gate as gate
from pilot_eval.sampling_sft_sampler import clean_gold
from pilot_eval.scoring import score_gsm8k
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash


def words(text,policy):
    if policy=='unicode-words-v1':
        text=unicodedata.normalize('NFKC',text).casefold().replace('\u2019',"'")
        return re.findall(r"[^\W_]+(?:'[^\W_]+)*",text,flags=re.UNICODE)
    raise ValueError('Unknown predeclared word-normalization contract.')


def leakage(text,gold,policy):
    response=words(text,policy);reference=words(clean_gold(gold),policy)
    gold_grams={tuple(reference[i:i+4]) for i in range(max(0,len(reference)-3))}
    grams=[tuple(response[i:i+4]) for i in range(max(0,len(response)-3))]
    hits=sum(g in gold_grams for g in grams)
    return dict(calculator_marker='<<' in text,word_count=len(response),word4_hits=hits,word4_total=len(grams),
        word4_overlap=hits/len(grams) if grams else None,
        word4_status='defined' if grams else 'undefined-fewer-than-four-words')


def rate(n,d):return dict(numerator=n,denominator=d,rate=n/d if d else None)


def collection_views(root,config,coverage):
    _,gen,folder=generation.load(root/config['generation_plan'],root)
    output={}
    for k in generation.ARMS:
        attempts=coverage['arms'][str(k)]['attempts'];initial=[a for a in attempts if a['retry_index']==0]
        final=coverage['arms'][str(k)]
        decisions=[]
        for path in sorted((folder/f'k{k}/rounds').glob('*/blocks/*/scientific.jsonl')):
            for record in generation.read_lines(path):
                if 'decision' in record:decisions.append(record['decision'])
        output['k'+str(k)]=dict(initial_auto_correct=rate(sum(a['eligible'] for a in initial),len(initial)),
            initial_unfinished=rate(sum(a['end_state'] in ['capped','block_budget_exhausted'] for a in initial),len(initial)),
            attempted_auto_correct=rate(sum(a['eligible'] for a in attempts),len(attempts)),
            attempted_unfinished=rate(sum(a['end_state'] in ['capped','block_budget_exhausted'] for a in attempts),len(attempts)),
            retried_problems=rate(len({a['problem_id'] for a in attempts if a['retry_index']>0}),512),
            reviewed_coverage=rate(final['covered_count'],512),attempts=len(attempts),
            coverage_decision=final['coverage_decision'],
            acceptance=rate(sum(d['accepted'] for d in decisions),len(decisions)),
            acceptance_by_step={str(i):rate(sum(d['accepted'] for d in decisions if d['step']==i),
                sum(d['step']==i for d in decisions)) for i in range(k)},
            acceptance_by_cut_decile={str(i):rate(sum(d['accepted'] for d in decisions
                if min(9,int(d['cut']/len(d['old_ids'])*10))==i),sum(min(9,int(d['cut']/len(d['old_ids'])*10))==i for d in decisions)) for i in range(10)})
    return output


def cohort(rows):
    value=stats.summarize_rows(rows)
    n=len(rows)
    value['strict_format']=rate(sum(row['strict_format'] for row in rows),n)
    value['contains_boxed_marker']=rate(sum('\\boxed{' in row['text'] for row in rows),n)
    value['contains_final_marker']=rate(sum('####' in row['text'] for row in rows),n)
    return value


def build(root,config,folder,scores):
    coverage=review.verify(root/config['review_plan'],root)
    rows=[];config_sha256=_hash(config)
    for index in range(512):
        value=gate.unit(folder,index,config,config_sha256)
        for request,score in zip(value['requests'],value['scores']):
            leak=leakage(request['text'],request['gold'],config['word_policy']['version'])
            rows.append(dict(**request,logprobs=score['logprobs'],positions=score['positions'],
                nll=-math.fsum(score['logprobs'])/len(score['logprobs']),
                strict_format=score_gsm8k(request['text'],request['gold'])['strict']['status']=='valid',**leak))
    groups={m:[r for r in rows if r['method']==m] for m in gate.METHODS}
    common=set(coverage['common_covered_ids'])
    paired={m:sorted([r for r in group if r['source_id'] in common],key=lambda r:r['problem_id']) for m,group in groups.items()}
    identity=[r['problem_id'] for r in paired['gold']]
    if any([r['problem_id'] for r in paired[m]]!=identity for m in gate.METHODS):raise ValueError('Paired gate cohorts are not aligned.')
    contrasts={}
    for left,right in itertools.combinations(gate.METHODS,2):
        contrasts[left+'-minus-'+right]=dict(
            nll=stats.paired_bootstrap([r['nll'] for r in paired[left]],[r['nll'] for r in paired[right]],'nll'),
            length=stats.paired_bootstrap([len(r['response_ids']) for r in paired[left]],
                                        [len(r['response_ids']) for r in paired[right]],'length'))
    report=dict(protocol='sampling-sft-gate-report-v1',execution_kind=scores['execution_kind'],
        own_cohort={m:cohort(group) for m,group in groups.items()},common_cohort={m:cohort(group) for m,group in paired.items()},
        common_problem_count=len(identity),common_problem_ids=identity,paired_contrasts=contrasts,
        collection=collection_views(root,config,coverage),coverage=dict(blocked=coverage['coverage_blocked'],
            pairwise_flags=coverage['pairwise_uncovered_differences'],
            uncovered_ids={k:a['uncovered_ids'] for k,a in coverage['arms'].items()},
            t16_required={k:a['t16_required'] for k,a in coverage['arms'].items()}),
        word_policy=config['word_policy'],word4_counting='response occurrences whose tuple occurs anywhere in cleaned gold',
        baseline='not measured',relative_sharpening='not measured',
        claim='Data-recipe comparison; descriptive diagnostics, no causal attribution or automatic equivalence/merge.',
        bootstrap=config['bootstrap'],runtime_hash=config['runtime_hash'],parents=config['parents'],
        score_evidence_sha256=file_hash(folder/'scores/complete.json'),training_authorized=False)
    return report,rows


def render(folder,report,rows):
    """Stable SVG artifacts plus raw CSV/JSON, all based on the common cohort."""
    os.environ.setdefault('MPLCONFIGDIR',str(folder/'plot-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.hashsalt':'sampling-sft-gate-v1','font.size':10,'font.family':'DejaVu Sans'})
    plots=folder/'plots';plots.mkdir(exist_ok=True)
    manifest=[]
    colors=['#8b9699','#6e9eb4','#7fae9e','#bd9e74']
    for metric,title,units in [('nll','Common-cohort answer NLL','nats / response token'),
                               ('length_median','Common-cohort median length','response tokens, stop included')]:
        fig,ax=plt.subplots(figsize=(6.5,3.8));values=[report['common_cohort'][m][metric] for m in gate.METHODS]
        if all(v is None for v in values):ax.text(0.5,0.5,'No common cohort',ha='center',va='center',transform=ax.transAxes)
        else:
            ax.bar(gate.METHODS,[v if v is not None else 0 for v in values],color=colors,width=0.6)
            for i,v in enumerate(values):
                if v is not None:ax.annotate(f'{v:.3g}',(i,v),ha='center',va='bottom',xytext=(0,3),textcoords='offset points')
        ax.set_title(title);ax.set_ylabel(units);ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',alpha=0.15)
        fig.tight_layout();name='common-nll.svg' if metric=='nll' else 'common-length.svg'
        fig.savefig(plots/name,metadata={'Date':None});plt.close(fig)
        manifest.append(dict(path=name,metric=metric,cohort='three-arm common',n_problems=report['common_problem_count'],values=values))
    for metric,units in [('nll','difference in equal-problem NLL (nats/token)'),('length','difference in median response length (tokens)')]:
        intervals={pair:value[metric] for pair,value in report['paired_contrasts'].items()}
        fig,ax=plt.subplots(figsize=(7.5,4.2))
        for index,(pair,value) in enumerate(intervals.items()):
            if value['estimate'] is None:continue
            color='#6e9eb4' if value['estimate']>=0 else '#bd8b81'
            ax.hlines(index,value['ci_low'],value['ci_high'],color=color,lw=2)
            ax.plot(value['estimate'],index,'o',color=color)
        ax.axvline(0,color='#8b9699',lw=1);ax.set_yticks(range(len(intervals)),list(intervals));ax.invert_yaxis()
        ax.spines[['top','right']].set_visible(False);ax.set_xlabel(units)
        ax.set_title('Paired left − right estimates · 95% percentile intervals')
        ax.grid(axis='x',alpha=0.15);ax.set_axisbelow(True);fig.tight_layout()
        name='paired-'+metric+'.svg';fig.savefig(plots/name,metadata={'Date':None});plt.close(fig)
        manifest.append(dict(path=name,metric=metric,cohort='three-arm common',n_problems=report['common_problem_count'],intervals=intervals))
    (plots/'manifest.json').write_text(json.dumps(dict(report_sha256=_hash(report),plots=manifest),indent=2)+'\n')
    with (folder/'results/summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['cohort','method','n_answers','n_tokens','nll','length_median','length_p90'])
        writer.writeheader()
        for scope in ['own_cohort','common_cohort']:
            for method,value in report[scope].items():writer.writerow(dict(cohort=scope,method=method,**{k:value[k] for k in writer.fieldnames[2:]}))
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [m,d['n_answers'],d['nll'],d['length_median'],d['length_p90']])+'</tr>'
                  for m,d in report['common_cohort'].items())
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sampling SFT data gate</title><style>body{max-width:1000px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#293e44;background:#fbfcf8}h1{font-family:Georgia}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #d7dfda;text-align:left}.plots{display:flex;flex-wrap:wrap}.plots img{width:48%;min-width:300px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>
<h1>Sampling SFT · data gate</h1><p>'''+html.escape(report['claim'])+'''</p><p>Execution evidence: <strong>'''+report['execution_kind']+'''</strong>. Training is not authorized by this report.</p>
<p>The table and plots below use the same problems across gold and all three k arms. The raw report also includes each arm's own covered cohort.</p><table><tr><th>Method</th><th>Problems</th><th>NLL</th><th>Median length</th><th>P90 length</th></tr>'''+table+'''</table>
<div class="plots"><img alt="Common-cohort NLL" src="plots/common-nll.svg"><img alt="Common-cohort median length" src="plots/common-length.svg"></div>
<div class="plots"><img alt="Paired NLL contrasts and 95% intervals" src="plots/paired-nll.svg"><img alt="Paired length contrasts and 95% intervals" src="plots/paired-length.svg"></div>
<p>Typical-sample baseline and relative sharpening: not measured. No baseline-adjusted quantity is computed.</p>
<p><a href="results/report.json">Structured report</a> · <a href="results/answers.jsonl">Answer/token evidence</a> · <a href="results/summary.csv">Summary CSV</a> · <a href="plots/manifest.json">Figure inputs</a></p>
<details><summary>Paired estimates and 95% percentile intervals</summary><pre>'''+html.escape(json.dumps(report['paired_contrasts'],indent=2))+'''</pre></details>
<details><summary>Coverage, retries, acceptance and descriptive diagnostics</summary><pre>'''+html.escape(json.dumps(dict(coverage=report['coverage'],collection=report['collection'],own_cohort=report['own_cohort'],common_cohort=report['common_cohort'],word_policy=report['word_policy']),indent=2))+'''</pre></details></html>'''
    (folder/'report.html').write_text(page)
