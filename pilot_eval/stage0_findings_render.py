"""Inspectable CPU primary report and exportable figures; all contrasts visible."""
import html
import io
import json
from pathlib import Path

from pilot_eval.stage0_edits import VARIANTS
from pilot_eval.rescoring import _save_bytes


def _plot(directory, name, series, xlabel, title, *, margin=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'svg.hashsalt': 'stage0-primary-findings-v1', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, ax = plt.subplots(figsize=(10, max(4.5, .32*len(series)+1.4)), layout='constrained')
        for i, (label, value) in enumerate(series):
            estimate = value['estimate']; limits = value['interval_95']
            color = '#437f9a' if estimate >= 0 else '#b87464'
            ax.plot(limits, [i, i], color=color, linewidth=2)
            ax.scatter([estimate], [i], color=color, s=24, zorder=3)
        ax.set_yticks(range(len(series)), [name for name, _ in series]); ax.invert_yaxis()
        ax.axvline(0, color='#a4aaa8', linewidth=1)
        if margin: ax.axvline(-5, color='#9d7564', linestyle='--', label='Non-inferiority boundary: −5 pp')
        ax.set_xlabel(xlabel); ax.set_title(title, loc='left', fontsize=12, pad=14)
        if margin: ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.16), fontsize=9)
        buffer = io.BytesIO(); fig.savefig(buffer, format='svg', metadata={'Date': None})
        png = io.BytesIO(); fig.savefig(png, format='png', dpi=120, metadata={'Software': 'Stage 0 findings'})
        plt.close(fig)
    _save_bytes(directory/'plots'/name.replace('.svg','.png'), png.getvalue())
    _save_bytes(directory/'plots'/name, buffer.getvalue())


def plots(result, directory):
    scaled = lambda value: dict(estimate=value['estimate']*100, interval_95=[v*100 for v in value['interval_95']])
    _plot(directory, 'accuracy.svg', [(c['id'], scaled(c['behaviour']['flexible']['difference'])) for c in result['contrasts']],
        'Edited/comparator − reference accuracy (percentage points); pointwise paired 95% interval',
        'Automatic flexible-v3 accuracy · all declared comparisons', margin=True)
    _plot(directory, 'length.svg', [(c['id'], c['behaviour']['length']['median_paired_change']) for c in result['contrasts']],
        'Median per-item generated content-token change; paired 95% interval',
        'Observed response length · edited/comparator − reference')
    cells = sorted({(v['corpus'], v['view']) for c in result['contrasts'] for v in c['drift']})
    for corpus, view in cells:
        series = [(c['id'], next(v for v in c['drift'] if (v['corpus'], v['view']) == (corpus, view))['kl']['difference'])
                  for c in result['contrasts']]
        _plot(directory, f'kl-{corpus}-{view}.svg', series,
            'Mean per-document KL change (nats); pointwise paired document 95% interval',
            f'Fixed-context KL(variant || untuned) · {corpus} / {view}')
    _save_bytes(directory/'plots/inputs.json', (json.dumps(dict(
        scope='Figures derived from the saved contrast estimates and interval_95 fields',
        comparisons=[dict(id=c['id'], accuracy=c['behaviour']['flexible']['difference'],
            length=c['behaviour']['length']['median_paired_change'],
            kl=[dict(corpus=v['corpus'], view=v['view'], **v['kl']['difference']) for v in c['drift']])
            for c in result['contrasts']]), sort_keys=True, indent=2)+'\n').encode())


def _interval(value, multiplier=1):
    lo, hi = value['interval_95']
    return f'{value["estimate"]*multiplier:+.3g} [{lo*multiplier:+.3g}, {hi*multiplier:+.3g}]'


def _table(headers, rows):
    escape = lambda x: html.escape(str(x))
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+escape(h)+'</th>' for h in headers)+\
        '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(v)+'</td>' for v in row)+'</tr>' for row in rows)+\
        '</tbody></table></div>'


def render(config, result, directory):
    directory = Path(directory); plots(result, directory)
    synthetic = result['execution_kind'] != 'real-cuda'
    sections = []
    sections.append('<p class="eyebrow">STAGE 0 · CONTROLLED ADAPTER INTERVENTIONS</p><h1>What changes when adapter components are removed?</h1>')
    sections.append('<p class="lead">Paired accuracy, response length and fixed-context drift on saved rank-one adapters. Every declared comparison is retained.</p>')
    if synthetic: sections.append('<aside><strong>Synthetic software fixture.</strong> These values test the pipeline; they are not research findings or GPU qualification.</aside>')
    sections.append('<div class="meta">Registration: '+html.escape(result['registration']['status'])+' · Review: '+html.escape(result['review']['state'])+'</div>')
    sections.append('<h2>Question and design</h2><p>Does removing the principal component of the SFT update change its length and distributional drift while retaining answer accuracy? Magnitude controls and individual random-subspace edits test alternative explanations; the GRPO edit supplies a second-arm comparison.</p>')
    if config.get('analysis_kind')=='conditional-intruders':
        sections[-1]='<h2>Conditional intruder intervention</h2><p>Does halving flagged singular components change accuracy, length or fixed-context KL relative to the original adapter and nearby non-flagged controls? Full, matched and ratio-filtered edits remain separate. Both original references are re-evaluated in this execution.</p>'
        sections.append(_table(['Variant','Arm','Scope','Merged modules'],[(name,row['source_arm'],row.get('mode','reference'),row['modules_edited']) for name,row in config['variant_definitions'].items()]))
        sections.append('<details><summary>Eligibility, undecidable/skipped modules and matched exclusions</summary><pre>'+html.escape(json.dumps(config['catalogue_scope']['arms'],indent=2))+'</pre></details>')
        sections.append('<p>The frozen analysis plan retains every original matched pair, its singular-value ratio and sensitivity mask. Full V6 is not restricted to matched modules. Empty sensitivity arms are not applicable. The rank-one factor norm-product statistic is unavailable for full merged-matrix edits; direct effective writes use the edited operator minus W0.</p>')
    else:
        sections.append(_table(['Variant', 'Source arm', 'Operation'], [(v, arm, operation) for v, (arm, operation) in VARIANTS.items()]))
    sections.append('<p>Behavior uses the same 150 held-out prompts, greedy decoding and 1,024-token cap. KL uses the fixed 150 GSM8K gold and 150 FineWeb sequences, with corpus/view results kept separate. All paired products bind one qualified execution, exact runtime and adapter payloads.</p>')
    schedules=config.get('inference_schedules')
    if schedules:
        sections.append('<p>Recorded inference schedules: behavioral batch '+str(schedules['behaviour_batch_size'])+
            '; fixed-context KL/effective-write batch '+str(schedules['internals_batch_size'])+
            '. Different schedules require the saved reviewed batch-agreement receipt.</p>')
    sections.append('<h2>Accuracy</h2><p>Flexible-v3 is primary; strict scores remain alongside it in the tables. “Preserved” requires the lower 95% bound to be <em>strictly above −5 percentage points</em>. At 150 items, inconclusive outcomes are plausible. The intervals are pointwise, not simultaneous across this family.</p><img src="plots/accuracy.svg" alt="All paired accuracy estimates and intervals">')
    sections.append(_table(['Comparison', 'Flexible difference, pp [95%]', 'Preservation test', 'Gains / losses', 'Exact McNemar p', 'Strict difference, pp [95%]'],
        [(c['id'], _interval(c['behaviour']['flexible']['difference'], 100), c['behaviour']['flexible']['noninferiority'],
          f"{c['behaviour']['flexible']['gains']} / {c['behaviour']['flexible']['losses']}",
          f"{c['behaviour']['flexible']['mcnemar_exact_two_sided']:.4g}", _interval(c['behaviour']['strict']['difference'],100))
         for c in result['contrasts']]))
    sections.append('<h2>Response length</h2><p>This is the median of paired item-level length changes, not the difference of marginal medians. Length is descriptive; it does not certify reasoning quality. Counts exclude EOS; capped responses retain the frozen zero-reward policy.</p><img src="plots/length.svg" alt="Paired median response-length changes">')
    sections.append(_table(['Comparison', 'Median paired tokens [95%]', 'Share shorter', 'Reference caps', 'Edited caps'],
        [(c['id'], _interval(c['behaviour']['length']['median_paired_change']),
          f"{100*c['behaviour']['length']['shorter_share']:.1f}%", c['behaviour']['length']['reference_cap_count'],
          c['behaviour']['length']['edited_cap_count']) for c in result['contrasts']]))
    sections.append('<h2>Fixed-context KL and effective writes</h2><p>KL is full-vocabulary KL(variant || untuned), in nats, averaged within each document before paired bootstrap. Token-weighted means remain separately available. Direct module-write sums were collected in the same runtime; module differences are descriptive, not independent-subject tests.</p>')
    resolution=result.get('numerical_resolution')
    if resolution:
        floor=resolution['observed_max_chunk_disagreement_nats']
        sections.append('<p>Observed maximum chunk disagreement: '+('not measured' if floor is None else f'{floor:.6g} nats')+
            '. This is an empirical numerical-resolution diagnostic, not statistical uncertainty or a universal bound.</p>')
        sections.append(_table(['Corpus','View','Variant pair','Smallest absolute token-mean KL difference','Observed chunk disagreement'],
            [(r['corpus'],r['view'],r['left']+' / '+r['right'],r['absolute_difference'],floor)
             for r in resolution['smallest_token_mean_variant_differences']]))
    for corpus, view in sorted({(v['corpus'], v['view']) for c in result['contrasts'] for v in c['drift']}):
        sections.append(f'<h3>{html.escape(corpus)} / {html.escape(view)}</h3><img src="plots/kl-{corpus}-{view}.svg" alt="Paired fixed-context KL changes">')
    sections.append('<p>KL is a distributional change on supplied contexts; lower KL alone is not demonstrated behavioral improvement. <a href="results/writes.csv">Module write differences</a> preserve all module identities.</p>')
    sections.append('<h2>Intended-answer review</h2><p>'+html.escape(result['review'].get('scope',''))+'</p>')
    if any('targeted_overlay' in c['behaviour'] for c in result['contrasts']):
        sections.append('<p>Confirmed numeric labels apply only to exactly matching prompt/response text. Unreviewed and nonnumeric cases retain automatic scores; this mixed overlay is <strong>not whole-cohort human accuracy</strong> and receives no preservation claim.</p>')
        sections.append(_table(['Comparison', 'Overlay correct: ref / edited', 'Confirmed numeric cases: ref / edited', 'Mixed overlay difference, pp'],
            [(c['id'], f"{c['behaviour']['targeted_overlay']['reference_correct']} / {c['behaviour']['targeted_overlay']['edited_correct']}",
              f"{c['behaviour']['targeted_overlay']['reviewed_reference']} / {c['behaviour']['targeted_overlay']['reviewed_edited']}",
              _interval(c['behaviour']['targeted_overlay']['difference'],100)) for c in result['contrasts']]))
    else:
        sections.append('<p>No confirmed numeric label overlay is applied in this snapshot. The raw report can complete while targeted review remains pending.</p>')
    sections.append('<h2>Scope and evidence</h2><ul>'+''.join('<li>'+html.escape(line)+'</li>' for line in result['limitations'])+'</ul>')
    sections.append('<p>'+html.escape(result['optional_work'])+'. '+html.escape(result['conditional_v6'])+'.</p>')
    sections.append('<p>Raw results and derivations: <a href="results/results.json">structured results</a> · <a href="results/behaviour.csv">accuracy CSV</a> · <a href="results/length.csv">length CSV</a> · <a href="results/kl.csv">KL CSV</a> · <a href="results/writes.csv">write CSV</a> · <a href="results/paired-items.jsonl">paired items</a> · <a href="plots/inputs.json">figure inputs</a> · <a href="config.json">frozen analysis plan</a>.</p>')
    prefix = result.get('evidence_base', '')
    sections.append('<h3>Comparison evidence</h3><p><a href="'+prefix+'inputs/responses.jsonl">Original response records</a> · <a href="'+prefix+'inputs/drift-documents.jsonl">Original document/module sums</a></p><ul>')
    for c in result['contrasts']:
        base = prefix+'units/'+c['id']+'/'
        sections.append('<li>'+html.escape(c['id'])+': <a href="'+base+'paired-items.jsonl">paired question records</a> · <a href="'+base+'paired-documents.jsonl">paired documents</a> · <a href="'+base+'result.json">estimates and saved draw paths</a></li>')
    sections.append('</ul>')
    if result['review'].get('numeric_overlay_records', 0):
        sections.append('<p><a href="results/review-evidence.jsonl">Exact confirmed-label source and response evidence</a></p>')
    sections.append('<details><summary>Source completion receipts and frozen hypotheses</summary><pre>'+html.escape(json.dumps(dict(source_seals=result['source_seals'],hypotheses=config['hypotheses'],registration=config['registration']),indent=2))+'</pre></details>')
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 0 intervention findings</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f7f8f5;color:#243b42;font:16px/1.65 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:52px 32px 80px}.eyebrow{letter-spacing:.14em;font-size:12px;color:#66817f}h1{font:46px/1.12 Georgia,serif;max-width:850px;margin:12px 0 20px}h2{font:29px/1.2 Georgia,serif;margin:48px 0 16px}h3{font-size:17px}.lead{font-size:20px;max-width:850px}.meta{font:13px monospace;color:#60716f;margin:24px 0}aside{padding:18px 22px;background:#fff0d8;border-left:4px solid #af8b4d}p,li{max-width:960px}img{display:block;width:100%;height:auto;background:white;border:1px solid #dbe2de;margin:20px 0}.table{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #dae2dd}th{background:#eaf0ec;white-space:nowrap}a{color:#276d86}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#edf1ed;padding:20px}details{margin-top:24px}@media(max-width:700px){main{padding:28px 16px}h1{font-size:34px}h2{font-size:25px}}
</style></head><body><main>'''+''.join(sections)+'</main></body></html>'
    (directory/'report.html').write_text(page)
    (directory/'results/report.md').write_text('# Stage 0 primary findings\n\nExecution: '+result['execution_kind']+
        '\n\nRegistration: '+result['registration']['status']+'\n\nReview: '+result['review']['state']+
        '\n\nFull comparison tables and figures are linked in ../report.html.\n\n'+'\n'.join('- '+x for x in result['limitations'])+'\n')
