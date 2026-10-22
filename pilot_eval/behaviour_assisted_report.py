"""Partial extraction-correction analysis using explicitly confirmed numbers only.

This analyst report neither freezes nor reveals the original human label journal.
Unconfirmed automatic proposals never affect the reported endpoint totals.
"""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
from datetime import datetime, timezone

from pilot_eval.scoring import _numeric_value


def summarize(pairs, mapping, assistance, confirmations):
    rows={r['response_id']:r for r in assistance['rows']}
    answers={rid:r['human_label']['answer'] for rid,r in rows.items()
             if r['human_label']['status']=='numeric' and r['human_label'].get('answer')}
    for document in confirmations:
        for label in document['labels']:
            rid=label['response_id']
            if rid not in rows or label['status']!='numeric':
                raise ValueError('unknown or nonnumeric confirmation')
            answer=label['answer'];_numeric_value(answer)
            if rid in answers and _numeric_value(answers[rid])!=_numeric_value(answer):
                raise ValueError('conflicting confirmed transcriptions')
            answers[rid]=answer
    indexed={}
    for pair in pairs:
        for endpoint in ['before','after']:
            r=pair[endpoint];key=(r['method'],r['step'],r['item_id'])
            if key in indexed:raise ValueError('duplicate endpoint')
            indexed[key]=r
    corrections={};cases=[];changes=[]
    for group in mapping:
        rid=group['response_id']
        if rid not in answers:continue
        answer=answers[rid]
        for ref in group['references']:
            key=(ref['method'],ref['step'],ref['item_id'])
            record=indexed[key]
            if record['generated_text']!=ref['record']['generated_text'] or record['gold']!=ref['record']['gold']:
                raise ValueError('audit reference does not match transition record')
            gold=record['gold'].rsplit('####',1)[-1].strip()
            correct=_numeric_value(answer)==_numeric_value(gold)
            if key in corrections:raise ValueError('duplicate correction reference')
            corrections[key]=correct
            case=dict(response_number=rows[rid]['index'],response_id=rid,method=key[0],step=key[1],
                      item_id=key[2],intended_answer=answer,gold_answer=gold,
                      original_correct=bool(record['scores']['flexible_v3']['correct']),
                      confirmed_answer_correct=correct)
            cases.append(case)
            if case['original_correct']!=correct:changes.append(case)
    methods={}
    for method in sorted({p['before']['method'] for p in pairs}):
        cohort=[p for p in pairs if p['before']['method']==method]
        scenarios={}
        for name,use_confirmed in [('original',False),('confirmed_corrections_only',True)]:
            before=[];after=[]
            for pair in cohort:
                for endpoint,values in [('before',before),('after',after)]:
                    r=pair[endpoint];key=(r['method'],r['step'],r['item_id'])
                    old=bool(r['scores']['flexible_v3']['correct'])
                    values.append(corrections.get(key,old) if use_confirmed else old)
            scenarios[name]=dict(n=len(cohort),before_correct=sum(before),after_correct=sum(after),
                                 gain=sum(after)-sum(before),gain_pp=100*(sum(after)-sum(before))/len(cohort),
                                 gross_gains=sum(not b and a for b,a in zip(before,after)),
                                 gross_losses=sum(b and not a for b,a in zip(before,after)))
        methods[method]=scenarios
    return dict(schema_version=1,scope='partial confirmed-case correction overlay; not whole-cohort human accuracy',
                scorer='flexible_v3',unique_confirmed_numeric=len(answers),
                human_classification_counts=dict(Counter(r['human_label']['status'] for r in rows.values())),
                unconfirmed_proposals_used=0,methods=methods,confirmed_references=cases,changes=changes,
                original_scores_modified=False,original_audit_frozen=False,
                limitations=['Targeted assisted review; flagged cases and scorer outcomes were discussed before numeric confirmation.',
                             'Unconfirmed proposals and absent/ambiguous classifications do not alter endpoint totals.',
                             'All other responses retain their original scorer values; this is not an exhaustive correction.',
                             'Differences in counts do not establish reasoning quality, causal attribution or a population error rate.',
                             'Original bootstrap intervals do not apply to these revised totals; no revised intervals are claimed.'])


def markdown(result):
    lines=['# Pilot 4: confirmed extraction corrections','',
           '**Partial confirmed-case overlay. Automatic proposals do not change these totals.**','',
           '| Method | Original step 0 → 64 | With confirmed corrections | Original gain | Revised gain |',
           '|---|---|---|---|---|']
    for method,row in result['methods'].items():
        a=row['original'];b=row['confirmed_corrections_only']
        lines.append(f'| {method.upper()} | {a["before_correct"]}/{a["n"]} → {a["after_correct"]}/{a["n"]} | '
                     f'{b["before_correct"]}/{b["n"]} → {b["after_correct"]}/{b["n"]} | '
                     f'{a["gain"]:+d} ({a["gain_pp"]:+.1f} pp) | {b["gain"]:+d} ({b["gain_pp"]:+.1f} pp) |')
    lines+=['','## Interpretation','',
            'The confirmed missed answers affect the untuned baseline more than the final adapters. '
            'Baseline undercounting inflated the original net training gain. The remaining net difference '
            'is answer-level evidence, not proof of improved reasoning.', '',
            'The shared untuned responses appear in both method comparisons. Corrections propagate to both '
            'references; those are not independent discoveries.', '',
            f'Confirmed numeric transcriptions: {result["unique_confirmed_numeric"]} unique responses. '
            f'Correctness-changing endpoint references: {len(result["changes"])}.', '',
            '## Cases that change a flexible-v3 correctness result','',
            '| Response # | Method | Step | Item | Intended answer | Gold | Original correct | Correct after confirmation |',
            '|---|---|---:|---|---|---|---|---|']
    for c in result['changes']:
        lines.append(f'| {c["response_number"]} | {c["method"]} | {c["step"]} | {c["item_id"]} | '
                     f'{c["intended_answer"]} | {c["gold_answer"]} | {c["original_correct"]} | {c["confirmed_answer_correct"]} |')
    lines+=['','## Limits','']+['- '+s for s in result['limitations']]
    return '\n'.join(lines)+'\n'


def render(result):
    # Small standalone analyst page. Gold is intentionally visible after targeted
    # confirmations, separate from the original blinded reviewer packet.
    tables=[]
    for method,row in result['methods'].items():
        a=row['original'];b=row['confirmed_corrections_only']
        tables.append(f'<tr><td>{method.upper()}</td><td>{a["before_correct"]} → {a["after_correct"]}</td>'
                      f'<td>{b["before_correct"]} → {b["after_correct"]}</td>'
                      f'<td>{a["gain_pp"]:+.1f} pp</td><td>{b["gain_pp"]:+.1f} pp</td></tr>')
    return ('<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Pilot 4 — confirmed extraction corrections</title><style>body{max-width:1050px;margin:45px auto;padding:0 24px;'
            'font:16px/1.65 system-ui;background:#f6f7f2;color:#203b42}h1{font:38px Georgia}table{border-collapse:collapse;width:100%;'
            'background:white}td,th{padding:14px;border-bottom:1px solid #d0dad4;text-align:left}.note{background:#e5eee8;padding:20px}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace}a{color:#286b61}</style>'
            '<h1>What changes when we fix confirmed extraction failures?</h1>'
            '<p class="note">A partial, assisted answer audit. Only human-confirmed numeric transcriptions replace scorer outcomes. '
            'All unconfirmed proposals are excluded. Counts below use 150 questions per method.</p>'
            '<table><tr><th>Method</th><th>Original counts<br>step 0 → 64</th><th>Confirmed corrections<br>step 0 → 64</th>'
            '<th>Original gain</th><th>Revised gain</th></tr>'+''.join(tables)+'</table>'
            '<h2>Reading the change</h2><p>The confirmed corrections recover more missed correct answers in the untuned model '
            'than in the final adapters. This reduces the measured GRPO gain and increases the measured SFT decline. '
            'It does not establish why either model changed its answers.</p>'
            '<p>These revised totals are conditional on confirmed cases only. They are not exhaustive human-scored accuracy. '
            'The original sampled-reliability analysis is not rescored here.</p>'
            '<h2>Counts, cases and limitations</h2><p><a href="results.json">Structured results</a> · '
            '<a href="report.md">Markdown report</a></p><pre>'+html.escape(markdown(result))+'</pre></html>')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['assistance','audit-directory','transitions','output']:
        p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--confirmation',type=Path,action='append',default=[])
    args=p.parse_args()
    assistance=json.loads(args.assistance.read_text())
    audit=args.audit_directory
    packet_seal=json.loads((audit/'packet.complete.json').read_text())
    transition_seal=json.loads((args.transitions/'complete.json').read_text())
    for base,seal in [(audit,packet_seal),(args.transitions,transition_seal)]:
        for name,digest in seal['files'].items():
            file=(base/name).resolve()
            if not file.is_relative_to(base.resolve()) or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:
                raise ValueError('sealed source changed: '+name)
    confirmation_docs=[]
    source_digest=hashlib.sha256(args.assistance.read_bytes()).hexdigest()
    for path in args.confirmation:
        doc=json.loads(path.read_text())
        if doc['packet_sha256']!=assistance['packet_sha256'] or doc['assistance_results_sha256']!=source_digest:
            raise ValueError('confirmation source identity changed')
        confirmation_docs.append(doc)
    mapping_path=audit/'private/mapping.json'
    pairs_path=args.transitions/'results/paired-items.jsonl'
    pairs=[json.loads(line) for line in pairs_path.read_text().splitlines()]
    result=summarize(pairs,json.loads(mapping_path.read_text()),assistance,confirmation_docs)
    sources=[args.assistance,mapping_path,pairs_path,audit/'packet.complete.json',args.transitions/'complete.json',
             Path(__file__),*args.confirmation]
    config=dict(mode='confirmed-case-extraction-overlay',sources={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    outputs={'results.json':json.dumps(result,indent=2)+'\n','config.json':json.dumps(config,indent=2)+'\n',
             'report.md':markdown(result),'report.html':render(result)}
    args.output.mkdir(parents=True,exist_ok=True)
    for name,text in outputs.items():
        f=args.output/name
        if f.exists() and f.read_text()!=text:raise ValueError('report already differs; use a new output name')
    for name,text in outputs.items():(args.output/name).write_text(text)
    (args.output/'status.json').write_text(json.dumps(dict(state='partial-confirmed-case-analysis-completed',
        updated_at=datetime.now(timezone.utc).isoformat(),files={name:hashlib.sha256((args.output/name).read_bytes()).hexdigest() for name in outputs}),indent=2)+'\n')
    print(json.dumps(result['methods'],indent=2));print(args.output/'report.html')


if __name__=='__main__':main()
