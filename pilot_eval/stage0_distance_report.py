"""Token-weighted dataset estimates with saved whole-problem bootstrap draws."""
import csv
import html
import json
from pathlib import Path

import numpy as np

from pilot_eval.stage0_distance import load,read_unit,verify_measurement
from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_workflow import _writer,_state
from pilot_eval.run import _write_json
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash


def calculate(config,directory,answers,memberships):
    arrays={a['answer_id']:read_unit(config,directory,a) for a in answers}
    datasets=json.loads((directory/'inputs/datasets.json').read_text());summaries=[];problems=[];windows=[]
    folder=directory/'bootstrap';folder.mkdir(parents=True,exist_ok=True)
    for name,rows in memberships.items():
        groups={}
        for member in rows:
            a=arrays[member['answer_id']];key=member['problem_id'];g=groups.setdefault(key,np.zeros(4))
            g += [len(a['surprise']),a['surprise'].sum(),a['entropy'].sum(),1]
        ids=sorted(groups)
        if not ids:
            summaries.append(dict(dataset=name,n_answers=0,n_tokens=0,nll=None,entropy=None,excess_surprise=None,ci_low=None,ci_high=None,excess_surprise_ci_low=None,excess_surprise_ci_high=None,neg_log_p_correct=None,state='no-eligible-problems'));continue
        sums=np.array([groups[i] for i in ids]);rng=np.random.default_rng(42);indices=rng.integers(0,len(ids),size=(10000,len(ids)))
        target=folder/(name+'.npz')
        if target.exists():
            with np.load(target,allow_pickle=False) as old:
                if not np.array_equal(old['indices'],indices):raise ValueError('Saved bootstrap indices differ.')
        else:np.savez_compressed(target,indices=indices)
        total=sums.sum(0);samples=sums[indices].sum(1)
        nll=total[1]/total[0];entropy=total[2]/total[0]
        bounds=np.percentile(samples[:,1]/samples[:,0],[2.5,97.5]);ebounds=np.percentile((samples[:,1]-samples[:,2])/samples[:,0],[2.5,97.5])
        neg=[datasets['neg_log_p_correct'][i] for i in ids if datasets['neg_log_p_correct'][i] is not None]
        summaries.append(dict(dataset=name,n_answers=int(total[3]),n_tokens=int(total[0]),nll=float(nll),entropy=float(entropy),
            excess_surprise=float(nll-entropy),ci_low=float(bounds[0]),ci_high=float(bounds[1]),
            excess_surprise_ci_low=float(ebounds[0]),excess_surprise_ci_high=float(ebounds[1]),
            neg_log_p_correct=float(np.mean(neg)) if name in ['own_correct','own_correct_one'] and neg else None,state='measured'))
        for i,g in zip(ids,sums):problems.append(dict(dataset=name,problem_id=i,n_answers=int(g[3]),n_tokens=int(g[0]),nll=float(g[1]/g[0]),entropy=float(g[2]/g[0]),excess_surprise=float((g[1]-g[2])/g[0]),neg_log_p_correct=datasets['neg_log_p_correct'][i] if name.startswith('own_correct') else None))
    gold={a['answer_id']:a for a in answers};top=[]
    for member in memberships['gold']:
        answer=gold[member['answer_id']];a=arrays[answer['answer_id']]
        for p,(t,s,e,r) in enumerate(zip(answer['target_ids'],a['surprise'],a['entropy'],a['rank'])):
            top.append(dict(problem_id=member['problem_id'],answer_id=answer['answer_id'],answer_position=p,context_position=answer['positions'][p],token_id=t,surprise=float(s),entropy=float(e),rank=int(r)))
    top=sorted(top,key=lambda r:(-r['surprise'],r['problem_id'],r['answer_position']))[:50]
    own=next(r for r in summaries if r['dataset']=='own_all');gate=abs(own['excess_surprise'])<=.02
    return dict(schema_version=1,analysis_complete=True,real_gpu_measurement=config['execution_kind']=='real-cuda',
        summaries=summaries,per_problem=problems,top_gold_tokens=top,statistics=config['statistics'],
        own_all_gate=dict(threshold=.02,observed=own['excess_surprise'],passed=gate,state='passed' if gate else 'failed',
            premise=config['premise'],note='Finite-cohort descriptive gate; no automatic tolerance change or distance=KL claim.'),
        limitations=['One sampled cohort and one seed; intervals resample whole problems and are pointwise.',
            'Primary means are token-weighted ratios; duplicates retain their cohort multiplicity despite one scoring forward.',
            'Excess surprise is descriptive, not KL. Whole-answer negative-log correctness is not comparable to per-token NLL.',
            'A finite stopped-sequence token-weighted ratio is not an exact zero-expectation identity.'],
        membership_counts={k:len(v) for k,v in memberships.items()})


def _table(directory,name,rows):
    if not rows:return
    with (directory/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def report(path,output_root):
    root=Path(output_root).resolve();config,directory,answers,memberships=load(path,root);verify_measurement(path,root)
    with _writer(directory):
        if (directory/'complete.json').exists():return verify(path,root)
        result=calculate(config,directory,answers,memberships)
        _write_json(directory/'results/results.json',result)
        _table(directory,'results/summary.csv',result['summaries']);_table(directory,'results/per-problem.csv',result['per_problem']);_table(directory,'results/top-gold-tokens.csv',result['top_gold_tokens'])
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(result['summaries']),directory/'summary.parquet')
        esc=lambda x:html.escape(str(x));rows=''.join('<tr>'+''.join('<td>'+esc(r[k])+'</td>' for k in ['dataset','n_answers','n_tokens','nll','entropy','excess_surprise','ci_low','ci_high'])+'</tr>' for r in result['summaries'])
        notice='<aside>Synthetic software fixture, not scientific results.</aside>' if config['execution_kind']=='synthetic-cpu-fixture' else ''
        (directory/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Answer NLL and excess surprise</title><style>body{max-width:1100px;margin:40px auto;padding:24px;background:#f7f8f5;color:#263e44;font:15px/1.6 system-ui}h1{font:36px Georgia}table{border-collapse:collapse}th,td{padding:8px;border-bottom:1px solid #ddd}aside{background:#fff0d8;padding:16px}</style><h1>Untuned answer likelihood</h1>'+notice+'<p>Each unique prompt/answer token sequence is scored once. Saved memberships restore sample multiplicity. Prompt tokens are masked; observed EOS is supervised and capped sequences receive no added EOS.</p><table><tr>'+''.join('<th>'+s+'</th>' for s in ['dataset','answers','tokens','NLL','entropy','excess surprise','NLL CI low','NLL CI high'])+'</tr>'+rows+'</table><h2>Own-all check</h2><pre>'+esc(json.dumps(result['own_all_gate'],indent=2))+'</pre><p>NLL is primary. Excess surprise is not KL. Intervals resample whole problems; negative-log(k/8) is separately reported in whole-answer units.</p><ul>'+''.join('<li>'+esc(s)+'</li>' for s in result['limitations'])+'</ul><p><a href="summary.parquet">Summary Parquet</a> · <a href="results/per-problem.csv">Per-problem values</a> · <a href="results/top-gold-tokens.csv">Top50 surprising gold tokens</a> · <a href="results/results.json">Structured results</a> · <a href="config.json">Frozen configuration</a></p></html>')
        files=['report.html','summary.parquet','measurement.complete.json']+[str(p.relative_to(directory)) for sub in ['results','bootstrap'] for p in (directory/sub).rglob('*') if p.is_file()]
        _write_json(directory/'complete.json',dict(config_sha256=_hash(config),files={p:file_hash(directory/p) for p in files}))
        _state(directory,'complete' if result['own_all_gate']['passed'] else 'failed-gate',len(answers),len(answers),own_all_gate=result['own_all_gate']['passed'])
        if not result['own_all_gate']['passed']:raise ValueError('Own_all excess-surprise gate failed; retained results require review. No tolerance was changed.')
        return result


def verify(path,output_root):
    root=Path(output_root).resolve();config,directory,answers,memberships=load(path,root);verify_measurement(path,root)
    seal=json.loads((directory/'complete.json').read_text())
    if seal['config_sha256']!=_hash(config):raise ValueError('Distance report identity changed.')
    for p,digest in seal['files'].items():
        if file_hash(inside(directory,p))!=digest:raise ValueError('Distance report evidence changed.')
    result=json.loads((directory/'results/results.json').read_text())
    if result!=calculate(config,directory,answers,memberships):raise ValueError('Distance statistics do not reconstruct from saved token arrays.')
    import pyarrow.parquet as pq
    if pq.read_table(directory/'summary.parquet').to_pylist()!=result['summaries']:raise ValueError('Distance summary Parquet differs.')
    return result
