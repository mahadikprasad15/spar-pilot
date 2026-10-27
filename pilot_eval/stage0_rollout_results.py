"""CPU publication and reconstruction of complete R0 cohorts."""
import csv
import html
import json
from pathlib import Path

from pilot_eval.run import _write_json
from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_workflow import _state
from pilot_eval.stage0_evaluation import _jsonl
from pilot_eval.stage0_rollout_math import cohorts
from pilot_eval.stage0_rollout_plan import load
from pilot_eval.stage0_rollout_run import _execution,phase_saved
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash


def flat(row):
    return dict(problem_id=row['group_id'],sample_idx=row['draw_index'],source=row['origin'],seed=row['seed'],
        prompt_sha256=row['prompt_sha256'],text=row['text'],token_ids=row['token_ids'],generated_token_ids=row['generated_token_ids'],
        n_tokens=row['token_count'],truncated=row['capped'],stop_reason=row['stop_reason'],
        correct_v3=row['scores']['flexible_v3']['correct'],correct_v2=row['scores']['flexible_v2']['correct'],
        correct_strict=row['scores']['strict']['correct'],extracted_v3=row['scores']['flexible_v3']['extracted'],
        gold_answer=row['gold'],historical_training_reward=row['historical_training_reward'])


def _expected(config,execution,saved,training,historical):
    new=[r for group in saved for r in group]
    rows=historical+new if execution['route']=='reuse-historical-128' else new
    order={identifier:n for n,identifier in enumerate(config['training_order'])}
    rows=sorted(rows,key=lambda r:(order[r['group_id']],r['draw_index']))
    if len(rows)!=4096 or len({r['draw_id'] for r in rows})!=4096:raise ValueError('Expected unique complete 512x8 identities.')
    data=cohorts(rows,config['training_order'],selection_seed=config['selection_seed'])
    if data['counts']['questions']!=512:raise ValueError('Uniform target cohort is incomplete.')
    summary=dict(schema_version=1,measurement_complete=True,real_gpu_measurement=config['execution_kind']=='real-cuda',
        route=execution['route'],counts=data['counts'],historical_joined=execution['route']=='reuse-historical-128',
        historical_groups=128,production_groups=len(execution['units']),group_size=8,
        scoring=['gsm8k-strict','gsm8k-flexible-v2','gsm8k-flexible-v3'],training_reward_unchanged=True,
        historical_reward_rate=sum(r['reward_v3'] for r in historical)/len(historical),
        current_reward_rate=sum(r['reward_v3'] for r in rows)/len(rows),cap_count=sum(r['capped'] for r in rows),
        origins={o:sum(r['origin']==o for r in rows) for o in sorted({r['origin'] for r in rows})},
        rates_are_descriptive=True,source_runtime=execution['runtime'],selection_rule=data['selection_rule'],
        note='One frozen uniform target cohort. Historical reward-rate similarity never authorizes a join. Zero of eight is not zero true correctness probability.')
    return rows,data,summary


def finish(config,base,execution,saved,training,historical,root):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows,data,summary=_expected(config,execution,saved,training,historical)
    _jsonl(base/'results/rollouts.jsonl',rows);_jsonl(base/'results/gold.jsonl',training)
    _jsonl(base/'results/historical/rollouts.jsonl',historical)
    _write_json(base/'results/historical/datasets.json',cohorts(historical,config['training_order'],selection_seed=config['selection_seed']))
    _write_json(base/'results/datasets.json',data);_write_json(base/'results/results.json',summary)
    pq.write_table(pa.Table.from_pylist([flat(r) for r in rows]),base/'results/rollouts.parquet')
    _write_json(base/'results/table-format.json',dict(pyarrow=pa.__version__,rows=4096,format='flat scalar columns plus token lists'))
    with (base/'results/groups.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['problem_id','correct_count','p_correct','neg_log_p_correct']);writer.writeheader()
        for identifier,p in data['p_correct'].items():writer.writerow(dict(problem_id=identifier,correct_count=round(8*p),p_correct=p,neg_log_p_correct=data['neg_log_p_correct'][identifier]))
    esc=lambda v:html.escape(str(v))
    cells=''.join(f'<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>' for k,v in summary.items() if not isinstance(v,dict))
    notice='<aside>Synthetic software fixture; not research findings or GPU qualification.</aside>' if config['execution_kind']=='synthetic-cpu-fixture' else ''
    (base/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage 0 R0 cohorts</title><style>body{max-width:1000px;margin:40px auto;padding:24px;background:#f7f8f5;color:#263e44;font:16px/1.6 system-ui}h1{font:38px Georgia}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px;border-bottom:1px solid #ddd}aside{padding:16px;background:#fff0d8}</style><h1>Untuned rollouts and derived cohorts</h1>'+notice+'<p>Route: '+esc(summary['route'])+'. The target contains 512 questions × eight draws. Historical groups remain separately accessible.</p><table>'+cells+'</table><p>gold: all 512 saved gold answers. own_all: all 4096 target draws. own_correct: flexible-v3 correct uncapped draws. own_correct_one: one uniform seed-42 selection per eligible question, in saved order. Zero-success questions remain in own_all and are listed as dropped from the correct subsets.</p><p>Reward rates are descriptive. This stage does not measure NLL, entropy or gradients; those are later supporting tickets.</p><p><a href="results/rollouts.parquet">Raw flat Parquet</a> · <a href="results/rollouts.jsonl">Full JSONL scores and tokens</a> · <a href="results/datasets.json">Memberships</a> · <a href="results/groups.csv">Per-question counts</a> · <a href="results/historical/rollouts.jsonl">Historical archive</a> · <a href="config.json">Frozen settings</a></p></html>')
    files=['config.json','report.html','inputs/training.jsonl','inputs/historical.jsonl','collection/meta/final-integrity.json']
    files += [str(p.relative_to(base)) for p in (base/'results').rglob('*') if p.is_file()]
    files += [str(p.relative_to(base)) for p in (base/'collection/units').rglob('complete.json')]
    marker=dict(config_sha256=_hash(config),execution_sha256=_hash(execution),files={p:file_hash(base/p) for p in files})
    _write_json(base/'collection/complete.json',marker);_write_json(base/'complete.json',marker)
    _state(base,'complete',512,512);return summary


def verify(path,output_root):
    import pyarrow.parquet as pq
    root=Path(output_root).resolve();config,base,training,historical=load(path,root);by_id={r['id']:r for r in training}
    execution=_execution(config,base,Path(path).resolve().parent/'stage0.rollouts.execution.json',root,by_id,historical)
    marker=json.loads((base/'complete.json').read_text())
    if marker['config_sha256']!=_hash(config) or marker['execution_sha256']!=_hash(execution):raise ValueError('R0 completion identity changed.')
    if marker!=json.loads((base/'collection/complete.json').read_text()):raise ValueError('R0 collection completion receipt differs.')
    for p,digest in marker['files'].items():
        if file_hash(inside(base,p))!=digest:raise ValueError('R0 completed payload changed: '+p)
    if json.loads((base/'collection/meta/final-integrity.json').read_text()).get('base_unchanged') is not True:raise ValueError('R0 lacks successful final integrity.')
    saved=phase_saved(config,base,'collection',execution['units'],execution['runtime'],by_id)
    if any(r is None for r in saved):raise ValueError('Missing collected group.')
    rows,data,summary=_expected(config,execution,saved,training,historical)
    if (rows!=[json.loads(x) for x in (base/'results/rollouts.jsonl').read_text().splitlines()]
            or data!=json.loads((base/'results/datasets.json').read_text()) or summary!=json.loads((base/'results/results.json').read_text())
            or pq.read_table(base/'results/rollouts.parquet').to_pylist()!=[flat(r) for r in rows]):
        raise ValueError('R0 memberships, Parquet or summary disagree with sealed groups.')
    return summary
