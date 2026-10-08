"""CPU-only clustered paired bootstrap from verified saved GRPO continuations."""
import json
from pathlib import Path
import numpy as np

from pilot_eval.grpo_evaluation import verify_evaluation
from pilot_eval.grpo_baseline import _percentile
from pilot_eval.grpo_prepare import _within
from pilot_eval.training import file_hash,run_lock
from pilot_eval.workflow import _save_frozen,_hash
from pilot_eval.run import _write_json,_write_state
from pilot_eval.sft import safe_name

NOTE=('Intervals condition on this evaluation cohort, observed draws and one training seed. '
      'Resampling whole items preserves within-item draw clustering; it does not isolate '
      'repeated-decoding uncertainty on this fixed cohort or variation across training seeds. '
      'Shared seeds across checkpoints do not imply matched sampled answers. '
      'Sampled accuracy is average reward, not pass-at-eight. Positive drop means final is worse.')


def _summary(rows):
    lengths=[r['token_count'] for r in rows];n=len(rows)
    return dict(total=n,strict_accuracy=sum(r['strict']['correct'] for r in rows)/n,
        flexible_accuracy=sum(r['reward'] for r in rows)/n,
        strict_invalid_rate=sum(r['strict']['status']=='invalid' for r in rows)/n,
        flexible_invalid_rate=sum(r['flexible']['status']=='invalid' for r in rows)/n,
        cap_rate=sum(r['capped'] for r in rows)/n,mean_tokens=sum(lengths)/n,
        length_percentiles={str(q):_percentile(lengths,q) for q in [0,25,50,75,90,95,99,100]})


def _interval(values,indices):
    distribution=values[indices].mean(axis=1)
    return [float(v) for v in np.percentile(distribution,[2.5,97.5])]


def report_grpo_behaviour(path,root,name):
    root=Path(root).resolve();path=_within(root,path);source,records=verify_evaluation(path,root)
    directory=root/'reports'/safe_name(name)
    config=dict(protocol_version='pilot4-paired-behaviour-v1',source_path=str(path.relative_to(root)),
        source_sha256=file_hash(path),source_seal_sha256=file_hash(root/source['run_path']/'complete.json'),
        bootstrap_resamples=10000,bootstrap_seed=42,method='paired item-cluster percentile bootstrap',
        margin=source['margin'],scorer=source['scorer'],implementation_sha256=file_hash(Path(__file__)),numpy_version=np.__version__)
    with run_lock(directory):
        _save_frozen(directory/'config.json',config)
        _save_frozen(directory/'meta/run_manifest.json',dict(source_path=config['source_path'],
            source_sha256=config['source_sha256'],report_path=str(directory.relative_to(root)),
            bootstrap_resamples=10000,bootstrap_seed=42))
        if (directory/'complete.json').exists():
            marker=json.loads((directory/'complete.json').read_text())
            required={'config.json','results/results.json','results/report.md','results/bootstrap-indices.json','results/paired-items.jsonl'}
            if marker['config_sha256']!=_hash(config) or not required.issubset(marker['files']): raise ValueError('paired report config/seal changed')
            for rel,digest in marker['files'].items():
                if file_hash(_within(root,directory/rel))!=digest: raise ValueError('paired report corrupted')
            return json.loads((directory/'results/results.json').read_text())
        _write_state(directory,'running',0,7)
        ids=[r['id'] for r in source['items']]
        # Exact saved item order maps bootstrap integers to original stable IDs.
        indices=np.random.default_rng(42).integers(0,150,size=(10000,150))
        _write_json(directory/'results/bootstrap-indices.json',indices.tolist())
        groups={}
        for row in records: groups.setdefault((row['mode'],row['step'],row['item_id']),[]).append(row)
        summaries={};changes=[];paired=[];drops={}
        for policy in source['policies']:
            mode,step=policy['mode'],policy['step'];draws=policy['draws']
            ordered=[];values=[];strict=[];lengths=[]
            for item_id in ids:
                rows=sorted(groups[(mode,step,item_id)],key=lambda r:r['draw_index'])
                if [r['draw_index'] for r in rows]!=list(range(draws)): raise ValueError('missing or duplicate paired draws')
                ordered.extend(rows);values.append(sum(r['reward'] for r in rows)/draws)
                strict.append(sum(r['strict']['correct'] for r in rows)/draws)
                lengths.append(sum(r['token_count'] for r in rows)/draws)
                paired.append(dict(item_id=item_id,mode=mode,step=step,draws=rows))
            summaries[f'{mode}-{step}']=_summary(ordered)
            groups[(mode,step,'values')]=np.asarray(values)
            groups[(mode,step,'strict')]=np.asarray(strict)
            groups[(mode,step,'lengths')]=np.asarray(lengths)
        for policy in source['policies']:
            mode,step=policy['mode'],policy['step'];change=dict(mode=mode,step=step)
            for metric in ['values','strict','lengths']:
                delta=groups[(mode,step,metric)]-groups[(mode,0,metric)]
                change[{'values':'flexible_change','strict':'strict_change','lengths':'mean_token_change'}[metric]]=dict(
                    estimate=float(delta.mean()),interval_95=_interval(delta,indices))
            changes.append(change)
        for mode in ['greedy','sampled']:
            delta=groups[(mode,0,'values')]-groups[(mode,64,'values')]
            interval=_interval(delta,indices);margin=source['margin']
            outcome=('non_inferiority_passed' if interval[1]<margin else
                     'demonstrated_harm_beyond_margin' if interval[0]>margin else 'inconclusive')
            drops[mode]=dict(drop=float(delta.mean()),interval_95=interval,margin=margin,outcome=outcome,
                evidence_of_positive_decline=interval[0]>0,gate='upper 95% interval < frozen margin')
        result=dict(summaries=summaries,changes=changes,drops=drops,bootstrap=dict(resamples=10000,seed=42,
            item_ids=ids,indices_file='bootstrap-indices.json'),note=NOTE,source=source['run_path'])
        _write_json(directory/'results/results.json',result)
        lines=['# GRPO paired behaviour','',NOTE,'','| Setting | Strict | Flexible v3 | Mean tokens | Median | Cap rate |',
               '|---|---:|---:|---:|---:|---:|']
        for key,row in summaries.items():
            lines.append(f'| {key} | {row["strict_accuracy"]:.3f} | {row["flexible_accuracy"]:.3f} | {row["mean_tokens"]:.2f} | {row["length_percentiles"]["50"]:.2f} | {row["cap_rate"]:.3f} |')
        lines+=['','## Checkpoint 0 minus checkpoint 64','']
        for mode,row in drops.items():
            lines.append(f'- {mode}: drop {row["drop"]:+.3f}, 95% interval {row["interval_95"]}, margin {row["margin"]}; **{row["outcome"]}**.')
        lines+=['','A failed non-inferiority gate alone does not demonstrate damage. See positive-decline evidence separately.',
                '', '## Paired trajectory (checkpoint minus zero)', '',
                '| Setting | Flexible change (95%) | Strict change (95%) | Mean-token change (95%) |',
                '|---|---|---|---|']
        for row in changes:
            cells=[f'{row[k]["estimate"]:+.3f} {row[k]["interval_95"]}' for k in ['flexible_change','strict_change','mean_token_change']]
            lines.append(f'| {row["mode"]}-{row["step"]} | '+ ' | '.join(cells)+' |')
        (directory/'results/report.md').write_text('\n'.join(lines)+'\n')
        (directory/'results/paired-items.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in paired))
        files={str(p.relative_to(directory)):file_hash(p) for p in [directory/'config.json',*sorted((directory/'results').iterdir())]}
        _write_json(directory/'complete.json',dict(config_sha256=_hash(config),files=files))
        _write_state(directory,'completed',7,7)
        return result
