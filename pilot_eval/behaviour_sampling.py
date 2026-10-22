"""Observed reliability of eight saved GRPO draws; no generation or model imports."""
import io
import json
from pathlib import Path

import numpy as np

from pilot_eval.behaviour_breakdown import verify_behaviour, _jsonl, _normalize, _scored
from pilot_eval.behaviour_transitions import _bootstrap, _difference, _unit
from pilot_eval.recovery import _inside, _name
from pilot_eval.rescoring import _save_bytes
from pilot_eval.run import _write_state
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _save_frozen


SCORERS = ('flexible_v3', 'last_number')
LIMITS = ('Observed eight-draw coverage is not newly acquired ability. Intervals resample '
          'questions retaining all eight observed draws, conditional on this trained run '
          'and these draws; they do not estimate training-seed or fresh-decoding variation. '
          'Flat coverage with increased mean reward is compatible with sharpening, not proof of it.')


def _source(parent, root, index, greedy):
    if type(index) is not int or not 0 <= index < len(parent['sources']):
        raise ValueError('sampled-source-index does not identify a frozen source')
    source = parent['sources'][index]
    if source['method'] != 'grpo' or source['format'] != 'grpo-behaviour':
        raise ValueError('sampled source must be an explicitly selected native GRPO evaluation')
    files = {f['role']:_inside(root, parent['run_path']) / f['snapshot'] for f in source['files']}
    cfg = json.loads(files['config'].read_text())
    selected = [p for p in cfg.get('policies', []) if p.get('mode') == 'sampled' and p.get('step') in [0,64]]
    if len(selected) != 2 or {p['step'] for p in selected} != {0,64}:
        raise ValueError('one saved sampled policy per endpoint is required')
    fields = ['do_sample','draws','temperature','top_p','top_k','num_beams','completion_limit','use_cache','batch_size']
    policies = {p['step']:{k:p.get(k) for k in fields} for p in selected}
    if policies[0] != policies[64]:
        raise ValueError('sampled endpoint decoding contracts differ')
    policy = policies[0]
    if (policy['do_sample'] is not True or policy['draws'] != 8 or
        not isinstance(policy['temperature'], (float,int)) or policy['temperature'] <= 0 or
        not isinstance(policy['top_p'], (float,int)) or not 0 < policy['top_p'] <= 1 or
        type(policy['top_k']) is not int or policy['top_k'] < 0 or
        policy['num_beams'] != 1 or type(policy['completion_limit']) is not int or policy['completion_limit'] <= 0):
        raise ValueError('complete eight-draw sampling contract required')
    expected = sorted(i['id'] for i in cfg.get('items', []))
    reference = {r['item_id']:r for r in greedy if r['method']=='grpo' and r['step']==0}
    if not expected or len(set(expected))!=len(expected) or set(expected)!=set(reference):
        raise ValueError('sampled evaluation cohort differs from verified greedy cohort')
    if any(not i.startswith('gsm8k:test:') for i in expected):
        raise ValueError('sampled coverage requires held-out GSM8K test questions, not training draws')
    rows, seen, identities = [], set(), set()
    for line, raw in enumerate(_jsonl(files['responses'])):
        if raw.get('mode')!='sampled' or raw.get('step') not in [0,64]:
            continue
        draw = raw.get('draw_index')
        if type(draw) is not int or draw not in range(8):
            raise ValueError('sampled draw index must be 0 through 7')
        row = _normalize(raw, 'grpo', raw['step'], index, line)
        item = row['item_id']; key = (raw['step'], item, draw)
        if item not in reference or (row['prompt'],row['gold']) != (reference[item]['prompt'],reference[item]['gold']):
            raise ValueError('sampled question/prompt/gold identity mismatch')
        if key in seen:
            raise ValueError('duplicate sampled question/draw identity')
        seen.add(key)
        if 'draw_id' in raw and raw['draw_id'] != f'{item}:draw:{draw}':
            raise ValueError('sampled draw alias mismatch')
        if 'record_id' in raw:
            identity = raw['record_id']
            if identity in identities:
                raise ValueError('duplicate sampled record ID')
            identities.add(identity)
        row['draw_index'] = draw
        rows.append(row)
    required = {(s,i,d) for s in [0,64] for i in expected for d in range(8)}
    if seen != required:
        raise ValueError('fixed-eight comparison requires all eight distinct draws per item at both endpoints')
    return cfg, policy, expected, rows


def _metric(rows, ids, scorer, directory):
    counts = {s:{i:0 for i in ids} for s in [0,64]}
    for row in rows:
        counts[row['step']][row['item_id']] += int(row['scores'][scorer]['correct'])
    before = np.asarray([counts[0][i] for i in ids])
    after = np.asarray([counts[64][i] for i in ids])
    indices, relative = _bootstrap(directory, ids)
    table = [[0]*9 for _ in range(9)]
    for a,b in zip(before,after):
        table[int(a)][int(b)] += 1
    both = [i for i in ids if counts[0][i]>0 and counts[64][i]>0]
    return dict(per_item=[dict(item_id=i, correct_0=counts[0][i], correct_64=counts[64][i]) for i in ids],
        mean_draw_reward=dict(before=float(before.mean()/8), after=float(after.mean()/8),
            difference=_difference(before/8, after/8, indices)),
        observed_coverage=dict(before=float((before>0).mean()),after=float((after>0).mean()),
            difference=_difference(before>0,after>0,indices)),
        coverage_gain_ids=[i for i in ids if counts[0][i]==0 and counts[64][i]>0],
        coverage_loss_ids=[i for i in ids if counts[0][i]>0 and counts[64][i]==0],
        already_observed_solved=dict(item_ids=both,n=len(both),
            mean_count_change=float(np.mean([counts[64][i]-counts[0][i] for i in both])) if both else None),
        count_transition=table,resampling_path=relative)


def _plot(result, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'svg.hashsalt':'pilot4-observed-eight-v1','font.size':10}):
        fig, axes = plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
        common_max=max(1,max(max(row) for scorer in SCORERS for row in result['metrics'][scorer]['count_transition']))
        for ax, scorer in zip(axes, SCORERS):
            table=result['metrics'][scorer]['count_transition']
            image=ax.imshow(table,cmap='Blues',vmin=0,vmax=common_max)
            for a in range(9):
                for b in range(9):
                    if table[a][b]:
                        ax.text(b,a,str(table[a][b]),ha='center',va='center',
                                color='white' if table[a][b]>image.norm.vmax/2 else '#23373f')
            ax.set(xticks=range(9),yticks=range(9),xlabel='Correct draws at step 64 (of 8)',
                   ylabel='Correct draws at step 0 (of 8)',title=scorer)
        fig.suptitle('Observed sampled count transitions · number of held-out questions')
        buffer=io.BytesIO();fig.savefig(buffer,format='svg',metadata={'Date':None});plt.close(fig)
    _save_bytes(directory/'plots/observed-coverage.svg',buffer.getvalue())


def _markdown(result):
    lines=['# GRPO observed eight-draw reliability','',LIMITS,'',
        f"{result['n']} questions × eight draws × two endpoints = {result['response_count']} saved responses.",'',
        '| Scorer | Metric | Step 0 | Step 64 | Paired difference, 95% interval |',
        '|---|---|---:|---:|---|']
    for scorer, metric in result['metrics'].items():
        for name in ['mean_draw_reward','observed_coverage']:
            row=metric[name]; delta=row['difference']
            lines.append(f"| {scorer} | {name} | {row['before']:.4f} | {row['after']:.4f} | {delta['estimate']:+.4f}, {delta['interval_95']} |")
        lines += ['',f"## {scorer}: coverage changes",'',
            f"Observed coverage gains: {len(metric['coverage_gain_ids'])}; losses: {len(metric['coverage_loss_ids'])}.",
            f"Solved at both endpoints: {metric['already_observed_solved']['n']}; mean correct-draw count change: {metric['already_observed_solved']['mean_count_change']}.",'',
            '| Correct draws 0 → 64 | '+' | '.join(map(str,range(9)))+' |',
            '|---|'+'---:|'*9]
        lines += ['| '+str(i)+' | '+' | '.join(map(str,row))+' |' for i,row in enumerate(metric['count_transition'])]
    lines += ['', 'No extra untuned sampling run was pooled. Last-number results remain extraction-dependent.',
              '', '![Observed draw-count transitions](../plots/observed-coverage.svg)','']
    return '\n'.join(lines).encode()


def _verify(directory):
    seal=json.loads((directory/'complete.json').read_text())
    required={'results/results.json','results/per-item.jsonl','results/responses.jsonl',
              'results/report.md','plots/observed-coverage.svg'}
    if seal.get('config_sha256')!=file_hash(directory/'config.json') or not required.issubset(seal.get('files',{})):
        raise ValueError('sampled report inventory/config changed')
    for relative,digest in seal['files'].items():
        if file_hash(_inside(directory,relative))!=digest:
            raise ValueError('sampled report output changed')
    return json.loads((directory/'results/results.json').read_text())


def report_sampling(path, root, name, source_index, *, max_new_units=None):
    if max_new_units is not None and max_new_units<1:
        raise ValueError('max-new-units must be positive')
    root=Path(root).resolve();path=Path(path).resolve()
    verify_behaviour(path,root)
    parent=json.loads(path.read_text());parent_dir=_inside(root,parent['run_path'])
    greedy=_jsonl(parent_dir/'results/responses.jsonl')
    cfg,policy,ids,rows=_source(parent,root,source_index,greedy)
    directory=root/'reports'/_name(name)
    from pilot_eval import behaviour_breakdown, behaviour_transitions
    config=dict(protocol='pilot4-observed-eight-v1',parent_path=str(path.relative_to(root)),
        parent_sha256=file_hash(path),parent_seal_sha256=file_hash(parent_dir/'complete.json'),
        sampled_source_index=source_index,policy=policy,item_ids=ids,
        source_metadata={k:cfg.get(k) for k in ['model','model_revision','tokenizer_revision','runtime','prompt_template']},
        implementation={p.name:file_hash(p) for p in [Path(__file__),Path(behaviour_breakdown.__file__),Path(behaviour_transitions.__file__)]},
        bootstrap=dict(draws=10000,seed=42,percentiles=[2.5,97.5],unit='question with all eight observed draws'),
        numpy_version=np.__version__)
    with run_lock(directory):
        _save_frozen(directory/'config.json',config)
        _save_frozen(directory/'meta/run_manifest.json',dict(stage='observed-eight-draws',**config))
        if (directory/'complete.json').exists():
            result=_verify(directory);_write_state(directory,'completed',4,4);return result
        new_units=0;scored=[]
        for step in [0,64]:
            unit_name=f'sampled-{step}'
            is_new=not (directory/'units'/(unit_name+'.complete.json')).exists()
            if is_new and max_new_units is not None and new_units>=max_new_units:
                _write_state(directory,'incomplete',len(scored)//(8*len(ids)),4)
                return dict(state='incomplete')
            scored.extend(_unit(directory,unit_name,lambda step=step:[_scored(r) for r in rows if r['step']==step]))
            new_units+=int(is_new)
            _write_state(directory,'running',2 if step==64 else 1,4)
            print(f'Sampled scoring: step {step}; {len(scored)}/{len(rows)} responses',flush=True)
        metrics={}
        for scorer in SCORERS:
            unit_name='metric-'+scorer
            is_new=not (directory/'units'/(unit_name+'.complete.json')).exists()
            if is_new and max_new_units is not None and new_units>=max_new_units:
                _write_state(directory,'incomplete',2+len(metrics),4)
                return dict(state='incomplete')
            metrics[scorer]=_unit(directory,unit_name,lambda scorer=scorer:_metric(scored,ids,scorer,directory))
            new_units+=int(is_new)
        result=dict(n=len(ids),response_count=len(rows),draws_per_question=8,metrics=metrics,
            parent_identity=config,bootstrap=config['bootstrap'],limitations=LIMITS,
            evidence_kind=parent['source_evidence_kind'],execution_kind='cpu-saved-output-only')
        _save_frozen(directory/'results/results.json',result)
        per_item=[dict(item_id=i,counts={s:next(r for r in metrics[s]['per_item'] if r['item_id']==i) for s in SCORERS}) for i in ids]
        _save_bytes(directory/'results/per-item.jsonl',''.join(json.dumps(r,sort_keys=True)+'\n' for r in per_item).encode())
        _save_bytes(directory/'results/responses.jsonl',''.join(json.dumps(r,sort_keys=True)+'\n' for r in scored).encode())
        _plot(result,directory);_save_bytes(directory/'results/report.md',_markdown(result))
        files={str(p.relative_to(directory)):file_hash(p) for folder in ['units','results','plots','bootstrap']
               for p in (directory/folder).glob('*') if p.is_file()}
        _save_frozen(directory/'complete.json',dict(config_sha256=file_hash(directory/'config.json'),files=files))
        _write_state(directory,'completed',4,4)
        return result
