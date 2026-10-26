"""CPU reconstruction of raw greedy responses, scores and descriptive summaries."""
import statistics

from pilot_eval.scoring import score_gsm8k, score_gsm8k_flexible_v3
from pilot_eval.workflow import _hash

OUTPUT_FIELDS=('text','token_ids','generated_token_ids','stop_reason')


def records(config,unit,rows,outputs,*,runtime_sha256,input_sha256):
    """Validate termination before scoring; gold never chooses an extraction."""
    if len(outputs)!=len(rows) or [r['id'] for r in rows]!=unit['ids']:
        raise ValueError('Generated output count/order differs from the frozen batch.')
    limit=config['decoding']['max_new_tokens'];eos=config['decoding']['eos_token_id']
    result=[]
    for row,output in zip(rows,outputs):
        raw=output.get('generated_token_ids');content=output.get('token_ids')
        if (not isinstance(raw,list) or not raw or not isinstance(content,list)
                or any(type(t) is not int or t<0 for t in raw+content)
                or len(raw)>limit or not isinstance(output.get('text'),str)):
            raise ValueError('Invalid generated token/text metadata.')
        reason=output.get('stop_reason')
        if reason=='eos':
            if raw[-1]!=eos or eos in raw[:-1] or content!=raw[:-1]:
                raise ValueError('EOS/content token metadata disagree.')
        elif reason=='cap':
            if len(raw)!=limit or eos in raw or content!=raw:
                raise ValueError('Cap/content token metadata disagree.')
        else:
            raise ValueError('Greedy continuation must terminate at EOS or the frozen cap.')
        capped=reason=='cap'
        strict=dict(score_gsm8k(output['text'],row['gold'],capped)['strict'],scorer_version='gsm8k-strict')
        flexible=score_gsm8k_flexible_v3(output['text'],row['gold'],capped)
        if strict['correct'] and not flexible['correct']:
            raise ValueError('Strict-correct/flexible-incorrect scorer invariant failed.')
        result.append(dict(record_id=f'{unit["variant"]}:{row["id"]}',variant=unit['variant'],
            item_id=row['id'],unit_key=unit['key'],seed=unit.get('item_seeds',{}).get(row['id'],unit['seed']),
            prompt=row['prompt'],prompt_sha256=_hash(row['prompt']),gold=row['gold'],
            **{k:output[k] for k in OUTPUT_FIELDS},token_count=len(content),
            generated_token_count=len(raw),capped=capped,end_token_id=raw[-1] if not capped else None,
            scorer=config['scorer'],strict=strict,flexible=flexible,
            reward=0 if capped else int(flexible['correct']),runtime_sha256=runtime_sha256,
            input_sha256=input_sha256,adapter_sha256=config.get('adapter_files',{}).get(unit['variant'],{}).get('adapter_model.safetensors'),
            human_review_applied=False))
    return result


def summarize(config,rows):
    result={}
    for variant,definition in config['variants'].items():
        selected=[r for r in rows if r['variant']==variant]
        if not selected:
            raise ValueError('Missing variant responses.')
        n=len(selected);lengths=[r['token_count'] for r in selected]
        correct=sum(r['flexible']['correct'] for r in selected)
        strict=sum(r['strict']['correct'] for r in selected)
        caps=sum(r['capped'] for r in selected)
        result[variant]=dict(n=n,source_arm=definition['source_arm'],operation=definition['operation'],
            reference='V0a' if definition['source_arm']=='sft' else 'V0b',
            flexible_correct=correct,flexible_accuracy=correct/n,strict_correct=strict,strict_accuracy=strict/n,
            cap_count=caps,cap_rate=caps/n,flexible_invalid=sum(r['flexible']['status']=='invalid' for r in selected),
            strict_invalid=sum(r['strict']['status']=='invalid' for r in selected),
            total_content_tokens=sum(lengths),mean_content_tokens=statistics.mean(lengths),
            median_content_tokens=statistics.median(lengths))
    return dict(schema_version=1,mode='stage0-paired-greedy',measurement_complete=True,
        real_gpu_measurement=config['execution_kind']=='real-cuda',execution_kind=config['execution_kind'],
        response_count=len(rows),question_count=len(config['item_ids']),variants=result,
        scorer=config['scorer'],strict_scorer='gsm8k-strict',human_review_status='not-applied',
        interpretation='Raw automatic endpoint measurements. Paired intervals, targeted answer review and preservation claims belong to Ticket 07.')
