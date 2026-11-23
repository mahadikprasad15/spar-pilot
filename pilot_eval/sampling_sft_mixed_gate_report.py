"""CPU gate calculations and inspectable scientific figures; no model imports."""
import csv
import html
import itertools
import json
import math
import os
import re
import unicodedata

from pilot_eval.sampling_sft_gate_report import words,leakage,rate,cohort,render
from pilot_eval import sampling_sft_mixed_review as review
from pilot_eval import sampling_sft_mixed_generation as generation
from pilot_eval import sampling_sft_gate_math as stats
from pilot_eval import sampling_sft_mixed_gate as gate
from pilot_eval.sampling_sft_sampler import clean_gold
from pilot_eval.scoring import score_gsm8k
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash








def collection_views(root,config,coverage):
    _,gen,folder=generation.load(root/config['generation_plan'],root)
    output={}
    for k in generation.ARMS:
        attempts=coverage['arms'][str(k)]['attempts'];initial=[a for a in attempts if a['retry_index']==0]
        final=coverage['arms'][str(k)]
        decisions=[]
        for path in sorted((generation.arm_directory(root,gen,k)/'rounds').glob('*/blocks/*/scientific.jsonl')):
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


