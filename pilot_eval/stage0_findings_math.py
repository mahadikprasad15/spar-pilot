"""Paired document/item statistics; no model, labels or plotting dependencies."""
import math

import numpy as np

from pilot_eval.stage0_edits import VARIANTS

# Each tuple means edited/comparator minus reference. No random draw is pooled.
CONTRASTS = [
    dict(id=f'{v}-minus-{ref}', edited=v, reference=ref, family='edit-vs-own-reference')
    for v, (arm, operation) in VARIANTS.items() if operation != 'reference'
    for ref in ['V0a' if arm == 'sft' else 'V0b']
] + [
    dict(id=f'{v}-minus-{ref}', edited=v, reference=ref, family=family)
    for v, ref, family in [
        ('V0b', 'V0a', 'unedited-arm-comparison'),
        ('V1', 'V4', 'size-control'), ('V2', 'V4-input', 'size-control'),
        *[('V1', f'V3-{i}', 'random-subspace-control') for i in range(3)],
    ]
]
STATISTICS = dict(draws=10000, seed=42, percentiles=[2.5, 97.5],
    unit='paired item/document ID', generator='numpy.default_rng/PCG64',
    intervals='pointwise percentile; not simultaneous across the comparison family')
PREDICTIONS = [
    'SFT principal output/input removal may increase response length and lower fixed-context KL while preserving accuracy.',
    'Compare the principal output edit with every random-subspace draw and its per-module magnitude control; compare the input edit with its own magnitude control.',
    'The analogous GRPO output edit is expected to change less; report its own paired effects without inferring mechanism from a favorable individual result.',
]
LIMITS = [
    'Results condition on one-seed saved adapters and fixed cohorts; bootstrap intervals do not estimate training-seed variation.',
    'At n=150 the -5 percentage point non-inferiority criterion may be inconclusive.',
    'Intervals are pointwise, not simultaneous. Every declared contrast is shown; an unadjusted winner does not confirm the family or a causal mechanism.',
    'Generated length is an observed response property, not validated reasoning quality. Longer is not automatically restored untuned performance.',
    'KL uses supplied teacher-forced contexts; distributional change alone is not behavioral harm.',
    'Targeted intended-answer review is one-reviewer evidence, not whole-cohort human-corrected accuracy or reasoning validation.',
]


def noninferiority(interval):
    low, high = interval
    return 'preserved' if low > -.05 else ('loss' if high < -.05 else 'inconclusive')


def paired(reference, edited, indices, *, statistic='mean'):
    before, after = np.asarray(reference, dtype=float), np.asarray(edited, dtype=float)
    if before.ndim != 1 or before.shape != after.shape or not len(before) or not np.isfinite([before, after]).all():
        raise ValueError('Paired outcomes must have matching nonempty finite item vectors.')
    if (indices.shape != (10000, len(before)) or not np.issubdtype(indices.dtype, np.integer)
            or indices.min() < 0 or indices.max() >= len(before)):
        raise ValueError('Invalid saved bootstrap index mapping.')
    delta = after - before
    function = {'mean': np.mean, 'median': np.median}[statistic]
    draws = function(delta[indices], axis=1)
    return dict(estimate=float(function(delta)), interval_95=np.percentile(draws, [2.5, 97.5]).tolist())


def mcnemar(gains, losses):
    """Exact two-sided conditional binomial test, including no discordant pairs."""
    if any(type(v) is not int or v < 0 for v in [gains, losses]):
        raise ValueError('McNemar discordance counts must be nonnegative integers.')
    total = gains + losses
    return min(1., 2 * sum(math.comb(total, k) for k in range(min(gains, losses) + 1)) / 2 ** total)


def bin_(score):
    return 'C' if score['correct'] else ('W' if score['status'] == 'valid' else 'U')


def behaviour(reference, edited, ids, indices, *, overlay=None):
    if set(reference) != set(edited) or set(ids) != set(reference) or len(ids) != 150:
        raise ValueError('Paired behavior requires the same original 150 item identities.')
    result = dict(n=len(ids), item_ids=ids)
    for scorer in ['flexible', 'strict']:
        before = [bool(reference[i][scorer]['correct']) for i in ids]
        after = [bool(edited[i][scorer]['correct']) for i in ids]
        gains = sum(not b and a for b, a in zip(before, after))
        losses = sum(b and not a for b, a in zip(before, after))
        cells = {f'{a}:{b}': [] for a in 'CWU' for b in 'CWU'}
        for i in ids:
            cells[f'{bin_(reference[i][scorer])}:{bin_(edited[i][scorer])}'].append(i)
        difference = paired(before, after, indices)
        result[scorer] = dict(reference_correct=sum(before), edited_correct=sum(after),
            reference_accuracy=float(np.mean(before)), edited_accuracy=float(np.mean(after)),
            difference=difference, gains=gains, losses=losses, churn=gains+losses,
            mcnemar_exact_two_sided=mcnemar(gains, losses), transition_item_ids=cells,
            noninferiority=noninferiority(difference['interval_95']),
            reference_invalid=sum(reference[i][scorer]['status'] != 'valid' for i in ids),
            edited_invalid=sum(edited[i][scorer]['status'] != 'valid' for i in ids))
    before = [reference[i]['token_count'] for i in ids]
    after = [edited[i]['token_count'] for i in ids]
    result['length'] = dict(median_paired_change=paired(before, after, indices, statistic='median'),
        mean_paired_change=paired(before, after, indices), shorter_share=float(np.mean(np.asarray(after)<before)),
        reference_median=float(np.median(before)), edited_median=float(np.median(after)),
        reference_cap_count=sum(reference[i]['capped'] for i in ids), edited_cap_count=sum(edited[i]['capped'] for i in ids),
        definition='generated content-token length; median of per-item edited minus reference differences')
    if overlay is not None:
        before = [overlay.get(reference[i]['record_id'], int(reference[i]['flexible']['correct'])) for i in ids]
        after = [overlay.get(edited[i]['record_id'], int(edited[i]['flexible']['correct'])) for i in ids]
        result['targeted_overlay'] = dict(reference_correct=sum(before), edited_correct=sum(after),
            difference=paired(before, after, indices),
            reviewed_reference=sum(reference[i]['record_id'] in overlay for i in ids),
            reviewed_edited=sum(edited[i]['record_id'] in overlay for i in ids),
            scope='confirmed intended-answer labels where available; automatic flexible-v3 elsewhere; not whole-cohort human accuracy',
            preservation_claim=False)
    return result


def drift(reference, edited, ids, indices, modules):
    if set(reference) != set(edited) or set(ids) != set(reference):
        raise ValueError('KL documents must pair exactly by original corpus/view IDs.')
    if any(reference[i]['kl_count'] != edited[i]['kl_count'] or reference[i]['write_count'] != edited[i]['write_count'] for i in ids):
        raise ValueError('Paired KL/write counted-position denominators differ.')
    if any(not reference[i]['kl_count'] for i in ids):
        raise ValueError('KL comparison includes a document with no counted targets.')
    before = [reference[i]['kl_mean'] for i in ids]
    after = [edited[i]['kl_mean'] for i in ids]
    def ratio_rows(records):
        return np.asarray([records[i]['write_sums'] for i in ids], dtype=float)
    a, b = ratio_rows(reference), ratio_rows(edited)
    if a.shape != (len(ids), len(modules), 3) or b.shape != a.shape:
        raise ValueError('Document/module sufficient-sum mapping differs.')
    count = np.asarray([reference[i]['write_count'] for i in ids])
    changes = np.mean((b[:, :, 2]-a[:, :, 2])/count[:, None], axis=0)
    token_count = sum(reference[i]['kl_count'] for i in ids)
    return dict(n=len(ids), item_ids=ids, tokens=token_count,
        kl=dict(difference=paired(before, after, indices), reference_equal_document_mean=float(np.mean(before)),
            edited_equal_document_mean=float(np.mean(after)),
            reference_token_mean=sum(reference[i]['kl_sum'] for i in ids)/token_count,
            edited_token_mean=sum(edited[i]['kl_sum'] for i in ids)/token_count),
        writes=dict(scope='descriptive equal-document mean per-token ratios; modules are not independent subjects',
                    module_mean_ratio_changes=[dict(module=m,estimate=float(v)) for m,v in zip(modules,changes)]))
