"""Sparse rewiring: one CPU driver for the analyses over a finished run (tickets 10, 11, 12, 13).

The library code for each ticket lives in its own module; this file only feeds it the sealed outputs of a
finished run and seals the results where ``rewire_report`` looks for them::

    <root>/c-attribution/sparse-rewiring/analysis/
        standalone-<arm>/standalone_analysis.json   ticket 12 (D22), one per arm
        rank-comparison/rank_comparison.json        ticket 11 (D21), rank 1 vs rank 0
        dynamics/*.png (+ svg, csv, provenance)     ticket 13 (D20, D21), one per arm and side by side
        attribution/overlaps.json                   ticket 10 (D19), only with --attribution-root

Every input is verified before use (``load_arm_artifacts`` runs ``verify_arm``; every evaluation unit goes through
``verify_run``). A unit that is already sealed is kept, never rewritten; ``--redo NAME`` moves an existing unit aside
(``<name>.superseded-<ns>``) and computes it again. CPU only, no model, no GPU.

    python -m pilot_eval.rewire_analysis --root $R                                  # 11, 12, 13
    python -m pilot_eval.rewire_analysis --root $R --attribution-root $C_ARTIFACTS  # and 10 (+ three-way in 11)
    python -m pilot_eval.rewire_analysis --root $R --gold-adapter $GOLD_DIR         # adds the rank-1 cosine
    python -m pilot_eval.rewire_report build --root $R                              # then rebuild the report
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pilot_eval import attribution_units as au

SCHEMA = 'sparse-rewiring/analysis-driver-v1'
UNITS = ('standalone', 'rank-comparison', 'dynamics', 'attribution')
_TENSOR = re.compile(r'layers\.(\d+)\.mlp\.rewire_(W|A|B|log_alpha)(?:_(gate|up))?$')


class AnalysisError(ValueError):
    pass


def _j(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sealed_exists(directory: Path) -> bool:
    if not (directory / 'complete.json').is_file():
        return False
    au.verify_run(directory)   # a tampered unit fails loudly instead of being kept
    return True


def _supersede(directory: Path) -> None:
    if directory.exists():
        os.replace(directory, directory.with_name(f'{directory.name}.superseded-{time.time_ns()}'))


def _responses(unit: Path) -> dict[str, dict[str, Any]]:
    from pilot_eval import rewire_rank_comparison as rrc
    return rrc.load_verified_eval_responses(unit)[0]


def _by_layer(tensors: Mapping[str, Any]) -> dict[str, dict[int, np.ndarray]]:
    """``{'W_gate': {layer: array}, 'A_up': ..., 'log_alpha': ...}`` from checkpoint tensor names."""
    out: dict[str, dict[int, np.ndarray]] = {}
    for name, value in tensors.items():
        m = _TENSOR.search(name)
        if not m:
            continue
        layer, kind, part = int(m.group(1)), m.group(2), m.group(3)
        key = kind if part is None else f'{kind}_{part}'
        arr = value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
        out.setdefault(key, {})[layer] = arr
    return out


# --- loading one finished arm -------------------------------------------------------------------------

def load_arm(root: Path, record: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """Everything the analyses read for one arm, verified; checkpoint tensors reduced to what is used."""
    from pilot_eval import rewire_chain as rc
    from pilot_eval import rewire_report as rr
    from pilot_eval import rewire_standalone as rs
    spec = record['arms'][arm]
    arm_dir = rr.paths(root)['runs'] / spec['run_id']
    plan = _j(rr._sealed(arm_dir / rc.ARM_EVIDENCE_DIR) / 'plan.json')
    art = rs.load_arm_artifacts(arm_dir, plan)
    tensors = _by_layer(art.pop('final_tensors'))
    layers = sorted(tensors['log_alpha'])
    final = int(plan['checkpoints'][-1])
    survivors = [(int(s['layer']), int(s['neuron'])) for s in art['final_export']['survivors']]
    learned: dict[str, Any] = {}
    if int(plan['rank']) == 0:
        for part in ('gate', 'up'):
            learned[f'rows_{part}'] = np.stack([tensors[f'W_{part}'][l][n] for l, n in survivors]) if survivors \
                else np.zeros((0, tensors[f'W_{part}'][layers[0]].shape[1]))
    else:
        learned['a'] = {part: {l: tensors[f'A_{part}'][l] for l in layers} for part in ('gate', 'up')}
    val = {int(s): _responses(arm_dir / 'eval' / str(s) / 'greedy-val') for s in plan['checkpoints']}
    cohort = {int(s): _responses(arm_dir / 'eval' / str(s) / 'greedy-cohort150')
              for s in plan.get('cohort150_steps', ()) if (arm_dir / 'eval' / str(s) / 'greedy-cohort150').is_dir()}
    scores = {int(s): _j(arm_dir / 'eval' / str(s) / 'greedy-val' / 'scores.json') for s in plan['checkpoints']}
    cohort_scores = {s: _j(arm_dir / 'eval' / str(s) / 'greedy-cohort150' / 'scores.json') for s in cohort}
    art.pop('arm'); art.pop('rank')
    return dict(arm=arm, rank=int(plan['rank']), run_id=spec['run_id'], plan=plan, final=final,
                shape=(len(layers), int(tensors['log_alpha'][layers[0]].size)), learned=learned,
                val=val, cohort150=cohort, val_scores=scores, cohort150_scores=cohort_scores,
                export_path=arm_dir / 'masks' / f'step-{final}' / f'{final}.json', **art)


def load_references(root: Path) -> dict[str, dict[str, Any]]:
    from pilot_eval import rewire_report as rr
    rw = rr.paths(root)['rw']
    refs = _j(rr._sealed(rw / 'references' / 'summary') / 'references.json')
    return {name: dict(step=int(ref['step']), val_ce=float(ref['val_ce']),
                       val=_responses(rw / 'references' / f"{name}-step-{ref['step']}" / 'greedy-val'))
            for name, ref in refs.items()}


# --- ticket 12: standalone, per arm ---------------------------------------------------------------------

def standalone(a: Mapping[str, Any], refs: Mapping[str, Any], *, gold_adapter: str | Path | None = None,
               activation_shift: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from pilot_eval import rewire_standalone as rs
    n_layers, _ = a['shape']
    survivors = a['final_export']['survivors']
    freeze = int(a['plan']['settings']['training']['freeze_after_step'])
    snaps = a['snapshots']
    pre = [s for s in snaps if s <= freeze]   # after the freeze every log_alpha is +-20: the last snapshot before it
    hist_step = max(pre) if pre else None
    final_la = np.asarray(snaps[hist_step]['log_alpha'], dtype=np.float64).ravel() if pre else np.zeros(0)
    ckpt_masks = {s: snaps[s]['hard_mask'] for s in a['plan']['checkpoints'] if int(s) in snaps}
    selection = dict(
        survivor_count=len(survivors),
        per_layer=rs.per_layer_distribution(survivors, n_layers),
        log_alpha_histogram=dict(step=hist_step, **rs.log_alpha_histogram(final_la)),
        settle=rs.compute_settle_step(settle_records=a['settle_rows'],
                                      snapshot_masks={s: v['hard_mask'] for s, v in snaps.items()},
                                      final_survivors=survivors, freeze_step=freeze),
        stability=dict(source='16-step gate snapshots at checkpoint steps (hard mask, z_det > 0)',
                       checkpoints_without_snapshot=sorted(set(map(int, a['plan']['checkpoints'])) - set(ckpt_masks)),
                       **rs.survivor_stability(ckpt_masks, survivors)),
    )
    learned: dict[str, Any] = dict(change_sizes=rs.summarize_change_sizes(survivors, n_layers))
    if a['rank'] == 0:
        learned['effective_rank'] = {part: rs.effective_rank_of_rows(a['learned'][f'rows_{part}'])
                                     for part in ('gate', 'up')}
    elif gold_adapter is None:
        learned['rank1_shared_direction'] = 'not computed: no --gold-adapter given'
    else:
        try:
            gold = rs.load_gold_lora_a(gold_adapter, sorted(a['learned']['a']['gate']))
            learned['rank1_shared_direction'] = dict(adapter=str(gold_adapter), **rs.rank1_shared_direction_cosine(
                {p: {l: v.reshape(-1) for l, v in d.items()} for p, d in a['learned']['a'].items()}, gold))
        except Exception as exc:   # recorded, never silently dropped
            learned['rank1_shared_direction'] = f'not computed: {type(exc).__name__}: {exc}'
    learned['activation_shift'] = dict(activation_shift) if activation_shift else \
        'not yet run (GPU pass: python -m pilot_eval.rewire_activation_shift)'
    final = a['final']
    ids = sorted(a['val'][final])
    behaviour: dict[str, Any] = dict(
        validation=dict(question_ids=len(ids),
                        val_ce={int(r['step']): float(r['val_ce']) for r in a['val_ce_rows'] if 'val_ce' in r},
                        hardening=a['hardening'],
                        by_checkpoint={s: rs.summarize_behaviour_responses(a['val'][s], ids) for s in sorted(a['val'])}),
        flips_val128_final={name: rs.flip_table(a['val'][final], ref['val'], ids, arm_label=a['arm'], ref_label=name)
                            for name, ref in sorted(refs.items())},
    )
    if a['cohort150']:
        behaviour['cohort150'] = {s: dict(rs.summarize_behaviour_responses(r, sorted(r)), exploratory=True)
                                  for s, r in sorted(a['cohort150'].items())}
        behaviour['cohort150_flips'] = 'not computed: no reference responses on cohort150 in this run'
    return dict(arm=a['arm'], rank=a['rank'], run_id=a['run_id'], plan_sha256=a['plan']['plan_sha256'],
                final_step=final, final_export=dict(path=a['export_path'].name, sha256=_sha(a['export_path'])),
                selection=selection, what_survivors_learned=learned, behaviour=behaviour,
                labels=dict(settle='definitions marked proposed (D22)', cohort150='exploratory, evaluation only (D12)'))


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- ticket 11: rank 1 vs rank 0 ---------------------------------------------------------------------

def rank_comparison(r1: Mapping[str, Any], r0: Mapping[str, Any], refs: Mapping[str, Any], *,
                    manifest_entries=None) -> dict[str, Any]:
    from pilot_eval import rewire_rank_comparison as rrc
    if r1['shape'] != r0['shape']:
        raise AnalysisError(f"rank 1 and rank 0 differ in shape: {r1['shape']} vs {r0['shape']}")
    final1, final0 = r1['final'], r0['final']
    ids = sorted(r1['val'][final1])
    if sorted(r0['val'][final0]) != ids:
        raise AnalysisError('rank 1 and rank 0 validation responses cover different questions')
    paired = {'val128_final': dict(r1=r1['val'][final1], r0=r0['val'][final0], question_ids=ids,
                                   references={n: ref['val'] for n, ref in refs.items()})}
    if r1['cohort150'] and r0['cohort150']:
        c1, c0 = r1['cohort150'][max(r1['cohort150'])], r0['cohort150'][max(r0['cohort150'])]
        if sorted(c1) != sorted(c0):
            raise AnalysisError('rank 1 and rank 0 cohort150 responses cover different questions')
        paired['cohort150_final'] = dict(r1=c1, r0=c0, question_ids=sorted(c1))
    out = rrc.compare_ranks(r1_export=r1['final_export'], r0_export=r0['final_export'], shape=tuple(r1['shape']),
                            manifest_entries=manifest_entries, val_ce=(r1['val_ce_rows'], r0['val_ce_rows']),
                            hardening=(r1['hardening'], r0['hardening']), paired_evals=paired)
    out['inputs'] = {k: dict(run_id=a['run_id'], plan_sha256=a['plan']['plan_sha256'],
                             final_export_sha256=_sha(a['export_path'])) for k, a in (('r1', r1), ('r0', r0))}
    if manifest_entries is None:
        out['three_way_attribution'] = 'not computed: no --attribution-root given'
    return out


# --- ticket 13: dynamics figures -------------------------------------------------------------------

def _dynamics_input(a: Mapping[str, Any], refs: Mapping[str, Any]) -> dict[str, Any]:
    return dict(label=a['arm'], step_rows=a['step_rows'], val_ce_rows=[r for r in a['val_ce_rows'] if 'val_ce' in r],
                val_acc_rows=[dict(step=s, accuracy=float(v['accuracy'])) for s, v in sorted(a['val_scores'].items())],
                cohort150_rows=[dict(step=s, accuracy=float(v['accuracy']))
                                for s, v in sorted(a['cohort150_scores'].items())],
                settle_rows=[r for r in a['settle_rows'] if 'jaccard' in r],
                references=dict(untuned_val_ce=(refs.get('untuned') or {}).get('val_ce'),
                                gold_sft_val_ce=(refs.get('gold-sft') or {}).get('val_ce')))


def dynamics(arms: Mapping[str, Mapping[str, Any]], refs: Mapping[str, Any]) -> dict[str, bytes]:
    """Render every figure into a scratch folder and return the files to seal."""
    from pilot_eval import rewire_dynamics_figure as rdf
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, a in sorted(arms.items()):
            rdf.build_dynamics_figure(_dynamics_input(a, refs), tmp, name=f'dynamics-{name}', plan=a['plan'],
                                      require_cohort150=bool(a['cohort150']))
        ranks = {a['rank']: a for a in arms.values()}
        if set(ranks) == {0, 1}:
            rdf.build_side_by_side_dynamics_figure(_dynamics_input(ranks[1], refs), _dynamics_input(ranks[0], refs),
                                                   tmp, name='dynamics-r1-vs-r0', plan_left=ranks[1]['plan'],
                                                   plan_right=ranks[0]['plan'])
        return {p.name: p.read_bytes() for p in sorted(tmp.iterdir()) if p.is_file()}


# --- ticket 10: attribution overlaps ---------------------------------------------------------------

def attribution_manifest(attribution_root: str | Path, out_dir: Path):
    """Seal the production-1 manifest (``c_production_entries``) as an input unit and load it."""
    from pilot_eval import rewire_attribution as ra
    body = dict(schema=ra.MANIFEST_SCHEMA, root=str(Path(attribution_root)), entries=ra.c_production_entries())
    if not _sealed_exists(out_dir):
        au.seal_run(out_dir, {'manifest.json': body})
    if _j(out_dir / 'manifest.json') != json.loads(json.dumps(body)):
        raise AnalysisError(f'{out_dir} holds a different manifest; pass --redo attribution to replace it')
    return ra.load_manifest(out_dir / 'manifest.json', attribution_root)


# --- driver ---------------------------------------------------------------------------------------

def run(root: str | Path, *, gold_adapter: str | Path | None = None, attribution_root: str | Path | None = None,
        only: tuple[str, ...] = ('standalone', 'rank-comparison', 'dynamics', 'attribution'),
        redo: tuple[str, ...] = ()) -> dict[str, Any]:
    """Run the requested analyses over the finished run at ``root`` (the directory that holds ``c-attribution/``)."""
    from pilot_eval import rewire_report as rr
    root = Path(root)
    base = rr.paths(root)['analysis']
    record = rr.launch_record(root)
    arms = {arm: load_arm(root, record, arm) for arm in record['arms']}
    refs = load_references(root)
    done: dict[str, Any] = {}

    def unit(name: str, directory: Path, files) -> None:
        if name.split(':')[0] in redo:
            _supersede(directory)
        if _sealed_exists(directory):
            done[name] = f'kept (already sealed): {directory}'
            return
        au.seal_run(directory, files() if callable(files) else files)
        done[name] = str(directory)

    manifest = None
    if attribution_root is not None and ('attribution' in only or 'rank-comparison' in only):
        if 'attribution' in redo:
            _supersede(base / 'attribution-inputs')
        manifest = attribution_manifest(attribution_root, base / 'attribution-inputs')

    if 'standalone' in only:
        for name, a in sorted(arms.items()):
            shift_dir = base / f'activation-shift-{name}'
            shift = None
            if _sealed_exists(shift_dir):
                body = _j(shift_dir / 'activation_shift.json')
                shift = {k: body[k] for k in ('per_neuron_mean_shift', 'per_neuron_mean_abs_shift', 'tokens')}
            from pilot_eval import rewire_standalone as rs
            unit(f'standalone:{name}', base / f'standalone-{name}',
                 lambda a=a, shift=shift: {'standalone_analysis.json': dict(schema=rs.SCHEMA, **standalone(
                     a, refs, gold_adapter=gold_adapter, activation_shift=shift))})
    ranks = {a['rank']: a for a in arms.values()}
    if 'rank-comparison' in only:
        if set(ranks) != {0, 1}:
            done['rank-comparison'] = f'skipped: arms have ranks {sorted(ranks)}, not 0 and 1'
        else:
            unit('rank-comparison', base / 'rank-comparison',
                 lambda: {'rank_comparison.json': rank_comparison(ranks[1], ranks[0], refs,
                                                                  manifest_entries=(manifest or {}).get('entries'))})
    if 'dynamics' in only:
        unit('dynamics', base / 'dynamics', lambda: dynamics(arms, refs))
    if 'attribution' in only:
        if manifest is None:
            done['attribution'] = 'skipped: no --attribution-root given'
        else:
            from pilot_eval import rewire_attribution as ra
            if 'attribution' in redo:
                _supersede(base / 'attribution')
            if _sealed_exists(base / 'attribution'):
                done['attribution'] = f"kept (already sealed): {base / 'attribution'}"
            else:
                sets, sources = {}, {}
                for name, a in sorted(arms.items()):
                    sets[name], sources[name] = ra.load_survivors_with_provenance(a['export_path'])
                ra.seal(base / 'attribution', manifest, ra.run(manifest, sets), survivor_sources=sources)
                done['attribution'] = str(base / 'attribution')
    return done


def headline(root: str | Path) -> dict[str, Any]:
    """The few numbers worth pasting back, read from the sealed units (whatever exists)."""
    from pilot_eval import rewire_report as rr
    base = rr.paths(root)['analysis']
    out: dict[str, Any] = {}
    comp = base / 'rank-comparison' / 'rank_comparison.json'
    if comp.is_file():
        c = _j(comp)
        s = c['survivors']
        null = s['layer_matched_null']
        out['survivors'] = dict(r1=s['counts']['r1']['set_size'], r0=s['counts']['r0']['set_size'],
                                shared=s['intersection_count'], jaccard=s['jaccard'],
                                null_expected_shared=null['exact']['expected'], p_upper=null['exact']['p_upper'])
        out['paired'] = {name: dict(accuracy=p['accuracy'], r0_minus_r1_pp=p['r0_vs_r1']['delta_pp_b_minus_a'],
                                    mcnemar_p=p['r0_vs_r1']['mcnemar_exact_p'])
                         for name, p in c['paired_accuracy'].items()}
    for unit in sorted(base.glob('standalone-*')):
        if not (unit / 'standalone_analysis.json').is_file() or '.superseded-' in unit.name:
            continue
        b = _j(unit / 'standalone_analysis.json')
        sel, learned = b['selection'], b['what_survivors_learned']
        row = dict(settle={k: sel['settle'][k] for k in ('from_diagnostics', 'from_snapshots')},
                   flips_vs_gold=({k: b['behaviour']['flips_val128_final']['gold-sft'][k]
                                   for k in ('fixed', 'broken', 'mcnemar_exact_p')}
                                  if 'gold-sft' in b['behaviour']['flips_val128_final'] else None))
        if 'effective_rank' in learned:
            row['effective_rank'] = {p: dict(rows=e['num_rows'], participation_ratio=e['participation_ratio'],
                                             count_90pct=e['count_for_energy'])
                                     for p, e in learned['effective_rank'].items()}
        out[b['arm']] = row
    return out


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--root', required=True, help='the finished run directory that holds c-attribution/')
    parser.add_argument('--gold-adapter', help='gold-sft adapter directory (adapter_model.safetensors) for the '
                                               'rank-1 shared-direction cosine')
    parser.add_argument('--attribution-root', help='C-attribution artifact root (holds c-attribution/aggregate and '
                                                   'c-attribution/freeze) for ticket 10 and the three-way overlap')
    parser.add_argument('--only', nargs='+', choices=UNITS, default=list(UNITS))
    parser.add_argument('--redo', nargs='+', choices=UNITS, default=[],
                        help='move these sealed units aside and compute them again')
    args = parser.parse_args(argv)
    import logging
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)   # missing-font chatter, not an error
    done = run(args.root, gold_adapter=args.gold_adapter, attribution_root=args.attribution_root,
               only=tuple(args.only), redo=tuple(args.redo))
    print(json.dumps(dict(units=done, headline=headline(args.root)), indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
