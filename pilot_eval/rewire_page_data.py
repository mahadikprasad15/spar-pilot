"""Print a compact JSON of a finished sparse-rewiring run, small enough to paste into a chat (for the results page).

    python -m pilot_eval.rewire_page_data --root $R > /workspace/page_data.json; wc -c /workspace/page_data.json

Reads the run's own files (launch record, report, analysis units, step/validation logs); no verification, no
model. Step metrics are thinned to every ``--every`` steps; numbers are rounded to 5 significant figures.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _r(x: Any) -> Any:
    if isinstance(x, float):
        return float(f'{x:.5g}')
    if isinstance(x, dict):
        return {k: _r(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v) for v in x]
    return x


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.is_file() else []


def collect(root: str | Path, every: int = 16) -> dict[str, Any]:
    from pilot_eval import rewire_report as rr
    root = Path(root)
    p = rr.paths(root)
    launch = sorted((root / 'c-attribution' / 'launch').glob('*/launch_record.json'))[0]
    record = json.loads(launch.read_text())
    settings = {k: v for k, v in record.items() if k not in ('inputs', 'pod_id', 'volume_identity', 'pod_stop_commands',
                                                            'export_repo', 'timeouts_seconds', 'marker_stale_seconds')}
    report = json.loads((p['report'] / 'report.json').read_text())
    report.pop('analyses', None)
    out: dict[str, Any] = dict(settings=settings, report=report, arms={})
    a = p['analysis']
    for arm, spec in record['arms'].items():
        logs = p['runs'] / spec['run_id'] / 'arm_evidence' / 'logs'
        if not logs.is_dir():
            logs = p['runs'] / spec['run_id'] / 'logs'
        steps = _jsonl(logs / 'steps.jsonl')
        last = steps[-1]['step'] if steps else 0
        keep = [r for r in steps if r['step'] % every == 0 or r['step'] in (1, 2, 4, 8, last)]
        rows = [[r['step'], r.get('expected_active'), r.get('nonzero_z_det'), r.get('ce_loss'), r.get('l0_loss'),
                 r.get('lr_reads'), (r.get('norms') or {}).get('gate_survivors', {}).get('sum'),
                 (r.get('norms') or {}).get('up_survivors', {}).get('sum'),
                 (r.get('norms') or {}).get('gate_all', {}).get('sum'), (r.get('norms') or {}).get('up_all', {}).get('sum')]
                for r in keep]
        val = [[r['step'], r['val_ce']] for r in _jsonl(logs / 'validation.jsonl') if 'val_ce' in r]
        settle = [[r['step'], r.get('jaccard'), r.get('expected_active_rel_change'), r.get('switched_on'),
                   r.get('switched_off')] for r in _jsonl(logs / 'settle.jsonl')]
        sa = a / f'standalone-{arm}' / 'standalone_analysis.json'
        standalone = json.loads(sa.read_text()) if sa.is_file() else None
        if standalone:
            standalone['selection'].get('stability', {}).pop('per_survivor', None)
        final = rr.paths(root)['runs'] / spec['run_id'] / 'masks'
        exports = sorted(final.glob(f"step-{record['training']['steps']}/*.json"))
        survivors = [{k: s.get(k) for k in ('layer', 'neuron', 'log_alpha', 'norm_gate', 'norm_up')}
                     for s in json.loads(exports[0].read_text())['survivors']] if exports else None
        out['arms'][arm] = dict(step_columns=['step', 'expected_active', 'nonzero_z_det', 'train_ce', 'l0_loss',
                                              'lr_reads', 'gate_surv_norm_sum', 'up_surv_norm_sum',
                                              'gate_all_norm_sum', 'up_all_norm_sum'],
                                steps=rows, val_ce=val, settle_columns=['step', 'jaccard', 'rel_change', 'on', 'off'],
                                settle=settle, survivors=survivors, standalone=standalone)
    rc = a / 'rank-comparison' / 'rank_comparison.json'
    if rc.is_file():
        comp = json.loads(rc.read_text())
        comp['survivors']['per_layer'] = [r for r in comp['survivors'].get('per_layer', []) if r.get('union')]
        out['rank_comparison'] = comp
    return _r(out)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--root', required=True)
    parser.add_argument('--every', type=int, default=16)
    args = parser.parse_args(argv)
    print(json.dumps(collect(args.root, args.every), separators=(',', ':'), default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
