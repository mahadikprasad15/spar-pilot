"""Measurements a restored C-attribution run gives for planning the next record (ticket 11's list).

Reads only sealed files of a clean-root restore (``neuron_audit_export restore``) and writes one JSON:
floors per arm (finite difference, identity, batch invariance, cached-against-forced), peak memory, the
gross/net ratio distribution, stage seconds, attribution throughput (steady units, after the first unit of
each arm, which includes the model load), greedy seconds per batch, and the export time from the final
summary when its restore root is given. Nothing is recomputed from tensors.

    python scripts/c_attribution_measurements.py --root artifacts/imports/c-attribution-rehearsal-pod-1 \
        --final-root artifacts/imports/c-attribution-rehearsal-pod-1-final --out <file>.json
"""
import argparse
import glob
import json
from pathlib import Path

SUB = 'c-attribution'


def _j(path):
    return json.loads(Path(path).read_text())


def _sealed(path):
    return _j(Path(path) / 'complete.json')['sealed_unix']


def _stage_start(log, stage):
    import datetime as dt
    for line in log.splitlines():
        if line.endswith(f'stage {stage} start'):
            return dt.datetime.fromisoformat(line.split()[0]).replace(tzinfo=dt.timezone.utc).timestamp()
    return None


def measure(root, final_root=None):
    sub = Path(root) / SUB
    numerics = _j(sub / 'qualification/numerics.json')
    qual = _j(sub / 'qualification/qualification.json')
    record = _j(next(sub.glob('launch/*/launch_record.json')))
    log = (sub / 'logs/chain.log').read_text()
    out = dict(run_id=record['run_id'], code_commit=record['code_commit'], tolerances=record['tolerances'],
               checks=qual['checks'], signoff=_j(sub / 'signoff/signoff.json')['decision'], arms={})
    for arm in record['arms']:
        n, d = numerics['per_arm'][arm], qual['details'][arm]
        agg = _j(sub / f'aggregate/{arm}/numerics.json')
        mem = d['memory']
        units = sorted(glob.glob(str(sub / f'attribution/{arm}/unit-*')))
        metas = [_j(Path(u) / 'meta.json') for u in units]
        times = [_sealed(u) for u in units]
        steady = [dict(rows=m['rows'], tokens=m['tokens'], seconds=b - a)
                  for m, a, b in zip(metas[1:], times, times[1:])]
        out['arms'][arm] = dict(
            finite_difference=dict(median_rel_err=n['finite_difference']['median_rel_err'],
                                   max_rel_err=n['finite_difference']['max_rel_err'],
                                   floors=n['finite_difference']['floors']),
            identity_relative_to_gross=agg['identity'],
            batch_invariance_fraction=n['batch_invariance']['max_fraction'],
            cached_vs_forced=n['cached_vs_forced'],
            memory=dict(peak_bytes=mem['peak_bytes'], total_bytes=mem['total_bytes'],
                        fraction=mem['peak_bytes'] / mem['total_bytes'], token_budget=mem['budget']),
            gross_net_ratio=agg['gross_net_ratio'],
            attribution_units=[dict(rows=m['rows'], tokens=m['tokens']) for m in metas],
            attribution_steady=steady,
            tokens_per_second=(sum(s['tokens'] for s in steady) / sum(s['seconds'] for s in steady)) if steady else None)
    batches = sorted(glob.glob(str(sub / 'generation/*/*/batch-*')), key=_sealed)
    start = _stage_start(log, 'generation')
    previous, rows = start, []
    for b in batches:
        sealed = _sealed(b)
        config = _j(Path(b) / 'config.json')
        answers = sum(1 for _ in open(Path(b) / 'responses.jsonl'))
        rows.append(dict(batch=str(Path(b).relative_to(sub)), condition=config['condition']['name'], answers=answers,
                         seconds=None if previous is None else sealed - previous))
        previous = sealed
    out['generation'] = dict(batch_size=record['generation']['batch_size'], cap=record['generation']['cap'],
                             batches=rows, note='the first batch includes the model load')
    out['stage_seconds'] = {line.split()[2]: float(line.split()[4].rstrip('s'))
                            for line in log.splitlines() if ' elapsed ' in line}
    if final_root is not None:
        summary = _j(next(Path(final_root).glob(f'{SUB}/summaries/*/final/run_summary.json')))
        out['export'] = dict(seconds=summary['backups']['hf_export'].get('seconds'),
                             attempts=len(summary['backups']['hf_export'].get('attempts') or []),
                             statement=summary['backups'].get('statement'))
        out['wall_seconds'] = summary['wall_seconds']
        out['resume'] = summary.get('resume')
        out['restored_from_volume'] = summary.get('restored_from_volume')
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--root', required=True)
    p.add_argument('--final-root')
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    result = measure(args.root, args.final_root)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(args.out)


if __name__ == '__main__':
    main()
