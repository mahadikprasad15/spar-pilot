"""Audit automatic generation coverage against gold's saved batch membership."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile


def batch_histogram(batches, missing):
    return dict(sorted(Counter(sum(item not in missing for item in batch) for batch in batches).items()))


def audit(archive):
    prefix = 'runs/sampling-sft/sampling-sft-v1/'
    generation = prefix + 'generation/a40-detsched-generation-v1/'
    names = [prefix + 'results/source-audit.json'] + [generation + f'k{k}/{name}' for k in (0, 2, 5) for name in ('attempts.jsonl', 'dispositions.jsonl')]
    with tarfile.open(archive) as tar:
        payloads = {name: tar.extractfile(name).read() for name in names}
    batches = json.loads(payloads[names[0]])['training']['batches']
    order = [item for batch in batches for item in batch]
    if len(batches) != 64 or any(len(batch) != 8 for batch in batches) or len(set(order)) != 512:
        raise ValueError('Gold order must have 64 batches of eight distinct problems.')
    arms = {}
    for k in (0, 2, 5):
        def rows(name):
            return [json.loads(line) for line in payloads[generation + f'k{k}/{name}'].splitlines()]
        attempts, dispositions = rows('attempts.jsonl'), rows('dispositions.jsonl')
        if len(dispositions) != 512 or {r['source_id'] for r in dispositions} != set(order):
            raise ValueError('Dispositions do not match the saved gold cohort.')
        missing = {r['source_id'] for r in dispositions if r['disposition'] == 'uncovered'}
        arms[str(k)] = dict(attempts=len(attempts), covered=512-len(missing), uncovered_ids=sorted(missing),
            batch_size_histogram=batch_histogram(batches, missing),
            rejection_reasons=dict(Counter(r['rejection_reason'] or 'accepted' for r in attempts)),
            end_states=dict(Counter(r['end_state'] for r in attempts)),
            required_human_reviews=sum(r['end_state']=='finished' and r['automatic_flexible_v3']['status']=='invalid' for r in attempts))
    return dict(protocol='sampling-sft-automatic-coverage-audit-v1', source_archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        source_hf_commit='45d66ffb1e2e58357e0601a1097e29abfbe5fb43', input_sha256={name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
        scope='automatic dispositions; human review may alter coverage', gold_batches=batches, arms=arms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = audit(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
