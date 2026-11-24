"""Sparse-rewiring validation set (D11, ticket 04): 128 seeded GSM8K train questions disjoint from every
recorded train use.

Exclusions, each recorded with its hash and ID count in the exclusion record:
- the gold 512, read from gold's sealed SFT plan (``sft.load_sft``), checked against the seed-42 derivation
  (``sft.gold_training_indices``) and against the catalog's ``gsm8k-train-512`` digest;
- every catalog cohort that is not a test split (``catalog.load``), plus any train ID named anywhere in the
  catalog's cohort or response-set entries (prior uses, notes);
- every GSM8K train ID in the sealed ``sequences-v2`` run (``neuron_audit_sequences.verify`` + its manifest),
  across calibration, discovery and held-out;
- every GSM8K ID and question text in the listed run records (JSON or JSONL, parsed by
  ``neuron_audit_sequences._lineage_file``).

Train questions whose normalized text equals an excluded question, or a cohort150 question, are excluded too,
as are later copies of a text already in the pool. Anything unreadable, unsealed or malformed is refused:
the record never says "no overlap" for a source it could not read.

Items are built by gold's own builder (``sft.gold_item``): same prompt, tokens, completion-only labels and
end-turn mask as the gold training rows.

    python -m pilot_eval.sparse_validation --sft-config <plans/.../sft.config.json> --artifact-root <root> \
        --catalog catalog --sequences runs/joint-neuron-adapter-audit/sequences/sequences-v2 \
        [--run-record <path> ...] --output <run>/inputs
"""
import argparse
import json
import random
import sys
from pathlib import Path

from pilot_eval import catalog as cat
from pilot_eval import neuron_audit_sequences as seqs
from pilot_eval.sft import gold_item, gold_training_indices, load_sft
from pilot_eval.training import file_hash
from pilot_eval.workflow import HFDependencies, _hash, _save_frozen

SCHEMA = 'sparse-rewiring/validation-set-v1'
SEED = 20261122
SIZE = 128
GOLD_COHORT = 'gsm8k-train-512'


def _train_ids(ids, n_train, where):
    out = set()
    for value in ids:
        match = seqs.ID_PATTERN.fullmatch(str(value))
        if not match:
            raise ValueError(f'{where}: malformed question ID {value!r}')
        if match.group(1) == 'train':
            out.add(int(match.group(2)))
    return _bounded(out, n_train, where)


def _bounded(indices, n_train, where):
    bad = sorted(i for i in indices if i >= n_train)
    if bad:
        raise ValueError(f'{where}: train IDs outside the pinned train split ({n_train} rows): {bad[:5]}')
    return set(indices)


def _source(kind, indices, sha256, **extra):
    ids = [f'gsm8k:train:{i}' for i in sorted(indices)]
    return dict(kind=kind, sha256=sha256, train_ids=len(ids), train_ids_sha256=cat.ids_digest(ids), **extra)


def _catalog_sources(catalog_dir, roots, gold, n_train):
    catalog = cat.load(catalog_dir)
    sources, excluded = [], set()
    cohorts_sha = file_hash(Path(catalog_dir) / 'cohorts.yaml')
    gold_digest = cat.ids_digest(f'gsm8k:train:{i}' for i in gold)
    if catalog['cohorts'].get(GOLD_COHORT, {}).get('ids_sha256') != gold_digest:
        raise ValueError(f'catalog cohort {GOLD_COHORT}: ids_sha256 does not match the gold plan IDs')
    for name, cohort in sorted(catalog['cohorts'].items()):
        if cohort.get('split') == 'test':
            continue
        where = f'catalog cohort {name}'
        try:
            ids = cat._cohort_ids(cohort, catalog, roots)
        except (ValueError, KeyError, TypeError) as exc:  # JSONDecodeError is a ValueError
            raise ValueError(f'{where}: malformed source {cohort["source"]["path"]}: {exc}') from exc
        if ids is None:
            if cohort['ids_sha256'] != gold_digest:
                raise ValueError(f'{where}: source {cohort["source"]["path"]} is unreadable here')
            indices, resolved = set(gold), 'gold-plan'
        else:
            if len(ids) != cohort['size'] or len(set(ids)) != len(ids) or cat.ids_digest(ids) != cohort['ids_sha256']:
                raise ValueError(f'{where}: source IDs do not match size/ids_sha256')
            indices, resolved = _train_ids(ids, n_train, where), 'source'
        excluded |= indices
        sources.append(_source('catalog-cohort', indices, cohort['ids_sha256'], name=name, resolved_by=resolved,
                               catalog_cohorts_sha256=cohorts_sha))
    entries = {'cohorts': catalog['cohorts'],
               'responses': {k: {f: v for f, v in e.items() if f != '_file'} for k, e in catalog['responses'].items()}}
    text = json.dumps(entries, sort_keys=True, default=str)
    named = _bounded({int(i) for s, i in seqs.ID_PATTERN.findall(text) if s == 'train'}, n_train, 'catalog entries')
    excluded |= named
    sources.append(_source('catalog-prior-uses', named, _hash(entries), name='train IDs named in catalog entries'))
    return sources, excluded


def _sequences_source(run_dir, n_train):
    if not run_dir.is_dir():
        raise ValueError(f'sequences run {run_dir} is unreadable (missing)')
    check = seqs.verify(run_dir)
    if not check['ok']:
        raise ValueError(f'sequences run {run_dir} failed verification: {", ".join(check["problems"])}')
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    if manifest.get('schema') != seqs.MANIFEST_SCHEMA:
        raise ValueError(f'sequences run {run_dir}: unexpected manifest schema {manifest.get("schema")!r}')
    rows = []
    for n, line in enumerate((run_dir / 'sequences.jsonl').read_text().splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line)['id'])
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f'sequences run {run_dir}: malformed sequences.jsonl line {n}') from exc
    if rows != manifest['order']:
        raise ValueError(f'sequences run {run_dir}: sequences.jsonl IDs differ from the manifest order')
    gsm = [i for i in rows if str(i).startswith('gsm8k:')]
    indices = _train_ids(gsm, n_train, f'sequences run {run_dir}')
    if not indices:
        raise ValueError(f'sequences run {run_dir}: no GSM8K train IDs; refusing to record "no overlap"')
    return _source('sequences', indices, file_hash(run_dir / 'complete.json'), path=manifest.get('run_id'),
                   manifest_sha256=file_hash(run_dir / 'manifest.json')), indices


def _record_source(path, label, n_train):
    if not path.is_file():
        raise ValueError(f'run record {label} is unreadable (missing)')
    if path.suffix not in ('.json', '.jsonl'):
        raise ValueError(f'run record {label}: only .json and .jsonl records are parsed')
    ids, questions, errors = seqs._lineage_file(path)
    if errors:
        raise ValueError(f'run record {label}: malformed JSON at line {errors[0]["line"]}: {errors[0]["error"]}')
    if not ids['train'] and not ids['test'] and not questions:
        raise ValueError(f'run record {label}: no GSM8K IDs or questions; refusing to record "no overlap"')
    indices = _bounded(ids['train'], n_train, f'run record {label}')
    return _source('run-record', indices, file_hash(path), path=label, question_texts=len(questions)), indices, questions


def build(sft_config, artifact_root, *, catalog_dir, sequences_dir, run_records=(), catalog_roots=None,
          dependencies=None, seed=SEED, size=SIZE):
    """Return (items, exclusion record). Raises ValueError on any unreadable or inconsistent source."""
    root = Path(artifact_root).resolve()
    config, gold_rows, cohort_items = load_sft(sft_config, root)
    source = config['source_config']
    deps = dependencies or HFDependencies()
    tokenizer, context = deps.load_tokenizer(source['model'], source['tokenizer_revision'])
    if _hash(tokenizer.chat_template) != source['chat_template_sha256']:
        raise ValueError('chat template hash mismatch')
    data = deps.load_dataset(source['dataset_path'], source['dataset_config'], source['dataset_revision'])
    train = data['train']
    n_train = len(train)

    gold = {r['source_index'] for r in gold_rows}
    if sorted(gold) != gold_training_indices(n_train):
        raise ValueError('gold plan indices differ from the seed-42 derivation on this train split')
    sources = [_source('gold-plan', gold, file_hash(sft_config), path=str(Path(sft_config).resolve().relative_to(root)),
                       training_items_sha256=config['training_items_sha256'])]
    excluded = set(gold)
    catalog_sources, catalog_ids = _catalog_sources(catalog_dir, dict(catalog_roots or {'local': root}), gold, n_train)
    sources += catalog_sources
    excluded |= catalog_ids
    seq_source, seq_ids = _sequences_source(seqs._inside(root, sequences_dir), n_train)
    sources.append(seq_source)
    excluded |= seq_ids
    protected_text = set()
    for label in run_records:
        rec_source, rec_ids, questions = _record_source(seqs._inside(root, label), str(label), n_train)
        sources.append(rec_source)
        excluded |= rec_ids
        protected_text |= questions

    protected_text |= {seqs.question_hash(train[i]['question']) for i in excluded}
    test_text = {seqs.question_hash(data['test'][item['source_index']]['question']) for item in cohort_items}
    dropped = dict(duplicate_text_of_protected=[], matches_test_cohort=[], duplicate_text_within_pool=[])
    eligible, seen = [], set()
    for index, row in enumerate(train):
        if index in excluded:
            continue
        digest = seqs.question_hash(row['question'])
        reason = ('duplicate_text_of_protected' if digest in protected_text else
                  'matches_test_cohort' if digest in test_text else
                  'duplicate_text_within_pool' if digest in seen else None)
        if reason:
            dropped[reason].append(f'gsm8k:train:{index}')
            continue
        seen.add(digest)
        eligible.append(index)
    if len(eligible) < size:
        raise ValueError(f'only {len(eligible)} eligible train questions for a {size}-question set')
    chosen = sorted(random.Random(seed).sample(eligible, size))
    if set(chosen) & excluded:  # cannot happen by construction; never return an excluded ID regardless
        raise ValueError('sampler selected an excluded ID')
    limit = min(context, source['context_limit'])
    items = [gold_item(tokenizer, train[i], i, dataset_revision=source['dataset_revision'], context_limit=limit)
             for i in chosen]
    ids = [item['id'] for item in items]
    record = dict(
        schema=SCHEMA, seed=seed, size=size, split='train',
        selection_method='python-random-sample-without-replacement-sorted, random.Random(seed).sample(eligible)',
        dataset=dict(path=source['dataset_path'], config=source['dataset_config'],
                     revision=source['dataset_revision'], train_rows=n_train),
        model=source['model'], tokenizer_revision=source['tokenizer_revision'],
        builder='pilot_eval.sft.gold_item (gold prompt, tokens, completion-only labels, end-turn)',
        sources=sources,
        excluded=dict(by_id=len(excluded), ids_sha256=cat.ids_digest(f'gsm8k:train:{i}' for i in excluded),
                      **dropped),
        eligible=len(eligible), ids=ids, ids_sha256=cat.ids_digest(ids), items_sha256=_hash(items),
        overlap='none with every listed source; every source was read in full (an unreadable one stops the build)')
    return items, record


def freeze(output_dir, items, record):
    """Write ``validation.items.json`` and its exclusion record; an existing different set is refused."""
    if record['items_sha256'] != _hash(items) or record['ids'] != [i['id'] for i in items]:
        raise ValueError('exclusion record items_sha256/ids do not match the items')
    output_dir = Path(output_dir)
    paths = dict(items=output_dir / 'validation.items.json', record=output_dir / 'validation.exclusions.json')
    _save_frozen(paths['items'], items)
    _save_frozen(paths['record'], record)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--sft-config', required=True)
    parser.add_argument('--artifact-root', required=True)
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--sequences', required=True)
    parser.add_argument('--run-record', action='append', default=[])
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    items, record = build(args.sft_config, args.artifact_root, catalog_dir=args.catalog,
                          sequences_dir=args.sequences, run_records=args.run_record)
    paths = freeze(args.output, items, record)
    print(json.dumps(dict(ids_sha256=record['ids_sha256'], eligible=record['eligible'],
                          excluded=record['excluded']['by_id'], **{k: str(v) for k, v in paths.items()}), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
