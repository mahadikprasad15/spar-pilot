"""Outcome-blind token normalization and task-reference construction."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata


class TokenInventory:
    """Explicit ID-to-decoded-string inventory; no model or network access."""

    def __init__(self, decoded, *, added_ids=(), special_ids=()):
        self.decoded = dict(decoded)
        self.excluded = set(added_ids) | set(special_ids)
        self.normalized = {}
        for token_id, text in self.decoded.items():
            value = unicodedata.normalize('NFKC', text).strip().lower()
            self.normalized[token_id] = (value if token_id not in self.excluded
                and value and '\ufffd' not in value else None)

    def normalize(self, token_id):
        if token_id not in self.normalized:
            raise ValueError(f'Token ID {token_id} is absent from the pinned inventory.')
        return self.normalized[token_id]

    def count(self, token_ids):
        return dict(Counter(value for token in token_ids
                            if (value := self.normalize(token)) is not None))


def log_odds(y_a, y_b, n_a, n_b, alpha0=1000):
    """Specified informative-prior weighted log-odds z, in Python FP64."""
    if (not math.isfinite(alpha0) or alpha0 <= 0 or n_a <= 0 or n_b <= 0
            or not 0 <= y_a <= n_a or not 0 <= y_b <= n_b or y_a + y_b == 0):
        raise ValueError('Invalid log-odds counts or prior mass.')
    prior = alpha0 * (y_a + y_b) / (n_a + n_b)
    a, b = y_a + prior, y_b + prior
    other_a, other_b = n_a + alpha0 - a, n_b + alpha0 - b
    if min(a, b, other_a, other_b) <= 0:
        raise ValueError('Log-odds requires nondegenerate token support.')
    return (math.log(a / other_a) - math.log(b / other_b)) / math.sqrt(1/a + 1/b)


def build_reference(task_counts, background_counts, *, alpha0=1000, size=200):
    """Build a reference from normalized occurrence counts, without vector inputs."""
    for counts in (task_counts, background_counts):
        if (not isinstance(counts, dict) or not counts or any(
                not isinstance(word, str) or not word or type(count) is not int or count < 0
                for word, count in counts.items())):
            raise ValueError('Reference counts must be nonnegative integer occurrence counts.')
    if type(size) is not int or size <= 0:
        raise ValueError('Reference size must be positive.')
    n_a, n_b = sum(task_counts.values()), sum(background_counts.values())
    scores = []
    for word in sorted(set(task_counts) | set(background_counts)):
        a, b = task_counts.get(word, 0), background_counts.get(word, 0)
        if a + b:
            scores.append(dict(string=word, y_a=a, y_b=b,
                               z=log_odds(a, b, n_a, n_b, alpha0)))
    scores.sort(key=lambda row: (-row['z'], -row['y_a'], row['string']))
    selected = [row['string'] for row in scores if row['y_a'] >= 5 and row['z'] > 0][:size]
    words = [value for value in selected if re.search(r'[^\W\d_]', value)]
    return dict(strings=selected, word=words,
                format_number=[value for value in selected if value not in set(words)],
                scores=scores, n_a=n_a, n_b=n_b, alpha0=alpha0,
                minimum_count=5, requested_size=size, short_reference=len(selected) < 100,
                tie_rule='descending-z,descending-y_a,ascending-string')


def freeze_references(directory, references, *, parameters, input_files):
    """Seal references, exact parameters and byte hashes of their source inputs."""
    from pilot_eval.training import run_lock, file_hash
    from pilot_eval.workflow import _save_frozen, _hash
    directory = Path(directory).resolve()
    if set(references) != {'ref_questions', 'ref_gold', 'ref_grpo'}:
        raise ValueError('Freeze requires all three named references.')
    parents = {name: dict(path=str(Path(path).resolve()), sha256=file_hash(Path(path)))
               for name, path in input_files.items()}
    payload = dict(schema_version=1, protocol='pilot5-overlap-v1.1',
                   references=references, parameters=parameters, inputs=parents)
    with run_lock(directory):
        path = directory/'freeze.json'
        _save_frozen(path, payload)
        _save_frozen(directory/'freeze-seal.json', dict(sha256=file_hash(path), content_sha256=_hash(payload)))
    load_references(path)
    return path


def load_references(path):
    """Verify a reference freeze before a supported scoring operation."""
    from pilot_eval.training import file_hash
    from pilot_eval.workflow import _hash
    path = Path(path).resolve()
    if not path.is_file() or not (path.parent/'freeze-seal.json').is_file():
        raise ValueError('A valid reference freeze is required before scoring.')
    seal = json.loads((path.parent/'freeze-seal.json').read_text())
    if file_hash(path) != seal['sha256']:
        raise ValueError('Reference freeze byte hash mismatch.')
    payload = json.loads(path.read_text())
    if _hash(payload) != seal['content_sha256'] or payload.get('protocol') != 'pilot5-overlap-v1.1':
        raise ValueError('Reference freeze content hash/protocol mismatch.')
    for item in payload['inputs'].values():
        source = Path(item['path'])
        if not source.is_file() or file_hash(source) != item['sha256']:
            raise ValueError('Reference parent hash mismatch.')
    return payload


def prepare_background(directory, documents, tokenizer, *, measurement_indices,
                       tokenizer_identity, scan_limit=10000, token_target=2000000,
                       seed=20261019, chunk_size=250):
    """Bounded, resumable source scan and exact raw-token background selection.

    On resume the source stream is replayed to the cursor, but sealed chunks are
    not retokenized. No learned vectors, lens tokens or model libraries are used.
    """
    import itertools
    import random
    from pilot_eval.training import file_hash, run_lock
    from pilot_eval.workflow import _hash, _save_frozen
    from pilot_eval.run import _write_json, _write_state
    directory = Path(directory).resolve()
    if (any(type(value) is not int or value <= 0 for value in (scan_limit,token_target,chunk_size))
            or type(seed) is not int or not measurement_indices
            or any(type(index) is not int or not 0 <= index < scan_limit for index in measurement_indices)):
        raise ValueError('Declare a bounded scan containing every measurement source index.')
    config = dict(schema_version=1, scan_limit=scan_limit, token_target=token_target,
        seed=seed, chunk_size=chunk_size, measurement_indices=sorted(set(measurement_indices)),
        tokenizer=tokenizer_identity, minimum_raw_tokens=64, maximum_document_tokens=1024,
        deduplication="sha256(' '.join(text.split()).encode('utf-8'))")
    with run_lock(directory):
        _save_frozen(directory/'config.json', config)
        complete=directory/'complete.json'
        if complete.exists():
            seal=json.loads(complete.read_text())
            for name,expected in seal['files'].items():
                if file_hash(directory/name)!=expected:
                    raise ValueError('Background payload hash mismatch.')
            return json.loads((directory/'results/background.json').read_text())
        _write_state(directory,'running',0,scan_limit)
        try:
            chunks=directory/'chunks';chunks.mkdir(exist_ok=True)
            scanned=[]
            for path in sorted(chunks.glob('*.json')):
                if path.name.endswith('.seal.json'): continue
                seal=path.with_suffix('.seal.json')
                if not seal.exists(): break  # interrupted unpublished chunk is safely replayed
                if file_hash(path)!=json.loads(seal.read_text())['sha256']:
                    raise ValueError('Background scan chunk hash mismatch.')
                block=json.loads(path.read_text())
                if [r['source_index'] for r in block]!=list(range(len(scanned),len(scanned)+len(block))):
                    raise ValueError('Background scan chunks have a gap or reordered cursor.')
                scanned.extend(block)
            print(f'Background scan: restoring {len(scanned)}/{scan_limit} source documents.',flush=True)
            stream=itertools.islice(documents,len(scanned),scan_limit)
            while len(scanned)<scan_limit:
                raw=list(itertools.islice(stream,min(chunk_size,scan_limit-len(scanned))))
                if not raw: raise ValueError('Pinned stream ended before the declared scan bound.')
                block=[]
                for offset,doc in enumerate(raw):
                    text=doc['text'];ids=tokenizer.encode(text,add_special_tokens=False)
                    block.append(dict(source_index=len(scanned)+offset,source_id=doc.get('id'),
                        full_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                        normalized_text_sha256=hashlib.sha256(' '.join(text.split()).encode()).hexdigest(),
                        token_ids=ids[:1024],raw_token_count=len(ids)))
                path=chunks/f'{len(scanned):06d}.json'
                _write_json(path,block)
                _write_json(path.with_suffix('.seal.json'),{'sha256':file_hash(path)})
                scanned.extend(block)
                _write_state(directory,'running',len(scanned),scan_limit)
                print(f'Background scan: {len(scanned)}/{scan_limit} documents sealed.',flush=True)
            excluded={scanned[i]['normalized_text_sha256'] for i in config['measurement_indices']}
            excluded_ids={scanned[i]['source_id'] for i in config['measurement_indices']
                          if scanned[i]['source_id'] is not None}
            seen=set();eligible=[]
            for row in scanned:
                key=row['normalized_text_sha256']
                if (row['source_index'] in config['measurement_indices'] or key in excluded
                        or row['source_id'] is not None and row['source_id'] in excluded_ids): continue
                if key in seen:continue
                seen.add(key)
                if row['raw_token_count']>=64:eligible.append(row)
            random.Random(seed).shuffle(eligible)
            selected=[];count=0
            for row in eligible:
                take=min(len(row['token_ids']),token_target-count)
                selected.append(dict(row,token_ids=row['token_ids'][:take],retained_tokens=take))
                count+=take
                if count==token_target:break
            if count!=token_target:
                raise ValueError('Bounded background pool cannot supply the approved token target; review required.')
            result=dict(config=config,documents=selected,token_count=count,
                        selected_indices=[row['source_index'] for row in selected],
                        selection_sha256=_hash([row['source_index'] for row in selected]))
            _save_frozen(directory/'results/background.json',result)
            payloads=[directory/'config.json',directory/'results/background.json',*sorted(chunks.glob('*.json'))]
            _write_json(complete,{'files':{str(p.relative_to(directory)):file_hash(p) for p in payloads}})
            _write_state(directory,'completed',scan_limit,scan_limit)
            return result
        except BaseException as exc:
            _write_state(directory,'failed',len(scanned) if 'scanned' in locals() else 0,scan_limit,str(exc))
            raise
