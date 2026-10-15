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
