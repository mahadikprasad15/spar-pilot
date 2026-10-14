"""Frozen raw Pilot 5 inputs. Tokenizer/data access is a CPU-only boundary."""
import itertools
import importlib.metadata
import json
import random
import shutil
from contextlib import contextmanager
from pathlib import Path

from pilot_eval.activation_prepare import _tokenize
from pilot_eval.run import _write_json, _write_state
from pilot_eval.sft import safe_name
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _hash, _save_frozen


class PinnedRawTokenizer:
    """Use the pinned tokenizer JSON directly, without importing model libraries."""

    def __init__(self, tokenizer_path, config_path):
        from tokenizers import Tokenizer
        self.engine = Tokenizer.from_file(str(tokenizer_path))
        token_data = json.loads(Path(tokenizer_path).read_text())
        if token_data.get('truncation') is not None or token_data.get('padding') is not None:
            raise ValueError('Pinned raw tokenizer must not contain automatic padding or truncation.')
        config = json.loads(Path(config_path).read_text())
        self.all_special_ids = sorted(token['id'] for token in token_data.get('added_tokens', [])
                                      if token.get('special'))
        special = {}
        for field in ('bos_token', 'eos_token', 'pad_token'):
            value = config.get(field)
            value = value.get('content') if isinstance(value, dict) else value
            token_id = self.engine.token_to_id(value) if value is not None else None
            if value is not None and (token_id is None or token_id not in self.all_special_ids):
                raise ValueError(f'Pinned tokenizer {field} disagrees with its special-token IDs.')
            special[field + '_id'] = token_id
        self.bos_token_id = special['bos_token_id']
        self.eos_token_id = special['eos_token_id']
        self.files = {'tokenizer.json': Path(tokenizer_path), 'tokenizer_config.json': Path(config_path)}
        self.metadata = dict(backend='tokenizers-json', version=importlib.metadata.version('tokenizers'),
            configured_add_bos_token=config.get('add_bos_token'), raw_add_special_tokens=False,
            file_sha256={name: file_hash(path) for name, path in self.files.items()})

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise ValueError('Raw corpus encoding must use add_special_tokens=False.')
        return self.engine.encode(text, add_special_tokens=False).ids

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=True):
        if add_special_tokens or not return_offsets_mapping:
            raise ValueError('Raw corpus requires explicit offsets and no inserted special tokens.')
        encoded = self.engine.encode(text, add_special_tokens=False)
        return {'input_ids': encoded.ids, 'offset_mapping': encoded.offsets}


class HFCorpusDependencies:
    """Pinned data/tokenizer access only. No model or Transformers imports."""

    def load_source(self, path, root):
        from pilot_eval.activation_prepare import load_prepared
        return load_prepared(path, root)

    def load_tokenizer(self, model, revision):
        from huggingface_hub import hf_hub_download
        print('Corpus: loading pinned tokenizer files (no model weights).', flush=True)
        return PinnedRawTokenizer(
            hf_hub_download(model, 'tokenizer.json', revision=revision),
            hf_hub_download(model, 'tokenizer_config.json', revision=revision))

    def stream_fineweb(self, config, revision):
        from datasets import load_dataset
        return load_dataset('HuggingFaceFW/fineweb', config, revision=revision,
                            split='train', streaming=True)


def _inside(root, path):
    path = Path(path)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Corpus artifact path escapes the selected artifact root.')
    return path


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    temporary.replace(path)


def _read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@contextmanager
def _preparing(directory, plan_dir):
    sealed = (plan_dir / 'complete.json').exists()
    if not sealed:
        _write_state(directory, 'running', 0, 200)
    try:
        yield
    except BaseException as exc:
        if not sealed:
            _write_state(directory, 'failed', 0, 200, str(exc))
            with (directory / 'logs/errors.jsonl').open('a') as stream:
                stream.write(json.dumps({'stage': 'corpus-preparation',
                    'type': type(exc).__name__, 'error': str(exc)}) + '\n')
        raise


def _raw(tokenizer, text):
    ids, offsets = _tokenize(tokenizer, text)
    special = set(tokenizer.all_special_ids)
    positions = [i for i, (token, span) in enumerate(zip(ids, offsets))
                 if token not in special and span[0] < span[1]]
    if len(positions) < 128:
        return None
    stop = positions[127]
    content = text[:offsets[stop][1]]
    prefix_ids, prefix_offsets = _tokenize(tokenizer, content)
    if prefix_ids != ids[:stop + 1] or prefix_offsets != offsets[:stop + 1]:
        raise ValueError('128-token raw prefix does not round-trip through tokenizer offsets.')
    return dict(content=content, content_sha256=_hash(content), input_ids=prefix_ids,
                offsets=[list(span) for span in prefix_offsets],
                attention_mask=[1] * len(prefix_ids), position_ids=list(range(len(prefix_ids))),
                content_positions=positions[:128], excluded_special_positions=[i for i, token in
                    enumerate(prefix_ids) if token in special],
                position_zero_kind='content' if positions[0] == 0 else 'excluded-special-or-empty',
                input_sha256=_hash({'input_ids': prefix_ids, 'content_positions': positions[:128]}))


def load_corpus(config_path, output_root):
    """Verify sealed corpus payloads and their input parents without network/tokenization."""
    root = Path(output_root).resolve()
    path = _inside(root, config_path)
    config = json.loads(path.read_text())
    if config.get('protocol_version') != 'pilot5-raw-corpus-v1':
        raise ValueError('Unsupported Pilot 5 corpus protocol.')
    marker = json.loads((path.parent / 'complete.json').read_text())
    if marker['config_sha256'] != _hash(config):
        raise ValueError('Corpus config hash mismatch.')
    required = {str(path.relative_to(root)), config['items_path'], config['pool_path'],
                config['run_path'] + '/meta/run_manifest.json'}
    required.update(row['path'] for row in config['tokenizer_files'].values())
    if not required.issubset(marker['files']):
        raise ValueError('Corpus completion marker omits required payloads.')
    for relative, expected in {**marker['files'], **config['source_files']}.items():
        payload = _inside(root, relative)
        if not payload.is_file() or file_hash(payload) != expected:
            raise ValueError(f'Corpus/source artifact hash mismatch: {relative}')
    rows = _read_rows(_inside(root, config['items_path']))
    if _hash(rows) != config['items_sha256']:
        raise ValueError('Corpus input identities changed.')
    return config, rows


def prepare_corpus(source_config, output_root, name, *, review, dependencies=None):
    """Prepare once; repeated calls verify existing inputs without sampling again."""
    if (review.get('schema_version') != 1 or review.get('reviewed') is not True
            or review.get('procedure') != 'same-pool-disjoint-random-v1'
            or not isinstance(review.get('seed'), int) or isinstance(review.get('seed'), bool)
            or not str(review.get('notes', '')).strip()
            or not str(review.get('prior_exposure', '')).strip()):
        raise ValueError('An explicitly reviewed corpus procedure, seed, notes and exposure record are required.')
    root = Path(output_root).resolve()
    name = safe_name(name)
    source_path = _inside(root, source_config)
    directory = root / 'runs' / 'pilot-5' / name / 'corpus'
    plan_dir = root / 'plans' / name
    path = plan_dir / 'traces.corpus.json'
    with run_lock(plan_dir), _preparing(directory, plan_dir):
        if (plan_dir / 'complete.json').exists():
            config, _ = load_corpus(path, root)
            if config['source_config'] != str(source_path.relative_to(root)) or config['review'] != review:
                raise ValueError('Frozen corpus selection/review changed; use a new named plan.')
            print('Corpus verified and reused; no tokenizer or dataset access.', flush=True)
            return path
        dependencies = dependencies or HFCorpusDependencies()
        print('Corpus: verifying existing input source.', flush=True)
        source, original = dependencies.load_source(source_path, root)
        source_rows = _inside(root, source['items_path'])
        if _hash(original) != source['items_sha256']:
            raise ValueError('Legacy source input hash mismatch.')
        legacy = [row for row in original if row['corpus'] == 'fineweb']
        if len(legacy) != 150 or len({row['source_index'] for row in legacy}) != 150:
            raise ValueError('Expected exactly 150 distinct legacy FineWeb identities.')
        if (source['fineweb']['dataset'] != 'HuggingFaceFW/fineweb'
                or source['fineweb']['source_indices'] != [row['source_index'] for row in legacy]
                or source['fineweb']['pool_limit'] != 2000
                or source['fineweb']['content_token_limit'] != 128):
            raise ValueError('Legacy FineWeb pool or membership differs from the reviewed source.')
        tokenizer = dependencies.load_tokenizer(source['model'], source['tokenizer_revision'])
        legacy_by_index = {row['source_index']: row for row in legacy}
        found, candidates, pool = {}, [], []
        seen_ids, seen_texts = set(), set()
        legacy_ids = {row.get('source_id') for row in legacy if row.get('source_id') is not None}
        legacy_texts = {row['full_document_sha256'] for row in legacy}
        limit = source['fineweb']['pool_limit']
        print(f'Corpus: replaying pinned FineWeb pool, up to {limit} documents.', flush=True)
        for index, doc in enumerate(itertools.islice(dependencies.stream_fineweb(
                source['fineweb']['config'], source['fineweb']['revision']), limit)):
            text_hash, source_id = _hash(doc['text']), doc.get('id')
            identity = dict(source_index=index, source_id=source_id, full_document_sha256=text_hash)
            if index in legacy_by_index:
                old = legacy_by_index[index]
                if old['full_document_sha256'] != text_hash or old.get('source_id') != source_id:
                    raise ValueError(f'Legacy document identity changed: {old["id"]}')
                raw = _raw(tokenizer, old['content'])
                full_raw = _raw(tokenizer, doc['text'])
                if (raw is None or full_raw is None or full_raw['content'] != old['content']
                        or raw['content_sha256'] != old['content_sha256'] or [
                        raw['input_ids'][position] for position in raw['content_positions']] != old['prefix_token_ids']):
                    raise ValueError(f'Legacy raw-token mapping changed: {old["id"]}')
                found[index] = dict(raw, **identity, id=old['id'], stratum='legacy-150',
                    source_url=old.get('source_url'), prior_exposure=review['prior_exposure'])
                identity['eligibility'] = 'legacy'
            elif text_hash in legacy_texts or text_hash in seen_texts or (
                    source_id is not None and (source_id in legacy_ids or source_id in seen_ids)):
                identity['eligibility'] = 'duplicate'
            else:
                raw = _raw(tokenizer, doc['text'])
                identity['eligibility'] = 'eligible' if raw is not None else 'too-short'
                if raw is not None:
                    candidates.append(dict(raw, **{k: v for k, v in identity.items() if k != 'eligibility'},
                        id=f'fineweb:train:{index}', stratum='new-50', source_url=doc.get('url'),
                        prior_exposure=review['prior_exposure']))
            seen_texts.add(text_hash)
            if source_id is not None:
                seen_ids.add(source_id)
            pool.append(identity)
            if (index + 1) % 250 == 0:
                print(f'Corpus pool: {index + 1}/{limit}; legacy {len(found)}/150; candidates {len(candidates)}.', flush=True)
        if len(pool) != limit:
            raise ValueError(f'Pinned FineWeb pool ended at {len(pool)} documents; expected {limit}.')
        if len(found) != 150:
            raise ValueError('Pinned stream did not recover all legacy documents; no replacements allowed.')
        if len(candidates) < 50:
            raise ValueError('Pinned pool has fewer than 50 disjoint eligible documents.')
        chosen = sorted(random.Random(review['seed']).sample(range(len(candidates)), 50))
        rows = [found[row['source_index']] for row in legacy] + [candidates[i] for i in chosen]
        items = directory / 'inputs/documents.jsonl'
        pool_path = directory / 'inputs/pool.jsonl'
        _write_rows(items, rows)
        _write_rows(pool_path, pool)
        source_files = {str(p.relative_to(root)): file_hash(p) for p in [source_path, source_rows]}
        metadata = dict(getattr(tokenizer, 'metadata', {}),
                        all_special_ids=sorted(tokenizer.all_special_ids),
                        bos_token_id=getattr(tokenizer, 'bos_token_id', None),
                        eos_token_id=getattr(tokenizer, 'eos_token_id', None), add_special_tokens=False)
        tokenizer_files = {}
        for filename, original_path in getattr(tokenizer, 'files', {}).items():
            if Path(filename).name != filename:
                raise ValueError('Tokenizer artifact filename must be a basename.')
            retained = directory / 'inputs/tokenizer' / filename
            retained.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original_path, retained)
            tokenizer_files[filename] = {'path': str(retained.relative_to(root)), 'sha256': file_hash(retained)}
        config = dict(schema_version=1, protocol_version='pilot5-raw-corpus-v1', status='prepared',
            name=name, form='raw', model=source['model'], model_revision=source['model_revision'],
            tokenizer_revision=source['tokenizer_revision'], tokenizer=metadata, fineweb=source['fineweb'],
            tokenizer_files=tokenizer_files,
            source_config=str(source_path.relative_to(root)), source_files=source_files, review=review,
            items_path=str(items.relative_to(root)), items_sha256=_hash(rows),
            pool_path=str(pool_path.relative_to(root)), run_path=str(directory.relative_to(root)),
            counts={'legacy-150': 150, 'new-50': 50}, content_tokens=128,
            selected_new_indices=[row['source_index'] for row in rows[150:]])
        _save_frozen(path, config)
        _save_frozen(directory / 'meta/run_manifest.json', config)
        payloads = [path, items, pool_path, directory / 'meta/run_manifest.json']
        payloads.extend(root / row['path'] for row in tokenizer_files.values())
        _write_json(plan_dir / 'complete.json', dict(config_sha256=_hash(config),
            files={str(p.relative_to(root)): file_hash(p) for p in payloads}))
        _write_state(directory, 'completed', 200, 200)
        print('Corpus prepared: 150 legacy + 50 new; 128 content tokens each. No model inference.', flush=True)
        return path
