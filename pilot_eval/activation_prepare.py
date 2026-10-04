"""Frozen activation inputs and their CPU-only audit; no model weights required."""

import itertools
import json
import random
from contextlib import contextmanager
from pathlib import Path

from pilot_eval.protocol import build_gsm8k_prompt, gsm8k_messages
from pilot_eval.run import _write_json
from pilot_eval.sft import read_artifact, safe_name
from pilot_eval.training import file_hash, run_lock, training_directory, verified_checkpoints
from pilot_eval.workflow import HFDependencies, _hash, _save_frozen

STEPS = [0, 8, 16, 32, 64]
PROJECTIONS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
VIEWS = ['question', 'solution', 'user']


@contextmanager
def _preparation_status(plan):
    """Record failures only for owned, unfinished preparations."""
    already_complete = (plan / 'prepare-complete.json').exists()
    if not already_complete:
        _write_json(plan / 'meta/status.json', {'state': 'running', 'stage': 'prepare'})
    try:
        yield
        if not already_complete:
            _write_json(plan / 'meta/status.json', {'state': 'prepared', 'stage': 'prepare'})
    except BaseException as exc:
        if not already_complete:
            _write_json(plan / 'meta/status.json', {'state': 'failed', 'stage': 'prepare',
                                                  'error': str(exc)})
            log = plan / 'logs/errors.jsonl'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('a') as stream:
                stream.write(json.dumps({'type': type(exc).__name__, 'error': str(exc)}) + '\n')
        raise


class ActivationInputDependencies(HFDependencies):
    def stream_fineweb(self, config, revision):
        from datasets import load_dataset
        return load_dataset('HuggingFaceFW/fineweb', config, revision=revision,
                            split='train', streaming=True)


def _source(source_path, root):
    config = json.loads(Path(source_path).read_text())
    safe_name(config.get('run_id', ''))
    if (config.get('protocol_version') != 'pilot2-sft-v1'
            or config.get('model') != 'Qwen/Qwen2.5-1.5B-Instruct'
            or config.get('dtype') != 'float32' or config.get('checkpoint_steps') != STEPS):
        raise ValueError('source is not the required completed Pilot 2 FP32 run')
    directory = training_directory(root, config)
    manifest = json.loads((directory / 'meta/run_manifest.json').read_text())
    if manifest['config'] != config:
        raise ValueError('source training manifest/config mismatch')
    result = json.loads((directory / 'results/results.json').read_text())
    preflight = json.loads((directory / 'results/preflight.json').read_text())
    base = result['base_sha256_before']
    if (base != result['base_sha256_after'] or base != preflight['base_sha256']
            or result.get('successful_steps') != 64):
        raise ValueError('source frozen-base hashes or completion evidence mismatch')
    checkpoints = verified_checkpoints(directory)
    if sorted(checkpoints) != STEPS:
        raise ValueError('source lacks all five verified checkpoints')
    evidence = {'base_sha256': base, 'source_config_sha256': _hash(config),
                'checkpoints': {}, 'files': {}}
    for path in [directory / 'meta/run_manifest.json', directory / 'meta/runtime.json',
                 directory / 'results/results.json', directory / 'results/preflight.json']:
        evidence['files'][str(path.relative_to(root))] = file_hash(path)
    for step, checkpoint in sorted(checkpoints.items()):
        adapter = json.loads((checkpoint / 'adapter_config.json').read_text())
        targets = adapter.get('target_modules')
        if (adapter.get('r') != 1 or adapter.get('lora_alpha') != 1
                or adapter.get('lora_dropout') != 0
                or (targets != 'all-linear' and set(targets or []) != set(PROJECTIONS))):
            raise ValueError('source adapter does not match rank-1 all-projection protocol')
        evidence['checkpoints'][str(step)] = {
            'path': str(checkpoint.relative_to(root)),
            'complete_sha256': file_hash(checkpoint / 'complete.json'),
            'adapter_sha256': file_hash(checkpoint / 'adapter_model.safetensors')}
    source = config['source_config']
    if any(config.get(key) != source.get(key) or not config.get(key)
           for key in ['model', 'model_revision', 'tokenizer_revision']):
        raise ValueError('source model/tokenizer pin mismatch')
    items = read_artifact(root, config['evaluation_items_path'])
    if (len(items) != 150 or len({row['id'] for row in items}) != 150
            or _hash(items) != config['evaluation_items_sha256']
            or _hash(items) != source['items_sha256']
            or [row['source_index'] for row in items] != source['prompt_indices']
            or [row['id'] for row in items] != [f'gsm8k:test:{i}' for i in source['prompt_indices']]):
        raise ValueError('source held-out cohort identity/hash mismatch')
    return config, items, evidence


def _tokenize(tokenizer, rendered):
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    tokens, offsets = list(encoded['input_ids']), list(encoded['offset_mapping'])
    if tokens != tokenizer.encode(rendered, add_special_tokens=False) or len(tokens) != len(offsets):
        raise ValueError('tokenizer offsets do not match final input IDs')
    if any(not 0 <= start <= end <= len(rendered) for start, end in offsets):
        raise ValueError('invalid tokenizer character offsets')
    return tokens, offsets


def _record(tokenizer, rendered, spans, context, **metadata):
    tokens, offsets = _tokenize(tokenizer, rendered)
    if len(tokens) > context:
        raise ValueError('complete measurement sequence context overflow; truncation forbidden')
    masks = {view: [False] * len(tokens) for view in VIEWS}
    ambiguous = {view: [] for view in VIEWS}
    for index, (start, end) in enumerate(offsets):
        if start == end:
            continue
        for view, (left, right) in spans.items():
            if left <= start < end <= right:
                masks[view][index] = True
            elif start < right and end > left:
                ambiguous[view].append(index)
    if any(sum(mask[index] for mask in masks.values()) > 1 for index in range(len(tokens))):
        raise ValueError('content masks overlap')
    return {**metadata, 'rendered': rendered, 'input_ids': tokens,
            'attention_mask': [1] * len(tokens), 'masks': masks,
            'content_spans': {view: list(span) for view, span in spans.items()},
            'counts': {view: sum(mask) for view, mask in masks.items()},
            'ambiguous_positions': ambiguous, 'offsets': [list(pair) for pair in offsets],
            'input_sha256': _hash({'tokens': tokens, 'masks': masks})}


def _audit(rows, manifest):
    lines = ['# Pilot 3 prepared input audit', '',
             'Prepared inputs only: no activation measurement or GPU validation has run.', '',
             f"Model: {manifest['model']} ({manifest['model_revision']})",
             f"FineWeb: {manifest['fineweb']['config']} ({manifest['fineweb']['revision']})",
             'Checkpoints: 0, 8, 16, 32, 64', '',
             '| Corpus | Examples | Longest input tokens | Counted tokens |',
             '|---|---:|---:|---:|']
    for corpus in ['gsm8k', 'fineweb']:
        subset = [row for row in rows if row['corpus'] == corpus]
        counts = {view: sum(row['counts'][view] for row in subset) for view in VIEWS}
        lines.append(f"| {corpus} | {len(subset)} | {max(len(row['input_ids']) for row in subset)} | {counts} |")
    for corpus in ['gsm8k', 'fineweb']:
        subset = [row for row in rows if row['corpus'] == corpus]
        samples = {row['id']: row for row in [subset[0], max(subset, key=lambda r: len(r['input_ids']))]}
        for row in samples.values():
            lines.extend(['', f"## {row['id']}", '', 'Rendered input:', '', '```text',
                          row['rendered'], '```'])
            for view, span in row['content_spans'].items():
                counted = ''.join(row['rendered'][start:end] for (start, end), chosen in
                                  zip(row['offsets'], row['masks'][view]) if chosen)
                lines.extend(['', f'Counted {view}:', '', '```text', counted, '```',
                              f"Count: {row['counts'][view]}; excluded boundary positions: {row['ambiguous_positions'][view]}"])
    lines.extend(['', 'See frozen records for every token ID, offset and mask.',
                  'FineWeb is sampled from a bounded stream prefix; cross-input comparisons remain descriptive.'])
    return '\n'.join(lines) + '\n'


def load_prepared(config_path, output_root):
    """Verify complete prepared payloads and current source evidence; no network."""
    root = Path(output_root).resolve()
    path = Path(config_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('prepared configuration must be within the artifact root')
    config = json.loads(path.read_text())
    if config.get('protocol_version') != 'pilot3-write-v1':
        raise ValueError('unsupported activation protocol')
    marker = json.loads((path.parent / 'prepare-complete.json').read_text())
    if marker['config_sha256'] != _hash(config):
        raise ValueError('prepared config hash mismatch')
    required = {str(path.relative_to(root)), config['items_path'], config['audit_path'],
                config['run_path'] + '/meta/run_manifest.json'}
    if not required.issubset(marker['files']):
        raise ValueError('preparation completion marker omits required payloads')
    for relative, digest in marker['files'].items():
        payload = (root / relative).resolve()
        if not payload.is_relative_to(root) or file_hash(payload) != digest:
            raise ValueError('prepared artifact hash mismatch')
    source_path = root / config['source_config_path']
    _, _, evidence = _source(source_path, root)
    if evidence != config['source_evidence']:
        raise ValueError('source evidence differs from frozen preparation')
    rows = read_artifact(root, config['items_path'])
    if _hash(rows) != config['items_sha256']:
        raise ValueError('prepared input hash mismatch')
    return config, rows


def prepare_activation(source_config, output_root, name, *, dependencies=None,
                       fineweb_config='sample-10BT', fineweb_revision=None):
    """Freeze complete task/control sequences and auditable tokenizer masks."""
    root = Path(output_root).resolve()
    name = safe_name(name)
    plan = root / 'plans' / name
    path = plan / 'activation.prepared.json'
    source_path = Path(source_config).resolve()
    if not source_path.is_relative_to(root):
        raise ValueError('source config must resolve within the artifact root')
    with run_lock(plan), _preparation_status(plan):
        if (plan / 'prepare-complete.json').exists():
            config, _ = load_prepared(path, root)
            if (config['source_config_path'] != str(source_path.relative_to(root))
                    or config['fineweb']['config'] != fineweb_config
                    or (fineweb_revision and config['fineweb']['revision'] != fineweb_revision)):
                raise ValueError('prepared plan mismatch; use a new named plan')
            return path
        config, items, evidence = _source(source_path, root)
        deps = dependencies or ActivationInputDependencies()
        tokenizer, context = deps.load_tokenizer(config['model'], config['tokenizer_revision'])
        source = config['source_config']
        if _hash(tokenizer.chat_template) != source['chat_template_sha256']:
            raise ValueError('source chat template mismatch')
        context = min(context, source['context_limit'])
        data = deps.load_dataset(source['dataset_path'], source['dataset_config'], source['dataset_revision'])
        rows = []
        for item in items:
            original = data['test'][item['source_index']]
            question, gold = original['question'], original['answer']
            prompt = build_gsm8k_prompt(question, tokenizer)
            if prompt != item['prompt'] or gold != item['gold']:
                raise ValueError('source prompt/gold differs from pinned held-out dataset')
            rendered = tokenizer.apply_chat_template(
                gsm8k_messages(question) + [{'role': 'assistant', 'content': gold}],
                tokenize=False, add_generation_prompt=False)
            if not rendered.startswith(prompt):
                raise ValueError('teacher-forced input does not preserve source prompt')
            user_content = gsm8k_messages(question)[0]['content']
            user_start = rendered.find(user_content)
            gold_start = len(prompt)
            if user_start < 0 or rendered[gold_start:gold_start + len(gold)] != gold:
                raise ValueError('rendered content cannot be mapped unambiguously')
            question_start = user_start + len(user_content) - len(question)
            rows.append(_record(tokenizer, rendered,
                {'question': (question_start, question_start + len(question)),
                 'solution': (gold_start, gold_start + len(gold))}, context,
                id=item['id'], corpus='gsm8k', source_index=item['source_index'],
                source_split='test', question=question, gold=gold,
                dataset_revision=source['dataset_revision']))
        revision = fineweb_revision or deps.resolve('HuggingFaceFW/fineweb', 'dataset')
        if not revision:
            raise ValueError('FineWeb revision is missing')
        eligible = []
        for index, doc in enumerate(itertools.islice(deps.stream_fineweb(fineweb_config, revision), 2000)):
            text = doc['text']
            tokens, offsets = _tokenize(tokenizer, text)
            if len(tokens) >= 128:
                prefix = text[:offsets[127][1]]
                eligible.append((index, doc, prefix, tokens[:128]))
        if len(eligible) < 150:
            raise ValueError('FineWeb prefix has fewer than 150 eligible documents')
        selected = sorted(random.Random(42).sample(range(len(eligible)), 150))
        for selected_index in selected:
            index, doc, content, prefix_tokens = eligible[selected_index]
            rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
                                                     tokenize=False, add_generation_prompt=False)
            start = rendered.find(content)
            if start < 0:
                raise ValueError('FineWeb rendered content cannot be mapped')
            rows.append(_record(tokenizer, rendered, {'user': (start, start + len(content))}, context,
                id=f'fineweb:train:{index}', corpus='fineweb', source_index=index,
                source_id=doc.get('id'), source_url=doc.get('url'), source_split='train',
                dataset_revision=revision, content=content, content_sha256=_hash(content),
                full_document_sha256=_hash(doc['text']), prefix_token_ids=prefix_tokens))
        run_dir = root / 'runs' / 'pilot-3' / config['model'].replace('/', '--') / \
            'gsm8k-fineweb' / 'heldout150-control150-seed42' / 'float32' / name
        inputs = run_dir / 'inputs/items.json'
        audit = run_dir / 'results/input-audit.md'
        manifest = {'protocol_version': 'pilot3-write-v1', 'schema_version': 1, 'run_id': name,
            'state': 'prepared', 'model': config['model'], 'model_revision': config['model_revision'],
            'tokenizer_revision': config['tokenizer_revision'], 'chat_template': tokenizer.chat_template,
            'chat_template_sha256': source['chat_template_sha256'], 'dtype': 'float32',
            'seed': 42, 'checkpoint_steps': STEPS, 'views': VIEWS, 'context_limit': context,
            'source_config_path': str(source_path.relative_to(root)), 'source_evidence': evidence,
            'fineweb': {'dataset': 'HuggingFaceFW/fineweb', 'config': fineweb_config,
                        'revision': revision, 'pool_limit': 2000, 'eligible_count': len(eligible),
                        'count': 150, 'content_token_limit': 128,
                        'selection': 'python-random-sample-without-replacement-sorted',
                        'source_indices': [eligible[i][0] for i in selected]},
            'items_path': str(inputs.relative_to(root)), 'items_sha256': _hash(rows),
            'audit_path': str(audit.relative_to(root)), 'run_path': str(run_dir.relative_to(root)),
            'execution': 'forward-only', 'scorer': None, 'decoding': None}
        _save_frozen(inputs, rows)
        _save_frozen(path, manifest)
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text(_audit(rows, manifest))
        _write_json(run_dir / 'meta/status.json', {'state': 'prepared', 'examples': len(rows)})
        _save_frozen(run_dir / 'meta/run_manifest.json', manifest)
        files = [inputs, audit, path, run_dir / 'meta/run_manifest.json']
        _write_json(plan / 'prepare-complete.json', {
            'config_sha256': _hash(manifest),
            'files': {str(p.relative_to(root)): file_hash(p) for p in files}})
        print(f'prepared: {len(rows)} fixed inputs; audit: {audit}', flush=True)
        return path
