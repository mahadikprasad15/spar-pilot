"""Ticket 02 of the joint neuron/adapter audit: frozen token sequences, masks and leakage report.

Reads a sealed ticket-01 inventory, pinned local GSM8K/UltraChat parquet files, the pinned
tokenizer files and the arms' lineage files. Writes sealed per-example token IDs and
position masks plus leakage, filter, review and decode reports. No model, Torch, GPU or
network access is used.

Formatting decisions (user-approved 2026-11-10):
- math: ``protocol.gsm8k_messages(question)`` as the user turn and the GSM8K reference
  answer as the assistant turn (the Gold SFT training format);
- background: first user/assistant turn of HuggingFaceH4/ultrachat_200k records, default
  Qwen system prompt from the chat template, same as the math rows.

Masks are position lists into ``input_ids``. An activation at position t is the state
after processing token t.
- ``response_content``: tokens wholly inside the assistant content characters.
- ``reply_start``: the last assistant-header token, which must contain the header newline.
- ``termination``: the last content token and the ``<|im_end|>`` token.
Padding is not part of a frozen sequence; batches and padding are frozen in 03b.
Prompt, header/template, EOS and trailing-newline tokens are in no mask. A token whose
character span crosses a content boundary makes the example ambiguous; it is rejected.

Usage:
    python -m pilot_eval.neuron_audit_sequences prepare --plan <plan.json> --artifact_root <root> \
        --output_root <out> --run_id <id>
    python -m pilot_eval.neuron_audit_sequences verify --run_dir <dir>
    python -m pilot_eval.neuron_audit_sequences review --run_dir <dir> --reviewer <name> \
        --decision <accept|reject> [--notes <text>]
"""
import argparse
import hashlib
import json
import os
import random
import re
import socket
import sys
import time
from pathlib import Path

from pilot_eval.protocol import gsm8k_messages
from pilot_eval.training import file_hash

PLAN_SCHEMA = 'joint-neuron-adapter-audit/sequences-plan-v1'
MANIFEST_SCHEMA = 'joint-neuron-adapter-audit/sequences-v1'
RUN_SUBDIR = 'runs/joint-neuron-adapter-audit/sequences'
SPLITS = ['discovery', 'heldout', 'calibration']
GROUPS = ['math', 'general', 'near_miss']
ID_PATTERN = re.compile(r'gsm8k:(train|test):(\d+)')

# Proposed, documented background filter. Applied to the first-turn prompt and response.
CODE_PATTERN = re.compile(
    r"```|\bdef \w+\(|\bclass \w+\s*[:(]|#include|^\s*(?:import|from) \w+|\bfunction\s*\w*\s*\(|=>"
    r"|</?[a-zA-Z][\w-]*[^>]{0,40}>|\bSELECT\b|printf\(|console\.log|\w+\(\)\s*[;{]", re.M)
MATH_PATTERN = re.compile(
    r"\d\s*[-+*/×÷^=<>%]\s*\d|\\frac|\\\(|\\\[|\$[^$\n]+\$|\\sum|\b(?:calculat\w*|equation\w*|solv\w*"
    r"|arithmetic|algebra\w*|geometr\w*|probabilit\w*|percent\w*|fraction\w*|integer\w*|sum of|product of"
    r"|average|how many|how much|formula\w*|math\w*|multipl\w*|divid\w*|subtract\w*|theorem\w*"
    r"|derivative\w*|integral\w*|equal\w*)\b", re.I)
DIGIT = re.compile(r'\d')
FILTER_DOCUMENTATION = dict(
    status='proposed-pending-human-review',
    order=['code', 'math', 'near_miss', 'digit-prompt-only', 'general'],
    code=CODE_PATTERN.pattern, math=MATH_PATTERN.pattern,
    near_miss='passes code and math filters and the response contains an ASCII digit',
    digit_prompt_only='passes code and math filters; digit only in the prompt (excluded from both groups)',
    general='passes code and math filters; no ASCII digit in prompt or response',
    confound='Filtering does not remove source, style, length or authorship differences from GSM8K math.')


# --- tokenizer ------------------------------------------------------------------------------

class ChatTokenizer:
    """Pinned Qwen tokenizer files rendered with the saved chat template; no transformers import."""

    def __init__(self, directory, expected_sha256=None):
        import tokenizers
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        self.directory = Path(directory)
        self.files = {n: file_hash(self.directory / n) for n in ['tokenizer.json', 'tokenizer_config.json']}
        for name, digest in (expected_sha256 or {}).items():
            if name in self.files and self.files[name] != digest:
                raise ValueError(f'Pinned tokenizer file {name} changed.')
        config = json.loads((self.directory / 'tokenizer_config.json').read_text())
        # Matches transformers' chat-template environment options.
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        self.template = environment.from_string(config['chat_template'])
        self.tokenizer = tokenizers.Tokenizer.from_file(str(self.directory / 'tokenizer.json'))
        self.eos_token = config['eos_token']
        self.eos_token_id = self.tokenizer.token_to_id(self.eos_token)
        # The pinned config.json has no pad_token_id; the tokenizer's pad token is what training padded with.
        self.pad_token = config.get('pad_token')
        self.pad_token_id = self.tokenizer.token_to_id(self.pad_token) if isinstance(self.pad_token, str) else None

    def render(self, messages, add_generation_prompt):
        return self.template.render(messages=messages, add_generation_prompt=add_generation_prompt, tools=None)

    def encode(self, text):
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        return list(encoding.ids), [tuple(o) for o in encoding.offsets]

    def token_text(self, token_id):
        return self.tokenizer.decode([token_id], skip_special_tokens=False)

    def identity(self):
        return dict(kind='tokenizers+jinja2', files=self.files)


def _default_tokenizer(spec, root):
    return ChatTokenizer(_inside(root, spec['dir']), spec.get('expected_sha256'))


# --- pure helpers ---------------------------------------------------------------------------

def math_messages(question, answer):
    return gsm8k_messages(question) + [dict(role='assistant', content=answer)]


def question_hash(text):
    return hashlib.sha256(' '.join(text.split()).casefold().encode()).hexdigest()


def first_turn(messages):
    if len(messages) < 2 or messages[0].get('role') != 'user' or messages[1].get('role') != 'assistant':
        raise ValueError('First-turn roles must be user then assistant.')
    prompt, response = messages[0].get('content') or '', messages[1].get('content') or ''
    if not prompt.strip() or not response.strip():
        raise ValueError('First-turn prompt or response is empty.')
    return dict(prompt=prompt, response=response, n_messages=len(messages))


def classify_background(prompt, response):
    text = prompt + '\n' + response
    if CODE_PATTERN.search(text):
        return 'code'
    if MATH_PATTERN.search(text):
        return 'math'
    if DIGIT.search(response):
        return 'near_miss'
    if DIGIT.search(prompt):
        return 'digit-prompt-only'
    return 'general'


def build_example(tokenizer, example_id, messages, *, max_tokens, overlength):
    """Tokenize one chat pair and derive masks from character offsets; never guess boundaries."""
    def rejected(reason):
        return dict(id=example_id, status='rejected', reason=reason)

    full = tokenizer.render(messages, add_generation_prompt=False)
    prefix = tokenizer.render(messages[:-1], add_generation_prompt=True)
    content = messages[-1]['content']
    start, end = len(prefix), len(prefix) + len(content)
    if not full.startswith(prefix) or full[start:end] != content:
        return rejected('rendered prompt/content does not align with the template')
    if not full.startswith(tokenizer.eos_token, end):
        return rejected('assistant content is not followed by the end-of-turn token')
    if not content:
        return rejected('empty assistant content')
    if not prefix.endswith('\n'):
        return rejected('assistant header does not end with a newline')
    ids, offsets = tokenizer.encode(full)
    for s, e in offsets:
        if s < start < e or s < end < e:
            return rejected('a token crosses an assistant-content boundary')
    response = [i for i, (s, e) in enumerate(offsets) if e > s and s >= start and e <= end]
    header = [i for i, (s, e) in enumerate(offsets) if e > s and e <= start]
    eos = [i for i, (s, e) in enumerate(offsets) if s == end and e == end + len(tokenizer.eos_token)]
    if not header or not (offsets[header[-1]][0] <= start - 1 < offsets[header[-1]][1]):
        return rejected('reply-start token does not contain the header newline')
    if len(eos) != 1 or ids[eos[0]] != tokenizer.eos_token_id:
        return rejected('end-of-turn token not found as a single token')
    if response and (response != list(range(response[0], response[-1] + 1)) or response[-1] + 1 != eos[0]
                     or response[0] != header[-1] + 1):
        return rejected('assistant content tokens are not contiguous between header and end-of-turn')
    masks = dict(response_content=response, reply_start=[header[-1]], termination=[eos[0] - 1, eos[0]])
    truncated, truncation = False, None
    if len(ids) > max_tokens:
        if overlength == 'exclude':
            return dict(id=example_id, status='excluded-overlength', reason=f'{len(ids)} > {max_tokens} tokens')
        if overlength != 'truncate':
            raise ValueError('overlength must be exclude or truncate')
        truncated, truncation = True, dict(original_total=len(ids), kept=max_tokens)
        ids = ids[:max_tokens]
        masks = {k: [i for i in v if i < max_tokens] for k, v in masks.items()}
        if len(masks['termination']) != 2:
            masks['termination'] = []
        if masks['reply_start'] == [] or not masks['response_content']:
            return rejected('empty primary mask after truncation')
    if not masks['response_content']:
        return rejected('empty primary mask')
    return dict(id=example_id, status='ok', input_ids=ids, masks=masks, truncated=truncated,
                truncation=truncation, lengths=dict(total=len(ids), response_content=len(masks['response_content'])))


# --- io helpers -----------------------------------------------------------------------------

def _inside(root, value):
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Input path is outside the artifact root: {value}')
    return path


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def _write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _write_json(path, value):
    _write_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + '\n').encode())


def _write_jsonl(path, rows):
    _write_bytes(path, ''.join(_canonical(r) + '\n' for r in rows).encode())


def _status(run_dir, state, **extra):
    _write_json(run_dir / 'meta/status.json', dict(state=state, updated_unix=time.time(),
                                                   host=socket.gethostname(), pid=os.getpid(), **extra))


def _read_parquet(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def _strings_under(value, key):
    if isinstance(value, dict):
        for k, v in value.items():
            if k == key and isinstance(v, str):
                yield v
            else:
                yield from _strings_under(v, key)
    elif isinstance(value, list):
        for v in value:
            yield from _strings_under(v, key)


def _lineage_file(path):
    text = path.read_text(errors='replace')
    ids = {'train': set(), 'test': set()}
    for split, index in ID_PATTERN.findall(text):
        ids[split].add(int(index))
    questions, errors = set(), []
    if path.suffix == '.json':
        chunks = [(1, text)]
    elif path.suffix == '.jsonl':
        chunks = [(n, l) for n, l in enumerate(text.splitlines(), start=1) if l.strip()]
    else:
        chunks = []  # non-JSON sources contribute IDs only
    for line_no, chunk in chunks:
        try:
            value = json.loads(chunk)
        except json.JSONDecodeError as exc:  # never drop a line silently: the source becomes unparseable
            errors.append(dict(line=line_no, error=str(exc)[:200]))
            continue
        questions.update(question_hash(q) for q in _strings_under(value, 'question'))
    return ids, questions, errors


# --- leakage --------------------------------------------------------------------------------

def _leakage(plan, root, arms, train_rows, test_rows):
    report = dict(arms={}, sources=[])
    train_ids, test_ids, hashes = set(), set(), set()
    for item in plan['lineage']:
        path = _inside(root, item['path'])
        entry = dict(arm=item['arm'], role=item['role'], path=item['path'], present=path.is_file())
        if entry['present']:
            ids, questions, errors = _lineage_file(path)
            entry.update(sha256=file_hash(path), train_ids=len(ids['train']), test_ids=len(ids['test']),
                         question_texts=len(questions), parse_errors=len(errors), parse_error_examples=errors[:5])
            train_ids |= ids['train']
            test_ids |= ids['test']
            hashes |= questions
        report['sources'].append(entry)
    for name, arm in sorted(arms.items()):
        if arm.get('role') == 'baseline':
            continue
        mine = [s for s in report['sources'] if s['arm'] == name]
        missing = [s['path'] for s in mine if not s['present']]
        unparseable = [s['path'] for s in mine if s.get('parse_errors')]
        if not mine:
            status = 'unknown-no-lineage-listed'
        elif missing:
            status = 'partial-some-sources-missing'
        elif unparseable:
            status = 'partial-some-sources-unparseable'
        else:
            status = 'available'
        report['arms'][name] = dict(status=status, missing_sources=missing, unparseable_sources=unparseable,
                                    roles=sorted({s['role'] for s in mine if s['present']}))
    bad = sorted(f'train:{i}' for i in train_ids if i >= len(train_rows)) + \
        sorted(f'test:{i}' for i in test_ids if i >= len(test_rows))
    if bad:
        raise ValueError(f'Lineage IDs exceed the pinned GSM8K splits: {bad[:5]}')
    protected = set(hashes)
    protected |= {question_hash(train_rows[i]['question']) for i in train_ids}
    test_hashes = {question_hash(test_rows[i]['question']) for i in test_ids}
    report.update(
        protected_train_ids=sorted(train_ids), protected_test_ids=sorted(test_ids),
        protected_question_hashes=len(protected), referenced_test_question_hashes=len(test_hashes),
        arms_with_unknown_provenance=sorted(a for a, r in report['arms'].items() if r['status'] != 'available'),
        cleanliness='no-overlap-with-available-sources',
        cleanliness_note='Only listed, present, fully parsed lineage files were checked; missing, unparseable or '
                         'unlisted sources mean cleanliness is unknown for those arms, not clean.')
    return report, train_ids, protected, test_hashes


# --- preparation ----------------------------------------------------------------------------

def _inventory(plan, root):
    directory = _inside(root, plan['inventory_dir'])
    seal = directory / 'complete.json'
    if not seal.is_file():
        raise ValueError('Ticket-01 inventory is not sealed; stop preparation.')
    if file_hash(directory / 'inventory.json') != json.loads(seal.read_text())['inventory_sha256']:
        raise ValueError('Ticket-01 inventory failed its seal check; stop preparation.')
    return json.loads((directory / 'inventory.json').read_text()), file_hash(seal)


def _take(candidates, counts, group, tokenizer, plan, seed):
    """Shuffle candidates by seed, tokenize in order and fill splits; report every skip."""
    order = sorted(candidates, key=lambda c: c['order_key'])
    random.Random(f'{seed}:{group}').shuffle(order)
    needed = [(split, counts.get(split, 0)) for split in SPLITS]
    selected, skipped, position = [], {}, 0
    for split, count in needed:
        taken = 0
        while taken < count and position < len(order):
            cand = order[position]
            position += 1
            ex = build_example(tokenizer, cand['id'], cand['messages'], max_tokens=plan['max_tokens'],
                               overlength=plan['overlength'])
            if ex['status'] != 'ok':
                skipped.setdefault(ex['status'] + ': ' + ex['reason'], []).append(cand['id'])
                continue
            selected.append(dict({k: v for k, v in cand.items() if k not in ('messages', 'order_key')},
                                 group=group, split=split, **{k: ex[k] for k in
                                 ['input_ids', 'masks', 'truncated', 'truncation', 'lengths']}))
            taken += 1
        if taken < count:
            raise ValueError(f'Insufficient eligible {group} examples for {split}: {taken}/{count}.')
    return selected, skipped


def _compute(plan_path, root, tokenizer_factory):
    plan = json.loads(Path(plan_path).read_text())
    if plan.get('schema') != PLAN_SCHEMA:
        raise ValueError(f'Expected plan schema {PLAN_SCHEMA}.')
    inventory, inventory_seal = _inventory(plan, root)
    tokenizer = tokenizer_factory(plan['tokenizer'], root)
    gsm, bg = plan['gsm8k'], plan['background']
    paths = dict(gsm8k_train=_inside(root, gsm['train_parquet']), gsm8k_test=_inside(root, gsm['test_parquet']),
                 background=_inside(root, bg['parquet']))
    inputs = dict(plan_sha256=file_hash(plan_path), inventory_seal_sha256=inventory_seal,
                  tokenizer=tokenizer.identity(), files={k: file_hash(p) for k, p in paths.items()},
                  lineage={i['path']: file_hash(_inside(root, i['path'])) if _inside(root, i['path']).is_file()
                           else None for i in plan['lineage']})
    train_rows, test_rows = _read_parquet(paths['gsm8k_train']), _read_parquet(paths['gsm8k_test'])
    leakage, protected_ids, protected_hashes, test_hashes = _leakage(
        plan, root, inventory['arms'], train_rows, test_rows)
    excluded = dict(protected_id=[], duplicate_text_of_protected=[], matches_test_question=[],
                    duplicate_text_within_pool=[])
    math_candidates, seen = [], set()
    for index, row in enumerate(train_rows):
        digest = question_hash(row['question'])
        if index in protected_ids:
            excluded['protected_id'].append(index)
        elif digest in protected_hashes:
            excluded['duplicate_text_of_protected'].append(index)
        elif digest in test_hashes:
            excluded['matches_test_question'].append(index)
        elif digest in seen:
            excluded['duplicate_text_within_pool'].append(index)
        else:
            seen.add(digest)
            math_candidates.append(dict(id=f'gsm8k:train:{index}', source='gsm8k', source_ref=index,
                                        text_sha256=digest, n_messages=2, order_key=index,
                                        messages=math_messages(row['question'], row['answer'])))
    leakage['excluded'] = excluded
    leakage['eligible_math_candidates'] = len(math_candidates)
    pools = {'general': [], 'near_miss': []}
    labels, extraction, review_pool, bg_seen = {}, {}, {}, set()
    for index, row in enumerate(_read_parquet(paths['background'])):
        try:
            turn = first_turn(row['messages'])
        except ValueError as exc:
            extraction[str(exc)] = extraction.get(str(exc), 0) + 1
            continue
        label = classify_background(turn['prompt'], turn['response'])
        labels[label] = labels.get(label, 0) + 1
        review_pool.setdefault(label, []).append(dict(prompt_id=row['prompt_id'], label=label, **turn))
        digest = question_hash(turn['prompt'])
        if label not in pools:
            continue
        if digest in bg_seen:
            extraction['duplicate first-turn prompt'] = extraction.get('duplicate first-turn prompt', 0) + 1
            continue
        bg_seen.add(digest)
        pools[label].append(dict(id=f"ultrachat:{bg['split']}:{row['prompt_id']}", source='ultrachat_200k',
                                 source_ref=row['prompt_id'], text_sha256=digest, n_messages=turn['n_messages'],
                                 order_key=index, messages=[dict(role='user', content=turn['prompt']),
                                                            dict(role='assistant', content=turn['response'])]))
    seed = plan['seed']
    selected, skipped = [], {}
    for group, candidates in [('math', math_candidates), ('general', pools['general']),
                              ('near_miss', pools['near_miss'])]:
        rows, skip = _take(candidates, plan['counts'][group], group, tokenizer, plan, seed)
        selected += rows
        skipped[group] = {k: len(v) for k, v in skip.items()}
    if len({r['text_sha256'] for r in selected}) != len(selected):
        raise ValueError('Selected examples share normalized text across groups.')
    filters = dict(definition=FILTER_DOCUMENTATION, first_turn_extraction_failures=extraction,
                   background_labels=labels, eligible=dict(general=len(pools['general']),
                                                           near_miss=len(pools['near_miss'])),
                   tokenization_skips=skipped)
    rng = random.Random(f'{seed}:review')
    sample = []
    for label, count in sorted((plan.get('review_sample') or {}).items()):
        pool = review_pool.get(label, [])
        sample += rng.sample(pool, min(count, len(pool)))
    return plan, inputs, selected, leakage, filters, sample, tokenizer


def _decodes(tokenizer, selected, sample, plan):
    def view(title, example, note):
        lines = [f'## {title}', '', note, '', '| pos | token | masks |', '|---:|---|---|']
        ids = example['input_ids']
        marks = {}
        for name, positions in example['masks'].items():
            for p in positions:
                marks.setdefault(p, []).append(name)
        for p, token in enumerate(ids):
            text = tokenizer.token_text(token).replace('\n', '\\n').replace('|', '\\|')
            lines.append(f"| {p} | `{text}` | {', '.join(marks.get(p, [])) or '-'} |")
        lines += ['', f"truncated: {example.get('truncated')}  lengths: {example.get('lengths')}", '']
        return lines

    out = ['# Decoded token/mask views', '',
           'Activation at position t is after processing token t. Masks: response_content, reply_start, termination.',
           'Positions in no mask (prompt, header/template, trailing newline) are shown with "-".', '']
    math = next(r for r in selected if r['group'] == 'math')
    general = next(r for r in selected if r['group'] == 'general')
    out += view(f"Math example {math['id']}", math, 'Frozen math row (project GSM8K prompt + gold answer).')
    out += view(f"Background example {general['id']}", general, 'Frozen general-background first turn.')
    messages = [dict(role='user', content='Name a river.'), dict(role='assistant', content='The Nile is long.')]
    full = build_example(tokenizer, 'illustration', messages, max_tokens=10 ** 6, overlength='exclude')
    cut = build_example(tokenizer, 'illustration', messages, max_tokens=full['lengths']['total'] - 3,
                        overlength='truncate')
    out += view('Illustration: truncated boundary (not in the manifest)', cut,
                'Cut inside the response: termination is empty and truncated is True.')
    empty = build_example(tokenizer, 'illustration', [messages[0], dict(role='assistant', content='')],
                          max_tokens=10 ** 6, overlength='exclude')
    out += ['## Illustration: empty response (not in the manifest)', '',
            f"status: {empty['status']}; reason: {empty.get('reason')}", '']
    return '\n'.join(out)


def prepare(plan_path, artifact_root, output_root, run_id, *, tokenizer_factory=_default_tokenizer):
    root = Path(artifact_root).resolve()
    run_dir = Path(output_root).resolve() / RUN_SUBDIR / run_id
    seal_path = run_dir / 'complete.json'
    if seal_path.exists():
        check = verify(run_dir)
        if not check['ok']:
            raise ValueError('Sealed sequences failed their integrity check: ' + '; '.join(check['problems']))
    try:
        plan, inputs, selected, leakage, filters, sample, tokenizer = _compute(plan_path, root, tokenizer_factory)
    except ValueError as exc:
        if not seal_path.exists():
            _status(run_dir, 'failed', run_id=run_id, error=str(exc))
        raise
    sequences = ''.join(_canonical(r) + '\n' for r in selected).encode()
    if seal_path.exists():
        manifest = json.loads((run_dir / 'manifest.json').read_text())
        if manifest['inputs'] != json.loads(_canonical(inputs)) or \
                manifest['sequences_sha256'] != hashlib.sha256(sequences).hexdigest():
            raise ValueError('Inputs changed since the sealed sequences; use a new run_id.')
        print(f'Sequences verified and reused: {run_dir}', flush=True)
        return manifest
    _status(run_dir, 'running', run_id=run_id)
    _write_bytes(run_dir / 'sequences.jsonl', sequences)
    _write_json(run_dir / 'plan.json', plan)
    _write_json(run_dir / 'leakage_report.json', leakage)
    _write_json(run_dir / 'filter_report.json', filters)
    _write_jsonl(run_dir / 'review/background_sample.jsonl', sample)
    _write_bytes(run_dir / 'review/decodes.md', _decodes(tokenizer, selected, sample, plan).encode())
    counts = {}
    for r in selected:
        key = f"{r['group']}/{r['split']}"
        counts[key] = counts.get(key, 0) + 1
    manifest = dict(
        schema=MANIFEST_SCHEMA, run_id=run_id, inputs=inputs, seed=plan['seed'], max_tokens=plan['max_tokens'],
        overlength=plan['overlength'], counts=dict(sorted(counts.items())),
        sequences_sha256=hashlib.sha256(sequences).hexdigest(),
        order=[r['id'] for r in selected], truncated=sum(r['truncated'] for r in selected),
        lengths=dict(max_total=max(r['lengths']['total'] for r in selected),
                     total_tokens=sum(r['lengths']['total'] for r in selected),
                     response_content_tokens=sum(r['lengths']['response_content'] for r in selected)),
        formatting=dict(math='protocol.gsm8k_messages(question) + GSM8K reference answer',
                        background='first user/assistant turn; template default system prompt'),
        mask_semantics='activation at position t is after processing token t; padding is frozen later in 03b',
        sources=dict(gsm8k={k: v for k, v in plan['gsm8k'].items()},
                     background={k: v for k, v in plan['background'].items()}),
        background_review='pending',
        batch_manifest='not-frozen; 03b freezes membership, padding and position IDs after benchmarking')
    _write_json(run_dir / 'manifest.json', manifest)
    files = ['sequences.jsonl', 'plan.json', 'leakage_report.json', 'filter_report.json', 'manifest.json',
             'review/background_sample.jsonl', 'review/decodes.md']
    _write_json(seal_path, dict(files={n: file_hash(run_dir / n) for n in files}))
    _status(run_dir, 'complete', run_id=run_id, counts=manifest['counts'])
    print(f"Sequences sealed: {run_dir} {manifest['counts']}", flush=True)
    return manifest


def _review_state(run_dir):
    path = run_dir / 'review/review.json'
    if not path.is_file():
        return 'pending'
    record = json.loads(path.read_text())
    if record.get('sample_sha256') != file_hash(run_dir / 'review/background_sample.jsonl'):
        return 'invalid-sample-changed'
    return record['decision']


def verify(run_dir):
    run_dir = Path(run_dir).resolve()
    problems = []
    seal = run_dir / 'complete.json'
    if not seal.is_file():
        return dict(ok=False, problems=['not sealed'], background_review='pending')
    for name, digest in json.loads(seal.read_text())['files'].items():
        if not (run_dir / name).is_file() or file_hash(run_dir / name) != digest:
            problems.append(f'{name} integrity')
    if not problems:
        manifest = json.loads((run_dir / 'manifest.json').read_text())
        if file_hash(run_dir / 'sequences.jsonl') != manifest['sequences_sha256']:
            problems.append('sequences.jsonl differs from manifest')
    return dict(ok=not problems, problems=problems, background_review=_review_state(run_dir))


def record_review(run_dir, *, reviewer, decision, notes=''):
    run_dir = Path(run_dir).resolve()
    if decision not in ('accept', 'reject'):
        raise ValueError('decision must be accept or reject')
    if not verify(run_dir)['ok']:
        raise ValueError('Sealed sequences failed verification; cannot record review.')
    path = run_dir / 'review/review.json'
    if path.exists():
        raise ValueError('A background review is already recorded; keep it and investigate instead.')
    record = dict(reviewer=reviewer, decision=decision, notes=notes, recorded_unix=time.time(),
                  sample_sha256=file_hash(run_dir / 'review/background_sample.jsonl'),
                  filter_report_sha256=file_hash(run_dir / 'filter_report.json'))
    _write_json(path, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    for name in ['--plan', '--artifact_root', '--output_root', '--run_id']:
        p.add_argument(name, required=True)
    v = sub.add_parser('verify')
    v.add_argument('--run_dir', required=True)
    r = sub.add_parser('review')
    r.add_argument('--run_dir', required=True)
    r.add_argument('--reviewer', required=True)
    r.add_argument('--decision', required=True, choices=['accept', 'reject'])
    r.add_argument('--notes', default='')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        manifest = prepare(args.plan, args.artifact_root, args.output_root, args.run_id)
        print(json.dumps(dict(counts=manifest['counts'], lengths=manifest['lengths']), indent=2))
        return 0
    if args.command == 'verify':
        result = verify(args.run_dir)
        print(json.dumps(result, indent=2))
        return 0 if result['ok'] else 1
    print(json.dumps(record_review(args.run_dir, reviewer=args.reviewer, decision=args.decision,
                                   notes=args.notes), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
