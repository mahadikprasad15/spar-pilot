"""A small, real-format input world for the C-attribution whole-chain CPU test (ticket 10).

Built the way the real inputs are:

- a 28-layer Qwen2 snapshot written by ``save_pretrained`` (the exact bigram base of ``attribution_realpath_fixture``:
  after the assistant header it answers ``Step\\n#### 7`` and stops);
- two rank-1 PEFT arms on all seven projections (online and offline GRPO), saved with ``save_pretrained``;
- a real ``tokenizers`` tokenizer with a Qwen-shaped chat template, and the pinned real ``generation_config.json``;
- behaviour ``responses.jsonl`` and source ``config.json`` per arm in the real record format (EOS, capped and invalid
  rows; all-same and mixed groups), re-scored by the real scorer, with units, policy, checkpoint and template hashes;
- an effective-write table whose union rule gives the real hot layers {11, 15, 16, 24};
- a catalog checkout (git-committed) whose ``catalog verify`` passes the bound entries;
- the input bundle (ticket 08) with its catalog revision, exported and restored through a fake Hub;
- the launch record sealed by the real ``finalize_draft`` from the actual rehearsal YAML, with CPU sizes.
"""
import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import yaml

import attribution_realpath_fixture as fx
from pilot_eval import c_attribution_bundle as cb
from pilot_eval import c_attribution_chain as cc
from pilot_eval import c_attribution_rows as cr
from pilot_eval.attribution_chain import _commit
from pilot_eval.scoring import score_gsm8k_flexible_v3
from pilot_eval.workflow import _hash

REPO = Path(__file__).resolve().parents[1]
DRAFT = REPO / '.scratch/c-attribution-pilot/launch-record-rehearsal.yaml'
GENERATION_CONFIG = REPO / 'tests/fixtures/c_attribution_checks/generation_config.json'
SOURCE_CONFIG = REPO / 'tests/fixtures/c_attribution_rows/online/config.json'
CAP = 12
HOT = 'analysis/results-hub-v1/data/writes_modules.json'
COHORT = 'plans/pilot2-sft-fp32-l4-batch8-v1/evaluation.items.json'
RESPONSES = {'online-grpo': 'imports/world/online/responses.jsonl', 'offline-grpo': 'imports/world/offline/responses.jsonl'}
CONFIGS = {'online-grpo': 'imports/world/online/config.json', 'offline-grpo': 'imports/world/offline/config.json'}
ADAPTERS = {'online-grpo': f'{cb.ONLINE}/checkpoint-64', 'offline-grpo': f'{cb.OFFLINE}/checkpoint-64'}
# c correct, w wrong, i invalid (no number), k capped. One string of 8 draws per question.
PATTERNS = {
    'online-grpo': ['cccccccc', 'wwwwwwww', 'ccwiccwc', 'wwwwiwwc', 'cwcwcwcw', 'cccccccw', 'wiwwwwww', 'ccwwccww'],
    'offline-grpo': ['cccccccc', 'kwwwiwcw', 'ciciwcwc', 'wwwwwwww', 'ccwcccwc', 'wcwcwcwc', 'iwwwwwwc', 'cwcwwwcc'],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _question_ids():
    return yaml.safe_load(DRAFT.read_text())['scope']['generation_questions']['online-grpo']


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _hot_table():
    rows = []
    for arm, downs in (('online', [11, 15, 16]), ('offline', [15, 16, 24])):
        ranked = [(layer, 'down') for layer in downs] + [(layer, 'q') for layer in range(7)]
        ranked += [(layer, 'down') for layer in (1, 2, 3)]  # outside the top 10
        for rank, (layer, module) in enumerate(ranked):
            rows.append(dict(arm=arm, context='gsm8k_solution', step=64, layer=layer, module=module,
                             write=1.0 - 0.01 * rank))
    return {'rows': rows}


def _rows(arm, ids, prompts, tok, vocab, eos, adapter_sha):
    pieces = {'c': ['Step', '\n', '####', ' ', '7'], 'w': ['Step', '\n', '####', ' ', '5'], 'i': ['Step']}
    out = []
    for qi, (qid, pattern) in enumerate(zip(ids, PATTERNS[arm])):
        for d, kind in enumerate(pattern):
            if kind == 'k':
                content = [vocab['Step'], vocab[' ']] * (CAP // 2)
                generated, stop, capped = content, 'cap', True
            else:
                content = [vocab[p] for p in pieces[kind]]
                generated, stop, capped = content + [eos], 'eos', False
            text = tok.decode(content, skip_special_tokens=True)
            flexible = score_gsm8k_flexible_v3(text, 'Count .\n#### 7', capped=capped)
            out.append(dict(adapter_sha256=adapter_sha, capped=capped, draw_id=f'{qid}:draw:{d}', draw_index=d,
                            flexible=flexible, generated_token_ids=generated, gold='Count .\n#### 7', group_id=qid,
                            item_id=qid, mode='sampled', prompt=prompts[qid], record_id=f'sampled:64:{qid}:draw:{d}',
                            reward=0 if capped else int(flexible['correct']), scorer=cr.SCORER, seed=42 + qi // 4,
                            step=64, stop_reason=stop, text=text, token_count=len(generated),
                            token_ids=generated, unit_key=f'sampled-64-batch-{qi // 4:04d}'))
    return out


def _config(ids, prompts, tok, eos, adapter_sha, chat_template):
    config = json.loads(SOURCE_CONFIG.read_text())
    policy = dict(cr.POLICY, completion_limit=CAP)
    config['policies'] = [policy]
    config['units'] = [dict(key=f'sampled-64-batch-{b:04d}', policy=policy, ids=ids[4 * b:4 * b + 4], seed=42 + b)
                       for b in range(len(ids) // 4)]
    config['items'] = [dict(id=q, prompt=prompts[q], gold='Count .\n#### 7',
                            prompt_tokens=len(tok(prompts[q], add_special_tokens=False)['input_ids'])) for q in ids]
    config['checkpoints'] = {'64': dict(adapter_sha256=adapter_sha)}
    config['implementation'] = dict(config['implementation'], engine=sha(REPO / 'pilot_eval/grpo_eval_engine.py'),
                                    scorer=cr.SCORER_SHA256)
    config['prompt_template'] = dict(config['prompt_template'], chat_template_sha256=_hash(chat_template))
    config['runtime'] = dict(config['runtime'], eos_token_id=eos)
    return config


def _catalog(code, art, ids, adapters_sha, responses):
    catalog = code / 'catalog'
    (catalog / 'responses').mkdir(parents=True)
    shutil.copy(REPO / 'catalog/locations.yaml', catalog / 'locations.yaml')
    arms = {arm: dict(description=f'world {arm}', checkpoints={64: dict(location='local', path=f'{ADAPTERS[arm]}/adapter_model.safetensors',
                                                                         sha256=adapters_sha[arm])})
            for arm in ('online-grpo', 'offline-grpo')}
    _write(catalog / 'arms.yaml', yaml.safe_dump({'arms': arms}))
    from pilot_eval.catalog import ids_digest
    real = yaml.safe_load((REPO / 'catalog/cohorts.yaml').read_text())['cohorts']['gsm8k-test-cohort150']
    cohort = dict(real, size=len(ids), ids_sha256=ids_digest(ids), source=dict(location='local', path=COHORT, id_field='id'))
    _write(catalog / 'cohorts.yaml', yaml.safe_dump({'cohorts': {'gsm8k-test-cohort150': cohort}}))
    for arm in ('online-grpo', 'offline-grpo'):
        entry_id = f'{arm}.s64.sampled-x8.test150'
        entry = yaml.safe_load((REPO / f'catalog/responses/{entry_id}.yaml').read_text())
        entry['files'] = [dict(location='local', path=RESPONSES[arm], sha256=sha(art / RESPONSES[arm]),
                               rows_total=len(responses[arm]), select=dict(step=64, mode='sampled'),
                               rows_selected=len(responses[arm]))]
        entry['policy'] = dict(entry['policy'], max_new_tokens=CAP)
        _write(catalog / f'responses/{entry_id}.yaml', yaml.safe_dump(entry))
    git = ['git', '-C', str(code), '-c', 'user.email=t@t', '-c', 'user.name=t']
    subprocess.run(['git', 'init', '-q', str(code)], check=True)
    subprocess.run(git + ['add', 'catalog'], check=True)
    subprocess.run(git + ['commit', '-qm', 'world catalog'], check=True)
    return code


def build(tmp, *, record_changes=None, export_client=None):
    """Original artifacts, the restored bundle root, the world code checkout, and the sealed record."""
    from test_neuron_audit_export import FakeHub
    tmp = Path(tmp)
    art = tmp / 'artifacts'
    ids = _question_ids()
    questions = {q: f'How many pears does {fx.NAMES[i % 10]} keep in box {i} now ?' for i, q in enumerate(ids)}
    prompts = {q: f'<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n' for q, text in questions.items()}
    corpus = list(prompts.values()) + ['Step\n#### 7', 'Step\n#### 5', 'Count .\n#### 7', 'user\n', 'assistant\n']
    tok_dir = art / cb.MODEL_DIR
    vocab = fx.tokenizer_files(tok_dir, corpus)
    tok_config = json.loads((tok_dir / 'tokenizer_config.json').read_text())
    (tok_dir / 'tokenizer_config.json').write_text(json.dumps(dict(tok_config, tokenizer_class='PreTrainedTokenizerFast')))
    shutil.copy(GENERATION_CONFIG, tok_dir / 'generation_config.json')
    model_dir = tmp / 'hf-home/snapshots/base'
    base = fx.base_snapshot(model_dir, vocab, layers=28, intermediate=32)
    eos, pad = vocab['<|im_end|>'], vocab['<|endoftext|>']
    adapters_sha = {}
    for seed, arm in enumerate(('online-grpo', 'offline-grpo'), start=1):
        fx.arm_adapter(base, art / ADAPTERS[arm], seed)
        adapters_sha[arm] = sha(art / ADAPTERS[arm] / 'adapter_model.safetensors')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
    responses = {}
    for arm in ('online-grpo', 'offline-grpo'):
        responses[arm] = _rows(arm, ids, prompts, tok, vocab, eos, adapters_sha[arm])
        _write(art / RESPONSES[arm], ''.join(json.dumps(r) + '\n' for r in responses[arm]))
        _write(art / CONFIGS[arm], json.dumps(_config(ids, prompts, tok, eos, adapters_sha[arm], tok.chat_template)))
    _write(art / COHORT, json.dumps([{'id': q} for q in ids]))
    _write(art / HOT, json.dumps(_hot_table()))
    code = _catalog(tmp / 'code', art, ids, adapters_sha, responses)

    entries = [{'role': f'responses:{a}', 'path': RESPONSES[a]} for a in RESPONSES] + \
              [{'role': f'run-config:{a}', 'path': CONFIGS[a]} for a in CONFIGS] + \
              [{'role': 'cohort-ids:gsm8k-test-cohort150', 'path': COHORT}, {'role': 'hot-layer-source', 'path': HOT},
               {'role': 'generation-config', 'path': f'{cb.MODEL_DIR}/generation_config.json'},
               {'role': 'tokenizer:tokenizer.json', 'path': f'{cb.MODEL_DIR}/tokenizer.json'},
               {'role': 'tokenizer:tokenizer_config.json', 'path': f'{cb.MODEL_DIR}/tokenizer_config.json'}]
    for arm in ADAPTERS:
        for name in ('adapter_model.safetensors', 'adapter_config.json'):
            entries.append({'role': f'adapter:{arm}:64:{name}', 'path': f'{ADAPTERS[arm]}/{name}'})
    bundle = cb.build(art, entries, repo_root=code, model_dir=model_dir)
    hub = export_client or FakeHub()
    (tmp / 'tmp').mkdir(exist_ok=True)
    receipt = cb.export(art, bundle, label='world', client=hub, temp_dir=tmp / 'tmp', on_event=lambda e: None)
    restored = tmp / 'restored'
    out = cb.restore(receipt, restored, client=hub, on_event=lambda e: None)
    bundle_path = restored / receipt['bundle_manifest']

    body = yaml.safe_load(DRAFT.read_text())
    body['hourly_cost_usd'] = 1.0
    body['generation'] = dict(body['generation'], cap=CAP, eos_token_id=eos, batch_size=4)
    body['attribution'] = dict(body['attribution'], token_budget=256)
    body['qualification'] = dict(body['qualification'], batch_invariance_budgets=[256], noop_greedy_tokens=4)
    body['statistics'] = dict(body['statistics'], resamples=500)
    body['hot_layers']['source']['sha256'] = sha(art / HOT)
    for dotted, value in (record_changes or {}).items():
        parent, keys = body, dotted.split('.')
        for k in keys[:-1]:
            parent = parent[k]
        parent[keys[-1]] = value
    draft = _write(tmp / 'draft.yaml', yaml.safe_dump(body))
    (tmp / 'work').mkdir(exist_ok=True)
    record_path = cc.finalize_draft(draft, tmp / 'launch_record.json', code_commit=_commit(), pod_id='podcpu1',
                                    fill={'inputs.artifact_root': str(restored), 'inputs.bundle': str(bundle_path),
                                          'inputs.model_dir': str(model_dir), 'inputs.work_dir': str(tmp / 'work')})
    return dict(artifacts=art, restored=restored, code=code, model_dir=model_dir, vocab=vocab, eos=eos, pad=pad,
                bundle=out['bundle'], record_path=record_path, hub=hub,
                record=cc.load_launch_record(record_path, code_commit=_commit()), draft=draft, ids=ids)
