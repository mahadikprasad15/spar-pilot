"""A small but real input world for the whole-chain CPU test (review round 6, "next step").

Everything the real ``TransformersBackend`` reads is produced the way the Pod's inputs were:

- a Qwen2 snapshot written by ``save_pretrained`` (safetensors, config, generation config), loaded through the real
  loader (``AutoModelForCausalLM.from_pretrained``, eager attention, local files only);
- two rank-1 PEFT arms on all seven projections, saved with ``save_pretrained``;
- a real ``tokenizers`` tokenizer (``tokenizer.json``) with a Qwen-shaped chat template in ``tokenizer_config.json``,
  read by the real ``ChatTokenizer``;
- GSM8K-shaped train/test parquets and an UltraChat-shaped background parquet;
- a sealed inventory-v2 and a sequences-v2 run built by the real ``neuron_audit_sequences.prepare``;
- the launch record sealed by the real ``finalize_draft`` from a fresh copy of the rehearsal YAML.

The base model's attention and MLP outputs (``o_proj``, ``down_proj``) are zero, so the untuned model is an exact
bigram model: its embedding rows are orthonormal and ``lm_head`` maps each token to a chosen successor. After the
assistant header it always answers ``Step\\n#### 7`` and stops, which the real flexible-v3 scorer marks correct
against gold answers ending ``#### 7``. The arms' random rank-1 adapters write through those zeroed projections, so
every adapter has a real, differentiable effect.
"""
import hashlib
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
DRAFT = REPO / '.scratch/adapter-neuron-attribution/launch-record-pod-rehearsal.yaml'
PROJ = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
SPECIAL = ['<|endoftext|>', '<|im_start|>', '<|im_end|>']
PATTERN = r'assistant\n|\n|[^\s<]+|[ \t]+|<'
TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
ANSWER = ['Step', '\n', '####', ' ', '7']  # the untuned model's answer after the assistant header
INV = 'runs/joint-neuron-adapter-audit/inventory/inventory-v2'
NAMES = ['Tom', 'Ana', 'Raj', 'Mei', 'Leo', 'Ivy', 'Sam', 'Zoe', 'Kai', 'Una']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _texts():
    from pilot_eval.protocol import gsm8k_messages
    train = [dict(question=f'How many apples does {NAMES[i % 10]} have in basket {i} today ?',
                  answer=f'{NAMES[i % 10]} counts the apples .\n#### 7') for i in range(80)]
    test = [dict(question=f'How many pears does {NAMES[i % 10]} keep in box {i} now ?', answer='Count .\n#### 7')
            for i in range(200)]
    background = []
    for i in range(12):
        name = chr(97 + i // 26) + chr(97 + i % 26)
        background.append(dict(prompt_id=f'g{i}', prompt='x', messages=[
            dict(role='user', content=f'Tell me about rivers {name}'), dict(role='assistant', content='Rivers flow .')]))
        background.append(dict(prompt_id=f'n{i}', prompt='x', messages=[
            dict(role='user', content=f'When was bridge {name} built'), dict(role='assistant', content='In 1889 .')]))
    corpus = [m['content'] for r in train + test for m in gsm8k_messages(r['question'])]
    corpus += [r['answer'] for r in train + test]
    corpus += [m['content'] for r in background for m in r['messages']]
    corpus += ['user\n', 'assistant\n', 'Name a river.', 'The Nile is long.']  # decodes.md illustration
    return train, test, background, corpus


def tokenizer_files(directory, corpus):
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers
    pieces = sorted({p for text in corpus for p in re.findall(PATTERN, text)} | set(ANSWER) | {'user', 'system'})
    vocab = {t: i for i, t in enumerate(SPECIAL + ['[UNK]'] + [p for p in pieces if p not in SPECIAL])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(PATTERN), behavior='isolated')
    tok.decoder = decoders.Fuse()
    tok.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in SPECIAL])
    directory.mkdir(parents=True, exist_ok=True)
    tok.save(str(directory / 'tokenizer.json'))
    (directory / 'tokenizer_config.json').write_text(json.dumps(dict(
        chat_template=TEMPLATE, eos_token='<|im_end|>', pad_token='<|endoftext|>', model_max_length=4096)))
    return vocab


def base_snapshot(directory, vocab, *, layers=12, intermediate=96):
    """The exact bigram base model described in the module docstring, saved like a hub snapshot."""
    from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM
    size = len(vocab)
    hidden = -(-size // 16) * 16
    eos, pad = vocab['<|im_end|>'], vocab['<|endoftext|>']
    config = Qwen2Config(vocab_size=size, hidden_size=hidden, intermediate_size=intermediate,
                         num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=1,
                         max_position_embeddings=4096, tie_word_embeddings=False, eos_token_id=eos,
                         pad_token_id=pad, bos_token_id=None, rms_norm_eps=1e-6, attn_implementation='eager')
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config)
    successor = {t: eos for t in range(size)}
    chain = ['assistant\n'] + ANSWER
    for here, nxt in zip(chain, chain[1:] + ['<|im_end|>']):
        successor[vocab[here]] = vocab[nxt]
    with torch.no_grad():
        basis, _ = torch.linalg.qr(torch.randn(hidden, hidden))
        model.model.embed_tokens.weight.copy_(basis[:size])
        for layer in model.model.layers:
            layer.self_attn.o_proj.weight.zero_()
            layer.mlp.down_proj.weight.zero_()
        head = torch.zeros(size, hidden)
        for token, nxt in successor.items():
            head[nxt] += (12.0 / hidden ** 0.5) * basis[token]  # successor logit ~12: confident, not saturated
        model.lm_head.weight.copy_(head)
    model.generation_config = GenerationConfig(eos_token_id=eos, pad_token_id=pad)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    return model


def arm_adapter(base, directory, seed):
    from peft import LoraConfig, get_peft_model
    import copy
    model = get_peft_model(copy.deepcopy(base), LoraConfig(r=1, lora_alpha=1, lora_dropout=0.0,
                                                           target_modules=PROJ))
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'lora_' in name:
                p.normal_(0, 0.03)  # small nudges, like trained step-64 adapters
    model.save_pretrained(directory)
    return directory


def build(tmp, *, sizes=None, draft_path=None, arms=('gold-sft', 'online-grpo'), math=None):
    """The prep root, inputs.json and the sealed launch record; returns paths and the record."""
    from pilot_eval import attribution_chain as ch
    from pilot_eval import neuron_audit_sequences as seq
    root = tmp / 'prep'
    train, test, background, corpus = _texts()
    write_parquet(root / 'data/gsm8k-train.parquet', train)
    write_parquet(root / 'data/gsm8k-test.parquet', test)
    write_parquet(root / 'data/ultrachat.parquet', background)
    tok_dir = root / 'inputs/models/tok-rev'
    vocab = tokenizer_files(tok_dir, corpus)
    model_dir = tmp / 'hf/snapshots/base-rev'
    base = base_snapshot(model_dir, vocab)
    adapters = {arm: str(arm_adapter(base, root / 'runs/arms' / arm / 'checkpoint-64', seed))
                for seed, arm in enumerate(arms, start=1)}
    (root / 'lineage').mkdir(parents=True)
    (root / 'lineage/gold.json').write_text(json.dumps([dict(id='gsm8k:train:3')]))
    (root / 'lineage/grpo.jsonl').write_text(json.dumps(dict(item='gsm8k:train:5')) + '\n')
    inv = root / INV
    inv.mkdir(parents=True)
    inventory = dict(audit_step=64, arms={arm: dict(checkpoints={'64': dict(status='verified', path=d, files={
        n: dict(sha256=sha(Path(d) / n)) for n in ('adapter_model.safetensors', 'adapter_config.json')})})
        for arm, d in adapters.items()},
        base=dict(model='fixture/qwen2-bigram', model_revision='base-rev', tokenizer_revision='tok-rev',
                  expected_file_sha256={n: sha(model_dir / n) for n in
                                        ('config.json', 'generation_config.json', 'model.safetensors')}))
    (inv / 'inventory.json').write_text(json.dumps(inventory))
    (inv / 'sources.json').write_text(json.dumps(dict(note='fixture')))
    (inv / 'complete.json').write_text(json.dumps(dict(inventory_sha256=sha(inv / 'inventory.json'),
                                                       sources_sha256=sha(inv / 'sources.json'))))
    plan = dict(
        schema=seq.PLAN_SCHEMA, seed=42, max_tokens=400, overlength='exclude', inventory_dir=INV,
        tokenizer=dict(dir='inputs/models/tok-rev', expected_sha256={n: sha(tok_dir / n) for n in
                                                                     ('tokenizer.json', 'tokenizer_config.json')}),
        gsm8k=dict(dataset='openai/gsm8k', revision='r' * 40, license='mit', train_parquet='data/gsm8k-train.parquet',
                   test_parquet='data/gsm8k-test.parquet', prompt='protocol.gsm8k_messages+gold-answer'),
        background=dict(dataset='HuggingFaceH4/ultrachat_200k', revision='u' * 40, license='mit', split='test_sft',
                        parquet='data/ultrachat.parquet'),
        counts=dict(math=math or dict(discovery=2, heldout=2, calibration=14), general=dict(discovery=1, heldout=1, calibration=1),
                    near_miss=dict(discovery=1, heldout=1)),
        lineage=[dict(arm='gold-sft', role='training', path='lineage/gold.json'),
                 dict(arm='online-grpo', role='rollout', path='lineage/grpo.jsonl')],
        review_sample=dict(general=1, near_miss=1))
    (root / 'plan.json').write_text(json.dumps(plan))
    seq.prepare(root / 'plan.json', root, root, 'sequences-v2')
    inputs = tmp / 'inputs.json'
    inputs.write_text(json.dumps(dict(prep=str(root), model_dir=str(model_dir), tokenizer_dir=str(tok_dir),
                                      adapter_dirs=adapters, model_bytes=0, hf_home=str(tmp / 'hf'))))
    draft = yaml.safe_load(Path(draft_path or DRAFT).read_text())
    draft.update(hourly_cost_usd=1.0, **(sizes or {}))
    draft_path = tmp / 'draft.yaml'
    draft_path.write_text(yaml.safe_dump(draft))
    record_path = ch.finalize_draft(draft_path, tmp / 'launch_record.json', code_commit=ch._commit(),
                                    inputs_json=inputs, pod_id='podcpu1', work_dir=str(tmp / 'work'))
    (tmp / 'work').mkdir(exist_ok=True)
    return dict(root=root, record_path=record_path, record=ch.load_launch_record(record_path, code_commit=ch._commit()),
                vocab=vocab, model_dir=model_dir, tokenizer_dir=tok_dir, adapters=adapters)
