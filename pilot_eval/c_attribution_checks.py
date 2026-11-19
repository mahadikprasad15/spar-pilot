"""C-attribution implementation, policy and numerical checks (ticket 05).

Used by CPU tests and, unchanged, by GPU qualification. Each check returns a JSON-ready record; a pass rule
is applied only when its tolerance is supplied (tolerances come from the launch record and never loosen
production-1's). Report-only checks never return a pass flag.

- ``resolve_processors``: the logits processors and warpers transformers resolves for the exact arguments
  ``grpo_gpu.GPUPreflight._generate`` passes, on both generation-config bases the source runs could have
  had (the pinned ``generation_config.json`` and the backend's fresh defaults). Passes only if every resolved
  processor leaves the logits unchanged.
- ``surrogate_finite_difference``: F(λ) = Σ_k w_k M_k(λ) with frozen weights; a central difference over
  branch or neuron gates against dJdλ from the instrument. The reference is always FP64 (``full_fp64``).
- ``cached_vs_forced``: per-token log p from left-padded prefill plus KV-cache decoding of the saved tokens,
  against the right-padded teacher-forced forward.
- ``batch_invariance``: ``seqlogp`` attributions, singleton against packed batches.
- ``noop_check``: an intervention context with an empty removal set is bit-identical and restores weights.
- ``replay_report``: token identity of a regenerated batch (report-only).
- ``numerics_record``: the measured floors for ``numerics.json``.
"""
import hashlib
import json
import types
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from pilot_eval import attribution_instrument as ai
from pilot_eval import attribution_metrics as am
from pilot_eval.neuron_audit_qualify import pack

PRODUCTION_1 = {'fd_median_rel': 0.05, 'fd_max_rel': 0.20, 'identity_rtol': 1e-4, 'batch_invariance_fraction': 1e-4}


class CheckFailure(RuntimeError):
    pass


class _Captured(Exception):
    pass


# --- logits processors ------------------------------------------------------------------------------

def _identity_on(processor, vocab, *, torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (2, 7), generator=g)
    scores = torch.randn(2, vocab, generator=g)
    out = processor(ids, scores.clone())
    return bool(torch.equal(out, scores))


def resolve_processors(model, tokenizer, *, settings, base, generation_config_path=None, overrides=None):
    """Resolve the processors for ``_generate``'s sampled call and test each one for identity.

    ``base`` is 'pinned-file' (``generation_config_path`` loaded as the model's generation config) or
    'fresh-defaults' (``backend.load_hf_backend``'s replacement config). ``overrides`` injects extra
    generate arguments (tests use it to show a repetition penalty is caught).
    """
    import torch
    from transformers import GenerationConfig
    from pilot_eval.grpo_gpu import GPUPreflight
    if base == 'pinned-file':
        model.generation_config = GenerationConfig.from_pretrained(Path(generation_config_path).parent,
                                                                   Path(generation_config_path).name)
    elif base == 'fresh-defaults':
        model.generation_config = GenerationConfig(
            do_sample=False, num_beams=1, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id, repetition_penalty=1.0, no_repeat_ngram_size=0)
    else:
        raise ValueError(f'unknown base {base!r}')
    captured = {}
    original_get = model._get_logits_processor
    original_generate = model.generate

    def get(*args, **kwargs):
        processors = original_get(*args, **kwargs)
        captured['processors'] = list(processors)
        captured['generation_config'] = kwargs.get('generation_config', args[0] if args else None)
        raise _Captured()

    def generate(**kwargs):
        merged = {**kwargs, **(overrides or {})}
        captured['arguments'] = {k: v for k, v in merged.items() if k not in ('input_ids', 'attention_mask')}
        return original_generate(**merged)

    fake = types.SimpleNamespace(torch=torch, tokenizer=tokenizer, model=model)
    model._get_logits_processor = get
    model.generate = generate
    try:
        GPUPreflight._generate(fake, [{'prompt': 'Question: 1 + 1?'}], settings, 42, draws=2)
    except _Captured:
        pass
    finally:
        model._get_logits_processor = original_get
        model.generate = original_generate
    if 'processors' not in captured:
        raise CheckFailure('generate did not reach logits-processor resolution')
    vocab = model.config.vocab_size
    names = [type(p).__name__ for p in captured['processors']]
    identity = [_identity_on(p, vocab, torch=torch) for p in captured['processors']]
    config = captured['generation_config']
    resolved = {k: getattr(config, k, None) for k in ('do_sample', 'temperature', 'top_k', 'top_p',
                                                      'repetition_penalty', 'no_repeat_ngram_size', 'num_beams')}
    return {'base': base, 'processors': names, 'identity': identity, 'passed': all(identity),
            'arguments': {k: (v if isinstance(v, (int, float, bool, str, type(None))) else str(v))
                          for k, v in captured['arguments'].items()},
            'resolved_generation_config': resolved}


def processor_check(model, tokenizer, *, settings, generation_config_path):
    """Both bases must pass for the provenance item to move from unverified to verified."""
    results = [resolve_processors(model, tokenizer, settings=settings, base='pinned-file',
                                  generation_config_path=generation_config_path),
               resolve_processors(model, tokenizer, settings=settings, base='fresh-defaults')]
    return {'bases': results, 'passed': all(r['passed'] for r in results),
            'scope': ('resolution on the current runtime from grpo_gpu._generate as it is on this branch; '
                      'grpo_gpu.py was not hashed by the source runs, so this verifies the code path, '
                      'not the bytes that ran')}


# --- surrogate finite differences --------------------------------------------------------------------

def _f(reference, ids, attention, positions, masks, weights, gates=None, neuron=None, adapter='default'):
    import torch
    with torch.no_grad(), ai.full_fp64(reference), ai.branch_gates(reference, gates or {}, adapter), \
            ai.neuron_gate(reference, neuron, adapter):
        logits = reference(input_ids=ids, attention_mask=attention, position_ids=positions, use_cache=False).logits
    m = am.seqlogp(logits, ids, masks).detach().cpu().numpy()
    return float(np.dot(weights, m)), m


def surrogate_finite_difference(inst, reference, rows, weights, *, epsilon, adapters=5, neurons=3,
                                median_tol=None, max_tol=None):
    """Central differences of F(λ) = Σ w M(λ) (FP64 reference) against dJdλ = -Σ w attr (FP32 instrument)."""
    import torch
    if any(p.dtype != torch.float64 for p in reference.parameters()):
        raise CheckFailure('the finite-difference reference must be an FP64 copy of the arm')
    weights = np.asarray(weights, dtype=np.float64)
    out = inst.attribute(rows)
    attr = out['attr']['seqlogp']
    d_adapter = -(weights[:, None] * attr).sum(0)
    device = next(reference.parameters()).device
    ids, attention, positions = (t.to(device) for t in inst.batch(rows))
    masks = [r['masks'] for r in rows]
    n = len(rows)
    base_f, base_m = _f(reference, ids, attention, positions, masks, weights, adapter=inst.adapter)
    items = []

    def gate(value):
        return torch.full((n, 1, 1), value, dtype=torch.float64, device=device)

    def record(kind, target, got, up, down):
        fd = (up - down) / (2 * epsilon)
        rel = abs(got - fd) / abs(fd) if fd != 0 else float('inf')
        items.append({'kind': kind, 'target': target, 'dJdlambda': float(got), 'finite_difference': float(fd),
                      'rel_err': float(rel)})

    for index in np.argsort(-np.abs(d_adapter), kind='mergesort')[:adapters]:
        up = _f(reference, ids, attention, positions, masks, weights, {int(index): gate(1 + epsilon)}, adapter=inst.adapter)[0]
        down = _f(reference, ids, attention, positions, masks, weights, {int(index): gate(1 - epsilon)}, adapter=inst.adapter)[0]
        record('adapter', {'adapter': int(index)}, d_adapter[index], up, down)
    layer = max(inst.down_index, key=lambda l: abs(d_adapter[inst.down_index[l]]))
    d_neuron = -(weights[:, None] * out['neuron']['seqlogp'][layer]).sum(0)
    for j in np.argsort(-np.abs(d_neuron), kind='mergesort')[:neurons]:
        up = _f(reference, ids, attention, positions, masks, weights, neuron=(layer, int(j), gate(1 + epsilon)), adapter=inst.adapter)[0]
        down = _f(reference, ids, attention, positions, masks, weights, neuron=(layer, int(j), gate(1 - epsilon)), adapter=inst.adapter)[0]
        record('neuron', {'layer': int(layer), 'neuron': int(j)}, d_neuron[j], up, down)
    errors = np.array([i['rel_err'] for i in items])
    floors = {'fp32_ulp_M_over_eps': float(np.max(np.spacing(np.abs(base_m).astype(np.float32)))) / epsilon,
              'fp64_ulp_M_over_eps': float(np.max(np.spacing(np.abs(base_m)))) / epsilon,
              'max_abs_M': float(np.max(np.abs(base_m)))}
    result = {'epsilon': epsilon, 'F': base_f, 'median_rel_err': float(np.median(errors)),
              'max_rel_err': float(errors.max()), 'items': items, 'floors': floors,
              'reference': 'fp64 via full_fp64', 'note': 'implementation check of dJdλ, not of the estimator'}
    if median_tol is not None and max_tol is not None:
        result['passed'] = bool(result['median_rel_err'] <= median_tol and result['max_rel_err'] <= max_tol)
    return result


# --- cached against teacher-forced -------------------------------------------------------------------

def cached_vs_forced(model, rows, prompt_lengths, *, pad_token_id):
    """Left-padded prefill + incremental KV-cache decoding of the saved tokens, against full forcing.

    ``rows`` are instrument rows (prompt ids then generated ids; ``seq_targets`` = generated positions);
    ``prompt_lengths`` gives each row's prompt length.
    """
    import torch
    device = next(model.parameters()).device
    prompts = [r['input_ids'][:p] for r, p in zip(rows, prompt_lengths)]
    generated = [r['input_ids'][p:] for r, p in zip(rows, prompt_lengths)]
    width = max(len(p) for p in prompts)
    ids = torch.tensor([[pad_token_id] * (width - len(p)) + p for p in prompts], device=device)
    attention = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=device)
    cached = [[] for _ in rows]
    steps = max(len(g) for g in generated)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attention,
                    position_ids=(attention.cumsum(1) - 1).clamp(min=0), use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1]
        for t in range(steps):
            logp = torch.log_softmax(logits.float(), -1)
            tokens = []
            for r, g in enumerate(generated):
                if t < len(g):
                    cached[r].append(float(logp[r, g[t]]))
                tokens.append(g[t] if t < len(g) else pad_token_id)
            if t == steps - 1:
                break
            live = torch.tensor([[1 if t < len(g) else 0] for g in generated], device=device)
            attention = torch.cat([attention, live], 1)
            out = model(input_ids=torch.tensor(tokens, device=device)[:, None], attention_mask=attention,
                        position_ids=(attention.sum(1, keepdim=True) - 1), past_key_values=past, use_cache=True)
            past, logits = out.past_key_values, out.logits[:, -1]
        inst = ai.Instrument(model, stop_ids=[0], pad_token_id=pad_token_id, identity_rtol=1.0)
        fids, fatt, fpos = inst.batch(rows)
        flogits = model(input_ids=fids, attention_mask=fatt, position_ids=fpos, use_cache=False).logits
    per_token, per_row = [], []
    for r, row in enumerate(rows):
        positions = am.logit_positions(row['masks']['seq_targets'], length=len(row['input_ids']))
        logp = torch.log_softmax(flogits[r, positions].float(), -1)
        forced = [float(logp[k, row['input_ids'][p + 1]]) for k, p in enumerate(positions)]
        diffs = np.abs(np.array(forced) - np.array(cached[r]))
        per_token.extend(diffs.tolist())
        per_row.append(abs(float(np.sum(forced, dtype=np.float64) - np.sum(cached[r], dtype=np.float64))))
    tokens = np.array(per_token)
    return {'per_token_abs': {'max': float(tokens.max()), 'median': float(np.median(tokens)),
                              'p99': float(np.percentile(tokens, 99))},
            'per_row_abs_M': {'max': float(max(per_row)), 'median': float(np.median(per_row))},
            'rows': [r['id'] for r in rows], 'generated_lengths': [len(g) for g in generated],
            'note': 'reconstructed cached decoding on this runtime; the original generation scores were not saved'}


# --- batch invariance ---------------------------------------------------------------------------------

def batch_invariance(inst, rows, budgets, *, tolerance=None):
    """|Δ attr_seqlogp| between singleton and packed batches, absolute and as a fraction of max |attr|."""
    single = {r['id']: inst.attribute([r])['attr']['seqlogp'][0] for r in rows}
    scale = float(np.max(np.abs(np.stack(list(single.values())))))
    worst_abs = 0.0
    for budget in budgets:
        for batch in pack(rows, budget):
            out = inst.attribute(batch)['attr']['seqlogp']
            for k, row in enumerate(batch):
                worst_abs = max(worst_abs, float(np.max(np.abs(out[k] - single[row['id']]))))
    result = {'max_abs': worst_abs, 'max_fraction': worst_abs / scale if scale > 0 else 0.0,
              'scale_max_abs_attr': scale, 'budgets': list(budgets)}
    if tolerance is not None:
        result['passed'] = result['max_fraction'] <= tolerance
    return result


# --- no-op and restore ----------------------------------------------------------------------------------

def adapter_hash(model):
    digest = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        if 'lora_' in name:
            digest.update(name.encode())
            digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def noop_check(model, rows, context_factory, *, greedy_tokens=4, pad_token_id=0):
    """``context_factory()`` builds the intervention context with an empty removal set."""
    import torch
    inst = ai.Instrument(model, stop_ids=[0], pad_token_id=pad_token_id, identity_rtol=1.0)
    ids, attention, positions = inst.batch(rows)
    before = adapter_hash(model)

    def logits():
        with torch.no_grad():
            return model(input_ids=ids, attention_mask=attention, position_ids=positions, use_cache=False).logits

    def greedy():
        prompt = torch.tensor([rows[0]['input_ids'][:4]])
        with torch.no_grad():
            return model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), do_sample=False,
                                  max_new_tokens=greedy_tokens, min_new_tokens=greedy_tokens,
                                  pad_token_id=pad_token_id, num_beams=1)[0].tolist()

    intact, intact_tokens = logits(), greedy()
    with context_factory():
        inside, inside_tokens = logits(), greedy()
    after = adapter_hash(model)
    result = {'logits_bit_identical': bool(torch.equal(intact, inside)),
              'greedy_tokens_identical': intact_tokens == inside_tokens,
              'adapter_hash_before': before, 'adapter_hash_after': after, 'restored': before == after}
    result['passed'] = result['logits_bit_identical'] and result['greedy_tokens_identical'] and result['restored']
    return result


# --- replay (report-only) --------------------------------------------------------------------------------

def replay_report(sealed, regenerated):
    """``sealed`` and ``regenerated`` map row id -> generated token ids. Report-only: no pass flag."""
    rows = {}
    for rid, tokens in sealed.items():
        other = regenerated.get(rid)
        if other is None:
            rows[rid] = {'identical': False, 'first_divergence': None, 'missing': True}
            continue
        first = next((i for i, (a, b) in enumerate(zip(tokens, other)) if a != b), None)
        if first is None and len(tokens) != len(other):
            first = min(len(tokens), len(other))
        rows[rid] = {'identical': first is None, 'first_divergence': first}
    identical = sum(r['identical'] for r in rows.values())
    return {'rows': rows, 'identical_rows': identical, 'total_rows': len(rows),
            'review': identical < len(rows),
            'note': 'report-only; a divergence is material only if it points to a policy or checkpoint mismatch'}


# --- numerics record ------------------------------------------------------------------------------------

def numerics_record(*, batch_invariance=None, identity=None, gross_net_ratios=None, finite_difference=None,
                    cached_vs_forced=None):
    """Assemble ``numerics.json``. These inform launch-record tolerances; they never loosen historical ones."""
    record = {'historical_tolerances': PRODUCTION_1}
    if batch_invariance is not None:
        record['batch_invariance'] = {k: batch_invariance[k] for k in ('max_abs', 'max_fraction', 'budgets')}
    if identity is not None:
        record['identity'] = identity
    if gross_net_ratios is not None:
        r = np.asarray(gross_net_ratios, dtype=np.float64)
        r = r[np.isfinite(r)]
        record['gross_net_ratio'] = ({q: float(np.percentile(r, p)) for q, p in
                                      (('min', 0), ('p01', 1), ('p10', 10), ('median', 50), ('p90', 90), ('max', 100))}
                                     if r.size else None)
    if finite_difference is not None:
        record['finite_difference'] = {k: finite_difference[k] for k in ('epsilon', 'median_rel_err', 'max_rel_err', 'floors')}
    if cached_vs_forced is not None:
        record['cached_vs_forced'] = {k: cached_vs_forced[k] for k in ('per_token_abs', 'per_row_abs_M')}
    return record


def write_numerics(path, record):
    Path(path).write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')


def empty_context():
    return ExitStack()
