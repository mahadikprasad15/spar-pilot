"""Pre-update, stopped-sequence fixed-pool log-ratio diagnostics."""
import math


def summarize(log_mu, log_pi):
    if len(log_mu) != len(log_pi) or not log_mu:
        raise ValueError('aligned nonempty sequences required')
    sums = []
    tokens = 0
    for mu, pi in zip(log_mu, log_pi):
        if not mu or len(mu) != len(pi) or any(not math.isfinite(x) for x in mu + pi):
            raise ValueError('nonfinite or misaligned token log probabilities')
        sums.append(sum(a-b for a, b in zip(mu, pi))); tokens += len(mu)
    ordered = sorted(sums)
    def percentile(q):
        position = (len(ordered)-1)*q
        lo = int(position); hi = min(lo+1, len(ordered)-1)
        return ordered[lo] + (ordered[hi]-ordered[lo])*(position-lo)
    return dict(sequence_sums=sums, sequence_mean=sum(sums)/len(sums),
                token_rate=sum(sums)/tokens, token_count=tokens,
                percentiles={str(q): percentile(q/100) for q in (0, 5, 50, 95, 100)},
                negative_sequences=sum(x < 0 for x in sums),
                interpretation='empirical fixed-data log ratio; KL(mu || pi) requires fixed, unselected fresh mu samples')


def score_window(model, rows, *, pad_token_id, batch_size=8):
    """Score generated tokens including EOS with both policies before update.

    Right padding is masked, and the logits immediately preceding each generated
    token are gathered. Values are emitted per draw, never as vocabulary arrays.
    """
    import random
    import numpy as np
    import torch
    if batch_size <= 0: raise ValueError('positive diagnostic batch size required')
    flat = [(r, d) for r in rows for d in r['draws']]
    if len(flat) != 64: raise ValueError('one window of 64 draws required')
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    modes = [(m, m.training) for m in model.modules()]
    device = next(model.parameters()).device
    pad = pad_token_id
    if pad is None: raise ValueError('diagnostic pad token required')
    def score(disabled):
        from contextlib import nullcontext
        context = model.disable_adapter() if disabled else nullcontext()
        answers = []
        with context, torch.inference_mode():
            for start in range(0, len(flat), batch_size):
                batch = flat[start:start+batch_size]
                seqs = [r['input_ids'] + d['generated_token_ids'] for r,d in batch]
                width = max(map(len, seqs))
                ids = torch.tensor([s+[pad]*(width-len(s)) for s in seqs], device=device)
                attention = torch.tensor([[1]*len(s)+[0]*(width-len(s)) for s in seqs], device=device)
                logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
                for i, (row, draw) in enumerate(batch):
                    count = len(draw['generated_token_ids']); offset = len(row['input_ids'])-1
                    selected = logits[i,offset:offset+count].float().log_softmax(-1)
                    targets = torch.tensor(draw['generated_token_ids'], device=device)
                    values = selected.gather(-1, targets[:, None]).squeeze(-1)
                    if not bool(torch.isfinite(values).all()): raise ValueError('nonfinite diagnostic log probabilities')
                    answers.append(values.cpu().tolist())
        return answers
    try:
        model.eval(); log_mu = score(True); log_pi = score(False)
        summary = summarize(log_mu, log_pi)
        summary['group_means'] = [sum(summary['sequence_sums'][i:i+8])/8 for i in range(0,64,8)]
        return dict(protocol='offline-logratio-v1', timing='before-update',
            draw_ids=[d['draw_id'] for _,d in flat], question_ids=[r['id'] for r,_ in flat],
            generated_token_ids=[d['generated_token_ids'] for _,d in flat],
            masks=[d['completion_mask'] for _,d in flat], log_mu=log_mu, log_pi=log_pi,
            diagnostic_batch_size=batch_size, mu_fidelity='pending-real-generation-score-check', summary=summary)
    finally:
        for module, training in modes: module.training = training
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        if rng[3] is not None: torch.cuda.set_rng_state_all(rng[3])


def independent_push(model, prompts, completions, advantages, names):
    """Initial detached-ratio gradient, independently accumulated per sequence."""
    import torch
    params = dict(model.named_parameters())
    selected = [params[n] for n in names]
    total = sum(map(len, completions)); device = next(model.parameters()).device
    accumulated = [torch.zeros_like(p) for p in selected]
    for i, (tokens, advantage) in enumerate(zip(completions, advantages)):
        if advantage == 0: continue
        prefix = prompts[i // 8]; full = prefix + tokens
        ids = torch.tensor([full], device=device)
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        positions = logits[0, len(prefix)-1:len(prefix)-1+len(tokens)].float()
        targets = torch.tensor(tokens, device=device)
        logp = positions.log_softmax(-1).gather(-1, targets[:,None]).squeeze(-1)
        loss = -logp.sum() * float(advantage) / total
        grads = torch.autograd.grad(loss, selected, allow_unused=True)
        for dst, grad in zip(accumulated, grads):
            if grad is not None: dst.add_(grad.detach())
    return -torch.cat([g.reshape(-1).cpu().double() for g in accumulated])


def measure_generation_scores(model, prompts, *, eos_token_id, pad_token_id, max_new_tokens,
                              long_min_tokens):
    """Measure actual generation scores against teacher forcing on new draws.

    Caller owns authorization and runtime identity. No tolerance is inferred;
    differences are observations, not an automatic numerical pass.
    """
    import random
    import numpy as np
    import torch
    if not prompts or not 1 <= long_min_tokens <= max_new_tokens:
        raise ValueError('explicit bounded prompts and long-answer criterion required')
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    modes = [(m, m.training) for m in model.modules()]
    device = next(model.parameters()).device
    rows = []
    try:
        model.eval()
        with model.disable_adapter(), torch.inference_mode():
            for prompt in prompts:
                ids = torch.tensor([prompt], device=device)
                result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    do_sample=True, temperature=1., top_p=1., top_k=0, num_beams=1,
                    max_new_tokens=max_new_tokens, pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id, return_dict_in_generate=True, output_scores=True)
                tokens = result.sequences[0, len(prompt):].tolist()
                if not tokens or len(tokens) != len(result.scores): raise ValueError('generation score alignment failed')
                generation = [float(s[0].float().log_softmax(-1)[token]) for s,token in zip(result.scores,tokens)]
                full = result.sequences
                logits = model(input_ids=full, attention_mask=torch.ones_like(full), use_cache=False).logits
                target = torch.tensor(tokens, device=device)
                teacher = logits[0,len(prompt)-1:len(prompt)-1+len(tokens)].float().log_softmax(-1)
                teacher = teacher.gather(-1,target[:,None]).squeeze(-1).cpu().tolist()
                if any(not math.isfinite(x) for x in generation+teacher): raise ValueError('nonfinite fidelity observation')
                rows.append(dict(prompt_token_ids=prompt, generated_token_ids=tokens,
                    generation_log_probs=generation, teacher_forced_log_probs=teacher,
                    differences=[a-b for a,b in zip(generation,teacher)],
                    eos=tokens[-1] == eos_token_id, capped=len(tokens)==max_new_tokens and tokens[-1]!=eos_token_id))
        if not any(r['eos'] for r in rows) or not any(r['capped'] or len(r['generated_token_ids'])>=long_min_tokens for r in rows):
            raise ValueError('fidelity observations require EOS and long or capped draw')
        return dict(protocol='offline-mu-fidelity-observation-v1', long_min_tokens=long_min_tokens,
                    eos_token_id=eos_token_id, pad_token_id=pad_token_id,
                    max_new_tokens=max_new_tokens, observations=rows,
                    comparison='recorded differences; no numerical agreement claimed')
    finally:
        for module, training in modes: module.training = training
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
        if state[3] is not None: torch.cuda.set_rng_state_all(state[3])
