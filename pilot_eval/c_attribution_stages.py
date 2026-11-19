"""C-attribution stage bodies (ticket 09), composed from tickets 01-08.

Each stage takes the chain ``Context``, keeps its state in sealed run directories under ``c-attribution/``, and
returns ``outputs`` (the sealed directories the chain re-verifies before skipping it on resume). Model access
goes through a backend (real: ``c_attribution_backend.TransformersBackend``; the CPU chain test uses the same
class on a tiny saved Qwen2), so the same code runs on CPU and on the Pod.

- **preflight**: backend facts (disk, CUDA, HF_HOME off the volume, volume identity, private export repo), the
  input bundle against its seal, the base-model files, the catalog revision and ``catalog verify``.
- **qualification** (rows_per_arm rows per arm): (a) cached against forced, (c) surrogate finite differences
  against the FP64 reference, (d) batch invariance at the token budget, (e) the neuron-sum identity, (f) no-op
  interventions and logits processors, and peak memory at the token budget. (b) replay is report-only and
  is recorded as not run (see ``REPLAY_NOTE``). Returns checks for the automatic sign-off.
- **rows**: every arm's rows, re-validated and tokenized with the pinned tokenizer, sealed.
- **attribution**: one sealed unit per block of whole questions (all 8 draws together), micro-batched by the
  token budget, with per-question S_q (adapters and down_proj reads), draw-half sums and selection measures.
- **aggregate_freeze**: aggregate, hot-layer re-derivation, splits, bootstraps, selection and matched
  controls, the freeze (with the fixed generation plan). Analysis outcomes are reported, never raised.
- **generation**: the frozen conditions over one fixed batch plan; the untuned base once.
- **report**: per-arm condition summaries and the online/offline comparison, sealed.
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from pilot_eval import attribution_units as au
from pilot_eval import c_attribution as ca
from pilot_eval import c_attribution_bundle as cb
from pilot_eval import c_attribution_checks as ck
from pilot_eval import c_attribution_generation as cg
from pilot_eval import c_attribution_rows as cr
from pilot_eval import c_attribution_selection as sel
from pilot_eval import c_attribution_stability as st
from pilot_eval.attribution_chain import StopBeforeProduction
from pilot_eval.c_attribution_chain import SUBTREE, record_sha256
from pilot_eval.neuron_audit_qualify import pack

PREFLIGHT_DIR = f'{SUBTREE}/preflight'
QUAL_DIR = f'{SUBTREE}/qualification'
ROWS_DIR = f'{SUBTREE}/rows'
ATTR_DIR = f'{SUBTREE}/attribution'
AGG_DIR = f'{SUBTREE}/aggregate'
REPORT_DIR = f'{SUBTREE}/report'
UNTUNED_ARM = 'untuned'
# The settings grpo_gpu._generate read for the source runs' sampled step-64 policy (c_attribution_rows.POLICY).
SAMPLED_SETTINGS = {'evaluation_batch_size': cr.POLICY['batch_size'], 'generation_groups': cr.POLICY['batch_size'],
                    'completion_limit': cr.POLICY['completion_limit'], 'top_p': cr.POLICY['top_p'],
                    'top_k': cr.POLICY['top_k']}
REPLAY_NOTE = ('not run: replaying a sealed sampled batch needs the source run\'s sampled generation path '
               '(grpo_gpu._generate with its seed schedule); report-only, so it does not gate. Open for review.')


def _npz(**arrays):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _read_npz(path):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _sealed(directory):
    au.verify_run(directory)
    return directory


def _jsonl(text):
    return [json.loads(line) for line in text.splitlines() if line]


class Stages:
    def __init__(self, record, local_root, backend):
        self.record, self.backend = record, backend
        self.root = Path(local_root)
        self.record_sha256 = record_sha256(record)
        self.arms = list(record['arms'])
        self.artifacts = Path(record['inputs']['artifact_root'])
        self._bundle = None
        self._loaded = {}

    def mapping(self):
        return {name: getattr(self, name) for name in ['preflight', 'qualification', 'rows', 'attribution',
                                                        'aggregate_freeze', 'generation', 'report']}

    # --- inputs -----------------------------------------------------------------------------------

    def bundle(self):
        if self._bundle is None:
            self._bundle = json.loads(Path(self.record['inputs']['bundle']).read_text())
        return self._bundle

    def file(self, role):
        entry = next((e for e in self.bundle()['files'] if e['role'] == role), None)
        if entry is None:
            raise StopBeforeProduction(f'the input bundle has no {role}')
        return entry

    def source(self, arm):
        spec = self.record['inputs']['sources'][arm]
        responses, config = self.file(spec['responses_role']), self.file(spec['config_role'])
        adapter = self.file(f"{spec['adapter_role']}:adapter_model.safetensors")
        return {'arm': arm, 'catalog_id': spec['catalog_id'],
                'responses': {'path': self.artifacts / responses['path'], 'sha256': responses['sha256']},
                'config': {'path': self.artifacts / config['path'], 'sha256': config['sha256']},
                'adapter_sha256': adapter['sha256'], 'eos_token_id': self.record['generation']['eos_token_id'],
                # The sampled policy the runs recorded, with the record's cap (production requires 1024).
                'policy': dict(cr.POLICY, completion_limit=self.record['generation']['cap'])}

    def loaded(self, arm):
        """Rows re-validated against the bound inputs (refusals are input-integrity stops) and tokenized."""
        if arm not in self._loaded:
            try:
                loaded = cr.load(self.source(arm))
                rows, manifest = cr.instrument_rows(loaded, self.backend.encode,
                                                    tokenizer_identity=self.backend.tokenizer_identity())
            except cr.RowsError as exc:
                raise StopBeforeProduction(f'{arm} rows: {exc}')
            self._loaded[arm] = dict(loaded=loaded, rows=rows, manifest=manifest)
        return self._loaded[arm]

    def scope(self, part, arm, available):
        chosen = self.record['scope'][part][arm]
        if chosen == 'all':
            return list(available)
        missing = sorted(set(chosen) - set(available))
        if missing:
            raise StopBeforeProduction(f'scope.{part}.{arm} names unknown questions {missing}')
        return list(chosen)

    def rewards(self, rows):
        out = {}
        for r in sorted(rows, key=lambda r: (r['question_id'], r['draw_index'])):
            out.setdefault(r['question_id'], []).append(int(r['reward']))
        return out

    def instrument(self, model):
        from pilot_eval.attribution_instrument import Instrument
        return Instrument(model, stop_ids=[self.backend.eos_token_id], pad_token_id=self.backend.pad_token_id,
                          identity_rtol=float(self.record['tolerances']['neuron_sum_identity']),
                          head_chunk=self.record['qualification']['head_chunk'], metric_order=('seqlogp',))

    # --- preflight ----------------------------------------------------------------------------------

    def preflight(self, ctx):
        problems = list(self.backend.preflight())
        record = {}
        try:
            record['bundle'] = cb.verify(self.artifacts, self.bundle())
        except (cb.BundleError, OSError, ValueError) as exc:
            problems.append(f'input bundle: {exc}')
        try:
            cb.check_base_model(self.record['inputs']['model_dir'], self.bundle())
            record['base_model'] = 'pass'
        except cb.BundleError as exc:
            problems.append(str(exc))
        hot = self.record['hot_layers']['source']
        try:
            entry = self.file(hot['role'])
            if entry['sha256'] != hot['sha256'] or entry['path'] != hot['path']:
                problems.append('the launch record hot-layer source differs from the bundled file')
        except StopBeforeProduction as exc:
            problems.append(str(exc))
        try:
            ids = [self.record['inputs']['sources'][a]['catalog_id'] for a in self.arms] + ['cohort:gsm8k-test-cohort150']
            record['catalog'] = cb.pod_catalog_check(self.backend.repo_root, self.bundle(), self.artifacts, ids)
        except Exception as exc:  # a missing or differing catalog is an input-integrity stop
            problems.append(f'catalog: {type(exc).__name__}: {exc}')
        if problems:
            raise StopBeforeProduction(f'preflight: {problems}')
        record['runtime'] = self.backend.runtime()
        au.seal_run(self.root / PREFLIGHT_DIR, {'preflight.json': record})
        return dict(outputs=[PREFLIGHT_DIR])

    # --- qualification ------------------------------------------------------------------------------

    def _qualification_rows(self, rows):
        """rows_per_arm rows spread over length, always including every capped row the arm has."""
        n = self.record['qualification']['rows_per_arm']
        ordered = sorted(rows, key=lambda r: (len(r['input_ids']), r['id']))
        capped = [r for r in ordered if len(r['masks']['seq_targets']) >= self.record['generation']['cap']]
        picks = {r['id'] for r in capped}
        for i in np.linspace(0, len(ordered) - 1, num=min(n, len(ordered))).round().astype(int):
            if len(picks) >= n:
                break
            picks.add(ordered[int(i)]['id'])
        for r in ordered:
            if len(picks) >= n:
                break
            picks.add(r['id'])
        chosen = [r for r in ordered if r['id'] in picks]
        return chosen, len(capped)

    def _fd_rows(self, rows):
        """fd_rows rows from mixed-reward questions (nonzero leave-one-out weights)."""
        rewards = self.rewards(rows)
        mixed = [q for q, r in rewards.items() if 0 < sum(r) < len(r)]
        pool = [r for r in rows if r['question_id'] in mixed]
        picked = sorted(pool, key=lambda r: (len(r['input_ids']), r['id']))[:self.record['qualification']['fd_rows']]
        req = ca.reduction_request(picked, rewards)
        return picked, req['coefficients']

    def qualification(self, ctx):
        tol, q = self.record['tolerances'], self.record['qualification']
        checks, flags, details = {}, [], {}
        budget = self.record['attribution']['token_budget']
        for arm in self.arms:
            ctx.log(f'qualification {arm}')
            arm_rows = self.loaded(arm)['rows']
            rows, capped = self._qualification_rows(arm_rows)
            model = self.backend.load(arm)
            inst = self.instrument(model)
            d = details[arm] = {'rows': [r['id'] for r in rows], 'capped_rows_available': capped}
            if capped == 0:
                d['capped_note'] = 'this arm has no capped rows; recorded, not required'
            # (a) cached against forced
            prompt_lengths = [r['masks']['seq_targets'][0] for r in rows]
            cvf = ck.cached_vs_forced(model, rows, prompt_lengths, pad_token_id=self.backend.pad_token_id)
            d['cached_vs_forced'] = cvf
            limits = tol['cached_vs_forced']
            if limits['per_token_abs_max'] == 'measure-only':
                d['cached_vs_forced_rule'] = 'measure-only (rehearsal)'
            else:
                checks[f'{arm}:cached_vs_forced'] = bool(
                    cvf['per_token_abs']['max'] <= limits['per_token_abs_max']
                    and cvf['per_row_abs_M']['max'] <= limits['per_row_abs_M_max'])
            # (b) replay: report-only
            d['replay'] = {'status': 'not-run', 'note': REPLAY_NOTE}
            # (c) surrogate finite differences, FP64 reference
            fd_rows, weights = self._fd_rows(arm_rows)
            reference = self.backend.reference(arm)
            fd = ck.surrogate_finite_difference(inst, reference, fd_rows, weights,
                                                epsilon=tol['finite_difference']['epsilon'],
                                                adapters=q['fd_adapters'], neurons=q['fd_neurons'],
                                                median_tol=tol['finite_difference']['median_rel_err_max'],
                                                max_tol=tol['finite_difference']['max_rel_err_max'])
            self.backend.release(reference)
            d['finite_difference'] = fd
            checks[f'{arm}:finite_difference'] = bool(fd['passed'])
            # (d) batch invariance at the token budget
            bi = ck.batch_invariance(inst, rows, q['batch_invariance_budgets'], tolerance=tol['batch_invariance'])
            d['batch_invariance'] = bi
            checks[f'{arm}:batch_invariance'] = bool(bi['passed'])
            # (e) identity: the reduced path raises on a violation; record the measured errors
            rewards = self.rewards(arm_rows)
            whole = [r for r in arm_rows if r['question_id'] in {x['question_id'] for x in rows[:2]}]
            identity = []
            for batch in pack(whole, budget):
                out = inst.attribute(batch, neuron_reduction=ca.reduction_request(batch, rewards))
                identity += [dict(layer=int(layer), **{k: np.asarray(v).tolist() for k, v in
                                                       block['identity'].items()})
                             for layer, block in out['neuron']['seqlogp'].items()]
            d['identity'] = identity
            checks[f'{arm}:identity'] = True
            # (f) no-op interventions, then the logits processors the source runs resolved
            noop = ck.noop_check(model, rows[:2], ck.empty_context, greedy_tokens=q['noop_greedy_tokens'],
                                 pad_token_id=self.backend.pad_token_id)
            d['noop'] = noop
            checks[f'{arm}:noop'] = bool(noop['passed'])
            proc = ck.processor_check(model, self.backend.hf_tokenizer(), settings=SAMPLED_SETTINGS,
                                      generation_config_path=self.backend.generation_config_path)
            d['processors'] = proc
            checks[f'{arm}:processors'] = bool(proc['passed'])
            # peak memory at the token budget
            self.backend.reset_peak()
            longest = max(pack(arm_rows, budget), key=lambda b: len(b) * max(len(r['input_ids']) for r in b))
            inst.attribute(longest, neuron_reduction=ca.reduction_request(longest, rewards))
            peak, total = self.backend.peak_memory(), self.backend.total_memory()
            d['memory'] = {'peak_bytes': peak, 'total_bytes': total, 'budget': budget}
            if total:
                checks[f'{arm}:memory'] = bool(peak <= tol['memory_fraction'] * total)
            self.backend.release(model)
            ctx.boundary(arm=arm)
        numerics = ck.numerics_record(
            batch_invariance=details[self.arms[0]]['batch_invariance'],
            finite_difference=details[self.arms[0]]['finite_difference'],
            cached_vs_forced=details[self.arms[0]]['cached_vs_forced'])
        numerics['per_arm'] = {a: ck.numerics_record(batch_invariance=details[a]['batch_invariance'],
                                                     finite_difference=details[a]['finite_difference'],
                                                     cached_vs_forced=details[a]['cached_vs_forced'])
                               for a in self.arms}
        result = dict(checks=checks, review_flags=flags, arms=list(details))
        au.seal_run(self.root / QUAL_DIR, {'qualification.json': dict(result, details=details),
                                           'numerics.json': numerics})
        return dict(result, outputs=[QUAL_DIR])

    # --- rows ---------------------------------------------------------------------------------------

    def rows(self, ctx):
        outputs, counts = [], {}
        for arm in self.arms:
            data = self.loaded(arm)
            body = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in data['rows'])
            manifest = dict(data['manifest'], rows_file_sha256=hashlib.sha256(body.encode()).hexdigest())
            directory = f'{ROWS_DIR}/{arm}'
            au.seal_run(self.root / directory, {'rows.jsonl': body, 'manifest.json': manifest})
            outputs.append(directory)
            counts[arm] = manifest['counts']
            ctx.boundary(arm=arm)
        return dict(outputs=outputs, counts=counts)

    def sealed_rows(self, arm):
        directory = _sealed(self.root / ROWS_DIR / arm)
        rows = _jsonl((directory / 'rows.jsonl').read_text())
        manifest = json.loads((directory / 'manifest.json').read_text())
        if cr.rows_sha256(rows) != manifest['rows_sha256']:
            raise StopBeforeProduction(f'{arm}: sealed rows do not match rows_sha256')
        return rows, manifest

    # --- attribution --------------------------------------------------------------------------------

    def units(self, arm, rows):
        order = []
        for r in rows:
            if r['question_id'] not in order:
                order.append(r['question_id'])
        chosen = self.scope('attribution_questions', arm, order)
        size = self.record['attribution']['questions_per_unit']
        return [chosen[i:i + size] for i in range(0, len(chosen), size)]

    def attribution(self, ctx):
        outputs = []
        budget = self.record['attribution']['token_budget']
        for arm in self.arms:
            rows, _ = self.sealed_rows(arm)
            rewards = self.rewards(rows)
            plan = self.units(arm, rows)
            model = None
            for index, questions in enumerate(plan):
                directory = f'{ATTR_DIR}/{arm}/unit-{index:04d}'
                outputs.append(directory)
                if (self.root / directory / 'complete.json').is_file():
                    meta = json.loads((_sealed(self.root / directory) / 'meta.json').read_text())
                    if meta['question_ids'] != questions:
                        raise StopBeforeProduction(f'{directory}: sealed unit holds other questions')
                    continue
                if model is None:
                    model = self.backend.load(arm)
                    inst = self.instrument(model)
                    layers = sorted(inst.down_index)
                    width = model.config.intermediate_size
                unit_rows = [r for r in rows if r['question_id'] in set(questions)]
                sums = ca.QuestionSums(questions, n_adapters=len(inst.adapters), layers=layers, width=width)
                term = np.zeros((len(layers), width))
                abs_c = np.zeros(len(inst.adapters))
                tokens, identity_max = 0, 0.0
                ratios = []
                for batch in pack(unit_rows, budget):
                    req = ca.reduction_request(batch, rewards)
                    out = inst.attribute(batch, neuron_reduction=req)
                    sums.add_adapters(req['question_ids'], [r['draw_index'] for r in batch], out['attr']['seqlogp'],
                                      req['coefficients'], req['half_coefficients'])
                    sums.add_reads(out['neuron']['seqlogp'], out['neuron_question_ids'])
                    for li, layer in enumerate(layers):
                        term[li] += out['term_size'][layer]
                        ident = out['neuron']['seqlogp'][layer]['identity']
                        err, gross = np.asarray(ident['error']), np.asarray(ident['gross'])
                        identity_max = max(identity_max, float(np.max(err / np.maximum(gross, 1e-300))))
                        ratios += (np.abs(np.asarray(ident['net'])) / np.maximum(gross, 1e-300)).tolist()
                    abs_c += np.asarray(out['sum_abs_c']).sum(0)
                    tokens += int(np.asarray(out['token_count']).sum())
                sums.complete(len(unit_rows))
                meta = dict(arm=arm, question_ids=questions, rows=len(unit_rows), tokens=tokens,
                            identity_max_relative=identity_max, net_over_gross=ratios, layers=layers,
                            adapters=[list(a) for a in inst.adapters], launch_record_sha256=self.record_sha256)
                au.seal_run(self.root / directory, {
                    'meta.json': meta,
                    'sums.npz': _npz(adapters=sums.adapters, adapter_halves=sums.adapter_halves, reads=sums.reads,
                                     read_halves=sums.read_halves, term_size=term, sum_abs_c=abs_c)})
                ctx.boundary(arm=arm, unit=index, of=len(plan))
            if model is not None:
                self.backend.release(model)
        return dict(outputs=outputs)

    # --- aggregate, split, bootstrap, selection, freeze ------------------------------------------

    def _arm_sums(self, arm):
        rows, _ = self.sealed_rows(arm)
        plan = self.units(arm, rows)
        order = [q for unit in plan for q in unit]
        metas, arrays = [], []
        for index in range(len(plan)):
            directory = _sealed(self.root / ATTR_DIR / arm / f'unit-{index:04d}')
            metas.append(json.loads((directory / 'meta.json').read_text()))
            arrays.append(_read_npz(directory / 'sums.npz'))
        layers = metas[0]['layers']
        sums = ca.QuestionSums(order, n_adapters=arrays[0]['adapters'].shape[1], layers=layers,
                               width=arrays[0]['reads'].shape[2])
        sums.adapters = np.concatenate([a['adapters'] for a in arrays])
        sums.reads = np.concatenate([a['reads'] for a in arrays])
        sums.adapter_halves = sum(a['adapter_halves'] for a in arrays)
        sums.read_halves = sum(a['read_halves'] for a in arrays)
        measures = dict(term_size=sum(a['term_size'] for a in arrays), sum_abs_c=sum(a['sum_abs_c'] for a in arrays),
                        tokens=sum(m['tokens'] for m in metas))
        adapters = [tuple(a) for a in metas[0]['adapters']]
        down_index = {l: i for i, (l, p) in enumerate(adapters) if p == 'down_proj'}
        numerics = dict(identity_max_relative=max(m['identity_max_relative'] for m in metas),
                        net_over_gross=[r for m in metas for r in m['net_over_gross']])
        return sums, measures, adapters, down_index, layers, numerics, rows

    def aggregate_freeze(self, ctx):
        outputs, analysis, by_arm, adapter_S = [], {}, {}, {}
        hot = dict(self.record['hot_layers'])
        hot['source'] = dict(path=str(self.artifacts / hot['source']['path']), sha256=hot['source']['sha256'])
        sel_cfg, stats = self.record['selection'], self.record['statistics']
        for arm in self.arms:
            sums, measures, adapters, down_index, layers, numerics, rows = self._arm_sums(arm)
            summary = ca.summary(sums, adapters=adapters, down_index=down_index)
            summary['launch_record_sha256'] = self.record_sha256
            bound = sel.check_hot_layer_binding({'hot_layers': hot}, summary, launch_record_sha256=self.record_sha256)
            split = st.question_split(sums.order, seed=stats['split_seed'])
            stability = st.stability(sums, hot_layers=bound['layers'], split=split)
            counts = st.resample_counts(len(sums.order), B=stats['resamples'], seed=stats['bootstrap_seed'])
            neuron_boot = st.neuron_bootstrap(sums.reads, counts, tiers=('per_layer',))
            b_norms, scales = self.backend.adapter_norms(arm, adapters)
            adapter_size = sel.adapter_measures(b_norms=b_norms, scale=scales, sum_abs_c=measures['sum_abs_c'],
                                                token_total=measures['tokens'])
            read_size = sel.read_measures(term_size_total=measures['term_size'], token_total=measures['tokens'])
            selection = sel.select_arm(summary, adapters=adapters, layers=layers, adapter_size=adapter_size,
                                       read_size=read_size, hot_layers=bound['layers'], seeds=sel_cfg['seeds'],
                                       tolerance=sel_cfg['tolerance'], k_adapters=sel_cfg['k_adapters'],
                                       k_reads=sel_cfg['k_reads'], k_harmful=sel_cfg['k_harmful'])
            gen_ids = self.scope('generation_questions', arm, [r['question_id'] for r in rows])
            gen_ids = list(dict.fromkeys(gen_ids))
            plan = cg.batches(gen_ids, self.record['generation']['batch_size'])
            freeze = sel.build_freeze(arm, selection, launch_record_sha256=self.record_sha256, hot=bound,
                                      split={'seed': stats['split_seed'], 'halves': split},
                                      inputs={'rows_sha256': self.sealed_rows(arm)[1]['rows_sha256']})
            freeze['generation_plan'] = plan
            directory = f'{AGG_DIR}/{arm}'
            ratios = np.asarray(numerics['net_over_gross'])
            au.seal_run(self.root / directory, {
                'adapters.json': {k: v for k, v in summary.items() if k != 'reads'} | {
                    'adapters': {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                                 for k, v in summary['adapters'].items()}},
                'neurons.npz': _npz(dJdlambda=np.asarray(summary['reads']['dJdlambda']),
                                    removal_J=np.asarray(summary['reads']['removal_J']), S=sums.reads,
                                    read_halves=sums.read_halves, read_size=read_size),
                'split.json': {'seed': stats['split_seed'], 'halves': split, 'stability': stability},
                'bootstrap.json': {'neurons': neuron_boot, 'resamples': stats['resamples'],
                                   'seed': stats['bootstrap_seed']},
                'identity.json': summary.get('identity'),
                'numerics.json': ck.numerics_record(identity=numerics['identity_max_relative'],
                                                    gross_net_ratios=ratios)})
            sel.seal_freeze(self.root, arm, freeze)
            outputs += [directory, f'{sel.FREEZE_DIR}/{arm}']
            by_arm[arm], adapter_S[arm] = summary, sums.adapters
            analysis[arm] = {'controls_unavailable': [g for g, v in selection['groups'].items()
                                                      if not v['control']['available']],
                             'shortfalls': {g: v['shortfall'] for g, v in selection['groups'].items()}}
            ctx.boundary(arm=arm)
        # Paired across arms by question ID (the arms' row files may order questions differently).
        first = by_arm[self.arms[0]]['question_ids']
        if all(set(by_arm[a]['question_ids']) == set(first) for a in self.arms):
            paired = {}
            for a in self.arms:
                position = {q: i for i, q in enumerate(by_arm[a]['question_ids'])}
                paired[a] = (first, adapter_S[a][[position[q] for q in first]])
            adapter_boot = st.adapter_bootstrap(paired, B=stats['resamples'], seed=stats['bootstrap_seed'])
        else:
            adapter_boot = {'status': 'not-run', 'reason': 'the arms cover different questions'}
        comparison = st.compare_adapters(np.asarray(by_arm[self.arms[0]]['adapters']['dJdlambda']),
                                         np.asarray(by_arm[self.arms[1]]['adapters']['dJdlambda']))
        au.seal_run(self.root / f'{AGG_DIR}/comparison', {'comparison.json': {'adapters': comparison,
                                                                             'adapter_bootstrap': adapter_boot}})
        outputs.append(f'{AGG_DIR}/comparison')
        return dict(outputs=outputs, analysis=analysis)

    # --- generation ---------------------------------------------------------------------------------

    def _questions(self, arm, ids):
        items = self.loaded(arm)['loaded']['items']
        return {q: {'prompt_ids': [int(t) for t in self.backend.encode(items[q]['prompt'])],
                    'prompt': items[q]['prompt'], 'gold': items[q]['gold']} for q in ids}

    def generation(self, ctx):
        wanted = self.record['scope']['generation_conditions']
        outputs, status, untuned_done = [], {}, False
        gen = self.record['generation']
        kw = dict(decode=self.backend.decode, pad_token_id=self.backend.pad_token_id,
                  eos_token_id=gen['eos_token_id'], cap=gen['cap'])
        for arm in self.arms:
            freeze = sel.read_freeze(self.root, arm)
            plan = freeze['generation_plan']
            ids = [q for b in plan for q in b]
            if plan != cg.batches(ids, gen['batch_size']):
                raise StopBeforeProduction(f'{arm}: the frozen batch plan does not follow the record batch size')
            questions = self._questions(arm, ids)
            model = self.backend.load(arm)
            for condition in cg.conditions(freeze):
                if wanted != 'all' and condition['name'] not in wanted:
                    continue
                status[f"{arm}/{condition['name']}"] = cg.run_condition(self.root, arm, condition, model, questions,
                                                                        plan, **kw)
                if condition['status'] == 'run':
                    outputs += [f'{sel.GENERATION_DIR}/{arm}/{condition["name"]}/batch-{b:04d}' for b in range(len(plan))]
                ctx.boundary(arm=arm, condition=condition['name'])
            if not untuned_done and (wanted == 'all' or 'untuned' in wanted):
                status['untuned'] = cg.run_condition(self.root, UNTUNED_ARM, cg.untuned_condition(), model, questions,
                                                     plan, **kw)
                outputs += [f'{sel.GENERATION_DIR}/{UNTUNED_ARM}/untuned/batch-{b:04d}' for b in range(len(plan))]
                untuned_done = True
                ctx.boundary(arm=UNTUNED_ARM, condition='untuned')
            self.backend.release(model)
        return dict(outputs=outputs, notes={'conditions': status})

    # --- report -------------------------------------------------------------------------------------

    def report(self, ctx):
        wanted = self.record['scope']['generation_conditions']
        arms, headline = {}, {}
        stats = self.record['statistics']
        for arm in self.arms:
            freeze = sel.read_freeze(self.root, arm)
            plan = freeze['generation_plan']
            ids = [q for b in plan for q in b]
            conds = [c for c in cg.conditions(freeze) if wanted == 'all' or c['name'] in wanted]
            results = {c['name']: cg.load_condition(self.root, arm, c['name'], plan) for c in conds if c['status'] == 'run'}
            counts = st.resample_counts(len(ids), B=stats['resamples'], seed=stats['bootstrap_seed'])
            arms[arm] = cg.summarize_arm(results, ids, conds, counts=counts)
            headline[arm] = {name: {k: v.get(k) for k in ('accuracy', 'delta_pp_vs_intact')}
                             for name, v in arms[arm]['conditions'].items() if isinstance(v, dict) and 'accuracy' in v}
        comparison = json.loads((_sealed(self.root / AGG_DIR / 'comparison') / 'comparison.json').read_text())
        body = dict(schema='c-attribution-report-v1', mode=self.record['mode'], label=self.record['label'],
                    launch_record_sha256=self.record_sha256, arms=arms, comparison=comparison,
                    labels=list(cg.LABELS))
        title = 'REHEARSAL — machinery test; numbers are not results' if self.record['mode'] == 'rehearsal' else \
            'C-attribution pilot report'
        markdown = '\n'.join([f'# {title}', '', *[f'- {label}' for label in cg.LABELS], '', '```',
                              json.dumps(headline, indent=2, default=str), '```', ''])
        au.seal_run(self.root / REPORT_DIR, {'report.json': body, 'report.md': markdown})
        return dict(outputs=[REPORT_DIR], headline=headline)


def build(record, local_root, volume_root, *, backend=None):
    if backend is None:
        from pilot_eval.c_attribution_backend import TransformersBackend
        backend = TransformersBackend(record, volume_root)
    return Stages(record, local_root, backend).mapping()
