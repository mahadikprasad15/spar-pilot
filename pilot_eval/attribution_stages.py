"""Ticket 11 (with 06b-09): the chain's seven stages, composed from tickets 01-05 and 10.

Every stage takes the chain ``Context`` and keeps its state in sealed run directories or units, so a resumed
chain rebuilds it from disk. Model access goes through a ``backend`` (real: ``attribution_backend`` loading the
inventory-v2 arms offline; tests: a tiny random Qwen2), so the same code runs on CPU fixtures and the Pod.

- **preflight**: backend facts (disk, volume identity, offline load, export check); any problem stops.
- **qualification** (calibration rows and rehearsal items only): calibration traces (ticket 01); every
  expected arm loads and passes routing; finite differences and batch invariance against the record's fixed
  tolerances; token-budget sweep (largest candidate with measured peak <= ``memory_fraction`` of the device;
  an OOM fails only that candidate); generation batch qualification on the record's rehearsal items. Returns
  checks / review flags / arms for the chain's automatic sign-off.
- **base_traces**: blind discovery and heldout traces at the qualified generation batch; minimums enforced;
  only counts are returned.
- **attribution_and_freeze**: pass 1 + pass 2 per (arm, batch) unit over discovery, then the discovery-only
  freeze.
- **stage_c**: <= 19 conditions per arm and metric on confirmation rows, read only after the freeze.
- **stage_d**: 9 generation runs on the ADR 0001 cohort from edited checkpoints at the qualified batch size.
- **report**: ``attribution_report.build`` + ``render``, sealed.

**Rehearsal mode** (record ``mode: rehearsal``) reads only visible inputs: the calibration rows stand in for
discovery *and* confirmation, and the record's 20 rehearsal items stand in for the ADR 0001 cohort. No blind
trace is generated, the freeze is labelled ``rehearsal`` (refused by production), there is no pilot comparison,
and the report is titled REHEARSAL. It exercises the machinery; its numbers mean nothing scientifically.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from pilot_eval import attribution_ablation as ab
from pilot_eval import attribution_gpu as gpu
from pilot_eval import attribution_instrument as ai
from pilot_eval import attribution_metrics as am
from pilot_eval import attribution_report as ar
from pilot_eval import attribution_traces as at
from pilot_eval import attribution_units as au
from pilot_eval.attribution_chain import StopBeforeProduction, run_conditions, run_units
from pilot_eval.neuron_audit_qualify import pack
from pilot_eval.scoring import score_gsm8k_flexible_v3

QUAL_DIR = 'attribution/qualification'
STAGE_D_DIR = 'attribution/stage_d'
REPORT_DIR = 'attribution/report'
STAGE_D_RUNS = {'gold-sft': ['full', 'top10-trace', 'match10-trace', 'top10-stop', 'match10-stop'],
                'online-grpo': ['full', 'top10-trace', 'match10-trace']}
METRICS = ('trace', 'stop')
COMPUTE_DIR = 'attribution/compute'
# (forward passes, activation-only backward passes) per token, for the MFU estimate
PASSES = dict(preflight=(1, 0), qualification=(1, 2), base_traces=(1, 0), attribution_and_freeze=(2, 2),
              stage_c=(2, 0), stage_d=(1, 0), report=(1, 0))


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _read(directory, name):
    au.verify_run(directory)
    return json.loads((Path(directory) / name).read_text())


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


class Stages:
    def __init__(self, record, local_root, backend, *, code_sha256):
        self.record, self.backend = record, backend
        self.root = Path(local_root)
        self.q = record['qualification']
        self.arms = list(record['expected_arms'])
        self.rehearsal = record['mode'] == 'rehearsal'
        self.units = au.Units(self.root, plan_sha256=_sha(record), code_sha256=code_sha256,
                              runtime_sha256=_sha(backend.runtime()))
        self.mon = None

    def mapping(self):
        return {name: self._monitored(name, getattr(self, name)) for name in
                ['preflight', 'qualification', 'base_traces', 'attribution_and_freeze', 'stage_c', 'stage_d',
                 'report']}

    def _monitored(self, name, fn):
        """Run a stage under a ``StageMonitor``; its record (and first-unit profiler trace) is sealed per attempt."""
        def run(ctx):
            stamp = time.time_ns()
            staging = self.root / 'profiles' / f'{name}-{stamp}.json'  # outside the export scope until sealed
            forward, backward = PASSES[name]
            self.mon = gpu.StageMonitor(name, nvml=getattr(self.backend, 'nvml', 'auto'),
                                        sync=getattr(self.backend, 'synchronize', lambda: None),
                                        parameters=getattr(self.backend, 'parameter_count', lambda: None)(),
                                        peak_flops=self.record['compute']['fp32_peak_flops'],
                                        forward_passes=forward, backward_passes=backward, profile_path=staging)
            self.mon.start()
            try:
                return fn(ctx)
            finally:
                record = self.mon.finish()
                files = {'compute.json': record}
                if staging.is_file():
                    files['trace.json'] = staging.read_bytes()
                    record['profile_trace'] = 'trace.json'
                directory = self.root / COMPUTE_DIR / f'{name}-{stamp}'
                au.seal_run(directory, files)
                ctx.expect(runs=[str(directory.relative_to(self.root))])
        return run

    def _unit(self, rows, padded_width=None):
        """Account one batch (unpadded and padded tokens) and profile it if it is the stage's first unit."""
        lengths = [len(r['input_ids']) if 'input_ids' in r else len(r['prompt_ids']) for r in rows]
        if self.mon is not None:
            self.mon.add_batch(lengths=lengths, padded_width=padded_width or max(lengths))
            return self.mon.unit()
        from contextlib import nullcontext
        return nullcontext()

    def _generate(self, generator, items, batches):
        """Generation one batch at a time so the first batch is profiled and generated tokens are counted."""
        by_id = {r['id']: r for r in items}
        out = []
        for index, batch in enumerate(batches):
            rows = [by_id[i] for i in batch]
            with self.mon.unit() if self.mon is not None else _null():
                part = at.generate(generator, rows, [batch])
            lengths = [len(r['prompt_ids']) + len(o['generated_ids']) for r, o in zip(rows, part)]
            if self.mon is not None:
                self.mon.add_batch(lengths=lengths, padded_width=max(lengths))
            out += [dict(o, batch=index) for o in part]
        return out

    # --- helpers ----------------------------------------------------------------------------------

    def _stop(self):
        return at.seal_stop_ids(self.backend.decoder(), self.backend.token_text)

    def _pad(self):
        return self.backend.decoder()['pad_token_id']

    def _instrument(self, model):
        return ai.Instrument(model, stop_ids=self._stop()['stop_ids'], pad_token_id=self._pad(),
                             identity_rtol=float(self.record['tolerances']['neuron_sum_identity']),
                             head_chunk=self.q['head_chunk'])

    def _traces(self, split, batch_size, *, minimum):
        rows = self.backend.prompt_rows(split)
        batches = [[r['id'] for r in rows[i:i + batch_size]] for i in range(0, len(rows), batch_size)]
        directory = at.split_dir(self.root, split) / 'generation'
        if not (directory / 'complete.json').is_file():
            responses = self._generate(self.backend.untuned_generator(batch_size), rows, batches)
            config = dict(model=self.backend.runtime().get('model'), adapter=None, decoding=self.backend.decoding(),
                          prompt_template='protocol.gsm8k_messages + chat template (frozen sequences-v2 prefix)',
                          prompt_ids=[r['id'] for r in rows], batches=batches, scorer='gsm8k_flexible_v3',
                          seed='greedy (no sampling)')
            at.seal_responses(self.root, split, responses, config=config, sources=self.backend.sources(),
                              stop=self._stop())
        rows_dir = at.split_dir(self.root, split) / 'rows'
        if (rows_dir / 'complete.json').is_file():
            manifest = _read(rows_dir, 'manifest.json')
            at.check_minimum(split, manifest['counts'], minimum=minimum)
            return manifest
        return at.seal_rows(self.root, split, gold=self.backend.gold(split), decode=self.backend.decode,
                            minimum=minimum)['manifest']

    # --- stages -------------------------------------------------------------------------------------

    def preflight(self, ctx):
        problems = self.backend.preflight()
        if problems:
            raise StopBeforeProduction(f'preflight: {problems}')
        return dict(runtime=self.backend.runtime())

    def qualification(self, ctx):
        tol = self.record['tolerances']
        manifest = self._traces('calibration', 1, minimum=self.q['calibration_minimum'])
        rows = at.load_rows(self.root, 'calibration')
        checks, flags, details, arms = {}, [], {}, []
        for arm in self.arms:
            model = self.backend.load(arm)
            inst = self._instrument(model)
            try:
                inst.check_routing()
            except ai.InstrumentFailure as exc:
                flags.append(f'{arm}: {exc}')
                continue
            arms.append(arm)
            if arm in self.q['fd_arms']:
                fd = finite_difference_check(model, inst, rows[:self.q['fd_rows']], self._stop()['stop_ids'],
                                             epsilon=tol['finite_difference']['epsilon'],
                                             adapters=self.q['fd_adapters'], neurons=self.q['fd_neurons'])
                details[f'fd:{arm}'] = fd
                checks[f'finite_differences:{arm}'] = bool(
                    fd['median_rel_err'] <= tol['finite_difference']['median_rel_err_max']
                    and fd['max_rel_err'] <= tol['finite_difference']['max_rel_err_max'])
            ctx.boundary()
        sweep = self._sweep(rows)
        details['sweep'] = sweep
        if sweep['selected'] is None:
            raise StopBeforeProduction('no token budget meets the memory policy')
        budgets = sorted({min(self.record['token_budget_sweep']), sweep['selected']})
        for arm in arms:
            bi = batch_invariance(self._instrument(self.backend.load(arm)), rows, budgets)
            details[f'batch_invariance:{arm}'] = bi
            checks[f'batch_invariance:{arm}'] = bool(bi['max_fraction'] <= tol['batch_invariance'])
            ctx.boundary()
        generation = self._generation_batches()
        details['generation_batches'] = generation
        floor = max([d['max_abs'] for k, d in details.items() if k.startswith('batch_invariance:')] or [0.0])
        result = dict(checks=checks, review_flags=flags, arms=arms, budget=sweep['selected'],
                      generation_batch=generation['selected'], calibration=manifest['counts'],
                      resolution_floor=dict(rule=tol['resolution_floor'], measured_batch_invariance=floor))
        au.seal_run(self.root / QUAL_DIR, {'qualification.json': result, 'details.json': details})
        ctx.expect(runs=[QUAL_DIR])
        return result

    def _sweep(self, rows):
        device = self.backend.device_memory()
        records, selected = [], None
        inst = self._instrument(self.backend.load(self.arms[0]))
        for budget in sorted(self.record['token_budget_sweep']):
            probe = longest_probe(rows, budget)
            if probe is None:
                records.append(dict(budget=budget, outcome='longest-example-exceeds-budget'))
                continue
            self.backend.reset_peak()
            try:
                inst.attribute(probe)
            except Exception as exc:  # an OOM fails this candidate only
                if not self.backend.is_oom(exc):
                    raise
                records.append(dict(budget=budget, outcome='oom'))
                continue
            peak = self.backend.peak_memory()
            fits = peak <= self.q['memory_fraction'] * device
            records.append(dict(budget=budget, outcome='ok', peak_bytes=peak, fits=fits))
            if fits:
                selected = budget
        return dict(candidates=records, selected=selected, device_bytes=device,
                    memory_fraction=self.q['memory_fraction'])

    def _generation_batches(self):
        rule = self.record['generation_batch_rule']
        items = self.backend.rehearsal_items()
        if len(items) != rule['items']:
            raise StopBeforeProduction(f"expected {rule['items']} rehearsal generation items, got {len(items)}")
        ordered = sorted(items, key=lambda r: (-len(r['prompt_ids']), r['id']))  # longest prompts first
        gold = self.backend.gold('rehearsal')
        reference, records, selected = None, [], 1
        for size in sorted(rule['candidates']):
            batches = [[r['id'] for r in ordered[i:i + size]] for i in range(0, len(ordered), size)]
            out = {o['id']: o for o in at.generate(self.backend.untuned_generator(size), ordered, batches)}
            correct = {i: score_gsm8k_flexible_v3(o['text'], gold[i], capped=o['stop_reason'] != 'eos')['correct']
                       for i, o in out.items()}
            if reference is None:
                reference = (out, correct)
                records.append(dict(batch=size, outcome='reference'))
                continue
            flips = sum(correct[i] != reference[1][i] for i in correct)
            stops = sum(out[i]['stop_reason'] != reference[0][i]['stop_reason'] for i in out)
            ok = flips <= rule['max_flips'] and stops <= rule['max_stop_changes']
            records.append(dict(batch=size, flips=flips, stop_changes=stops, qualifies=ok))
            if not ok:
                break
            selected = size
        return dict(candidates=records, selected=selected, rule=rule)

    def base_traces(self, ctx):
        qual = _read(self.root / QUAL_DIR, 'qualification.json')
        out = {}
        for split in ('discovery', 'heldout'):
            if self.rehearsal:  # calibration rows (traced in qualification) stand in; nothing blind is generated
                manifest = _read(at.split_dir(self.root, 'calibration') / 'rows', 'manifest.json')
                at.check_minimum(split, manifest['counts'], minimum=self.record['minimums'][split])
                out[split] = dict(manifest['counts'], stands_in='calibration')
                ctx.boundary()
                continue
            manifest = self._traces(split, qual['generation_batch'], minimum=self.record['minimums'][split])
            ctx.expect(runs=[str(at.split_dir(self.root, split).relative_to(self.root) / d)
                             for d in ('generation', 'rows')])
            out[split] = manifest['counts']  # counts only; no confirmation output is inspected
            ctx.boundary()
        return out

    def _split(self, role):
        """The sequences split playing ``role`` (discovery or heldout); calibration for both in a rehearsal."""
        return 'calibration' if self.rehearsal else role

    def _cohort(self):
        """Stage D items and gold: the ADR 0001 cohort, or the 20 rehearsal items in a rehearsal."""
        if self.rehearsal:
            return self.backend.rehearsal_items(), self.backend.gold('rehearsal')
        return self.backend.cohort_items(), self.backend.gold('cohort')

    def _discovery_batches(self):
        qual = _read(self.root / QUAL_DIR, 'qualification.json')
        rows = at.load_rows(self.root, self._split('discovery'))
        return rows, [[r['id'] for r in b] for b in pack(rows, qual['budget'])]

    def attribution_and_freeze(self, ctx):
        rows, batches = self._discovery_batches()
        by_id = {r['id']: r for r in rows}
        input_sha = _sha([r['id'] for r in rows])
        ctx.expect(units=[au.unit_key('ab', arm, batch=i) for arm in self.arms for i in range(len(batches))])
        arms = {}
        for arm in self.arms:
            inst = self._instrument(self.backend.load(arm))

            def compute(batch, inst=inst):
                rows_b = [by_id[i] for i in batch]
                with self._unit(rows_b):
                    blocks, counts, results = ai.unit_blocks(inst.attribute(rows_b))
                return dict(blocks=blocks, counts=counts, results=results)

            run_units(self.units, arm, batches, compute, boundary=ctx.boundary, input_sha256=input_sha)
            arms[arm] = self._discovery_summary(arm, len(batches), inst)
        selection = self.record['selection']
        freeze = ab.build_freeze(arms, sources=[dict(split=self._split('discovery'), rows_sha256=input_sha)],
                                 rehearsal=self.rehearsal,
                                 analysis=dict(statistics=self.record['statistics'],
                                               tolerances=self.record['tolerances']),
                                 tolerance=selection['tolerance'], seed=selection['seed'],
                                 random_seeds=selection['random_seeds'], top_group=selection['top_group'],
                                 top_singles=selection['top_singles'], read_size=selection['read_size'])
        ab.seal_freeze(self.root, freeze)
        ctx.expect(runs=[ab.FREEZE_DIR])
        return dict(arms=sorted(arms), batches=len(batches))

    def _discovery_summary(self, arm, n_batches, inst):
        parts = [self.units.read(au.unit_key('ab', arm, batch=i))[0] for i in range(n_batches)]

        def cat(key):
            return np.concatenate([p[key] for p in parts]).astype(np.float64)

        tokens = cat(('examples', 0, 'token_count'))[:, 0].sum()
        b_norms = [float(m.lora_B[inst.adapter].weight.norm()) for _, _, m in inst.modules]
        scale = float(inst.modules[0][2].scaling[inst.adapter])
        layers = sorted(inst.down_index)
        group = {'trace': 'examples', 'stop': 'stop'}
        return dict(adapters=inst.adapters,
                    mean_attr={m: np.nanmean(cat((group[m], 0, f'attr_{m}')), 0) for m in METRICS},
                    write=ab.discovery_write(cat(('examples', 0, 'sum_abs_c')).sum(0),
                                             np.full(len(inst.adapters), tokens), b_norms, scale),
                    signed_sum_c=cat(('examples', 0, 'signed_sum_c')).sum(0), token_total=np.full(len(layers), tokens),
                    neuron={m: {l: np.nanmean(cat((group[m], l, f'neuron_{m}')), 0) for l in layers} for m in METRICS},
                    term_size={l: cat(('examples', l, 'term_size')).sum(0) for l in layers})

    def stage_c(self, ctx):
        freeze_dir = self.root / ab.FREEZE_DIR
        au.verify_run(freeze_dir)  # Stage C never runs before a sealed freeze
        rows = at.load_rows(self.root, self._split('heldout'), freeze_dir=freeze_dir)
        ids = [r['id'] for r in rows]
        order = {i: k for k, i in enumerate(ids)}
        batches = pack(rows, _read(self.root / QUAL_DIR, 'qualification.json')['budget'])
        out = {}
        for arm in self.arms:
            stage = ab.StageC(self.backend.load(arm), stop_ids=self._stop()['stop_ids'], pad_token_id=self._pad(),
                              head_chunk=self.q['head_chunk'])
            for metric in METRICS:
                conditions = ab.conditions(freeze_dir, arm, metric, rehearsal=self.rehearsal)

                def evaluate(condition, stage=stage):
                    merged = {k: np.full(len(ids), np.nan) for k in ('delta_trace', 'delta_stop', 'kl')}
                    for batch in batches:
                        with self._unit(batch):
                            part = stage.evaluate(batch, condition)
                        for key in merged:
                            merged[key][[order[i] for i in part['example_ids']]] = part[key]
                    return dict(blocks={('examples', 0, 'delta_trace'): merged['delta_trace'][:, None].astype(np.float32),
                                        ('examples', 0, 'kl'): merged['kl'][:, None].astype(np.float32),
                                        ('stop', 0, 'delta_stop'): merged['delta_stop'][:, None].astype(np.float32)},
                                counts=dict(examples=[1] * len(ids),
                                            stop=(~np.isnan(merged['delta_stop'])).astype(int).tolist()))

                ctx.expect(units=[au.unit_key('c', arm, condition=c['name']) for c in conditions
                                  if c['status'] == 'run'])
                result = run_conditions(self.units, arm, conditions, evaluate, example_ids=ids,
                                        boundary=ctx.boundary, input_sha256=_sha(ids))
                out[f'{arm}|{metric}'] = dict(skipped=result['skipped'], conditions=len(conditions))
        return out

    def _compute_records(self):
        out = {}
        for directory in sorted((self.root / COMPUTE_DIR).glob('*')) if (self.root / COMPUTE_DIR).is_dir() else []:
            if (directory / 'complete.json').is_file():
                record = _read(directory, 'compute.json')
                out[record['stage']] = dict(record, sealed_dir=str(directory.relative_to(self.root)))  # latest attempt
        return out

    def _neuron_means(self, n_batches):
        group = {'trace': 'examples', 'stop': 'stop'}
        out = {}
        for arm in self.arms:
            parts = [self.units.read(au.unit_key('ab', arm, batch=i))[0] for i in range(n_batches)]
            out[arm] = {m: {key[1]: np.nanmean(np.concatenate([p[key] for p in parts]).astype(np.float64), 0)
                            for key in parts[0] if key[0] == group[m] and key[2] == f'neuron_{m}'} for m in METRICS}
        return out

    def _stage_d_report(self):
        base = self.root / STAGE_D_DIR
        runs = {}
        for config in sorted(base.glob('*/*/complete.json')):
            directory = config.parent
            au.verify_run(directory)
            runs[f'{directory.parent.name}|{directory.name}'] = _jsonl(directory / 'responses.jsonl')
        if 'untuned|reference' not in runs:
            return dict(available=False, reason='untuned reference run not sealed')
        rule = self.record['stage_d']['early_stop_rule']
        try:
            reference = ar.early_stop_reference(runs['untuned|reference'], rule=rule)
        except ValueError as exc:  # e.g. every untuned response capped: reported, never fatal
            reference = None
            unavailable = dict(available=False, rule=rule, reason=str(exc))
        out = dict(early_stop_reference=reference if reference is not None else unavailable,
                   outcomes={k: ar.stage_d_outcomes(v, early_reference=reference) for k, v in runs.items()},
                   pilot_comparison={})
        if self.rehearsal:
            out['pilot_comparison'] = dict(available=False, reason='rehearsal: no pilot items')
            return out
        root = Path(self.record['inputs'].get('artifact_root', '.'))
        _, gold = self._cohort()
        for arm, spec in sorted(self.record['stage_d'].get('pilot_results', {}).items()):
            if f'{arm}|full' not in runs:
                continue
            try:
                path = Path(spec['path'])
                path = path if path.is_absolute() else root / path
                pilot = ar.load_pilot_rows(path, filters=spec.get('filters', {}))
                out['pilot_comparison'][arm] = ar.pilot_comparison(runs[f'{arm}|full'], pilot, gold)
            except (OSError, ValueError, KeyError, TypeError) as exc:  # an analysis gap is reported, never fatal
                out['pilot_comparison'][arm] = dict(available=False, reason=f'{type(exc).__name__}: {exc}')
        return out

    def stage_d(self, ctx):
        freeze = _read(self.root / ab.FREEZE_DIR, 'freeze.json')
        size = _read(self.root / QUAL_DIR, 'qualification.json')['generation_batch']
        items, gold = self._cohort()
        batches = [[r['id'] for r in items[i:i + size]] for i in range(0, len(items), size)]
        runs = [('untuned', 'reference')] + [(arm, c) for arm, cs in STAGE_D_RUNS.items() if arm in self.arms
                                              for c in cs]
        out = {}
        for arm, condition in runs:
            directory = self.root / STAGE_D_DIR / arm / condition
            ctx.expect(runs=[str(directory.relative_to(self.root))])
            if (directory / 'complete.json').is_file():
                out[f'{arm}|{condition}'] = 'sealed'
                continue
            adapters = None if condition in ('full', 'reference') else stage_d_adapters(freeze, arm, condition)
            if adapters is False:
                self.root.joinpath(STAGE_D_DIR).mkdir(parents=True, exist_ok=True)
                out[f'{arm}|{condition}'] = 'skipped: matched-control-unavailable'
                continue
            generator, edit = self.backend.stage_d_generator(None if arm == 'untuned' else arm, adapters, size,
                                                             self.root / 'attribution/edited' / arm / condition)
            scored = []
            for r in self._generate(generator, items, batches):
                score = score_gsm8k_flexible_v3(r['text'], gold[r['id']], capped=r['stop_reason'] != 'eos')
                scored.append(dict(r, correct=bool(score['correct']), status=score['status'],
                                   tokens=len(r['generated_ids']), capped=r['stop_reason'] != 'eos'))
            au.seal_run(directory, {'responses.jsonl': ''.join(json.dumps(s, sort_keys=True) + '\n' for s in scored),
                                    'config.json': dict(arm=arm, condition=condition, adapters=adapters, edit=edit,
                                                        decoding=self.backend.decoding(), batches=batches,
                                                        scorer='gsm8k_flexible_v3',
                                                        cohort='rehearsal-items' if self.rehearsal else 'adr-0001',
                                                        model=self.backend.runtime().get('model'))})
            out[f'{arm}|{condition}'] = dict(accuracy=float(np.mean([s['correct'] for s in scored])))
            ctx.boundary()
        return out

    def report(self, ctx):
        freeze = _read(self.root / ab.FREEZE_DIR, 'freeze.json')
        stats = self.record['statistics']
        _, batches = self._discovery_batches()
        discovery = {m: {} for m in METRICS}
        for arm in self.arms:
            parts = [self.units.read(au.unit_key('ab', arm, batch=i))[0] for i in range(len(batches))]
            discovery['trace'][arm] = np.concatenate([p[('examples', 0, 'attr_trace')] for p in parts])
            discovery['stop'][arm] = np.concatenate([p[('stop', 0, 'attr_stop')] for p in parts])
        stage_c = {}
        for arm in self.arms:
            stage_c[arm] = {}
            for metric in METRICS:
                stage_c[arm][metric] = {}
                for c in ab.conditions(self.root / ab.FREEZE_DIR, arm, metric, rehearsal=self.rehearsal):
                    key = au.unit_key('c', arm, condition=c['name'])
                    if c['status'] != 'run' or self.units.state(key) != 'sealed':
                        continue
                    arrays, _ = self.units.read(key)
                    stage_c[arm][metric][c['name']] = dict(delta_trace=arrays[('examples', 0, 'delta_trace')][:, 0],
                                                           delta_stop=arrays[('stop', 0, 'delta_stop')][:, 0],
                                                           kl=arrays[('examples', 0, 'kl')][:, 0])
        stage_d = {}
        gold_dir = self.root / STAGE_D_DIR / 'gold-sft'
        if (gold_dir / 'full' / 'complete.json').is_file():
            full = _jsonl(gold_dir / 'full' / 'responses.jsonl')
            for condition in STAGE_D_RUNS['gold-sft'][1:]:
                directory = gold_dir / condition
                metric = condition.split('-')[1]
                if (directory / 'complete.json').is_file() and condition in stage_c.get('gold-sft', {}).get(metric, {}):
                    au.verify_run(directory)
                    stage_d[condition] = (full, _jsonl(directory / 'responses.jsonl'))
        concordance_c = {name: values for metric in METRICS for name, values in
                         stage_c.get('gold-sft', {}).get(metric, {}).items()}
        inputs = dict(settings={k: stats[k] for k in ('resamples', 'seed', 'alpha', 'margin', 'sparse_lower',
                                                      'dense_upper', 'gate_singles_min')},
                      provenance=self.backend.sources()['provenance'], discovery=discovery,
                      unresolved_arms=self.backend.unresolved_arms(),
                      freeze={arm: freeze['arms'][arm] for arm in self.arms}, stage_c=stage_c, stage_d={},
                      neuron={}, compute=self._compute_records(), a_vs_a0=None)
        report = ar.build(inputs)
        report['rehearsal'] = self.rehearsal
        qual = _read(self.root / QUAL_DIR, 'qualification.json')
        floor_rule = qual['resolution_floor']
        floor = ar.resolution_floor(floor_rule['rule'], floor_rule['measured_batch_invariance'])
        frozen = {arm: freeze['arms'][arm] for arm in self.arms}
        report['descriptive'] = dict(
            report['descriptive'], resolution_floor=dict(floor_rule, value=floor),
            concentration=ar.concentration_table(frozen, self._neuron_means(len(batches)),
                                                 null_orders=stats['null_orders'], null_seed=stats['null_seed'],
                                                 resolution_floor=floor),
            a_vs_a0=ar.a_vs_a0(self.record['inputs'].get('init_check_rows'), frozen))
        report['stage_d'] = self._stage_d_report()
        if stage_d:  # concordance over gold-sft's generated removal conditions
            report['concordance'] = ar.concordance(concordance_c, stage_d,
                                                   ar.bootstrap_indices(len(next(iter(concordance_c.values()))['delta_trace']),
                                                                        resamples=stats['resamples'], seed=stats['seed']),
                                                   ar.bootstrap_indices(len(next(iter(stage_d.values()))[0]),
                                                                        resamples=stats['resamples'], seed=stats['seed']))
        au.seal_run(self.root / REPORT_DIR, {'report.json': report, 'report.md': ar.render(report)})
        ctx.expect(runs=[REPORT_DIR])
        return dict(gates={k: g['pass'] for k, g in report['gates'].items()}, primary1=report['primary1']['outcome'],
                    primary3={m: {a: r['classification'] for a, r in report['primary3'][m].items()} for m in METRICS})


def stage_d_adapters(freeze, arm, condition):
    kind, metric = condition.split('-')
    sel = freeze['arms'][arm][metric]
    if kind == 'top10':
        return list(sel['top10'])
    return False if sel['controls']['group'] is None else list(sel['controls']['group'])


def longest_probe(rows, budget):
    """The first (longest-first) packed batch at this budget, or None if the longest row exceeds it."""
    try:
        return pack(rows, budget)[0]
    except ValueError:
        return None


def batch_invariance(inst, rows, budgets):
    """Per-example |Δattr| between singleton and packed batches at each budget, as a fraction of max |attr|."""
    single = {}
    for row in rows:
        out = inst.attribute([row])
        single[row['id']] = {m: out['attr'][m][0] for m in METRICS}
    scale = {m: np.nanmax(np.abs([single[r['id']][m] for r in rows])) for m in METRICS}
    worst_fraction, worst_abs = 0.0, 0.0
    for budget in budgets:
        for batch in pack(rows, budget):
            out = inst.attribute(batch)
            for k, row in enumerate(batch):
                for m in METRICS:
                    diff = np.abs(out['attr'][m][k] - single[row['id']][m])
                    if np.isnan(diff).all():
                        continue
                    worst_abs = max(worst_abs, float(np.nanmax(diff)))
                    if scale[m] > 0:
                        worst_fraction = max(worst_fraction, float(np.nanmax(diff)) / scale[m])
    return dict(max_fraction=worst_fraction, max_abs=worst_abs, budgets=list(budgets))


def finite_difference_check(model, inst, rows, stop_ids, *, epsilon, adapters, neurons):
    """Central differences on explicit branch/neuron gates vs the instrument, for the top adapters and neurons."""
    import torch
    out = inst.attribute(rows)
    ids, attention, positions = inst.batch(rows)
    masks = [r['masks'] for r in rows]
    n, errors, checked = len(rows), [], []

    def metric(name, gates=None, neuron=None):
        with torch.no_grad(), ai.branch_gates(model, gates or {}, inst.adapter), \
                ai.neuron_gate(model, neuron, inst.adapter):
            logits = model(input_ids=ids, attention_mask=attention, position_ids=positions, use_cache=False).logits
        if name == 'trace':
            return am.trace_loglik(logits, ids, masks).numpy()
        return am.stop_pull(logits, masks, stop_ids=stop_ids)[0].numpy()

    def compare(name, got, kw_up, kw_down):
        fd = -(metric(name, **kw_up) - metric(name, **kw_down)) / (2 * epsilon)
        ok = ~np.isnan(fd) & (np.abs(fd) > 0)
        errors.extend(np.abs(got[ok] - fd[ok]) / np.abs(fd[ok]))

    for name in METRICS:
        mean = np.nanmean(out['attr'][name], 0)
        if np.isnan(mean).all():
            continue
        for index in ab.top_k(np.nan_to_num(mean), adapters):
            compare(name, out['attr'][name][:, index], dict(gates={index: torch.full((n, 1, 1), 1 + epsilon)}),
                    dict(gates={index: torch.full((n, 1, 1), 1 - epsilon)}))
            checked.append(dict(metric=name, adapter=int(index)))
        layer = max(inst.down_index, key=lambda l: abs(np.nan_to_num(mean[inst.down_index[l]])))
        values = np.nanmean(out['neuron'][name][layer], 0)
        for j in np.argsort(-np.abs(values))[:neurons]:
            compare(name, out['neuron'][name][layer][:, j],
                    dict(neuron=(layer, int(j), torch.full((n, 1, 1), 1 + epsilon))),
                    dict(neuron=(layer, int(j), torch.full((n, 1, 1), 1 - epsilon))))
            checked.append(dict(metric=name, layer=int(layer), neuron=int(j)))
    errors = np.asarray(errors, dtype=np.float64)
    return dict(median_rel_err=float(np.median(errors)) if len(errors) else float('inf'),
                max_rel_err=float(errors.max()) if len(errors) else float('inf'), checked=checked, epsilon=epsilon,
                note='FP32 central differences; resolution about ulp(|M|)/epsilon')


def build(record, local_root, volume_root, *, backend=None, code_sha256=None):
    if 'qualification' not in record or 'inputs' not in record:
        raise StopBeforeProduction('launch record lacks the qualification settings or inputs block')
    if backend is None:
        from pilot_eval.attribution_backend import TransformersBackend
        backend = TransformersBackend(record, volume_root)
    return Stages(record, local_root, backend, code_sha256=code_sha256 or record['code_commit']).mapping()


def _null():
    from contextlib import nullcontext
    return nullcontext()
