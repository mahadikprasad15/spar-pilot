"""Ticket 03b: qualify every arm on the real GPU and freeze the production measurement plan.

Reads the approved run plan (``neuron_audit_plan.load(plan, '03b')``), the sealed ticket-01
inventory and ticket-02 sequences, and a small runner config with the worker's paths. It runs
sealed, resumable stages under ``<output_root>/runs/joint-neuron-adapter-audit/qualification/<run_id>/``:

    preflight -> masks -> arms -> benchmark -> batch_agreement -> fp64_floor -> batch_freeze

Hard gates (plan ``hard_gates``) stop 03b with saved evidence. An arm-specific gate failure
marks that arm failed (evidence kept) and the study partial (plan ``partial_study``); global
gates (masks, logits batching, storage) stop everything. Review flags (plan ``review_flags``)
never stop 03b; ticket 04 refuses to start until a sign-off is recorded with ``signoff``.

The model work goes through a ``Backend`` (``TransformersBackend`` on the Pod; a tiny CPU
model in tests). Pure decision math (packing, flags, floor, scope, projections) is separate.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from pilot_eval import neuron_audit_collector as nc
from pilot_eval import neuron_audit_plan as plan_mod

RUN_SUBDIR = 'runs/joint-neuron-adapter-audit/qualification'
STAGES = ['preflight', 'masks', 'arms', 'benchmark', 'batch_agreement', 'fp64_floor', 'batch_freeze']
PRIMARY = 'response_content'
FLAG_STATS = ['mean_z', 'mean_abs_z']


class QualificationFailure(RuntimeError):
    def __init__(self, stage, message, evidence=None):
        super().__init__(f'{stage}: {message}')
        self.stage, self.evidence = stage, evidence or {}


# --- io ---------------------------------------------------------------------------------------

def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=_json_default)


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def _sha(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_sequences(run_dir):
    return [json.loads(line) for line in (Path(run_dir) / 'sequences.jsonl').read_text().splitlines()]


# --- pure decision math -----------------------------------------------------------------------

def pack(rows, budget):
    """Plan packing rule: sort by (length desc, id), greedily fill while batch_size * max_len <= budget."""
    ordered = sorted(rows, key=lambda r: (-len(r['input_ids']), r['id']))
    batches, current = [], []
    for row in ordered:
        width = max([len(row['input_ids'])] + [len(r['input_ids']) for r in current])
        if current and (len(current) + 1) * width > budget:
            batches.append(current)
            current = []
        if len(row['input_ids']) > budget:
            raise ValueError(f"sequence {row['id']} longer than the token budget {budget}")
        current.append(row)
    if current:
        batches.append(current)
    return batches


def batch_manifest(batches, budget):
    entries = [dict(index=i, ids=[r['id'] for r in b], padded_width=max(len(r['input_ids']) for r in b),
                    lengths=[len(r['input_ids']) for r in b]) for i, b in enumerate(batches)]
    body = dict(budget_padded_tokens=budget, packing_rule='sort (length desc, id); greedy; size*max_len <= budget',
                position_ids='attention.cumsum(-1) - 1, clamped at 0; right padding with pad_token_id',
                batches=entries, examples=sum(len(e['ids']) for e in entries),
                padded_tokens=sum(e['padded_width'] * len(e['ids']) for e in entries))
    body['sha256'] = _sha(body)
    return body


def mask_problems(row, token_text, eos_id):
    """Decoded-mask test on a frozen sequence (independent of the ticket-02 text offsets)."""
    ids, masks, problems = row['input_ids'], row['masks'], []
    reply = masks['reply_start']
    if len(reply) != 1:
        return ['reply_start must be one position']
    r = reply[0]
    if '\n' not in token_text(ids[r]) or r == 0 or token_text(ids[r - 1]).strip() != 'assistant':
        problems.append('reply_start is not the assistant-header newline')
    content, term = masks['response_content'], masks['termination']
    if not content or content != list(range(r + 1, content[-1] + 1)):
        problems.append('response_content is empty or not contiguous after reply_start')
    if term:
        e = term[-1]
        if term != [e - 1, e] or ids[e] != eos_id or content[-1] != e - 1:
            problems.append('termination is not [last content token, end-of-turn]')
        if e + 1 < len(ids) and (e + 1) in content + reply + term:
            problems.append('trailing token after end-of-turn is masked')
    elif content[-1] != len(ids) - 1:
        problems.append('truncated row must end in content')
    if eos_id in [ids[p] for p in content]:
        problems.append('end-of-turn token inside response_content')
    return problems


def normalized_discrepancy(single, batched):
    """|diff_j| / RMS over neurons of the singleton statistic vector; returns the 99th percentile."""
    single, batched = np.asarray(single, np.float64), np.asarray(batched, np.float64)
    if np.isnan(single).all() and np.isnan(batched).all():
        return None
    rms = float(np.sqrt(np.mean(single ** 2)))
    diff = np.abs(batched - single)
    if rms == 0:
        return 0.0 if not diff.any() else float('inf')
    return float(np.percentile(diff / rms, 99))


def review_flags(records, flags_plan):
    """records: per (example, batch, padded, layer, group, stat) p99 values. Flags never stop 03b."""
    early, padding = flags_plan['early_layer'], flags_plan['padding']
    threshold, factor, floor = float(early['threshold']), float(padding['factor']), float(padding['unpadded_floor'])
    raised = []
    early_hits = [r for r in records if r['layer'] in early['layers'] and r['group'] == PRIMARY
                  and r['value'] is not None and r['value'] > threshold]
    if early_hits:
        raised.append(dict(flag='early_layer', threshold=threshold, hits=early_hits[:20], count=len(early_hits)))
    per_example = {}
    for r in records:
        if r['group'] == PRIMARY and r['value'] is not None:
            key = (r['arm'], r['batch'], r['example'])
            per_example[key] = max(per_example.get(key, 0.0), r['value'])
    padding_hits = []
    for arm, batch in sorted({(k[0], k[1]) for k in per_example}):
        padded = [v for (a, b, e), v in per_example.items() if a == arm and b == batch
                  and any(r['padded'] for r in records if (r['arm'], r['batch'], r['example']) == (a, b, e))]
        unpadded = [v for (a, b, e), v in per_example.items() if a == arm and b == batch
                    and not any(r['padded'] for r in records if (r['arm'], r['batch'], r['example']) == (a, b, e))]
        if padded and np.median(padded) > factor * max(max(unpadded, default=0.0), floor):
            padding_hits.append(dict(arm=arm, batch=batch, padded_median=float(np.median(padded)),
                                     unpadded_max=float(max(unpadded, default=0.0))))
    if padding_hits:
        raised.append(dict(flag='padding', rule=f'median(padded) > {factor} * max(max(unpadded), {floor})',
                           hits=padding_hits))
    return raised


def layer_curves(records):
    """Discrepancy vs layer (absolute p99 values), padded and unpadded separately, for the reviewer."""
    curves = {}
    for r in records:
        if r['group'] != PRIMARY or r['value'] is None:
            continue
        key = ('padded' if r['padded'] else 'unpadded', r['stat'])
        curves.setdefault(key, {}).setdefault(r['layer'], []).append(r['value'])
    return {f'{k[0]}/{k[1]}': {str(layer): dict(max=float(max(v)), median=float(np.median(v)))
                                for layer, v in sorted(vals.items())} for k, vals in curves.items()}


def d_values(base_means, arm_means):
    """Primary D per neuron: |mean over examples of (arm - base) per-example mean |z||."""
    diff = np.asarray(arm_means, np.float64) - np.asarray(base_means, np.float64)
    return np.abs(np.nanmean(diff, axis=0))


def floor_from(d32, d64):
    error = np.abs(np.asarray(d32, np.float64) - np.asarray(d64, np.float64))
    p99 = float(np.percentile(error, 99))
    return dict(p99=p99 if p99 > 0 else None, status='measured' if p99 > 0 else 'unresolved',
                median=float(np.median(error)), max=float(error.max()))


def projected_bytes(*, arms_trained, arms_total, examples, layers, width, groups, union_tokens,
                    heads_layers=0, heads=0):
    base_stats, adapter_stats = 2, 2
    dense = arms_total * groups * layers * examples * width * base_stats * 4
    dense += arms_trained * groups * layers * examples * width * adapter_stats * 4
    tokens = arms_trained * (layers * union_tokens * 4 + union_tokens * (4 + 4 + 1))
    head = groups * heads_layers * examples * heads * 4 * (arms_total + 2 * arms_trained)
    return int(dense + tokens + head)


def choose_scope(measure, *, all_layers, subset_layers, max_ratio, fits_storage):
    """Plan 1B rule: all layers if time ratio and storage fit; else measured subset; else omit.

    ``fits_storage(total_bytes)`` applies the Global-Volume-era rules (local working set and HF quota).
    """
    tried = []
    for scope, layers in [('all', all_layers), ('subset', [l for l in subset_layers if l in all_layers])]:
        if not layers:
            tried.append(dict(scope=scope, available=False))
            continue
        m = measure(layers)
        fits = (m['time_ratio'] is not None and m['time_ratio'] <= max_ratio
                and fits_storage(m['projected_bytes']))
        tried.append(dict(scope=scope, layers=layers, fits=fits, **m))
        if fits:
            return dict(scope=scope, layers=layers, tried=tried)
    return dict(scope='omit', layers=[], tried=tried)


# --- backend ------------------------------------------------------------------------------------

class TransformersBackend:
    """Real loader for the Pod: local files only, eager attention, explicit dtype and device."""

    def __init__(self, model_dir, adapter_dirs, tokenizer_dir, reference_arm, device='cuda'):
        self.model_dir, self.adapter_dirs, self.tokenizer_dir = model_dir, adapter_dirs, tokenizer_dir
        self.reference_arm, self.device = reference_arm, device

    def tokenizer(self):
        from pilot_eval.neuron_audit_sequences import ChatTokenizer
        return ChatTokenizer(self.tokenizer_dir)

    def fresh_base(self, dtype, device=None):
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(self.model_dir, torch_dtype=getattr(torch, dtype),
                                                     attn_implementation='eager', local_files_only=True)
        return model.to(device or self.device).eval()

    def load(self, arm, dtype, device=None):
        from peft import PeftModel
        adapter = self.adapter_dirs[self.reference_arm if arm == 'untuned' else arm]
        import torch
        # Explicit torch_device: never let PEFT infer a device for the saved factors.
        model = PeftModel.from_pretrained(self.fresh_base(dtype, device), adapter, is_trainable=False,
                                          torch_device=device or self.device)
        # Saved FP32 adapter factors are cast with the base for the FP64 reference only.
        return model.to(getattr(torch, dtype)).eval()

    def release(self, model):
        import gc
        import torch
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def device_info(self):
        import torch
        props = torch.cuda.get_device_properties(0)
        return dict(name=props.name, total_memory=props.total_memory)

    def reset_peak(self):
        import torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def peak_memory(self):
        import torch
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    def synchronize(self):
        import torch
        torch.cuda.synchronize()

    @staticmethod
    def is_oom(exc):
        import torch
        return isinstance(exc, torch.cuda.OutOfMemoryError) or 'out of memory' in str(exc).lower()


# --- runner -------------------------------------------------------------------------------------

class Qualification:
    def __init__(self, plan, config, backend, *, preflight=None):
        self.plan, self.config, self.backend = plan, config, backend
        self.preflight_fn = preflight
        self.root = Path(config['artifact_root'])
        self.rehearsal = bool(plan.get('_rehearsal'))
        if self.rehearsal != config['run_id'].startswith('rehearsal-'):
            raise QualificationFailure('setup', 'rehearsal runs (and only they) must use a run_id starting with rehearsal-')
        self.run_dir = Path(config['output_root']) / RUN_SUBDIR / config['run_id']
        self.identity = dict(plan_sha256=plan['_plan_sha256'], config=_sha({k: v for k, v in config.items()
                                                                             if k not in ('local_temp',)}))
        self.inventory = json.loads((self.root / plan['inputs']['inventory_run'] / 'inventory.json').read_text())
        self.sequences = read_sequences(self.root / plan['inputs']['sequences_run'])
        numeric = plan['hard_gates']['identities']['numeric']
        self.tolerances = dict(z_reconstruction_relative_to_layer_max=float(numeric['z_relative_to_layer_max']),
                               identity_atol=float(numeric['atol']), identity_rtol=float(numeric['rtol']))
        self.arms = list(plan['arms'])
        self.trained = [a for a in self.arms if a != 'untuned']

    # stage machinery
    def _stage_dir(self, name):
        return self.run_dir / 'stages' / name

    def _sealed(self, name):
        directory = self._stage_dir(name)
        seal = directory / 'complete.json'
        if not seal.is_file():
            return None
        record = json.loads(seal.read_text())
        if record['identity'] != self.identity or _file_sha(directory / 'result.json') != record['result_sha256']:
            raise QualificationFailure(name, 'sealed stage identity or integrity differs; investigate')
        return json.loads((directory / 'result.json').read_text())

    def _seal(self, name, result):
        directory = self._stage_dir(name)
        _write_json(directory / 'result.json', result)
        _write_json(directory / 'complete.json', dict(identity=self.identity,
                                                      result_sha256=_file_sha(directory / 'result.json')))
        self._progress(name, 'sealed')
        return result

    def _fail(self, name, message, evidence):
        _write_json(self._stage_dir(name) / f'failed-{time.time_ns()}.json',
                    dict(stage=name, message=message, evidence=evidence))
        self._status('failed', stage=name, error=message)
        raise QualificationFailure(name, message, evidence)

    def _status(self, state, **extra):
        _write_json(self.run_dir / 'meta/status.json', dict(state=state, updated_unix=time.time(),
                                                            run_id=self.config['run_id'], pid=os.getpid(), **extra))

    def _progress(self, stage, event, **extra):
        line = dict(time=time.time(), stage=stage, event=event, **extra)
        (self.run_dir / 'logs').mkdir(parents=True, exist_ok=True)
        with (self.run_dir / 'logs' / 'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(line, default=_json_default) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(line, default=_json_default), flush=True)

    # helpers
    def _rows(self, split):
        return [r for r in self.sequences if r['split'] == split]

    def _allocation(self):
        calibration = sorted(self._rows('calibration'), key=lambda r: r['id'])
        alloc = self.plan['calibration_allocation']
        fp64 = [r for r in calibration if r['group'] == 'math'][: alloc['fp64_reference']['math']] + \
            [r for r in calibration if r['group'] == 'general'][: alloc['fp64_reference']['general']]
        rest = [r for r in calibration if r['id'] not in {x['id'] for x in fp64}]
        calibrate = rest[: alloc['adr0008_calibrate']]
        validate = rest[alloc['adr0008_calibrate']: alloc['adr0008_calibrate'] + alloc['adr0008_validate']]
        return dict(all=calibration, fp64=fp64, calibrate=calibrate, validate=validate)

    def _arm_record(self, arm):
        return self.inventory['arms'][arm]

    def _scale(self, arm):
        return self._arm_record(arm)['checkpoints'][str(self.inventory['audit_step'])]['config']['scale']

    def _factors(self, arm, layers, projection='down_proj'):
        return nc.load_saved_factors(self.config['adapter_dirs'][arm], layers, self._scale(arm), projection)

    def _expected_keys(self, arm):
        rel = self._arm_record(arm)['checkpoints'][str(self.inventory['audit_step'])]['key_list_path']
        path = self.root / rel
        return json.loads(path.read_text())['keys']

    def _collector(self, model, dtype='float32'):
        expected = self.inventory['base'].get('expected_base_parameter_sha256') if dtype == 'float32' else None
        if not self.config.get('check_base_hash', True):
            expected = None
        return nc.NeuronCollector(model, tolerances=self.tolerances, expected_base_hash=expected, dtype=dtype)

    # stages
    def stage_preflight(self):
        if self.preflight_fn is None:
            self._fail('preflight', 'no preflight function supplied', {})
        try:
            return self.preflight_fn()
        except Exception as exc:
            self._fail('preflight', str(exc), getattr(exc, 'record', {}))

    def stage_masks(self):
        tok = self.backend.tokenizer()
        problems = {}
        for row in self.sequences:
            found = mask_problems(row, tok.token_text, tok.eos_token_id)
            if found:
                problems[row['id']] = found
        if problems:
            self._fail('masks', 'decoded-mask test failed', dict(examples=dict(list(problems.items())[:20]),
                                                                 count=len(problems)))
        return dict(rows_checked=len(self.sequences), problems=0)

    def stage_arms(self):
        rep = self._allocation()['calibrate'][:4]
        results, failed = {}, {}
        reference = self.backend.fresh_base('float32')
        layers = None
        for arm in self.arms:
            self._progress('arms', 'start', arm=arm)
            collector, problems = None, []
            if arm != 'untuned':
                saved = Path(self.config['adapter_dirs'][arm]) / 'adapter_model.safetensors'
                recorded = self._arm_record(arm)['checkpoints'][str(self.inventory['audit_step'])]['files']
                if not saved.is_file() or _file_sha(saved) != recorded['adapter_model.safetensors']['sha256']:
                    failed[arm] = dict(problems=['adapter file on the worker differs from the sealed inventory'])
                    self._progress('arms', 'done', arm=arm, problems=failed[arm]['problems'])
                    continue
            model = self.backend.load(arm, 'float32')
            try:
                collector = self._collector(model)
                layers = list(range(len(collector.layers)))
                inputs, _ = collector.batch(rep)
                import torch
                with torch.no_grad():
                    ref_logits = reference(**inputs, use_cache=False).logits
                effect = nc.adapter_effect(model, inputs, reference_logits=ref_logits)
                record = dict(runtime=collector.runtime(), disabled_vs_fresh_base=effect['disabled_minus_reference_max_abs'])
                problems = []
                if effect['disabled_minus_reference_max_abs'] != 0.0:
                    problems.append('disabled adapter logits differ from the freshly loaded base (torch.equal)')
                if arm == 'untuned':
                    out = collector.collect(rep, factors=None, validate=True)
                else:
                    live = nc.check_live_adapter(model, self.config['adapter_dirs'][arm], expected_scale=self._scale(arm),
                                                 expected_keys=self._expected_keys(arm))
                    record['live_adapter'] = live
                    if live['status'] != 'pass':
                        problems += live['reasons']
                    if live['zero_b_modules']:
                        problems.append('zero lora_B at the audited step is unresolved')
                    record['logit_effect'] = effect['classification']
                    if effect['classification'] != 'changes-logits':
                        problems.append('active adapter does not change logits on representative examples')
                    factors = self._factors(arm, layers)
                    heads = self._factors(arm, layers, 'o_proj')
                    out = collector.collect(rep, factors=factors, validate=True, heads=layers, head_factors=heads)
                    write = max(float(np.abs(out['tokens']['ct'][i]).max()) * float(np.linalg.norm(factors['layers'][l]['b']))
                                for i, l in enumerate(layers)) * abs(factors['scale'])
                    record['max_direct_write'] = write
                    if write <= 0:
                        problems.append('active down_proj branch does not change the direct module output')
                record.update(routing=out['routing'], checks=len(out['validation']),
                              worst=max((r['max_ratio'] or 0) for r in out['validation'] if r.get('max_ratio') is not None))
            except nc.CollectorFailure as exc:
                problems = [f'{exc}']
                record = dict(evidence=exc.evidence)
            finally:
                if collector is not None:
                    collector.close()
                self.backend.release(model)
            record['problems'] = problems
            (failed if problems else results)[arm] = record
            self._progress('arms', 'done', arm=arm, problems=problems)
        self.backend.release(reference)
        if 'untuned' in failed:
            self._fail('arms', 'untuned/reference qualification failed; no arm can be compared', failed)
        for arm, record in failed.items():
            _write_json(self._stage_dir('arms') / f'arm-failed-{arm}.json', record)
        return dict(qualified=sorted(results), failed=sorted(failed), details=dict(results, **failed),
                    layers=layers, partial=bool(failed),
                    partial_label=('partial study; missing arms: ' + ', '.join(sorted(failed))) if failed else None)

    def _time_run(self, arm, batches, *, heads=None):
        model = self.backend.load(arm, 'float32')
        collector = self._collector(model)
        layers = list(range(len(collector.layers)))
        factors = None if arm == 'untuned' else self._factors(arm, layers)
        head_factors = None if arm == 'untuned' or not heads else self._factors(arm, heads, 'o_proj')
        try:
            self.backend.reset_peak()
            started = time.perf_counter()
            tokens = padded = 0
            for batch in batches:
                collector.collect(batch, factors=factors, validate=False, heads=heads, head_factors=head_factors)
                tokens += sum(len(r['input_ids']) for r in batch)
                padded += len(batch) * max(len(r['input_ids']) for r in batch)
            self.backend.synchronize()
            seconds = time.perf_counter() - started
            return dict(seconds=seconds, tokens=tokens, padded_tokens=padded, peak_bytes=self.backend.peak_memory(),
                        padded_tokens_per_second=padded / seconds if seconds > 0 else None)
        finally:
            collector.close()
            self.backend.release(model)

    def stage_benchmark(self, arms_result, storage_inputs):
        bench = self.plan['benchmark']
        examples = self._allocation()['all'][: bench['examples'][1]]
        if len(examples) < bench['examples'][0]:
            self._fail('benchmark', 'too few calibration examples for the benchmark', dict(count=len(examples)))
        arm = next((a for a in self.trained if a in arms_result['qualified']), None)
        if arm is None:
            self._fail('benchmark', 'no qualified adapted arm to benchmark', {})
        layers = arms_result['layers']
        device = self.backend.device_info()
        limit = float(bench['vram_fraction_max']) * device['total_memory']
        probes = []
        for budget in bench['token_budgets_padded']:
            batches = pack(examples, budget)
            entry = dict(budget=budget, batches=len(batches))
            try:
                for hooks, heads in [('1A', None), ('1A+1B', layers)]:
                    for a in ['untuned', arm]:
                        entry[f'{hooks}/{a}'] = self._time_run(a, batches, heads=heads)
            except Exception as exc:
                if not self.backend.is_oom(exc):
                    raise
                entry['oom'] = f'{type(exc).__name__}'
                probes.append(entry)
                break
            entry['peak_1A'] = max(entry[f'1A/{a}']['peak_bytes'] for a in ['untuned', arm])
            entry['peak_1A+1B'] = max(entry[f'1A+1B/{a}']['peak_bytes'] for a in ['untuned', arm])
            probes.append(entry)
            if entry['peak_1A'] > limit:
                break
        feasible_1a = [p for p in probes if 'oom' not in p and p['peak_1A'] <= limit]
        if not feasible_1a:
            self._fail('benchmark', 'no probed budget fits 80% of device memory', dict(probes=probes))
        production = self._rows('discovery') + self._rows('heldout')
        union_tokens = sum(len(set(r['masks']['response_content']) | set(r['masks']['reply_start']) |
                               set(r['masks']['termination'])) for r in production)
        model_cfg = arms_result['details'][arm]['runtime']
        n_heads = model_cfg['num_attention_heads']
        common = dict(arms_trained=len(arms_result['qualified']) - ('untuned' in arms_result['qualified']),
                      arms_total=len(arms_result['qualified']), examples=len(production), layers=len(layers),
                      width=model_cfg['intermediate_size'], groups=3, union_tokens=union_tokens)
        n_arms = len(arms_result['qualified'])

        def storage_check(total_bytes):
            return self._confirm_storage(storage_inputs, total_bytes, n_arms)

        def fits_storage(total_bytes):
            return not storage_check(total_bytes)[1]

        def seconds_for(probe, hooks):
            rate = min(probe[f'{hooks}/{a}']['padded_tokens_per_second'] for a in ['untuned', arm])
            padded = batch_manifest(pack(production, probe['budget']), probe['budget'])['padded_tokens']
            return padded / rate

        feasible_both = [p for p in feasible_1a if p['peak_1A+1B'] <= limit]
        scope_rule = bench['phase_1b_scope']
        top_both = feasible_both[-1] if feasible_both else None

        def measure(selected):
            if top_both is None:
                return dict(time_ratio=None, projected_bytes=float('inf'))
            if selected == layers:
                ratio = sum(top_both[f'1A+1B/{a}']['seconds'] for a in ['untuned', arm]) / \
                    sum(top_both[f'1A/{a}']['seconds'] for a in ['untuned', arm])
            else:
                batches = pack(examples, top_both['budget'])
                ratio = sum(self._time_run(a, batches, heads=selected)['seconds'] for a in ['untuned', arm]) / \
                    sum(top_both[f'1A/{a}']['seconds'] for a in ['untuned', arm])
            return dict(time_ratio=ratio, projected_bytes=projected_bytes(**common, heads_layers=len(selected),
                                                                           heads=n_heads), budget=top_both['budget'])

        scope = choose_scope(measure, all_layers=layers, subset_layers=list(scope_rule['subset_layers']),
                             max_ratio=float(scope_rule['time_ratio_max']), fits_storage=fits_storage)
        chosen = top_both if scope['scope'] != 'omit' else feasible_1a[-1]
        hooks = '1A+1B' if scope['scope'] != 'omit' else '1A'
        need = projected_bytes(**common, heads_layers=len(scope['layers']), heads=n_heads)
        per_arm = seconds_for(chosen, hooks)
        projection = dict(seconds_per_arm=per_arm, total_hours=per_arm * len(arms_result['qualified']) / 3600,
                          projected_bytes=need, budget=chosen['budget'], hooks=hooks)
        storage, problems = storage_check(need)
        if problems:
            self._fail('benchmark', 'storage capacity check failed', dict(storage=storage, problems=problems))
        return dict(device=device, vram_limit_bytes=limit, probes=probes, benchmark_arm=arm,
                    selected_budget=chosen['budget'], phase_1b=scope, projection=projection, storage=storage)

    def _confirm_storage(self, inputs, total_bytes, n_arms):
        from pilot_eval.neuron_audit_preflight import confirm_storage
        quota = self.plan['storage']['hf_quota']
        hf = dict(observed_private_bytes=inputs['hf_observed_private_bytes'], quota_bytes=quota['private_quota_bytes'])
        per_arm = total_bytes / max(n_arms, 1)
        return confirm_storage(inputs['local_disk'], hf, per_arm_bytes=per_arm, total_upload_bytes=total_bytes,
                               model_cache_bytes=inputs.get('model_cache_bytes', 0),
                               local_fraction=float(self.plan['benchmark']['phase_1b_scope']['disk_fraction_max']),
                               hf_fraction=0.85)

    def stage_batch_agreement(self, arms_result, benchmark):
        alloc = self._allocation()
        budget = benchmark['selected_budget']
        heads = benchmark['phase_1b']['layers'] or None
        records, logit_max, logit_evidence = [], {}, []
        bound = float(self.plan['hard_gates']['batched_vs_singleton_logits']['bound'])
        sets = dict(calibrate=alloc['calibrate'], validate=alloc['validate'])
        for arm in arms_result['qualified']:
            model = self.backend.load(arm, 'float32')
            collector = self._collector(model)
            layers = list(range(len(collector.layers)))
            factors = None if arm == 'untuned' else self._factors(arm, layers)
            head_factors = None if arm == 'untuned' or not heads else self._factors(arm, heads, 'o_proj')
            worst = 0.0
            try:
                for set_name, rows in sets.items():
                    for b_index, batch in enumerate(pack(rows, budget)):
                        width = max(len(r['input_ids']) for r in batch)
                        out = collector.collect(batch, factors=factors, validate=False, heads=heads,
                                                head_factors=head_factors)
                        for i, row in enumerate(batch):
                            single = collector.collect([row], factors=factors, validate=False, heads=heads,
                                                       head_factors=head_factors)
                            n = len(row['input_ids'])
                            positions = sorted(set(row['masks']['response_content']) | set(row['masks']['reply_start'])
                                               | set(row['masks']['termination']))
                            diff = (out['logits'][i, positions] - single['logits'][0, positions]).abs().max().item()
                            worst = max(worst, diff)
                            if diff > bound:
                                logit_evidence.append(dict(arm=arm, example=row['id'], max_abs=diff, length=n,
                                                           padded=n < width))
                            for (group, layer, stat), value in single['arrays'].items():
                                if stat not in FLAG_STATS:
                                    continue
                                records.append(dict(arm=arm, set=set_name, batch=f'{set_name}-{b_index}',
                                                    example=row['id'], padded=n < width, layer=layer, group=group,
                                                    stat=stat, value=normalized_discrepancy(
                                                        value[0], out['arrays'][(group, layer, stat)][i])))
            finally:
                collector.close()
                self.backend.release(model)
            logit_max[arm] = worst
            self._progress('batch_agreement', 'arm', arm=arm, logits_max_abs=worst)
        _write_json(self._stage_dir('batch_agreement') / 'discrepancies.json', records)
        if logit_evidence:
            self._fail('batch_agreement', f'batched vs singleton logits exceed {bound} (ADR 0014 gate)',
                       dict(evidence=logit_evidence[:50], logits_max_abs=logit_max))
        flags = review_flags(records, self.plan['review_flags'])
        curves = layer_curves(records)
        self._plot(curves)
        return dict(budget=budget, logits_max_abs=logit_max, logit_bound=bound, review_flags=flags,
                    requires_signoff=True, curves=curves, records=len(records),
                    discrepancies_sha256=_file_sha(self._stage_dir('batch_agreement') / 'discrepancies.json'),
                    sets={k: [r['id'] for r in v] for k, v in sets.items()})

    def _plot(self, curves):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return
        fig, axes = plt.subplots(1, len(FLAG_STATS), figsize=(6 * len(FLAG_STATS), 4), squeeze=False)
        for ax, stat in zip(axes[0], FLAG_STATS):
            for kind in ['unpadded', 'padded']:
                data = curves.get(f'{kind}/{stat}', {})
                layers = sorted(int(k) for k in data)
                ax.plot(layers, [data[str(l)]['max'] for l in layers], marker='o', label=f'{kind} (max p99)')
            ax.set_yscale('symlog', linthresh=1e-8)
            ax.set_xlabel('layer')
            ax.set_ylabel('p99 |diff| / RMS (singleton)')
            ax.set_title(f'{stat}: batched vs singleton')
            ax.legend()
        fig.tight_layout()
        fig.savefig(self._stage_dir('batch_agreement') / 'discrepancy_vs_layer.png', dpi=120)
        plt.close(fig)

    def _d_matrix(self, arm, rows, dtype, *, device=None, batch_size=None):
        model = self.backend.load(arm, dtype, device=device)
        collector = self._collector(model, dtype)
        layers = list(range(len(collector.layers)))
        factors = self._factors(arm, layers)
        base, adapted = {l: [] for l in layers}, {l: [] for l in layers}
        try:
            batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)] if batch_size else [rows]
            for batch in batches:
                b = collector.collect(batch, factors=None, validate=False)
                a = collector.collect(batch, factors=factors, validate=False)
                for l in layers:
                    base[l].append(b['arrays'][(PRIMARY, l, 'mean_abs_z')])
                    adapted[l].append(a['arrays'][(PRIMARY, l, 'mean_abs_z')])
        finally:
            collector.close()
            self.backend.release(model)
        return {l: d_values(np.concatenate(base[l]), np.concatenate(adapted[l])) for l in layers}

    def stage_fp64_floor(self, arms_result):
        rows = self._allocation()['fp64']
        chain = self.plan['fp64_floor']['fallback_chain']
        floors = {}
        for arm in [a for a in arms_result['qualified'] if a != 'untuned']:
            d32 = self._d_matrix(arm, rows, 'float32')
            attempt, d64, label, used_rows = None, None, 'fp64-gpu', rows
            for step, (kind, kwargs) in enumerate([
                    ('fp64-gpu', dict()),
                    ('fp64-gpu-batch1-le512', dict(batch_size=1)),
                    ('fp64-cpu-3-examples', dict(device='cpu'))]):
                subset = rows if step == 0 else [r for r in rows if len(r['input_ids']) <= 512] if step == 1 else rows[:3]
                if not subset:
                    attempt = f'{kind}: no eligible examples'
                    continue
                try:
                    d64 = self._d_matrix(arm, subset, 'float64', **kwargs)
                    label, used_rows = kind, subset
                    if subset is not rows:
                        d32 = self._d_matrix(arm, subset, 'float32', **{k: v for k, v in kwargs.items() if k != 'device'})
                    break
                except Exception as exc:
                    if not self.backend.is_oom(exc):
                        raise
                    attempt = f'{kind}: {type(exc).__name__}'
                    self._progress('fp64_floor', 'fallback', arm=arm, failed=kind)
            per_layer = {}
            if d64 is None:
                label = 'proxy floor'
                singles = self._d_matrix(arm, rows, 'float32', batch_size=1)
                for l in d32:
                    per_layer[str(l)] = dict(floor_from(d32[l], singles[l]), kind='proxy floor')
            else:
                for l in d32:
                    per_layer[str(l)] = dict(floor_from(d32[l], d64[l]), kind=label)
            floors[arm] = dict(method=label, examples=[r['id'] for r in used_rows], last_failure=attempt,
                               layers=per_layer, unresolved_layers=[l for l, v in per_layer.items()
                                                                   if v['status'] == 'unresolved'])
            self._progress('fp64_floor', 'arm', arm=arm, method=label)
        return dict(floors=floors, fallback_chain=chain, exploratory_flag='|D| < 5 x layer floor (not a gate)',
                    zero_floor='unresolved')

    def stage_batch_freeze(self, benchmark):
        production = self._rows('discovery') + self._rows('heldout')
        manifest = batch_manifest(pack(production, benchmark['selected_budget']), benchmark['selected_budget'])
        _write_json(self._stage_dir('batch_freeze') / 'batch_manifest.json', manifest)
        return dict(manifest_sha256=manifest['sha256'], batches=len(manifest['batches']),
                    examples=manifest['examples'], padded_tokens=manifest['padded_tokens'])

    def run(self, *, stop_after=None):
        self._status('running', stages=STAGES)
        results = {}
        for name in STAGES:
            sealed = self._sealed(name)
            if sealed is not None:
                results[name] = sealed
                self._progress(name, 'reused')
            else:
                self._progress(name, 'start')
                if name == 'preflight':
                    value = self.stage_preflight()
                elif name == 'masks':
                    value = self.stage_masks()
                elif name == 'arms':
                    value = self.stage_arms()
                elif name == 'benchmark':
                    pre = results['preflight']
                    value = self.stage_benchmark(results['arms'], dict(
                        local_disk=pre['local_disk'],
                        hf_observed_private_bytes=(pre.get('hf_online', {}).get('usage') or {}).get('observed_private_bytes'),
                        model_cache_bytes=self.config.get('model_cache_bytes', 0)))
                elif name == 'batch_agreement':
                    value = self.stage_batch_agreement(results['arms'], results['benchmark'])
                elif name == 'fp64_floor':
                    value = self.stage_fp64_floor(results['arms'])
                else:
                    value = self.stage_batch_freeze(results['benchmark'])
                results[name] = self._seal(name, value)
            if name == stop_after:
                self._status('paused', stage=name)
                return results
        summary = self.summary(results)
        _write_json(self.run_dir / 'qualification.json', summary)
        self._status('complete', status=summary['status'])
        return summary

    def summary(self, results):
        arms = results['arms']
        flags = results['batch_agreement']['review_flags']
        blockers_04 = ['production wall-time cap not recorded in the plan',
                       'review-flag sign-off not recorded (required whether or not flags were raised)']
        return dict(
            status='partial' if arms['partial'] else 'qualified', partial_label=arms['partial_label'],
            rehearsal=self.rehearsal, label='REHEARSAL (not valid for production)' if self.rehearsal else None,
            plan_sha256=self.plan['_plan_sha256'], plan_core_sha256=self.plan.get('_plan_core_sha256'),
            run_id=self.config['run_id'],
            arms_qualified=arms['qualified'], arms_failed=arms['failed'],
            runtime=results['preflight'].get('precision_device'),
            selected_budget=results['benchmark']['selected_budget'], phase_1b=results['benchmark']['phase_1b'],
            projection=results['benchmark']['projection'], storage=results['benchmark']['storage'],
            logits_max_abs=results['batch_agreement']['logits_max_abs'], review_flags_raised=[f['flag'] for f in flags],
            fp64_floor_methods={a: v['method'] for a, v in results['fp64_floor']['floors'].items()},
            unresolved_floor_layers={a: v['unresolved_layers'] for a, v in results['fp64_floor']['floors'].items()},
            batch_manifest_sha256=results['batch_freeze']['manifest_sha256'],
            ready_for_04=False, blockers_for_04=blockers_04,
            note='Passing this small qualification is evidence within its coverage, not a guarantee for all neurons.')


def signoff(run_dir, *, reviewer, decision, notes=''):
    """Record the human review of 03b's flags; ticket 04 requires decision == 'accept'."""
    run_dir = Path(run_dir)
    if decision not in ('accept', 'reject'):
        raise ValueError('decision must be accept or reject')
    summary = run_dir / 'qualification.json'
    result = run_dir / 'stages/batch_agreement/result.json'
    if not summary.is_file() or not result.is_file():
        raise ValueError('qualification is not complete')
    path = run_dir / 'review' / 'signoff.json'
    if path.exists():
        raise ValueError('a sign-off is already recorded; keep it and investigate instead')
    record = dict(reviewer=reviewer, decision=decision, notes=notes, recorded_unix=time.time(),
                  qualification_sha256=_file_sha(summary), batch_agreement_sha256=_file_sha(result),
                  flags=json.loads(result.read_text())['review_flags'])
    _write_json(path, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--plan', required=True)
    r.add_argument('--config', required=True, help='runner config JSON (paths on the worker)')
    r.add_argument('--stop_after', choices=STAGES)
    r.add_argument('--rehearsal', action='store_true', help='Colab bug-finding run with a rehearsal plan')
    s = sub.add_parser('signoff')
    s.add_argument('--run_dir', required=True)
    s.add_argument('--reviewer', required=True)
    s.add_argument('--decision', required=True, choices=['accept', 'reject'])
    s.add_argument('--notes', default='')
    args = parser.parse_args(argv)
    if args.command == 'signoff':
        print(json.dumps(signoff(args.run_dir, reviewer=args.reviewer, decision=args.decision, notes=args.notes),
                         indent=2, default=_json_default))
        return 0
    from pilot_eval import neuron_audit_preflight as pf
    config = json.loads(Path(args.config).read_text())
    environment, env_problems = pf.prepare_environment()  # before any Hub import
    plan = plan_mod.load(args.plan, '03b', rehearsal=args.rehearsal)
    inputs = plan_mod.input_problems(plan, config['artifact_root'])
    if inputs:
        raise SystemExit('inputs not ready: ' + '; '.join(inputs))
    backend = TransformersBackend(config['model_dir'], config['adapter_dirs'], config['tokenizer_dir'],
                                  reference_arm=config['reference_arm'])

    def offline_load():
        tok = backend.tokenizer()
        model = backend.fresh_base('float32', device='cpu')
        record = dict(tokenizer_files=tok.identity(), model_class=type(model).__name__,
                      parameters=sum(p.numel() for p in model.parameters()))
        backend.release(model)
        return record

    def preflight():
        record = pf.run(plan, volume_dir=config['volume_dir'], local_temp=config['local_temp'],
                        offline_load=offline_load,
                        hf_check=lambda: pf.hf_online_subprocess(plan['preflight']['export_repo'], config['local_temp']),
                        rehearsal=args.rehearsal)
        record['environment_first_call'] = dict(environment, problems=env_problems)
        if env_problems:
            raise pf.PreflightFailure(env_problems, record)
        return record

    summary = Qualification(plan, config, backend, preflight=preflight).run(stop_after=args.stop_after)
    print(json.dumps(summary, indent=2, default=_json_default))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
