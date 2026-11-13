"""Ticket 06a: Colab rehearsal driver (bug-finding only; qualifies nothing; outputs labelled ``rehearsal-colab``).

Inputs are calibration rows and the record's rehearsal generation items only; discovery, confirmation and the
ADR 0001 cohort are never read. Covering set, per rehearsal arm (spec §Run tiers):

- calibration base traces (ticket 01); pass 1 + pass 2 with the Σ_j identity (ticket 03) as sealed units;
- real-model finite differences (``fd_adapters`` adapters, ``fd_neurons`` neurons);
- Stage C machinery on calibration rows: no-op hook, single removal, group removal, full ``down_proj``
  removal, natural 64-term read, constant read (ticket 04);
- full vs edited-checkpoint generation on 2 rehearsal items, edited logits checked against the gate;
- a corrupted and an interrupted unit, each detected;
- a token-budget probe from the smallest sweep budget upward with singleton and longest examples (OOM
  recorded per candidate);
- a sealed summary, then online upload with fresh readback and a clean-root restore of this partial output.

Every step's outcome is recorded; a failing step is recorded with its traceback and the driver continues
with the steps that do not depend on it, so one Colab session finds as many bugs as possible.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from pilot_eval import attribution_ablation as ab
from pilot_eval import attribution_stages as st
from pilot_eval import attribution_traces as at
from pilot_eval import attribution_units as au

LABEL_PREFIX = 'rehearsal-colab'


class Rehearsal:
    def __init__(self, record, root, backend, *, client=None, clean_root=None):
        if record['mode'] != 'rehearsal' or not str(record.get('rehearsal_label', '')).startswith(LABEL_PREFIX):
            raise ValueError(f'06a records must be rehearsals labelled {LABEL_PREFIX}-*')
        self.record, self.root, self.backend = record, Path(root), backend
        self.stages = st.Stages(record, root, backend, code_sha256=record['code_commit'])
        self.client, self.clean_root = client, clean_root
        self.steps, self.started = {}, time.monotonic()
        caps = dict(record['caps_seconds'])
        total = caps.pop('total')
        self.budget = au.Budget(self.root / 'attribution/rehearsal/budget.json', stage_caps=caps, total_cap=total)

    def step(self, name, fn):
        self.budget.start('qualification')
        started = time.monotonic()
        try:
            value = fn()
            self.steps[name] = dict(status='ok', seconds=time.monotonic() - started, result=value)
        except Exception as exc:  # recorded with its traceback; later independent steps still run
            self.steps[name] = dict(status='failed', seconds=time.monotonic() - started,
                                    error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()[-6000:])
        print(json.dumps(dict(step=name, status=self.steps[name]['status'],
                              error=self.steps[name].get('error')), default=str), flush=True)
        self.budget.boundary('qualification')
        return self.steps[name].get('result')

    def run(self):
        q = self.record['qualification']
        tol = self.record['tolerances']
        manifest = self.step('calibration_traces',
                             lambda: self.stages._traces('calibration', 1, minimum=q['calibration_minimum'])['counts'])
        rows = at.load_rows(self.root, 'calibration') if manifest is not None else []
        for arm in self.record['expected_arms']:
            self.step(f'attribution:{arm}', lambda arm=arm: self._attribution(arm, rows))
            self.step(f'finite_differences:{arm}', lambda arm=arm: self._fd(arm, rows, tol))
            self.step(f'conditions:{arm}', lambda arm=arm: self._conditions(arm, rows))
            self.step(f'generation:{arm}', lambda arm=arm: self._generation(arm))
        self.step('corrupt_and_interrupted_units', self._corruption)
        self.step('token_budget_probe', lambda: self._probe(rows))
        summary = dict(label=self.record['rehearsal_label'], code_commit=self.record['code_commit'],
                       steps={k: {kk: vv for kk, vv in v.items() if kk != 'traceback'} for k, v in self.steps.items()},
                       failed=[k for k, v in self.steps.items() if v['status'] != 'ok'],
                       wall_seconds=time.monotonic() - self.started, runtime=self.backend.runtime(),
                       note='rehearsal: finds bugs, qualifies nothing; never feeds production')
        au.seal_run(self.root / 'attribution/rehearsal/tracebacks',
                    {'tracebacks.json': {k: v.get('traceback') for k, v in self.steps.items()}})
        au.write_summary(self.root, summary=summary, markdown=_markdown(summary), missing_work=summary['failed'])
        if self.client is not None:
            summary['export'] = self._export()
        return summary

    # --- steps ------------------------------------------------------------------------------------

    def _attribution(self, arm, rows):
        inst = self.stages._instrument(self.backend.load(arm))
        units = self.stages.units
        out, results = None, []
        for index, row in enumerate(rows):
            key = au.unit_key('ab', arm, batch=index)
            result = inst.attribute([row])  # the Σ_j identity is enforced inside
            from pilot_eval.attribution_instrument import unit_blocks
            blocks, counts, records = unit_blocks(result)
            units.write(key, identity=units.identity('ab', arm, [row['id']], input_sha256='calibration', batch=index),
                        example_ids=[row['id']], blocks=blocks, counts=counts, results=records)
            out = result if out is None else out
            results.append(dict(id=row['id'], trace=float(result['metrics']['trace'][0])))
        return dict(units=len(rows), adapters=len(inst.adapters), padded_width=out['padded_width'] if out else None)

    def _fd(self, arm, rows, tol):
        q = self.record['qualification']
        model = self.backend.load(arm)
        inst = self.stages._instrument(model)
        fd = st.finite_difference_check(model, inst, rows[:q['fd_rows']], self.stages._stop()['stop_ids'],
                                        epsilon=tol['finite_difference']['epsilon'], adapters=q['fd_adapters'],
                                        neurons=q['fd_neurons'])
        fd['within_record_tolerances'] = bool(fd['median_rel_err'] <= tol['finite_difference']['median_rel_err_max']
                                              and fd['max_rel_err'] <= tol['finite_difference']['max_rel_err_max'])
        return fd

    def _calibration_selection(self, arm, rows):
        inst = self.stages._instrument(self.backend.load(arm))
        out = inst.attribute(rows)
        mean = np.nanmean(out['attr']['trace'], 0)
        down = max(inst.down_index.values(), key=lambda i: abs(mean[i]))
        layer = [l for l, i in inst.down_index.items() if i == down][0]
        position = sorted(inst.down_index).index(layer)
        cbar = float(out['signed_sum_c'][:, position].sum() / out['token_count'].sum())
        order = sorted(range(out['neuron']['trace'][layer].shape[1]),
                       key=lambda j: -np.sign(mean[down]) * np.nanmean(out['neuron']['trace'][layer][:, j]))
        return dict(top=ab.top_k(mean, self.record['selection']['top_group']), down=down, layer=layer, cbar=cbar,
                    order=order)

    def _conditions(self, arm, rows):
        sel = self._calibration_selection(arm, rows)
        stage = ab.StageC(self.backend.load(arm), stop_ids=self.stages._stop()['stop_ids'],
                          pad_token_id=self.stages._pad(), head_chunk=self.record['qualification']['head_chunk'])
        read = self.record['selection']['read_size']
        conditions = dict(noop=dict(kind='removal', adapters=[]), single=dict(kind='removal', adapters=sel['top'][:1]),
                          group=dict(kind='removal', adapters=sel['top']),
                          down_full=dict(kind='removal', adapters=[sel['down']]),
                          read64=dict(kind='read', layer=sel['layer'], keep=sel['order'][:read]),
                          constant=dict(kind='constant', adapter=sel['down'], cbar=sel['cbar']))
        out = {}
        for name, condition in conditions.items():
            result = stage.evaluate(rows, condition)
            out[name] = dict(mean_delta_trace=float(np.nanmean(result['delta_trace'])),
                             mean_kl=float(np.nanmean(result['kl'])),
                             stop_defined=int(np.sum(~np.isnan(result['delta_stop']))))
        if out['noop']['mean_kl'] != 0.0 or out['noop']['mean_delta_trace'] != 0.0:
            raise ValueError(f"no-op hook changed the metrics: {out['noop']}")
        return dict(conditions=out, selection='calibration-only (rehearsal); not a freeze')

    def _generation(self, arm):
        items = self.backend.rehearsal_items()[:2]
        sel_rows = at.load_rows(self.root, 'calibration')
        top = self._calibration_selection(arm, sel_rows)['top']
        full, _ = self.backend.stage_d_generator(arm, None, 2, None)
        edited, edit = self.backend.stage_d_generator(arm, top, 2,
                                                      self.root / f'attribution/rehearsal/edited/{arm}')
        batches = [[r['id'] for r in items]]
        a, b = at.generate(full, items, batches), at.generate(edited, items, batches)
        au.seal_run(self.root / f'attribution/rehearsal/generation/{arm}',
                    {'full.jsonl': ''.join(json.dumps(r) + '\n' for r in a),
                     'edited.jsonl': ''.join(json.dumps(r) + '\n' for r in b), 'edit.json': edit})
        return dict(items=len(items), edit_hash=edit['edited_sha256'] if edit else None,
                    logits_equal_gate=bool(edit and edit.get('logits_equal_gate', True)),
                    changed=[x['generated_ids'] != y['generated_ids'] for x, y in zip(a, b)])

    def _corruption(self):
        units = au.Units(self.root / 'attribution/rehearsal/corruption', plan_sha256='rehearsal',
                         code_sha256=self.record['code_commit'], runtime_sha256='rehearsal')
        key = au.unit_key('ab', 'probe', batch=0)
        units.write(key, identity=units.identity('ab', 'probe', ['x'], input_sha256='x', batch=0), example_ids=['x'],
                    blocks={('examples', 0, 'v'): np.ones((1, 2), dtype=np.float32)}, counts={'examples': [1]})
        blob = units.store.unit_dir(key) / 'blocks.f32'
        data = bytearray(blob.read_bytes())
        data[0] ^= 0xFF
        blob.write_bytes(bytes(data))
        interrupted = au.unit_key('ab', 'probe', batch=1)
        units.store.unit_dir(interrupted).mkdir(parents=True)
        fresh = au.Units(units.root, plan_sha256='rehearsal', code_sha256=self.record['code_commit'],
                         runtime_sha256='rehearsal')
        states = dict(corrupted=fresh.state(key), interrupted=fresh.state(interrupted))
        if states != dict(corrupted='corrupt', interrupted='interrupted'):
            raise ValueError(f'corruption/interruption not detected: {states}')
        import shutil
        shutil.rmtree(units.root)  # evidence recorded; keep the export scope clean
        return states

    def _probe(self, rows):
        if not rows:
            raise ValueError('no calibration rows')
        inst = self.stages._instrument(self.backend.load(self.record['expected_arms'][0]))
        longest = max(rows, key=lambda r: len(r['input_ids']))
        out = []
        for budget in sorted(self.record['token_budget_sweep']):
            for kind, batch in [('singleton-longest', [longest]), ('packed', st.longest_probe(rows, budget))]:
                if batch is None:
                    out.append(dict(budget=budget, probe=kind, outcome='longest-example-exceeds-budget'))
                    continue
                self.backend.reset_peak()
                started = time.monotonic()
                try:
                    inst.attribute(batch)
                except Exception as exc:
                    if not self.backend.is_oom(exc):
                        raise
                    out.append(dict(budget=budget, probe=kind, outcome='oom'))
                    continue
                out.append(dict(budget=budget, probe=kind, outcome='ok', rows=len(batch),
                                padded_tokens=len(batch) * max(len(r['input_ids']) for r in batch),
                                seconds=time.monotonic() - started, peak_bytes=self.backend.peak_memory(),
                                device_bytes=self.backend.device_memory()))
        return out

    def _export(self):
        label = self.record['rehearsal_label']
        try:
            receipt = au.export(self.root, label=label, include=['attribution'], expected_units=[],
                                expected_runs=[au.SUMMARY_DIR], client=self.client, repo=self.record['export_repo'],
                                temp_dir=self.root / 'tmp', on_event=lambda e: print(json.dumps(e), flush=True))
            restored = au.restore_summary(receipt, self.clean_root, client=self.client,
                                          on_event=lambda e: print(json.dumps(e), flush=True))
            return dict(status='ok', revision=receipt['revision'], completeness=receipt['completeness'],
                        restored_files=restored['proof']['verified_files'],
                        restored_summary_label=restored['summary']['label'])
        except Exception as exc:
            return dict(status='failed', error=f'{type(exc).__name__}: {exc}')


def _markdown(summary):
    lines = [f"# 06a rehearsal — {summary['label']}", '', summary['note'], '',
             '| step | status | seconds |', '| --- | --- | --- |']
    lines += [f"| {k} | {v['status']} | {v['seconds']:.1f} |" for k, v in summary['steps'].items()]
    if summary['failed']:
        lines += ['', 'Failed steps (tracebacks in attribution/rehearsal/tracebacks): ' + ', '.join(summary['failed'])]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--record', required=True)
    parser.add_argument('--work', required=True)
    parser.add_argument('--clean_root', required=True)
    args = parser.parse_args(argv)
    from pilot_eval.attribution_backend import TransformersBackend
    from pilot_eval.attribution_chain import _commit, load_launch_record
    from pilot_eval.neuron_audit_export import OnlineHubProcess
    record = load_launch_record(args.record, code_commit=_commit())
    backend = TransformersBackend(record, Path(args.work) / 'volume')
    summary = Rehearsal(record, Path(args.work) / 'local', backend, client=OnlineHubProcess(timeout=1800),
                        clean_root=args.clean_root).run()
    print('REHEARSAL SUMMARY:', json.dumps(dict(failed=summary['failed'], export=summary.get('export')), default=str))
    return 0 if not summary['failed'] and summary.get('export', {}).get('status') == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
