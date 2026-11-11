"""Ticket 04: resumable multi-arm production collection with per-arm verified HF packages.

Preconditions (all checked before any model load):
- the run plan loads for stage 04 (approved, wall-time cap recorded) and its frozen core equals the
  core the 03b qualification ran under (only the wall cap / approval may differ);
- the 03b qualification is complete, is not a rehearsal unless this run is, and has an *accept*
  sign-off bound to its current files;
- the frozen batch manifest matches its hash, and only discovery/held-out rows are collected.

Work happens on the Pod's local disk. Each sealed unit (arm x frozen batch) is mirrored to the Global
Volume (copy + verify, no rename or listing assumptions) so a later Pod can resume. When an arm
finishes, its deterministic package is uploaded to the private HF dataset under ``neuron-audit/`` and
verified by a fresh pinned-revision download; then its local units may be pruned. The final manifest
references the arm packages and is itself read back. The Pod may stop only after that (optional
``stop_command``); any failure leaves the Pod running and the local/volume work intact.

The advisory writer marker (host, PID, heartbeat) is evidence, not a lock: an existing marker from
another writer, live or stale, stops the run for manual investigation.

Before any collection the run re-establishes its inputs: sequences bytes, adapter files and the runtime
fingerprint must equal what 03b qualified, and the progress record's context (plan core, qualification,
batch manifest, sequences, label, repo) must equal the current one, so a reused run_id cannot mix contexts.
Wall time is cumulative across restarts (persisted) and checked after every unit; Hub calls run in a
separate online process (``OnlineHubProcess``) because the worker itself stays offline.
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

from pilot_eval import neuron_audit_collector as nc
from pilot_eval import neuron_audit_export as ex
from pilot_eval import neuron_audit_plan as plan_mod
from pilot_eval.neuron_audit_blocks import BlockStore, mirror_store
from pilot_eval.neuron_audit_qualify import RUN_SUBDIR as QUAL_SUBDIR, _file_sha, _write_json, read_sequences

RUN_SUBDIR = 'runs/joint-neuron-adapter-audit/production'


class ProductionRefused(RuntimeError):
    pass


class OverrunStop(RuntimeError):
    pass


def check_inputs(plan, config, summary, *, rehearsal, backend=None):
    """Re-establish in 04 that the files and runtime are the ones 03b qualified; returns the sequences sha."""
    root = Path(config['artifact_root'])
    seq_dir = root / plan['inputs']['sequences_run']
    problems = plan_mod.input_problems(plan, root)
    if rehearsal:  # bug-finding only, as in 03b: a pending background review is a warning
        problems = [p for p in problems if not p.startswith('background review')]
    if problems:
        raise ProductionRefused('inputs not ready: ' + '; '.join(problems))
    actual = _file_sha(seq_dir / 'sequences.jsonl')
    manifest = json.loads((seq_dir / 'manifest.json').read_text())['sequences_sha256']
    qualified = (summary.get('inputs') or {}).get('sequences_sha256')
    if actual != manifest or actual != qualified:
        raise ProductionRefused(f'sequences.jsonl ({actual[:12]}) differs from its manifest ({manifest[:12]}) or from '
                                f'what 03b qualified ({str(qualified)[:12]})')
    inventory = json.loads((root / plan['inputs']['inventory_run'] / 'inventory.json').read_text())
    step = str(inventory['audit_step'])
    recorded_adapters = (summary.get('inputs') or {}).get('adapters') or {}
    for arm in summary['arms_qualified']:
        if arm == 'untuned':
            continue
        path = Path(config['adapter_dirs'][arm]) / 'adapter_model.safetensors'
        expected = inventory['arms'][arm]['checkpoints'][step]['files']['adapter_model.safetensors']['sha256']
        got = _file_sha(path) if path.is_file() else None
        if got != expected or got != recorded_adapters.get(arm):
            raise ProductionRefused(f'adapter for {arm} at {path} differs from the sealed inventory or from 03b')
    if backend is not None:
        from pilot_eval.neuron_audit_preflight import runtime_fingerprint
        current = runtime_fingerprint(backend.device_info())
        if current != summary.get('runtime_fingerprint'):
            raise ProductionRefused(f'runtime differs from the qualified one: {current} vs '
                                    f"{summary.get('runtime_fingerprint')}; requalify (03b) on this runtime")
    return actual


# --- preconditions --------------------------------------------------------------------------------

def check_qualification(plan, qual_dir, *, rehearsal):
    qual_dir = Path(qual_dir)
    summary_path = qual_dir / 'qualification.json'
    if not summary_path.is_file():
        raise ProductionRefused('03b qualification is not complete')
    summary = json.loads(summary_path.read_text())
    if bool(summary.get('rehearsal')) != bool(rehearsal):
        raise ProductionRefused('rehearsal qualifications can only feed rehearsal runs, and vice versa')
    if summary.get('status') not in ('qualified', 'partial'):
        raise ProductionRefused(f"qualification status is {summary.get('status')}")
    if summary.get('plan_core_sha256') != plan['_plan_core_sha256']:
        raise ProductionRefused('plan content changed since 03b (beyond the wall-time cap and approval)')
    signoff = qual_dir / 'review' / 'signoff.json'
    if not signoff.is_file():
        raise ProductionRefused('03b review-flag sign-off is not recorded')
    record = json.loads(signoff.read_text())
    if record.get('decision') != 'accept':
        raise ProductionRefused('03b review-flag sign-off is not an accept')
    if record.get('qualification_sha256') != _file_sha(summary_path) or record.get('batch_agreement_sha256') != \
            _file_sha(qual_dir / 'stages/batch_agreement/result.json'):
        raise ProductionRefused('sign-off does not match the current qualification files')
    manifest_path = qual_dir / 'stages/batch_freeze/batch_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    from pilot_eval.neuron_audit_qualify import _sha
    if _sha({k: v for k, v in manifest.items() if k != 'sha256'}) != manifest['sha256'] or \
            manifest['sha256'] != summary['batch_manifest_sha256']:
        raise ProductionRefused('frozen batch manifest hash differs')
    return summary, manifest


class Marker:
    """Advisory single-writer marker (local + mirrored); never a distributed lock, never auto-takeover."""

    def __init__(self, paths, run_id):
        self.paths, self.run_id = [Path(p) for p in paths], run_id
        self.me = dict(run_id=run_id, host=socket.gethostname(), pid=os.getpid(), started_unix=time.time())

    def acquire(self, *, resume_own=False):
        for path in self.paths:
            if path.is_file():
                other = json.loads(path.read_text())
                if other.get('state') == 'released':
                    continue
                same = other.get('host') == self.me['host'] and other.get('pid') == self.me['pid']
                if not same and not resume_own:
                    age = time.time() - other.get('heartbeat_unix', 0)
                    raise ProductionRefused(f"writer marker exists ({other.get('host')}:{other.get('pid')}, "
                                            f'heartbeat {age:.0f}s ago); investigate before resuming, never take over')
        self.beat(state='running')

    def beat(self, state='running', **extra):
        record = dict(self.me, state=state, heartbeat_unix=time.time(), **extra)
        for path in self.paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('w') as stream:  # direct write: no rename semantics assumed on the volume
                stream.write(json.dumps(record, indent=2) + '\n')
                stream.flush()
                os.fsync(stream.fileno())

    def release(self, state):
        self.beat(state='released', final_state=state)


def overrun_check(done_fraction, elapsed_seconds, *, projected_seconds, cap_seconds, history, step=0.10):
    """Plan rule: at 10% of units and every further 10%, forecast linearly; two consecutive bad forecasts stop.

    ``elapsed_seconds`` is cumulative collection time, comparable with the 03b benchmark projection.
    """
    checkpoint = int(done_fraction / step + 1e-9)
    if checkpoint < 1 or checkpoint in history['seen']:
        return None
    history['seen'].add(checkpoint)
    forecast = elapsed_seconds / done_fraction
    bad = forecast > 2 * projected_seconds or forecast > cap_seconds
    history['consecutive'] = history['consecutive'] + 1 if bad else 0
    record = dict(checkpoint=checkpoint * step, forecast_seconds=forecast, bad=bad, consecutive=history['consecutive'])
    history['records'].append(record)
    if history['consecutive'] >= 2:
        raise OverrunStop(f'two consecutive forecasts exceed 2x the benchmark projection or the cap: {record}')
    return record


# --- runner ------------------------------------------------------------------------------------

class Production:
    def __init__(self, plan, config, backend, *, client=None, now=time.monotonic, stop_runner=subprocess.run,
                 preflight=None):
        self.plan, self.config, self.backend, self.client = plan, config, backend, client
        self.preflight = preflight
        self.rehearsal = bool(plan.get('_rehearsal'))
        run_id = config['run_id']
        if self.rehearsal != run_id.startswith('rehearsal-'):
            raise ProductionRefused('rehearsal runs (and only they) must use a run_id starting with rehearsal-')
        self.now, self.stop_runner = now, stop_runner
        self.session_started = now()
        self.local_dir = Path(config['local_root']) / RUN_SUBDIR / run_id
        self.volume_dir = Path(config['volume_root']) / RUN_SUBDIR / run_id
        self.qual_dir = Path(config['qualification_root']) / QUAL_SUBDIR / config['qualification_run_id']
        self.summary, self.manifest = check_qualification(plan, self.qual_dir, rehearsal=self.rehearsal)
        self.seq_sha = check_inputs(plan, config, self.summary, rehearsal=self.rehearsal, backend=backend)
        self.sequences = {r['id']: r for r in read_sequences(Path(config['artifact_root']) / plan['inputs']['sequences_run'])}
        for batch in self.manifest['batches']:
            for i in batch['ids']:
                if self.sequences[i]['split'] not in ('discovery', 'heldout'):
                    raise ProductionRefused(f'batch manifest contains a {self.sequences[i]["split"]} row')
        self.arms = [a for a in plan['arms'] if a in self.summary['arms_qualified']]
        self.label = config['label'] if not self.rehearsal else f"rehearsal-{config['label']}"
        self.repo = config.get('repo') or plan['preflight']['export_repo']
        if self.client is None:
            self.client = ex.OnlineHubProcess()
        self.inventory = json.loads((Path(config['artifact_root']) / plan['inputs']['inventory_run'] /
                                     'inventory.json').read_text())
        self.local_store = BlockStore(self.local_dir / 'blocks')
        self.marker = Marker([self.local_dir / 'meta/writer.json', self.volume_dir / 'meta/writer.json'], run_id)
        self.state_paths = [self.local_dir / 'meta/progress.json', self.volume_dir / 'meta/progress.json']

    # state
    def _context(self):
        return dict(plan_core_sha256=self.plan['_plan_core_sha256'], qualification_run_id=self.config['qualification_run_id'],
                    qualification_sha256=_file_sha(self.qual_dir / 'qualification.json'),
                    batch_manifest_sha256=self.manifest['sha256'], sequences_sha256=self.seq_sha, label=self.label,
                    repo=self.repo, rehearsal=self.rehearsal, arms=self.arms,
                    max_batches_per_arm=self.config.get('max_batches_per_arm'))

    def _load_state(self):
        state = None
        for path in self.state_paths[::-1]:  # the volume copy survives a Pod change
            if path.is_file():
                state = json.loads(path.read_text())
                break
        if state is None:
            return dict(context=self._context(), units={}, arm_packages={}, overrun=[], overrun_seen=[], events=[],
                        wall_seconds=0.0, collect_seconds=0.0)
        if state.get('context') != self._context():
            raise ProductionRefused('this run_id already holds progress from a different context '
                                    f"({state.get('context')}); use a new run_id or investigate")
        return state

    def _wall(self, state):
        return state.get('wall_seconds', 0.0) + (self.now() - self.session_started)

    def _save_state(self, state):
        state['wall_seconds_saved'] = self._wall(state)
        for path in self.state_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('w') as stream:
                stream.write(json.dumps(state, indent=2, sort_keys=True) + '\n')
                stream.flush()
                os.fsync(stream.fileno())

    def _event(self, state, kind, **extra):
        line = dict(time=time.time(), kind=kind, **extra)
        state['events'].append(line)
        print(json.dumps(line), flush=True)

    def _scale(self, arm):
        return self.inventory['arms'][arm]['checkpoints'][str(self.inventory['audit_step'])]['config']['scale']

    def _unit_identity(self, arm, batch):
        return dict(arm=arm, batch=batch['index'], rows=batch['ids'], sequences_sha256=self.seq_sha,
                    batch_manifest_sha256=self.manifest['sha256'], plan_core_sha256=self.plan['_plan_core_sha256'],
                    qualification=self.config['qualification_run_id'], rehearsal=self.rehearsal)

    def _restore_from_volume(self, keys):
        volume = BlockStore(self.volume_dir / 'blocks')
        present = [k for k in keys if volume.state(k) == 'sealed']  # explicit paths, no listing
        if present:
            mirror_store(self.volume_dir / 'blocks', self.local_dir / 'blocks', keys=present, source_listing=False)
        return present

    def _keys(self, arm, batches):
        return [f'{arm}-batch-{b["index"]:05d}' for b in batches]

    def _check_cap(self, state, where):
        if self._wall(state) > self.cap_seconds:
            raise OverrunStop(f'hard wall-time cap reached ({where}; cumulative across restarts)')

    def _collect_arm(self, arm, state, *, batches, history, total_units):
        heads = self.summary['phase_1b']['layers'] or None
        keys = self._keys(arm, batches)
        todo = [(k, b) for k, b in zip(keys, batches) if state['units'].get(k) != 'mirrored']
        if not todo:
            return keys
        self._restore_from_volume([k for k, _ in todo])
        model = self.backend.load(arm, 'float32')
        collector = nc.NeuronCollector(model, tolerances=self._tolerances(),
                                       expected_base_hash=self.inventory['base'].get('expected_base_parameter_sha256')
                                       if self.config.get('check_base_hash', True) else None)
        layers = list(range(len(collector.layers)))
        factors = None if arm == 'untuned' else nc.load_saved_factors(self.config['adapter_dirs'][arm], layers, self._scale(arm))
        head_factors = None if arm == 'untuned' or not heads else nc.load_saved_factors(
            self.config['adapter_dirs'][arm], heads, self._scale(arm), 'o_proj')
        validate_first = int(self.config.get('validate_batches', 2))
        try:
            for position, (key, batch) in enumerate(todo):
                rows = [self.sequences[i] for i in batch['ids']]
                unit_started = self.now()
                nc.collect_cohort(collector, self.local_store, rows, arm=arm, batch_size=len(rows),
                                  factors=factors, identity=self._unit_identity(arm, batch),
                                  validate=position < validate_first, heads=heads, head_factors=head_factors,
                                  key=key)
                state['collect_seconds'] = state.get('collect_seconds', 0.0) + (self.now() - unit_started)
                mirror_store(self.local_dir / 'blocks', self.volume_dir / 'blocks', keys=[key])
                state['units'][key] = 'mirrored'
                self._save_state(state)
                self.marker.beat(arm=arm, unit=key)
                self._check_cap(state, f'after unit {key}')
                done = sum(1 for v in state['units'].values() if v == 'mirrored') / total_units
                # Forecast collection time only (what 03b benchmarked); export/mirror time counts toward the hard cap.
                record = overrun_check(done, state['collect_seconds'], projected_seconds=self.projected_seconds,
                                       cap_seconds=self.cap_seconds, history=history)
                if record:
                    state['overrun'].append(record)
                    state['overrun_seen'] = sorted(history['seen'])
                    self._save_state(state)
        finally:
            collector.close()
            self.backend.release(model)
        return keys

    def _tolerances(self):
        numeric = self.plan['hard_gates']['identities']['numeric']
        return dict(z_reconstruction_relative_to_layer_max=float(numeric['z_relative_to_layer_max']),
                    identity_atol=float(numeric['atol']), identity_rtol=float(numeric['rtol']))

    def _package_arm(self, arm, state, keys):
        if arm in state['arm_packages']:
            return state['arm_packages'][arm]
        # A new Pod has none of the earlier units locally: bring back every mirrored one, then package exactly
        # the arm's frozen unit list (a missing unit refuses the package; it is never silently smaller).
        missing_local = [k for k in keys if self.local_store.state(k) != 'sealed']
        restored = self._restore_from_volume(missing_local)
        if set(missing_local) - set(restored):
            raise RuntimeError(f'arm {arm}: units neither local nor sealed on the volume: '
                               f'{sorted(set(missing_local) - set(restored))[:5]}')
        if restored:
            self._event(state, 'units-restored-from-volume', arm=arm, count=len(restored))
        package, manifest_path, manifest = ex.package_arm(self.local_dir / 'blocks', arm, self.label,
                                                          self.config['local_temp'], expected_keys=keys)
        try:
            record = ex.upload_package(package, manifest_path, manifest, repo=self.repo, label=self.label,
                                       client=self.client, temp_dir=self.config['local_temp'])
        finally:
            shutil.rmtree(package.parent, ignore_errors=True)
        state['arm_packages'][arm] = record
        self._save_state(state)
        self._event(state, 'arm-package-verified', arm=arm, bytes=record['bytes'], revision=record['revision'])
        if self.config.get('prune_local_after_verified_upload', True):
            for key in record['units']:
                shutil.rmtree(self.local_store.unit_dir(key), ignore_errors=True)
        return record

    def _finish(self, state, *, final_state):
        records = list(state['arm_packages'].values())
        final = ex.publish_final_manifest(
            records, repo=self.repo, label=self.label, required_arms=self.arms, client=self.client,
            temp_dir=self.config['local_temp'], expected_units=self.expected_units,
            extras=dict(qualification_run_id=self.config['qualification_run_id'],
                        qualification_sha256=_file_sha(self.qual_dir / 'qualification.json'),
                        batch_manifest_sha256=self.manifest['sha256'], plan_core_sha256=self.plan['_plan_core_sha256'],
                        sequences_sha256=self.seq_sha, run_state=final_state, rehearsal=self.rehearsal,
                        partial_label=self.summary.get('partial_label')))
        state['final_manifest'] = final
        state['safe_to_stop_pod'] = ex.may_stop_pod_final(final, records, required_arms=self.arms,
                                                          expected_units=self.expected_units)
        self._save_state(state)
        return final

    def _maybe_stop_after_failure(self, state):
        """Opt-in (config ``stop_after_verified_partial_backup``, default off): after a failure or overrun, stop
        the Pod only if every unit collected so far is sealed on the volume and every packaged arm is verified on
        HF; nothing then exists only on the container disk. Off by default: the user rule is to stop only after
        the required final backup."""
        if not (self.config.get('stop_after_verified_partial_backup') and self.config.get('stop_command')):
            return False
        volume = BlockStore(self.volume_dir / 'blocks')
        local = [k for k, s in self.local_store.status().items() if s == 'sealed']
        unmirrored = [k for k in local if volume.state(k) != 'sealed']
        final = state.get('final_manifest') or {}
        if unmirrored or not final.get('revision'):
            self._event(state, 'pod-left-running', reason='partial backup incomplete', unmirrored=unmirrored[:5])
            return False
        result = self.stop_runner(self.config['stop_command'], capture_output=True, text=True)
        state['pod_stop_requested'] = True
        self._event(state, 'pod-stop-requested-after-partial-backup', returncode=result.returncode)
        self._save_state(state)
        return True

    def run(self, *, resume_own=False):
        self.marker.acquire(resume_own=resume_own)
        state = self._load_state()
        if self.preflight is not None:  # the actual worker's interpreter, packages, offline load, storage
            state.setdefault('preflights', []).append(self.preflight())
            self._save_state(state)
        state['label'], state['arms'] = self.label, self.arms
        caps = self.plan['operations']
        cap = caps.get('production_wall_time_cap_hours')
        if cap is None and self.rehearsal:
            cap = self.config.get('rehearsal_cap_hours')
        if cap is None:
            raise ProductionRefused('production wall-time cap is not recorded')
        self.cap_seconds = float(cap) * 3600
        self.projected_seconds = float(self.summary['projection']['total_hours']) * 3600
        batches = self.manifest['batches'][: self.config.get('max_batches_per_arm') or None]
        if self.config.get('max_batches_per_arm') and not self.rehearsal:
            raise ProductionRefused('max_batches_per_arm is for rehearsals only')
        total_units = len(batches) * len(self.arms)
        self.expected_units = {arm: self._keys(arm, batches) for arm in self.arms}
        state['wall_seconds'] = state.get('wall_seconds_saved', state.get('wall_seconds', 0.0))
        history = dict(seen=set(state.get('overrun_seen', [])), consecutive=0, records=state.get('overrun', []))
        final_state = 'complete'
        try:
            self._check_cap(state, 'at start')
            for arm in self.arms:
                if arm in state['arm_packages']:
                    self._event(state, 'arm-already-verified', arm=arm)
                    continue
                self._event(state, 'arm-start', arm=arm)
                self._collect_arm(arm, state, batches=batches, history=history, total_units=total_units)
                self._package_arm(arm, state, self.expected_units[arm])
                self._check_cap(state, f'after packaging {arm}')
        except OverrunStop as exc:
            final_state = 'stopped-overrun'
            self._event(state, 'overrun-stop', reason=str(exc))
        except Exception as exc:
            self._event(state, 'failed', error=f'{type(exc).__name__}: {exc}')
            self._save_state(state)
            try:
                self._finish(state, final_state='failed')
            except Exception as export_exc:  # never stop the Pod without a verified backup
                self._event(state, 'partial-export-failed', error=f'{type(export_exc).__name__}: {export_exc}')
            self._maybe_stop_after_failure(state)
            self.marker.release('failed')
            raise
        final = self._finish(state, final_state=final_state)
        if final_state != 'complete':
            if not self._maybe_stop_after_failure(state):
                self._event(state, 'pod-left-running', reason=f'run ended {final_state}; billing continues')
        elif state['safe_to_stop_pod'] and self.config.get('stop_command'):
            result = self.stop_runner(self.config['stop_command'], capture_output=True, text=True)
            state['pod_stop_requested'] = True
            self._event(state, 'pod-stop-requested', returncode=result.returncode)
        elif self.config.get('stop_command'):
            self._event(state, 'pod-left-running', reason='final backup not verified complete; billing continues')
        self._save_state(state)
        self.marker.release(final_state)
        return dict(state=final_state, label=self.label, final_manifest=final, safe_to_stop_pod=state['safe_to_stop_pod'],
                    arms=self.arms, partial_label=self.summary.get('partial_label'), rehearsal=self.rehearsal)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--plan', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--rehearsal', action='store_true')
    parser.add_argument('--resume_own', action='store_true', help='resume after investigating this run\'s own marker')
    args = parser.parse_args(argv)
    from pilot_eval import neuron_audit_preflight as pf
    pf.prepare_environment()
    from pilot_eval.neuron_audit_qualify import TransformersBackend
    config = json.loads(Path(args.config).read_text())
    plan = plan_mod.load(args.plan, '03b' if args.rehearsal else '04', rehearsal=args.rehearsal)
    backend = TransformersBackend(config['model_dir'], config['adapter_dirs'], config['tokenizer_dir'],
                                  reference_arm=config['reference_arm'])

    def offline_load():
        model = backend.fresh_base('float32', device='cpu')
        record = dict(model_class=type(model).__name__, parameters=sum(p.numel() for p in model.parameters()))
        backend.release(model)
        return record

    def preflight():
        return pf.run(plan, volume_dir=config['volume_root'], local_temp=config['local_temp'], offline_load=offline_load,
                      hf_check=lambda: pf.hf_online_subprocess(config.get('repo') or plan['preflight']['export_repo'],
                                                               config['local_temp']), rehearsal=args.rehearsal)

    client = ex.OnlineHubProcess()  # the worker stays offline; every Hub call runs in an online child process
    print(json.dumps(Production(plan, config, backend, client=client, preflight=preflight).run(
        resume_own=args.resume_own), indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
