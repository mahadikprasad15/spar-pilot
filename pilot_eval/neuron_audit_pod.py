"""Pod driver for the joint neuron/adapter audit: facts, input preparation, the small Pod rehearsal, and configs.

Run inside the Pod's venv from the repo checkout (see .scratch/joint-neuron-adapter-audit/pod-runbook.md):

    python -m pilot_eval.neuron_audit_pod facts --volume_dir /path/on/sl-0/neuron-audit --work /root/audit
    python -m pilot_eval.neuron_audit_pod prepare --volume_dir ... --work /root/audit
    python -m pilot_eval.neuron_audit_pod rehearsal --volume_dir ... --work /root/audit --tag pod1
    python -m pilot_eval.neuron_audit_pod configs --volume_dir ... --work /root/audit --tag v1

``facts`` records what the plan's Pod-specific fields must say (interpreter, prefix, torch build, device,
volume mount, container disk, commit); it changes nothing. ``prepare`` restores the sealed prep snapshot and
the pinned base model onto the local container disk (online steps run in child processes with the Hub
online). ``rehearsal`` runs a small, labelled rehearsal against the *real* volume and runtime: 03b with
reduced sizes, sign-off, 04 with two batches per arm interrupted after three units, deletion of the local
working directory (a fresh container disk), resume from the volume, and a clean-root restore of the final
manifest. It never stops the Pod. ``configs`` writes the production qualification/production configs.
``chain`` runs 03b then 04 unattended under the plan's pre-registered automatic sign-off rule and ends with
the Pod stopped once the evidence is safe (see ``chain``).
Secrets come only from the environment (HF_TOKEN as a Pod secret) and are never written or printed.
"""
import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
PLAN = REPO_DIR / '.scratch/joint-neuron-adapter-audit/frozen_plan_v1.yaml'
PREP_RECEIPT = REPO_DIR / '.scratch/joint-neuron-adapter-audit/prep-v2.hf-upload.json'
SEQ_RUN = 'runs/joint-neuron-adapter-audit/sequences/sequences-v2'
INV_RUN = 'runs/joint-neuron-adapter-audit/inventory/inventory-v2'
PACKAGES = ['torch', 'transformers', 'peft', 'huggingface-hub', 'tokenizers', 'safetensors', 'accelerate']


def _env(online=False):
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONPATH=str(REPO_DIR))
    env.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    env.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
    if online:
        env.update(HF_HUB_OFFLINE='0', TRANSFORMERS_OFFLINE='0')
    return env


def _run(args, *, online=False, check=True, log=None):
    """Run one pipeline step as a child process (its own offline/online setting); stream and tee its output."""
    command = [sys.executable, '-m', *args]
    print('$', ' '.join(command), flush=True)
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=REPO_DIR, env=_env(online), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in process.stdout:
        print(line, end='', flush=True)
        lines.append(line)
        if log is not None:
            with open(log, 'a') as stream:
                stream.write(line)
    code = process.wait()
    record = dict(command=args[:2], returncode=code, seconds=round(time.monotonic() - started, 1),
                  tail=''.join(lines[-30:])[-4000:])
    if check and code:
        raise RuntimeError(f'step failed ({code}): {" ".join(args[:2])}')
    return record


# --- facts ----------------------------------------------------------------------------------------

def facts(volume_dir, work):
    import importlib.metadata as md
    from pilot_eval import neuron_audit_preflight as pf
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    out = dict(host=socket.gethostname(), pod_id=os.environ.get('RUNPOD_POD_ID'), sys_executable=sys.executable,
               sys_prefix=sys.prefix, python=sys.version.split()[0], recorded_unix=time.time())
    out['packages'] = {}
    for name in PACKAGES:
        try:
            out['packages'][name] = md.version(name)
        except md.PackageNotFoundError:
            out['packages'][name] = None
    try:
        import torch
        out['torch_runtime'] = dict(version=torch.__version__, cuda=torch.version.cuda,
                                    cudnn=torch.backends.cudnn.version(), cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            out['device'] = dict(name=props.name, total_memory=props.total_memory)
            out['runtime_fingerprint'] = pf.runtime_fingerprint(dict(name=props.name, total_memory=props.total_memory))
    except Exception as exc:  # report, never hide
        out['torch_error'] = f'{type(exc).__name__}: {exc}'
    try:
        smi = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30)
        out['nvidia_smi'] = smi.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out['nvidia_smi'] = None
    out['volume'], out['volume_problems'] = pf.volume_identity(volume_dir, expected_id='sl-0', local_dir=work,
                                                               require_sentinel=False)
    out['volume']['identity_file_present'] = (Path(volume_dir) / pf.VOLUME_SENTINEL).is_file()
    out['local_disk'] = pf.local_disk(work)
    out['disk_problems'] = disk_problems(out['local_disk'])
    out['code_commit'], out['commit_problems'] = pf.check_code_commit('unrecorded')
    torch_version = (out.get('torch_runtime') or {}).get('version') or out['packages'].get('torch') or ''
    local = torch_version.partition('+')[2]
    out['suggested_plan_values'] = {
        'runtime.expected_device': (out.get('device') or {}).get('name'),
        'runtime.code_commit': out['code_commit'].get('head'),
        'runtime.allowed_local_builds': {'torch': [local]} if local else {},
        'preflight.expected_sys_executable': sys.executable,
        'preflight.expected_sys_prefix': sys.prefix,
    }
    out['notes'] = [
        'Copy suggested_plan_values into frozen_plan_v1.yaml only after checking them.',
        'Local working set must be <= 60% of local_disk.free_bytes (about 50 GB free needed for the full run).',
    ]
    path = work / 'pod-facts.json'
    path.write_text(json.dumps(out, indent=2, default=str) + '\n')
    return out


# Before ``prepare``: the full run's local working set (~27 GB: one arm's units, package and verification
# download plus the model cache) must stay <= 60% of free disk at 03b, after the model and prep are added.
MIN_FREE_BYTES_AT_FACTS = 60 * 10**9


def disk_problems(local, minimum=MIN_FREE_BYTES_AT_FACTS):
    """Flag a container disk too small for the full run before anything is installed into it."""
    if local['free_bytes'] >= minimum:
        return []
    return [f"container disk has {local['free_bytes'] / 1e9:.0f} GB free (total {local['total_bytes'] / 1e9:.0f} GB); "
            f"the full run needs >= {minimum / 1e9:.0f} GB free here: recreate the Pod with a >= 120 GB container disk"]


# --- inputs ----------------------------------------------------------------------------------------

def prepare(work):
    """Restore the sealed prep snapshot and the pinned base model to local disk; returns paths."""
    work = Path(work)
    prep = work / 'prep'
    if not (prep / 'runs').is_dir():
        _run(['pilot_eval.neuron_audit_export', 'restore', '--upload_receipt', str(PREP_RECEIPT),
              '--clean_root', str(prep)], online=True)
    _run(['pilot_eval.neuron_audit_sequences', 'verify', '--run_dir', str(prep / SEQ_RUN)])
    inventory = json.loads((prep / INV_RUN / 'inventory.json').read_text())
    base = inventory['base']
    request = work / 'model_request.json'
    request.write_text(json.dumps(dict(model=base['model'], revision=base['model_revision'],
                                       expected=base['expected_file_sha256'], hf_home=str(work / 'hf'))))
    env = _env(online=True)
    env['HF_HOME'] = str(work / 'hf')
    result = subprocess.run([sys.executable, '-m', 'pilot_eval.neuron_audit_pod', '_download_model',
                             '--request', str(request)], cwd=REPO_DIR, env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('model download failed: ' + result.stderr[-2000:])
    model = json.loads(result.stdout.strip().splitlines()[-1])
    if model['mismatched']:
        raise RuntimeError(f"base model files differ from the pinned hashes: {model['mismatched']}")
    adapters = {arm: str(prep / rec['checkpoints'][str(inventory['audit_step'])]['path'])
                for arm, rec in inventory['arms'].items() if rec.get('checkpoints', {}).get(str(inventory['audit_step']))}
    paths = dict(prep=str(prep), model_dir=model['path'], model_bytes=model['bytes'],
                 tokenizer_dir=str(prep / 'inputs/models' / base['tokenizer_revision']), adapter_dirs=adapters,
                 hf_home=str(work / 'hf'))
    (work / 'inputs.json').write_text(json.dumps(paths, indent=2) + '\n')
    return paths


def _download_model(request_path):
    request = json.loads(Path(request_path).read_text())
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(request['model'], revision=request['revision']))
    bad = []
    for name, digest in request['expected'].items():
        digest_now = hashlib.sha256()
        with (path / name).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1 << 24), b''):
                digest_now.update(chunk)
        if digest_now.hexdigest() != digest:
            bad.append(name)
    size = sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
    print(json.dumps(dict(path=str(path), mismatched=bad, bytes=size)))


def write_configs(work, volume_dir, *, inputs, tag, rehearsal, stop_command=None, max_batches=None,
                  interrupt_after=None):
    work = Path(work)
    prefix = 'rehearsal-' if rehearsal else ''
    for d in ['out', 'tmp', 'local']:
        (work / d).mkdir(parents=True, exist_ok=True)
    qualify = dict(artifact_root=inputs['prep'], output_root=str(work / 'out'), run_id=f'{prefix}q-{tag}',
                   volume_dir=str(volume_dir), local_temp=str(work / 'tmp'), model_dir=inputs['model_dir'],
                   tokenizer_dir=inputs['tokenizer_dir'], reference_arm='gold-sft', adapter_dirs=inputs['adapter_dirs'],
                   model_cache_bytes=inputs['model_bytes'])
    production = dict(qualify, run_id=f'{prefix}p-{tag}', local_root=str(work / 'local'), volume_root=str(volume_dir),
                      qualification_root=str(work / 'out'), qualification_run_id=qualify['run_id'],
                      label=f'pod-{tag}')
    if rehearsal:
        production.update(max_batches_per_arm=max_batches or 2, rehearsal_cap_hours=1)
        if interrupt_after:
            production['rehearsal_interrupt_after_units'] = interrupt_after
    elif stop_command:
        production['stop_command'] = stop_command
    q_path, p_path = work / f'{prefix}qualify-{tag}.json', work / f'{prefix}production-{tag}.json'
    q_path.write_text(json.dumps(qualify, indent=2) + '\n')
    p_path.write_text(json.dumps(production, indent=2) + '\n')
    return q_path, p_path, qualify, production


# --- the small Pod rehearsal -------------------------------------------------------------------------

def rehearsal(work, volume_dir, *, tag, max_batches=2, interrupt_after=3):
    from pilot_eval import neuron_audit_plan as plan_mod
    work = Path(work)
    report = dict(tag=tag, started_unix=time.time(), steps={}, ok=False)
    report_path = work / f'pod-rehearsal-{tag}.json'
    log = work / f'pod-rehearsal-{tag}.log'

    def save():
        report_path.write_text(json.dumps(report, indent=2, default=str) + '\n')

    try:
        report['facts'] = facts(volume_dir, work)
        if report['facts']['volume_problems'] or not report['facts']['volume']['identity_file_present']:
            raise RuntimeError(f"volume not established: {report['facts']['volume_problems']} "
                               '(run init-volume once after confirming the sl-0 mount)')
        inputs = json.loads((work / 'inputs.json').read_text()) if (work / 'inputs.json').is_file() else prepare(work)
        plan = work / f'rehearsal-plan-{tag}.yaml'
        plan_mod.make_rehearsal_plan(PLAN, plan, small=True, note='Pod rehearsal: real volume/runtime, small sizes')
        q_path, p_path, qualify, production = write_configs(work, volume_dir, inputs=inputs, tag=tag, rehearsal=True,
                                                            max_batches=max_batches, interrupt_after=interrupt_after)
        report['steps']['03b'] = _run(['pilot_eval.neuron_audit_qualify', 'run', '--rehearsal', '--plan', str(plan),
                                       '--config', str(q_path)], log=log)
        qual_dir = work / 'out/runs/joint-neuron-adapter-audit/qualification' / qualify['run_id']
        summary = json.loads((qual_dir / 'qualification.json').read_text())
        report['qualification'] = {k: summary.get(k) for k in ['status', 'arms_qualified', 'arms_failed',
                                                               'selected_budget', 'projection', 'review_flags_raised',
                                                               'logits_max_abs', 'fp64_floor_methods',
                                                               'runtime_fingerprint', 'phase_1b']}
        if not (qual_dir / 'review/signoff.json').exists():
            _run(['pilot_eval.neuron_audit_qualify', 'signoff', '--run_dir', str(qual_dir), '--reviewer',
                  'pod-rehearsal', '--decision', 'accept', '--notes', 'rehearsal only'], log=log)
        save()
        first = _run(['pilot_eval.neuron_audit_production', '--rehearsal', '--plan', str(plan), '--config',
                      str(p_path)], check=False, log=log)
        report['steps']['04-interrupted'] = first
        if first['returncode'] == 0 or 'RehearsalInterruption' not in first['tail']:
            raise RuntimeError('the rehearsal interruption did not happen as planned')
        shutil.rmtree(work / 'local')  # a fresh container disk: only the volume and HF survive
        (work / 'local').mkdir()
        resumed = dict(production)
        resumed.pop('rehearsal_interrupt_after_units', None)
        p_path.write_text(json.dumps(resumed, indent=2) + '\n')
        report['steps']['04-resumed'] = _run(['pilot_eval.neuron_audit_production', '--rehearsal', '--plan', str(plan),
                                              '--config', str(p_path)], log=log)
        progress = Path(volume_dir) / 'runs/joint-neuron-adapter-audit/production' / production['run_id'] / \
            'meta/progress.json'
        state = json.loads(progress.read_text())
        restored = [e for e in state['events'] if e['kind'] == 'units-restored-from-volume']
        report['production'] = dict(final_manifest=state.get('final_manifest'), restored_events=restored,
                                    safe_to_stop_pod=state.get('safe_to_stop_pod'),
                                    arm_packages={a: dict(units=len(r['units']), bytes=r['bytes'])
                                                  for a, r in state['arm_packages'].items()})
        if (state.get('final_manifest') or {}).get('completeness') != 'complete' or not restored:
            raise RuntimeError('resume did not restore mirrored units and complete the rehearsal export')
        clean = work / f'clean-restore-{tag}'
        if clean.exists():
            shutil.rmtree(clean)
        report['steps']['restore-final'] = _run(['pilot_eval.neuron_audit_export', 'restore-final', '--progress',
                                                 str(progress), '--clean_root', str(clean), '--temp_dir',
                                                 str(work / 'tmp'), '--require_complete'], online=True, log=log)
        receipt = json.loads((work / f'clean-restore-{tag}.restore-final-receipt.json').read_text())
        report['restore'] = {k: receipt[k] for k in ['completeness', 'verified_units', 'final_sha256', 'revision']}
        report['ok'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['finished_unix'] = time.time()
        save()
        print('PASTE THIS BACK:', json.dumps(dict(ok=report['ok'], report=str(report_path),
                                                  error=report.get('error'))), flush=True)
    return report


# --- unattended 03b -> 04 chain ----------------------------------------------------------------------

AUTO_SIGNOFF_RULE = 'accept-if-qualified-and-no-flags'
AUTO_REVIEWER = 'automatic-rule (plan review_flags.automatic_signoff, pre-registered by the user 2026-11-12)'


def _copy_verified(source, destination):
    """Copy every file of ``source`` to ``destination`` and re-read each copy by full hash; returns problems."""
    source, destination = Path(source), Path(destination)
    problems = []
    for path in sorted(p for p in source.rglob('*') if p.is_file()):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if _sha_file(path) != _sha_file(target):
            problems.append(f'{path.relative_to(source)} differs on the volume')
    return problems


def _sha_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def stop_command(pod_id):
    """Stop this Pod with either runpodctl syntax: the Pod image's runpodctl accepts ``stop pod <id>`` (found
    2026-11-12, when ``pod stop <id>`` exited 1 after the v1 run); newer releases document ``pod stop <id>``."""
    if not pod_id or not str(pod_id).replace('-', '').isalnum():
        raise ValueError(f'unexpected Pod id {pod_id!r}')
    return ['sh', '-c', f'runpodctl stop pod {pod_id} || runpodctl pod stop {pod_id}']


def _upload_stopped(directory, plan, tag, work):
    from pilot_eval import neuron_audit_export as ex
    return ex.upload_directory_package(directory, repo=plan['preflight']['export_repo'], label=f'pod-{tag}',
                                       kind='qualification-stopped', client=ex.OnlineHubProcess(),
                                       temp_dir=str(Path(work) / 'tmp'))


def chain(work, volume_dir, *, tag, plan_path, runner=_run, stop_runner=subprocess.run, uploader=None, pod_id=None):
    """Run 03b then 04 unattended under the plan's pre-registered rules and always end with the Pod stopped
    once the evidence is safe.

    03b qualified with no review flags -> automatic sign-off (labelled as the rule, never as the user) -> 04,
    which stops the Pod itself after the verified final backup (or a verified partial backup). Any other 03b
    outcome (failure, ceiling, partial, any flag) -> the 03b run directory is copied to the volume and
    re-verified by hash, uploaded to HF if possible, and the Pod stops when the volume copy verified."""
    from pilot_eval import neuron_audit_plan as plan_mod
    from pilot_eval import neuron_audit_qualify as q
    work = Path(work)
    plan = plan_mod.load(plan_path, '04')  # approved, and the 04 cap is fixed before anything runs
    if plan['review_flags'].get('automatic_signoff') != AUTO_SIGNOFF_RULE:
        raise SystemExit(f'plan review_flags.automatic_signoff must be {AUTO_SIGNOFF_RULE!r} for an unattended chain')
    pod_id = pod_id or os.environ.get('RUNPOD_POD_ID')
    if not pod_id:
        raise SystemExit('RUNPOD_POD_ID is not set; the chain must be able to stop this Pod')
    stop = stop_command(pod_id)
    inputs = json.loads((work / 'inputs.json').read_text())
    q_path, p_path, qualify, production = write_configs(work, volume_dir, inputs=inputs, tag=tag, rehearsal=False,
                                                        stop_command=stop)
    log = work / f'chain-{tag}.log'
    report_path = work / f'chain-{tag}.json'
    report = dict(tag=tag, started_unix=time.time(), rule=AUTO_SIGNOFF_RULE)

    def save():
        report_path.write_text(json.dumps(report, indent=2, default=str) + '\n')

    report['03b'] = runner(['pilot_eval.neuron_audit_qualify', 'run', '--plan', str(plan_path), '--config',
                            str(q_path)], check=False, log=log)
    qual_dir = work / 'out' / q.RUN_SUBDIR / qualify['run_id']
    summary_path = qual_dir / 'qualification.json'
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    report['qualification'] = {k: summary.get(k) for k in ['status', 'arms_failed', 'review_flags_raised',
                                                           'logits_max_abs', 'projection', 'selected_budget']}
    reasons = []
    if report['03b']['returncode'] != 0:
        reasons.append(f"03b exited {report['03b']['returncode']}")
    if summary.get('status') != 'qualified':
        reasons.append(f"03b status is {summary.get('status')}")
    if summary.get('review_flags_raised'):
        reasons.append(f"review flags raised: {summary['review_flags_raised']}")
    if reasons:
        report['decision'] = dict(continue_to_04=False, reasons=reasons)
        if qual_dir.is_dir():
            backup = Path(volume_dir) / 'runs/joint-neuron-adapter-audit/qualification-stopped' / qualify['run_id']
            report['volume_copy_problems'] = _copy_verified(qual_dir, backup)
            report['volume_copy'] = str(backup)
            try:
                upload = uploader or (lambda d: _upload_stopped(d, plan, tag, work))
                report['hf_package'] = upload(qual_dir)
            except Exception as exc:  # the verified volume copy is the backup; HF is a second copy
                report['hf_error'] = f'{type(exc).__name__}: {exc}'
            safe = not report['volume_copy_problems']
        else:
            report['volume_copy'] = None
            safe = True  # 03b never created its run directory: nothing exists only on this disk
        report['finished_unix'] = time.time()
        if safe:
            report['pod_stop'] = 'requested after 03b stopped the chain'
            save()
            print('CHAIN STOPPED BEFORE 04:', json.dumps(dict(reasons=reasons, report=str(report_path))), flush=True)
            stop_runner(stop, capture_output=True, text=True)
        else:
            report['pod_stop'] = 'left running: the volume copy did not verify'
            save()
            print('CHAIN STOPPED BEFORE 04 (POD LEFT RUNNING):', json.dumps(dict(reasons=reasons)), flush=True)
        return report
    report['signoff'] = q.signoff(qual_dir, reviewer=AUTO_REVIEWER, decision='accept',
                                  notes='03b qualified with no review flags raised; recorded by the unattended chain')
    report['decision'] = dict(continue_to_04=True)
    save()
    report['04'] = runner(['pilot_eval.neuron_audit_production', '--plan', str(plan_path), '--config', str(p_path)],
                          check=False, log=log)  # 04 stops the Pod itself only after a verified (partial) backup
    report['finished_unix'] = time.time()
    save()
    print('CHAIN FINISHED:', json.dumps(dict(returncode_04=report['04']['returncode'], report=str(report_path))),
          flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['facts', 'prepare', 'rehearsal', 'configs', 'chain']:
        p = sub.add_parser(name)
        p.add_argument('--work', required=True, help='a directory on the container (local) disk')
        if name != 'prepare':
            p.add_argument('--volume_dir', required=True, help='audit directory on the mounted sl-0 volume')
        if name in ('rehearsal', 'configs', 'chain'):
            p.add_argument('--tag', required=True)
        if name == 'chain':
            p.add_argument('--plan', required=True, help='the approved plan (outside the checkout)')
        if name == 'configs':
            p.add_argument('--stop_pod', action='store_true', help='04 stops this Pod after the verified backup')
    d = sub.add_parser('_download_model')
    d.add_argument('--request', required=True)
    args = parser.parse_args(argv)
    if args.command == '_download_model':
        _download_model(args.request)
        return 0
    if args.command == 'facts':
        print(json.dumps(facts(args.volume_dir, args.work), indent=2, default=str))
    elif args.command == 'prepare':
        print(json.dumps(prepare(args.work), indent=2))
    elif args.command == 'rehearsal':
        rehearsal(args.work, args.volume_dir, tag=args.tag)
    elif args.command == 'chain':
        chain(args.work, args.volume_dir, tag=args.tag, plan_path=args.plan)
    else:
        inputs = json.loads((Path(args.work) / 'inputs.json').read_text())
        pod = os.environ.get('RUNPOD_POD_ID')
        if args.stop_pod and not pod:
            raise SystemExit('RUNPOD_POD_ID is not set; cannot build the stop command')
        stop = stop_command(pod) if args.stop_pod else None
        q_path, p_path, _, _ = write_configs(args.work, args.volume_dir, inputs=inputs, tag=args.tag, rehearsal=False,
                                             stop_command=stop)
        print(json.dumps(dict(qualify_config=str(q_path), production_config=str(p_path), stop_command=stop), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
