"""Rank-0 read-LR probe (ticket 15 follow-up, researcher decision 2026-11-25).

The rehearsal (``rehearsal-pod-1``) showed rank 0 at read LR 3e-4 (D1) taking validation accuracy from 103/128 to
36/128 at step 12 and 12/128 at step 16, and its 4-step qualification pass to CE 0.88 (base 0.57) with every answer
at the cap. This probe trains **rank 0 only** under the rehearsal's frozen plan, changing nothing but the read LR,
and reports validation CE at every checkpoint and greedy accuracy on the 128 validation questions at the chosen
steps. It is a measurement, not a result: no chain, no sign-off, and cohort150 is never touched.

Each LR trains in ``--work`` (container disk; rank-0 checkpoints are ~2.9 GB each and are deleted once evaluated);
logs, greedy responses and ``lr-probe.json`` are written under ``--work``/results and mirrored after each LR to
``--out`` (put it on the volume) with plain byte copies. With ``--stop-pod`` the Pod is
stopped when the probe ends, whether it succeeded or not.

    python -m pilot_eval.rewire_lr_probe --record $WORK/probe_record.json --volume $VOL --work $WORK/lr-probe \\
        --out $VOL/c-attribution/sparse-rewiring-lr-probe-1 --lrs 3e-5 1e-5 --stop-pod
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ARM = 'sparse-gold-sft-r0'


def probe_plan(plan: dict[str, Any], lr: float, *, steps: int | None = None,
               checkpoints: list[int] | None = None) -> dict[str, Any]:
    """The arm's frozen plan with only the read LR changed, under its own run ID and plan hash. With ``steps``, the
    plan is cut to its first ``steps`` steps (each step's IDs, mask groups, LR multiplier and mask phase unchanged,
    so a cut production plan follows production's own schedule) and checkpointed at ``checkpoints``."""
    from pilot_eval import sparse_rewiring_plan as sp
    cut = copy.deepcopy(plan)
    cut.pop('plan_sha256', None)
    if steps is not None:
        steps = int(steps)
        if not 0 < steps <= len(plan['steps']):
            raise ValueError(f'cut of {steps} steps does not fit the plan ({len(plan["steps"])} steps)')
        cut['steps'] = cut['steps'][:steps]
        points = sorted({0, steps, *(int(c) for c in (checkpoints or []))})
        if points[-1] != steps:
            raise ValueError(f'checkpoints {points} go past the cut at step {steps}')
        cut['checkpoints'] = points
        cut['cut_of_steps'] = len(plan['steps'])
    cut['run_id'] = f"{plan['run_id']}-lr{lr:g}".replace('.', 'p').replace('+', '')
    cut['optimizer']['reads']['lr'] = float(lr)
    cut.pop('read_lr', None)
    cut['probe_of'] = plan['plan_sha256']
    cut['plan_sha256'] = sp.plan_sha256(cut)
    return cut


def run_probe(record, backend, data, *, lrs, work, out, greedy_steps, code_commit, log=print, cut_steps=None,
              checkpoints=None):
    import torch
    from pilot_eval import rewire_checkpoint as rckpt
    from pilot_eval import rewire_evaluation as reval
    from pilot_eval import rewire_training as rt
    from pilot_eval.rewire_launch import base_hash_once
    from pilot_eval.workflow import _hash

    torch.use_deterministic_algorithms(True)
    rt_sha = _hash(backend.runtime())
    expected_base = base_hash_once(backend)
    base_plan = data['arm_plans'][ARM]
    # Everything is written on the container disk first, then mirrored to ``out`` with plain byte copies: the
    # network volume refuses metadata calls (utime/chmod), so shutil.copy2/copytree fail there (Pod 2026-11-25).
    work, volume_out, out = Path(work), Path(out), Path(work) / 'results'
    out.mkdir(parents=True, exist_ok=True)
    results = dict(arm=ARM, rank=int(base_plan['rank']), baseline_lr=base_plan['optimizer']['reads']['lr'],
                   steps=cut_steps or len(base_plan['steps']), cut_of=len(base_plan['steps']) if cut_steps else None, code_commit=code_commit,
                   runtime_sha256=rt_sha, lrs={})
    summary_path = out / 'lr-probe.json'

    for lr in lrs:
        plan = probe_plan(base_plan, lr, steps=cut_steps, checkpoints=checkpoints)
        run_dir = work / plan['run_id']
        if run_dir.exists():
            shutil.rmtree(run_dir)
        t0 = time.monotonic()
        log(f'lr {lr:g}: training {len(plan["steps"])} steps ({plan["run_id"]})')
        rckpt.run_arm(
            run_dir, plan,
            model_factory=lambda: backend.load_rewired_model(
                rank=int(plan['rank']), seed=int(record['seeds']['read_init_base']),
                mask_init=float(record['hard_concrete']['log_alpha_init'])),
            train=lambda model, p, sink: rt.train_rewire(
                model, data['training_rows'], run_dir, plan=p, sink=sink,
                mask_lr=float(record['optimizer']['masks']['lr']),
                betas=tuple(float(b) for b in record['optimizer']['reads']['betas']),
                eps=float(record['optimizer']['reads']['eps']),
                max_grad_norm=float(record['optimizer']['reads']['clip_norm']),
                snapshot_interval=int(record['cadences']['gate_snapshot_every']),
                settle_start=int(record['cadences']['settle']['first']),
                settle_interval=int(record['cadences']['settle']['every']),
                gradient_checkpointing=True, model_context_limit=32768, pad_token_id=backend.pad_token_id,
                expected_base_sha256=expected_base(), execution_kind='real-cuda' if backend.device != 'cpu' else 'cpu',
                val_ce_fn=lambda m, _step: reval.evaluate_validation_ce(m, data['val_items'])),
            runtime_sha256=rt_sha, code_commit=code_commit, files=('logs/steps.jsonl',))
        backend.release()
        train_seconds = time.monotonic() - t0
        lr_out = out / plan['run_id']
        (lr_out / 'logs').mkdir(parents=True, exist_ok=True)
        for name in ('steps.jsonl', 'validation.jsonl', 'settle.jsonl'):
            if (run_dir / 'logs' / name).is_file():
                shutil.copyfile(run_dir / 'logs' / name, lr_out / 'logs' / name)
        row = dict(run_id=plan['run_id'], plan_sha256=plan['plan_sha256'], train_seconds=train_seconds, evals={},
                   gates=gate_counts(run_dir / 'logs' / 'steps.jsonl', plan['checkpoints'][1:]))
        for step in plan['checkpoints'][1:]:
            model = backend.load_base_model()
            try:
                seal = reval.load_checkpoint_for_eval(model, rckpt.checkpoint_dir(run_dir, step), plan,
                                                      runtime_sha256=rt_sha, code_commit=code_commit)
                ev = dict(val_ce=reval.evaluate_validation_ce(model, data['val_items'])['val_ce'])
                if step in greedy_steps:
                    g = reval.evaluate_greedy(
                        model, data['val_questions'], data['val_question_ids'], lr_out / 'eval' / str(step),
                        arm=plan['run_id'], step=step, cohort='gsm8k-train-val128', decode=backend.decode,
                        pad_token_id=backend.pad_token_id, eos_token_id=backend.eos_token_id,
                        cap=int(record['evaluation']['cap']), batch_size=int(record['evaluation']['batch_size']),
                        final_step=int(record['training']['steps']), checkpoint_meta=seal)
                    s = g['scores']
                    ev.update(accuracy=s['accuracy'], capped=s['capped'], tokens_mean=s['tokens_mean'])
            finally:
                model = None
                backend.release()
            row['evals'][str(step)] = ev
            log(f'lr {lr:g} step {step}: {json.dumps(ev)}')
        shutil.rmtree(run_dir / 'checkpoints', ignore_errors=True)  # ~2.9 GB each; the numbers are kept
        results['lrs'][f'{lr:g}'] = row
        summary_path.write_text(json.dumps(results, indent=2, default=str) + '\n')
        mirror(out, volume_out, log)
    return results


def gate_counts(steps_log, steps):
    """Open-gate counts from the step log at ``steps`` (expected active, deterministic-gate nonzero, binary
    survivors): whether the masks prune selectively or wholesale."""
    wanted, out = {int(s) for s in steps}, {}
    if Path(steps_log).is_file():
        for line in Path(steps_log).read_text().splitlines():
            r = json.loads(line) if line else {}
            if r.get('step') in wanted:
                out[str(r['step'])] = {k: r.get(k) for k in ('expected_active', 'nonzero_z_det', 'binary_survivors',
                                                            'ce_loss', 'l0_loss')}
    return out


def mirror(src, dst, log=print):
    """Copy every file under ``src`` to ``dst`` by bytes only (no copystat), reporting rather than raising."""
    src, dst = Path(src), Path(dst)
    failed = []
    for path in sorted(p for p in src.rglob('*') if p.is_file()):
        target = dst / path.relative_to(src)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        except OSError as exc:
            failed.append(f'{path.relative_to(src)}: {exc}')
    if failed:
        log(f'mirror to {dst}: {len(failed)} files not copied, first: {failed[0]}')
    return failed


def _stop_pod(log):
    pod = os.environ.get('RUNPOD_POD_ID')
    if not pod:
        log('RUNPOD_POD_ID is not set: the Pod was NOT stopped')
        return
    for cmd in (['runpodctl', 'stop', 'pod', pod], ['runpodctl', 'pod', 'stop', pod]):
        try:
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except Exception as exc:  # noqa: BLE001 - report and try the other syntax
            log(f'{" ".join(cmd)}: {exc}')
            continue
        log(f'{" ".join(cmd)} -> {done.returncode}: {(done.stdout + done.stderr).strip()[:300]}')
        if done.returncode == 0:
            return


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--record', required=True, help='sealed rehearsal launch record (rewire_launch seal)')
    parser.add_argument('--volume', required=True)
    parser.add_argument('--work', required=True, help='container-disk scratch for rank-0 checkpoints')
    parser.add_argument('--out', required=True, help='where results go (on the volume)')
    parser.add_argument('--lrs', nargs='+', type=float, default=[3e-5, 1e-5])
    parser.add_argument('--greedy-steps', nargs='+', type=int, default=None,
                        help='checkpoints that get the 128-question greedy pass (default: the last)')
    parser.add_argument('--cut-steps', type=int, default=None,
                        help="run only the plan's first N steps (its own schedule), e.g. 64 of production's 1,024")
    parser.add_argument('--checkpoints', nargs='+', type=int, default=None,
                        help='checkpoints for a cut run (0 and the last step are always added)')
    parser.add_argument('--stop-pod', action='store_true')
    args = parser.parse_args(argv)

    from pilot_eval import attribution_chain as pc
    from pilot_eval import rewire_chain as rc
    from pilot_eval import rewire_launch as rl
    from pilot_eval import sparse_rewiring_plan as sp
    from pilot_eval.attribution_chain import LaunchRefused

    def log(msg):
        print(f'[lr-probe {time.strftime("%H:%M:%S")}] {msg}', flush=True)

    status = 1
    try:
        commit = pc._commit()
        record = sp.load_launch_record(args.record, code_commit=commit)
        if record['mode'] != 'rehearsal':
            raise LaunchRefused('the LR probe runs from a rehearsal record only')
        data = rl.load_inputs(record)
        backend = rc.RewireBackend(record, args.volume, device='cuda')
        problems = list(backend.preflight())
        if not os.environ.get('CUBLAS_WORKSPACE_CONFIG'):
            problems.append('CUBLAS_WORKSPACE_CONFIG must be set (D8)')
        if problems:
            raise RuntimeError('preflight failed: ' + '; '.join(map(str, problems)))
        last = args.cut_steps or record['checkpoints'][-1]
        greedy = set(args.greedy_steps or [last])
        results = run_probe(record, backend, data, lrs=args.lrs, work=args.work, out=args.out,
                            greedy_steps=greedy, code_commit=commit, log=log, cut_steps=args.cut_steps,
                            checkpoints=args.checkpoints)
        for lr, row in results['lrs'].items():
            log(f'RESULT lr {lr}: ' + json.dumps(row['evals']))
        status = 0
    except Exception as exc:  # noqa: BLE001 - always reach the stop below
        log(f'FAILED: {type(exc).__name__}: {exc}')
        import traceback
        traceback.print_exc()
    finally:
        if args.stop_pod:
            _stop_pod(log)
    return status


if __name__ == '__main__':
    sys.exit(main())
