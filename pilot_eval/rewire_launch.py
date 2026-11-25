"""Sparse rewiring Pod launcher: command line for ``scripts/rewire_launch.sh`` (tickets 14-15).

The same shape as ``c_attribution_chain``'s command line, reused rather than copied: ``precheck`` (refuse a record
for another Pod, a finished run ID, or a live/foreign writer), ``paths`` (for the launch script: the chain directory,
the stop commands and timeouts), ``watchdog-copy`` (the watchdog's sealed copy before its stop) and ``run``. The run
is the existing ``c_attribution_chain.Chain`` driven by ``rewire_chain.build_rewire_chain``:

    preflight -> qualification (automatic sign-off) -> references -> train:r1 -> eval:r1 -> train:r0 -> eval:r0
    -> verified volume copy -> HF export with readback -> summaries -> Pod stop

``run`` builds every input from the sealed launch record and the restored input bundle (``rewire_bundle``): the
512 gold training rows, gold's saved order and each arm's frozen plan (``sparse_rewiring_plan.plan_from_files``),
and ticket 04's validation set. A rehearsal never passes cohort150. A production record (ticket 16) also passes gold's
frozen cohort150 (``rewire_bundle.check_cohort150``), which the eval stage scores only at the final step (D12).

The Pod is never stopped unless ``run`` gets ``--stop-pod`` (researcher, 2026-11-25): without it, teardown still
copies, exports and verifies, then leaves the Pod running and prints the stop command.

    python -m pilot_eval.rewire_launch seal --draft <yaml> --out $WORK/launch_record.json --artifact-root ... \\
        --bundle ... --model-dir ... --work-dir ...
    python -m pilot_eval.rewire_launch {precheck,paths,watchdog-copy,run} --record ... --local ... --volume ...
    python -m pilot_eval.rewire_launch run ... --stop-pod      # stop the Pod after a complete verified backup
"""
import argparse
import json
import os
import sys
from pathlib import Path

from pilot_eval import attribution_chain as pc
from pilot_eval import c_attribution_bundle as cb
from pilot_eval import c_attribution_chain as cc
from pilot_eval import rewire_bundle as rb
from pilot_eval import sparse_rewiring_plan as sp
from pilot_eval.attribution_chain import LaunchRefused, ResumeRefused

ARM_ORDER = ('sparse-gold-sft-r1', 'sparse-gold-sft-r0')   # rank 1 then rank 0 (ticket 17)


def chain_record(record):
    from pilot_eval.rewire_chain import chain_run_id
    return dict(record, run_id=chain_run_id(record))


def load_inputs(record):
    """Everything the chain trains and evaluates on, from the sealed record and the restored bundle."""
    inputs = record['inputs']
    root = Path(inputs['artifact_root'])
    bundle = json.loads(Path(inputs['bundle']).read_text())
    cb.verify(root, bundle)  # every bundled file against the bundle seal before anything reads it
    items_path = rb.path_of(bundle, root, 'gold:training-items')
    order_path = rb.path_of(bundle, root, 'gold:order')
    if str(items_path.relative_to(root)) != json.loads(rb.path_of(bundle, root, 'gold:sft-config').read_text())[
            'training_items_path'] or str(order_path.relative_to(root)) != inputs['gold_order']:
        raise LaunchRefused("the bundled gold files are not the ones the launch record and gold's plan name")
    training_rows = json.loads(items_path.read_text())
    plans = {arm: sp.plan_from_files(record, arm, items_path, order_path) for arm in ARM_ORDER}
    val_items = rb.check_validation(bundle, root, questions=record['evaluation']['validation']['questions'])
    val_questions = {i['id']: {'prompt_ids': list(i['input_ids'][:int(i['prompt_tokens'])]), 'gold': i['gold']}
                     for i in val_items}
    cohort150 = rb.check_cohort150(bundle, root) if record['mode'] == 'production' else None
    return dict(training_rows=training_rows, arm_plans=plans, val_items=val_items, val_questions=val_questions,
                val_question_ids=[i['id'] for i in val_items], bundle=bundle, cohort150_items=cohort150)


def cohort150_questions(items, encode):
    """Gold's rendered cohort150 prompts, tokenized as C attribution does (no special tokens), in cohort order."""
    return ({i['id']: {'prompt_ids': [int(t) for t in encode(i['prompt'])], 'prompt': i['prompt'], 'gold': i['gold']}
             for i in items}, [i['id'] for i in items])


def base_hash_once(backend):
    """The frozen base-weight hash of the verified pinned model, computed once on CPU (``expected_base_sha256``)."""
    cache = {}

    def compute():
        if 'sha' not in cache:
            import torch
            from transformers import AutoModelForCausalLM
            from pilot_eval import rewire_module as rm
            from pilot_eval import sft_backend as sb
            cb.check_base_model(backend.model_dir, backend.bundle, pinned=backend.pinned_base_model)
            model = AutoModelForCausalLM.from_pretrained(backend.model_dir, local_files_only=True,
                                                         torch_dtype=torch.float32, attn_implementation='eager')
            cache['sha'] = sb.frozen_weight_hash(model, trained=rm.is_rewire_parameter)
            model = None
        return cache['sha']
    return compute


def build(record, local, volume, *, backend=None, probes=None, device='cuda', exporter=None, runner=None,
          volume_identity=None, current_pod=None, references_fn=None, gradient_checkpointing=True, stop_pod=False):
    """The chain for a sealed record (the CPU tests call this with the tiny model and stubs)."""
    from pilot_eval import rewire_chain as rc
    if record['mode'] not in ('rehearsal', 'production'):
        raise LaunchRefused(f"unknown launch mode {record['mode']!r}")
    data = load_inputs(record)
    backend = backend or rc.RewireBackend(record, volume, device=device, probes=probes)
    c150, c150_ids = (None, None)
    if data['cohort150_items'] is not None:
        c150, c150_ids = cohort150_questions(data['cohort150_items'], backend.encode)
    return rc.build_rewire_chain(
        record=record, backend=backend, local_root=local, volume=volume,
        training_rows=data['training_rows'], arm_plans=data['arm_plans'], val_items=data['val_items'],
        val_questions=data['val_questions'], val_question_ids=data['val_question_ids'],
        cohort150_questions=c150, cohort150_question_ids=c150_ids, arm_order=list(ARM_ORDER),
        expected_base_sha256=base_hash_once(backend), execution_kind='real-cuda',
        gradient_checkpointing=gradient_checkpointing, references_fn=references_fn,
        exporter=exporter, runner=runner, volume_identity=volume_identity, current_pod=current_pod,
        stop_pod=stop_pod)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    seal = sub.add_parser('seal', help='approved YAML draft -> sealed, typed JSON record (code commit and Pod ID '
                                       'filled from this checkout and RUNPOD_POD_ID)')
    seal.add_argument('--draft', required=True)
    seal.add_argument('--out', required=True)
    for key in ('artifact-root', 'bundle', 'model-dir', 'work-dir'):
        seal.add_argument(f'--{key}', required=True)
    for name in ('run', 'precheck', 'watchdog-copy', 'paths'):
        p = sub.add_parser(name)
        p.add_argument('--local', required=True)
        p.add_argument('--volume', required=True)
        p.add_argument('--record', required=True)
        if name == 'run':
            p.add_argument('--stop-pod', action='store_true',
                           help='stop the Pod after a complete verified backup (default: leave it running)')
    args = parser.parse_args(argv)
    if args.command == 'seal':
        fill = {f'inputs.{k}': getattr(args, k) for k in ('artifact_root', 'bundle', 'model_dir', 'work_dir')}
        print(sp.finalize_draft(args.draft, args.out, code_commit=pc._commit(),
                                pod_id=os.environ.get('RUNPOD_POD_ID'), fill=fill))
        return 0
    record = sp.load_launch_record(args.record, code_commit=pc._commit())
    crec = chain_record(record)
    root = cc.run_root(args.volume, crec['run_id'])
    if args.command == 'paths':
        print(json.dumps(dict(run_root=str(root), chain_dir=str(root / cc.CHAIN_DIR),
                              stop=[' '.join(c) for c in cc.stop_commands(crec)],
                              copy_timeout=crec['timeouts_seconds']['copy'],
                              stop_timeout=crec['timeouts_seconds']['stop_attempt'])))
        return 0
    if args.command == 'watchdog-copy':
        result = cc.copy_sealed(args.local, root)
        print(json.dumps(result, indent=2))
        return 0 if result['verified'] and not result['problems'] else 1
    if args.command == 'precheck':
        try:
            decision = cc.precheck(crec, args.volume)
        except ResumeRefused as exc:
            print(f'precheck refused: {exc}', file=sys.stderr)
            return 3
        print(json.dumps(dict(precheck='ok', run_id=crec['run_id'], decision=decision['decision'])))
        return 0
    identity = pc.observed_volume_identity(args.volume, args.local, expected=record['volume_identity'])
    chain = build(record, args.local, args.volume, volume_identity=identity, stop_pod=args.stop_pod)
    summary = chain.run()
    print(json.dumps(dict(state=summary['state'], backups=summary['backups']['statement'], stop=summary['stop']),
                     indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
