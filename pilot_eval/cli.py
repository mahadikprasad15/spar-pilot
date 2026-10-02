"""Command-line interface; optional GPU dependencies load only inside commands."""

import argparse
import json
import sys
from pathlib import Path

from pilot_eval.workflow import execute_config, prepare_plan
from pilot_eval.recovery import fork_plan, revise_logits
from pilot_eval.reporting import build_report
from pilot_eval.rescoring import rescore_gsm8k
from pilot_eval.sft import prepare_sft


def main(argv=None, *, dependencies=None):
    parser = argparse.ArgumentParser(description="Pilot 1 frozen benchmark evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="pin revisions and freeze five baseline cells")
    prepare.add_argument("--plan", required=True, help="unique immutable plan name")
    prepare.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    prepare.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    prepare.add_argument("--batch-size", type=int, default=4)
    prepare.add_argument("--adapter", help="local PEFT directory or Hub repository")
    run = commands.add_parser("run", help="execute/resume one frozen cell")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--audit-items", type=int, help="separate fixed-prefix audit run, e.g. 5")
    fork = commands.add_parser("fork-plan", help="copy frozen inputs into a new batch-size plan")
    fork.add_argument("--source", required=True)
    fork.add_argument("--plan", required=True)
    fork.add_argument("--batch-size", type=int, required=True)
    revise = commands.add_parser("revise-logits", help="version tie policy and reuse verified raw source logits")
    revise.add_argument("--source", required=True)
    revise.add_argument("--plan", required=True)
    report = commands.add_parser("report", help="verify and combine selected completed source runs")
    report.add_argument("--selection", type=Path, required=True, help="JSON mapping cell names to plan names")
    report.add_argument("--name", required=True)
    rescore = commands.add_parser("rescore-gsm8k", help="CPU-only flexible v2 rescoring of saved responses")
    for field in ("responses", "config", "summary"):
        rescore.add_argument("--" + field, type=Path, required=True)
    rescore.add_argument("--name", required=True)
    sft_prepare = commands.add_parser('sft-prepare', help='freeze Pilot 2 training inputs and analysis plan')
    sft_prepare.add_argument('--source-config', type=Path, required=True)
    sft_prepare.add_argument('--name', required=True)
    sft_prepare.add_argument('--hardware', choices=['T4', 'L4'], default='T4')
    sft_prepare.add_argument('--evaluation-batch-size', type=int, choices=[1, 2], default=1)
    sft_train = commands.add_parser('sft-train', help='preflight or resume Pilot 2 training')
    sft_train.add_argument('--config', type=Path, required=True)
    sft_train.add_argument('--preflight-only', action='store_true')
    sft_eval = commands.add_parser('sft-evaluate', help='evaluate a matched baseline or adapter checkpoint')
    sft_eval.add_argument('--config', type=Path, required=True)
    sft_eval.add_argument('--checkpoint', choices=['baseline', '0', '8', '16', '32', '64'], required=True)
    sft_compare = commands.add_parser('sft-compare', help='CPU-only paired checkpoint trajectory report')
    sft_compare.add_argument('--config', type=Path, required=True)
    for command in (prepare, run, fork, revise, report, rescore, sft_prepare, sft_train, sft_eval, sft_compare):
        command.add_argument("--output-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)
    try:
        if args.command == 'sft-compare':
            from pilot_eval.sft_reporting import compare_sft
            result = compare_sft(args.config, args.output_root)
            print(json.dumps({'mode': result['mode'], 'trajectory': result['trajectory']}, indent=2))
        elif args.command == 'sft-evaluate':
            from pilot_eval.sft_evaluation import evaluate_sft
            print(json.dumps(evaluate_sft(args.config, args.output_root, args.checkpoint,
                dependencies=dependencies), indent=2))
        elif args.command == 'sft-train':
            from pilot_eval.training import run_sft
            print(json.dumps(run_sft(args.config, args.output_root,
                preflight_only=args.preflight_only, dependencies=dependencies), indent=2))
        elif args.command == 'sft-prepare':
            print(prepare_sft(args.source_config, args.output_root, args.name, dependencies=dependencies,
                hardware=args.hardware, evaluation_batch_size=args.evaluation_batch_size))
        elif args.command == "prepare":
            paths = prepare_plan(args.output_root, args.plan, model=args.model, dtype=args.dtype,
                                 batch_size=args.batch_size, adapter=args.adapter, dependencies=dependencies)
            for path in paths:
                print(path)
        elif args.command == "fork-plan":
            for path in fork_plan(args.output_root, args.source, args.plan, args.batch_size):
                print(path)
        elif args.command == "revise-logits":
            for path in revise_logits(args.output_root, args.source, args.plan):
                print(path)
        elif args.command == "report":
            summary = build_report(args.output_root, args.name, json.loads(args.selection.read_text()))
            print(json.dumps(summary, indent=2, sort_keys=True))
        elif args.command == "rescore-gsm8k":
            summary = rescore_gsm8k(args.responses, args.config, args.summary, args.output_root, args.name)
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            summary = execute_config(args.config, args.output_root, audit_items=args.audit_items,
                                     dependencies=dependencies)
            print(json.dumps(summary, indent=2, sort_keys=True))
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"Evaluation stopped: {exc}", file=sys.stderr)
        return 1
    return 0
