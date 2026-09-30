"""Command-line interface; optional GPU dependencies load only inside commands."""

import argparse
import json
import sys
from pathlib import Path

from pilot_eval.workflow import execute_config, prepare_plan
from pilot_eval.recovery import fork_plan


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
    for command in (prepare, run, fork):
        command.add_argument("--output-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            paths = prepare_plan(args.output_root, args.plan, model=args.model, dtype=args.dtype,
                                 batch_size=args.batch_size, adapter=args.adapter, dependencies=dependencies)
            for path in paths:
                print(path)
        elif args.command == "fork-plan":
            for path in fork_plan(args.output_root, args.source, args.plan, args.batch_size):
                print(path)
        else:
            summary = execute_config(args.config, args.output_root, audit_items=args.audit_items,
                                     dependencies=dependencies)
            print(json.dumps(summary, indent=2, sort_keys=True))
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"Evaluation stopped: {exc}", file=sys.stderr)
        return 1
    return 0
