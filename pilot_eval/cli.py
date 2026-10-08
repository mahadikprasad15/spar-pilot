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
    sft_prepare.add_argument('--evaluation-batch-size', type=int, choices=[1, 2, 4, 8], default=1)
    write_report = commands.add_parser('grpo-writes-report', help='CPU comparison of verified SFT, GRPO and random mean writes')
    for field in ['grpo-report', 'control-report', 'sft-report']:
        write_report.add_argument('--' + field, type=Path, required=True)
    write_report.add_argument('--name', required=True)
    write_report.add_argument('--output-root', type=Path, default=Path('artifacts'))
    control = commands.add_parser('grpo-writes-control', help='save one norm-matched random control with exact fixed inputs')
    control.add_argument('--config', type=Path, required=True)
    control.add_argument('--name', required=True)
    control.add_argument('--output-root', type=Path, default=Path('artifacts'))
    writes = commands.add_parser('grpo-writes-prepare', help='bind verified GRPO checkpoints to exact Pilot 3 inputs')
    writes.add_argument('--config', type=Path, required=True)
    writes.add_argument('--measurement-config', type=Path, required=True)
    writes.add_argument('--name', required=True)
    writes.add_argument('--output-root', type=Path, default=Path('artifacts'))
    grpo_prepare = commands.add_parser('grpo-prepare', help='verify Pilot 4 sources and expose pending settings; CPU only')
    grpo_prepare.add_argument('--sft-config', type=Path, required=True)
    grpo_prepare.add_argument('--measurement-config', type=Path, required=True)
    grpo_prepare.add_argument('--name', required=True)
    grpo_preflight = commands.add_parser('grpo-preflight', help='collect disposable GPU evidence for an explicitly configured Pilot 4 plan')
    grpo_preflight.add_argument('--config', type=Path, required=True)
    grpo_preflight.add_argument('--baseline-config', type=Path, required=True)
    grpo_preflight.add_argument('--controls-name', required=True)
    grpo_preflight.add_argument('--settings', type=Path, required=True)
    grpo_preflight.add_argument('--name', required=True)
    grpo_preflight.add_argument('--output-root', type=Path, default=Path('artifacts'))
    grpo_freeze = commands.add_parser('grpo-freeze', help='freeze reviewed GPU evidence and dated preregistration')
    grpo_freeze.add_argument('--config', type=Path, required=True)
    grpo_freeze.add_argument('--review', type=Path, required=True)
    grpo_freeze.add_argument('--name', required=True)
    grpo_freeze.add_argument('--output-root', type=Path, default=Path('artifacts'))
    for name in ['grpo-evaluate','grpo-behaviour-report']:
        command=commands.add_parser(name,help='evaluate verified GRPO checkpoints or report paired held-out behaviour')
        command.add_argument('--config',type=Path,required=True)
        command.add_argument('--name',required=True)
        command.add_argument('--output-root',type=Path,default=Path('artifacts'))
    for name in ['grpo-train','grpo-verify-training']:
        command=commands.add_parser(name,help='run/recover or verify the sealed scientific GRPO arm')
        command.add_argument('--config',type=Path,required=True)
        command.add_argument('--output-root',type=Path,default=Path('artifacts'))
        if name=='grpo-train':
            command.add_argument('--name',required=True)
            command.add_argument('--length-change-definition',choices=['step-lag-8'],required=True)
    grpo_controls = commands.add_parser('grpo-controls', help='CPU-only independent algorithm checks and tiny real-model learning controls')
    grpo_controls.add_argument('--name', required=True)
    grpo_controls.add_argument('--output-root', type=Path, default=Path('artifacts'))
    grpo_baseline = commands.add_parser('grpo-baseline', help='sample/resume the explicit untuned 128x8 training baseline')
    grpo_baseline.add_argument('--config', type=Path, required=True)
    grpo_baseline.add_argument('--settings', type=Path, required=True)
    grpo_baseline.add_argument('--name', required=True)
    grpo_baseline.add_argument('--output-root', type=Path, default=Path('artifacts'))
    grpo_audit = commands.add_parser('grpo-audit', help='reverify and display the Pilot 4 source audit')
    grpo_audit.add_argument('--config', type=Path, required=True)
    grpo_ready = commands.add_parser('grpo-check-ready', help='reject unresolved Pilot 4 execution settings')
    grpo_ready.add_argument('--config', type=Path, required=True)
    for command in (grpo_prepare, grpo_audit, grpo_ready):
        command.add_argument('--output-root', type=Path, default=Path('artifacts'))
    sft_train = commands.add_parser('sft-train', help='preflight or resume Pilot 2 training')
    sft_train.add_argument('--config', type=Path, required=True)
    sft_train.add_argument('--preflight-only', action='store_true')
    sft_eval = commands.add_parser('sft-evaluate', help='evaluate a matched baseline or adapter checkpoint')
    sft_eval.add_argument('--config', type=Path, required=True)
    sft_eval.add_argument('--checkpoint', choices=['baseline', '0', '8', '16', '32', '64'], required=True)
    sft_benchmark = commands.add_parser('sft-benchmark', help='measure untuned generation speed at batch 1 and 2')
    sft_benchmark.add_argument('--config', type=Path, required=True)
    sft_compare = commands.add_parser('sft-compare', help='CPU-only paired checkpoint trajectory report')
    sft_compare.add_argument('--config', type=Path, required=True)
    score_audit = commands.add_parser('audit-sft-scores', help='CPU-only post-hoc flexible v3 audit of paired responses')
    score_audit.add_argument('--paired', type=Path, required=True)
    score_audit.add_argument('--name', required=True)
    activation_prepare = commands.add_parser('activation-prepare', help='freeze Pilot 3 sources, inputs and masks')
    activation_prepare.add_argument('--source-config', type=Path, required=True)
    activation_prepare.add_argument('--name', required=True)
    activation_prepare.add_argument('--fineweb-config', default='sample-10BT')
    activation_prepare.add_argument('--fineweb-revision')
    activation_audit = commands.add_parser('activation-audit', help='verify and display the saved Pilot 3 input audit')
    activation_audit.add_argument('--config', type=Path, required=True)
    activation_validate = commands.add_parser('activation-validate', help='validate one diagnostic batch across five checkpoints')
    activation_validate.add_argument('--config', type=Path, required=True)
    activation_validate.add_argument('--batch-size', type=int, default=2)
    activation_profile = commands.add_parser('activation-profile', help='profile validated fixed activation workloads')
    activation_profile.add_argument('--config', type=Path, required=True)
    activation_profile.add_argument('--calibration', type=Path)
    activation_validate.add_argument('--namespace', choices=['agreement-v2'])
    calibration_commands=[]
    for name in ['activation-calibration-prepare','activation-calibrate','activation-calibration-freeze','activation-calibration-validate']:
        command=commands.add_parser(name, help='versioned empirical numerical agreement calibration')
        command.add_argument('--config',type=Path,required=True)
        command.add_argument('--output-root',type=Path,default=Path('artifacts'))
        if name.endswith('prepare'): command.add_argument('--name',required=True)
        if name.endswith('freeze'): command.add_argument('--review-notes',required=True)
        calibration_commands.append(command)
    activation_freeze = commands.add_parser('activation-freeze', help='freeze an explicitly reviewed passing batch plan')
    activation_freeze.add_argument('--config', type=Path, required=True)
    activation_freeze.add_argument('--profile', type=Path, required=True)
    activation_freeze.add_argument('--batch-size', type=int, choices=[1, 2, 4, 8, 16], required=True)
    activation_freeze.add_argument('--name', required=True)
    activation_freeze.add_argument('--review-notes', required=True)
    activation_recover = commands.add_parser('activation-recover-measurement', help='verify saved units and authorize reviewed production hash scheduling recovery')
    activation_recover.add_argument('--config', type=Path, required=True)
    activation_recover.add_argument('--review-notes', required=True)
    activation_measure = commands.add_parser('activation-measure', help='execute/resume frozen activation batches')
    activation_measure.add_argument('--config', type=Path, required=True)
    activation_verify = commands.add_parser('activation-verify', help='CPU-only verification of completed activation measurement')
    activation_verify.add_argument('--config', type=Path, required=True)
    activation_report = commands.add_parser('activation-report', help='CPU scientific report from verified activation summaries')
    activation_report.add_argument('--config', type=Path, required=True)
    activation_report.add_argument('--name', required=True)
    for command in (prepare, run, fork, revise, report, rescore, sft_prepare, sft_train, sft_eval, sft_compare, sft_benchmark, score_audit, activation_prepare, activation_audit, activation_validate, activation_profile, activation_freeze, activation_recover, activation_measure, activation_verify, activation_report):
        command.add_argument("--output-root", type=Path, default=Path("artifacts"))
    token_prepare = commands.add_parser('tokens-prepare', help='freeze supplemental SFT/GRPO coefficient and KL inputs')
    token_prepare.add_argument('--sft-execution', type=Path, required=True)
    token_prepare.add_argument('--grpo-execution', type=Path, required=True)
    token_prepare.add_argument('--name', required=True)
    token_prepare.add_argument('--context-chunk', type=int, required=True)
    token_prepare.add_argument('--workspace-mib', type=int, required=True)
    token_prepare.add_argument('--output-root', type=Path, default=Path('artifacts'))
    for name in ['tokens-profile','tokens-freeze','tokens-measure','tokens-verify','tokens-report']:
        token = commands.add_parser(name, help='supplemental fixed-token coefficient/KL '+name.split('-')[1])
        token.add_argument('--config', type=Path, required=True)
        token.add_argument('--output-root', type=Path, default=Path('artifacts'))
        if name=='tokens-freeze': token.add_argument('--review-notes', required=True)
        if name=='tokens-report': token.add_argument('--name', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command.startswith('tokens-'):
            from pilot_eval.token_measurement_workflow import (prepare_tokens,profile_tokens,freeze_tokens,
                measure_tokens,verify_tokens,report_tokens)
            if args.command=='tokens-prepare':
                print(prepare_tokens(args.sft_execution,args.grpo_execution,args.output_root,args.name,
                    context_chunk=args.context_chunk,workspace_bytes=args.workspace_mib*2**20))
            elif args.command=='tokens-profile':
                result=profile_tokens(args.config,args.output_root,dependencies=dependencies)
                print(json.dumps({k:result[k] for k in ['profile_passed','estimated_main_forward_seconds',
                    'estimated_array_bytes','total_wall_seconds','estimate_note']},indent=2))
            elif args.command=='tokens-freeze':print(freeze_tokens(args.config,args.output_root,review_notes=args.review_notes))
            elif args.command=='tokens-measure':print(json.dumps(measure_tokens(args.config,args.output_root,dependencies=dependencies),indent=2))
            elif args.command=='tokens-verify':print(json.dumps(verify_tokens(args.config,args.output_root),indent=2))
            else:
                result=report_tokens(args.config,args.output_root,args.name)
                print(json.dumps(dict(report_complete=result['report_complete'],kl_trajectory=[{k:v for k,v in row.items() if k!='per_example'} for row in result['kl_trajectory']]),indent=2))
        elif args.command=='grpo-writes-report':
            from pilot_eval.grpo_write_report import report_grpo_writes
            result = report_grpo_writes(args.grpo_report,args.control_report,args.sft_report,args.output_root,args.name)
            print(json.dumps({k:result[k] for k in ['report_complete','defined_cosines','total_cosines','random_realizations']},indent=2))
        elif args.command=='grpo-writes-control':
            from pilot_eval.grpo_writes import prepare_random_writes
            print(prepare_random_writes(args.config,args.output_root,args.name))
        elif args.command=='grpo-writes-prepare':
            from pilot_eval.grpo_writes import prepare_grpo_writes
            print(prepare_grpo_writes(args.config,args.measurement_config,args.output_root,args.name))
        elif args.command=='grpo-evaluate':
            from pilot_eval.grpo_evaluation import evaluate_grpo
            print(json.dumps(evaluate_grpo(args.config,args.output_root,args.name,dependencies=dependencies),indent=2))
        elif args.command=='grpo-behaviour-report':
            from pilot_eval.grpo_behaviour_report import report_grpo_behaviour
            print(json.dumps(report_grpo_behaviour(args.config,args.output_root,args.name),indent=2))
        elif args.command in ['grpo-train','grpo-verify-training']:
            from pilot_eval.grpo_training import run_grpo,verify_training
            result=(run_grpo(args.config,args.output_root,args.name,length_change_definition=args.length_change_definition,dependencies=dependencies)
                    if args.command=='grpo-train' else verify_training(args.config,args.output_root))
            print(json.dumps(result,indent=2))
        elif args.command == 'grpo-preflight':
            from pilot_eval.grpo_preflight import collect_preflight
            print(json.dumps(collect_preflight(args.config,args.baseline_config,args.controls_name,args.settings,args.output_root,args.name,dependencies),indent=2))
        elif args.command == 'grpo-freeze':
            from pilot_eval.grpo_preflight import freeze_protocol
            print(freeze_protocol(args.config,args.review,args.output_root,args.name))
        elif args.command == 'grpo-controls':
            from pilot_eval.grpo_controls import run_controls
            print(json.dumps(run_controls(args.output_root, args.name), indent=2))
        elif args.command == 'grpo-baseline':
            from pilot_eval.grpo_baseline import sample_baseline
            print(json.dumps(sample_baseline(args.config, args.settings, args.output_root, args.name, dependencies), indent=2))
        elif args.command == 'grpo-prepare':
            from pilot_eval.grpo_prepare import prepare_grpo
            print(prepare_grpo(args.sft_config, args.measurement_config, args.output_root, args.name))
        elif args.command in ['grpo-audit', 'grpo-check-ready']:
            from pilot_eval.grpo_prepare import load_grpo_prepared, require_grpo_ready
            if args.command == 'grpo-check-ready':
                require_grpo_ready(args.config, args.output_root)
                print('GRPO plan has no unresolved choices.')
            else:
                config, *_ = load_grpo_prepared(args.config, args.output_root)
                print((args.output_root / config['audit_path']).read_text())
        elif args.command in ['activation-calibration-prepare','activation-calibrate','activation-calibration-freeze','activation-calibration-validate']:
            from pilot_eval.activation_calibration import prepare_calibration, collect_calibration, freeze_rule, validate_rule
            if args.command.endswith('prepare'):
                print(prepare_calibration(args.config,args.output_root,name=args.name,dependencies=dependencies))
            elif args.command=='activation-calibrate':
                print(json.dumps(collect_calibration(args.config,args.output_root,dependencies=dependencies)))
            elif args.command.endswith('freeze'):
                print(freeze_rule(args.config,args.output_root,review_notes=args.review_notes))
            else:
                print(json.dumps(validate_rule(args.config,args.output_root,dependencies=dependencies),indent=2))
        elif args.command == 'activation-report':
            from pilot_eval.activation_report import report_activation
            result = report_activation(args.config, args.output_root, name=args.name)
            print(json.dumps({'report_complete': result['report_complete'], 'name': result['name']}))
        elif args.command == 'activation-recover-measurement':
            from pilot_eval.activation_measurement import authorize_measurement_recovery
            result = authorize_measurement_recovery(args.config,args.output_root,review_notes=args.review_notes,dependencies=dependencies)
            print(json.dumps({'policy':result['policy'],'verified_completed_units':result['verified_completed_units'],'actual_runtime':result['actual_runtime']},indent=2))
        elif args.command in ['activation-measure', 'activation-verify']:
            from pilot_eval.activation_measurement import measure_activation, verify_measurement
            result = (measure_activation(args.config, args.output_root, dependencies=dependencies)
                      if args.command == 'activation-measure' else verify_measurement(args.config, args.output_root))
            print(json.dumps({key: value for key, value in result.items() if key != 'example_ids'}))
        elif args.command == 'activation-profile':
            from pilot_eval.activation_profile import profile_activation
            result = profile_activation(args.config, args.output_root, dependencies=dependencies, calibration=args.calibration)
            print(json.dumps({'profile_path': result['profile_path'],
                'provisional_fastest_batch': result['provisional_fastest_batch'],
                'measurements': [{k: v for k, v in row.items() if k != 'validation'}
                                 for row in result['measurements']]}, indent=2))
        elif args.command == 'activation-freeze':
            from pilot_eval.activation_profile import freeze_execution
            print(freeze_execution(args.config, args.output_root, profile=args.profile,
                batch_size=args.batch_size, name=args.name, review_notes=args.review_notes))
        elif args.command == 'activation-validate':
            from pilot_eval.activation_workflow import validate_activation
            print(json.dumps(validate_activation(args.config, args.output_root,
                batch_size=args.batch_size, dependencies=dependencies, namespace=args.namespace), indent=2))
        elif args.command == 'activation-prepare':
            from pilot_eval.activation_prepare import prepare_activation
            print(prepare_activation(args.source_config, args.output_root, args.name,
                dependencies=dependencies, fineweb_config=args.fineweb_config,
                fineweb_revision=args.fineweb_revision))
        elif args.command == 'activation-audit':
            from pilot_eval.activation_prepare import load_prepared
            config, _ = load_prepared(args.config, args.output_root)
            print((args.output_root / config['audit_path']).read_text())
        elif args.command == 'audit-sft-scores':
            from pilot_eval.sft_score_audit import audit_sft_scores
            result = audit_sft_scores(args.paired, args.output_root, args.name)
            print(json.dumps(result, indent=2))
        elif args.command == 'sft-benchmark':
            from pilot_eval.sft_benchmark import benchmark_sft
            print(json.dumps(benchmark_sft(args.config, args.output_root, dependencies=dependencies), indent=2))
        elif args.command == 'sft-compare':
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
