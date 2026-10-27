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
    for operation in ['prepare','runtime','run','status','forecast','freeze','verify-measurement','report','verify','recover-writer']:
        distance=commands.add_parser('stage0-distance-'+operation,help='optional teacher-forced NLL/entropy measurement')
        distance.add_argument('--output-root',type=Path,required=True)
        if operation=='prepare':
            distance.add_argument('--rollouts',type=Path,required=True);distance.add_argument('--name',required=True)
            distance.add_argument('--context-chunk',type=int,default=16);distance.add_argument('--workspace-bytes',type=int,default=536870912)
        else:distance.add_argument('--config',type=Path,required=True)
        if operation=='run':
            distance.add_argument('--approval',type=Path);distance.add_argument('--max-new-units',type=int);distance.add_argument('--no-progress',action='store_true')
        if operation=='recover-writer':
            distance.add_argument('--owner-token',required=True);distance.add_argument('--review-notes',required=True)
        if operation=='freeze':distance.add_argument('--review',type=Path,required=True)

    for operation in ['audit','prepare','runtime','profile','profile-verify','freeze','run','status','verify','recover-writer']:
        rollout=commands.add_parser('stage0-rollouts-'+operation,help='optional audited untuned rollouts and derived cohorts')
        rollout.add_argument('--output-root',type=Path,required=True)
        if operation in ['audit','prepare']:
            rollout.add_argument('--audit',type=Path,required=True)
            rollout.add_argument('--baseline',type=Path,required=True)
            rollout.add_argument('--name',required=True)
            if operation=='prepare':rollout.add_argument('--weights-dir',type=Path,required=True)
        else:rollout.add_argument('--config',type=Path,required=True)
        if operation in ['profile','run']:
            rollout.add_argument('--approval',type=Path)
            rollout.add_argument('--max-new-units',type=int)
            rollout.add_argument('--no-progress',action='store_true')
        if operation=='freeze':rollout.add_argument('--review',type=Path,required=True)
        if operation=='recover-writer':
            rollout.add_argument('--phase',choices=['profile','collection'],required=True)
            rollout.add_argument('--owner-token',required=True)
            rollout.add_argument('--review-notes',required=True)

    for operation in ['prepare','run','verify','status','recover-writer']:
        conditional=commands.add_parser('stage0-intruder-edits-'+operation,help='CPU conditional merged V6/V6c catalogues')
        conditional.add_argument('--output-root',type=Path,required=True)
        if operation=='prepare':
            conditional.add_argument('--edits',type=Path,required=True)
            conditional.add_argument('--geometry',type=Path,required=True)
            conditional.add_argument('--name',required=True)
        else:conditional.add_argument('--config',type=Path,required=True)
        if operation=='run':
            conditional.add_argument('--max-new-units',type=int)
            conditional.add_argument('--no-progress',action='store_true')
        if operation=='recover-writer':
            conditional.add_argument('--owner-token',required=True)
            conditional.add_argument('--review-notes',required=True)
    for operation in ['prepare','run','verify','status','recover-writer']:
        intruders=commands.add_parser('stage0-intruders-'+operation,help='CPU gap-aware subspace rotation and chance-aware intruder geometry')
        intruders.add_argument('--output-root',type=Path,required=True)
        if operation=='prepare':
            intruders.add_argument('--spectral',type=Path,required=True)
            intruders.add_argument('--name',required=True)
        else:intruders.add_argument('--config',type=Path,required=True)
        if operation=='run':
            intruders.add_argument('--max-new-units',type=int)
            intruders.add_argument('--no-progress',action='store_true')
        if operation=='recover-writer':
            intruders.add_argument('--owner-token',required=True)
            intruders.add_argument('--review-notes',required=True)
    for operation in ['prepare','run','verify','status','audit-prepare','audit-status','audit-save','audit-freeze','audit-reveal','report']:
        findings=commands.add_parser('stage0-findings-'+operation,help='CPU paired Stage 0 findings and targeted intended-answer review')
        findings.add_argument('--output-root',type=Path,required=True)
        if operation=='prepare':
            findings.add_argument('--evaluation',type=Path,required=True)
            findings.add_argument('--drift',type=Path,required=True)
            findings.add_argument('--name',required=True)
        else:findings.add_argument('--config',type=Path,required=True)
        if operation=='run':findings.add_argument('--max-new-units',type=int)
        if operation=='audit-prepare':
            findings.add_argument('--reuse-audit',type=Path)
            findings.add_argument('--reuse-revision')
            findings.add_argument('--reuse-assisted-report',type=Path)
        if operation=='audit-save':findings.add_argument('--labels',type=Path,required=True)
    for operation in ['prepare','run','verify','status','recover-writer']:
        drift=commands.add_parser('stage0-drift-'+operation,help='paired fixed-context KL and direct effective writes')
        drift.add_argument('--output-root',type=Path,required=True)
        if operation=='prepare':
            drift.add_argument('--execution',type=Path,required=True)
            drift.add_argument('--inputs',type=Path,required=True)
            drift.add_argument('--name',required=True)
            drift.add_argument('--batch-size',type=int,default=1)
            drift.add_argument('--batches-per-unit',type=int,default=8)
            drift.add_argument('--context-chunk',type=int,default=32)
            drift.add_argument('--workspace-bytes',type=int,default=536870912)
        else:drift.add_argument('--config',type=Path,required=True)
        if operation=='run':
            drift.add_argument('--approval',type=Path)
            drift.add_argument('--max-new-units',type=int)
            drift.add_argument('--no-progress',action='store_true')
        if operation=='recover-writer':
            drift.add_argument('--owner-token',required=True)
            drift.add_argument('--review-notes',required=True)
    for operation in ['prepare','run','verify','status','recover-writer']:
        evaluation=commands.add_parser('stage0-evaluation-'+operation,
            help='saved paired greedy intervention evaluation')
        evaluation.add_argument('--output-root',type=Path,required=True)
        if operation=='prepare':
            evaluation.add_argument('--execution',type=Path,required=True)
            evaluation.add_argument('--batch-size',type=int,default=1)
            evaluation.add_argument('--batches-per-unit',type=int,default=8)
        else:
            evaluation.add_argument('--config',type=Path,required=True)
        if operation=='run':
            evaluation.add_argument('--approval',type=Path)
            evaluation.add_argument('--max-new-units',type=int)
            evaluation.add_argument('--no-progress',action='store_true')
        if operation=='recover-writer':
            evaluation.add_argument('--owner-token',required=True)
            evaluation.add_argument('--review-notes',required=True)
    for operation in ['prepare', 'run', 'verify']:
        edits=commands.add_parser('stage0-edits-'+operation,help='CPU principal edits and matched controls')
        edits.add_argument('--output-root',type=Path,required=True)
        if operation=='prepare':
            edits.add_argument('--spectral',type=Path,required=True)
            edits.add_argument('--name',required=True)
        else:
            edits.add_argument('--config',type=Path,required=True)
        if operation=='run':edits.add_argument('--max-new-units',type=int)
    for operation in ['prepare', 'run', 'verify', 'report']:
        spectral = commands.add_parser('stage0-spectral-'+operation, help='CPU spectral cache and explicit overlap reveal')
        spectral.add_argument('--output-root', type=Path, required=True)
        if operation == 'prepare':
            spectral.add_argument('--audit', type=Path, required=True)
            spectral.add_argument('--weights-dir', type=Path, required=True)
            spectral.add_argument('--name', required=True)
        else:
            spectral.add_argument('--config', type=Path, required=True)
        if operation == 'run':
            spectral.add_argument('--max-new-units', type=int)
        if operation == 'report':
            spectral.add_argument('--review', type=Path, required=True)
            spectral.add_argument('--name', required=True)
    for operation in ['prepare', 'runtime', 'run', 'verify', 'approve', 'freeze', 'status', 'recover-writer']:
        qualification = commands.add_parser('stage0-qualification-' + operation,
            help='bounded mixed-loader qualification and measured budget freeze')
        qualification.add_argument('--output-root', type=Path, required=True)
        if operation == 'prepare':
            source = qualification.add_mutually_exclusive_group(required=True)
            source.add_argument('--edits', type=Path)
            source.add_argument('--catalogue', type=Path)
            qualification.add_argument('--cohort', type=Path)
            qualification.add_argument('--name', required=True)
        else:
            qualification.add_argument('--config', type=Path, required=True)
        if operation == 'run':
            qualification.add_argument('--approval', type=Path, required=True)
            qualification.add_argument('--max-new-units', type=int)
        if operation in ['approve', 'freeze']:
            qualification.add_argument('--review', type=Path, required=True)
        if operation == 'recover-writer':
            qualification.add_argument('--owner-token', required=True)
            qualification.add_argument('--review-notes', required=True)
    for operation in ['prepare', 'audit', 'status', 'budget-record', 'recover-writer']:
        stage0 = commands.add_parser('stage0-'+operation, help='CPU Stage 0 source readiness and cumulative budget')
        stage0.add_argument('--output-root', type=Path, required=True)
        if operation == 'prepare':
            stage0.add_argument('--selection', type=Path, required=True)
            stage0.add_argument('--name', required=True)
        else:
            stage0.add_argument('--config', type=Path, required=True)
        if operation == 'audit':
            stage0.add_argument('--max-new-units', type=int)
        if operation == 'budget-record':
            stage0.add_argument('--observation', type=Path, required=True)
        if operation == 'recover-writer':
            stage0.add_argument('--owner-token', required=True)
            stage0.add_argument('--review-notes', required=True)
            stage0.add_argument('--product', choices=['audit', 'budget'], default='audit')
    budget_settle = commands.add_parser('stage0-budget-settle', help='reconcile one Stage 0 GPU reservation from a newer cumulative usage reading')
    budget_settle.add_argument('--output-root', type=Path, required=True)
    budget_settle.add_argument('--reservation-id', required=True)
    findings=commands.add_parser('behaviour-findings',help='CPU combined behavioural findings and evidence handoff')
    findings.add_argument('--config',type=Path,required=True)
    findings.add_argument('--transitions',type=Path,required=True)
    findings.add_argument('--sampling',type=Path,required=True)
    findings.add_argument('--audit-config',type=Path)
    findings.add_argument('--assisted-report',type=Path)
    findings.add_argument('--name',required=True)
    findings.add_argument('--output-root',type=Path,default=Path('artifacts'))
    findings.add_argument('--max-new-units',type=int)
    audit=commands.add_parser('behaviour-audit-prepare',help='Select and export blinded intended-answer review')
    audit.add_argument('--transitions',type=Path,required=True)
    audit.add_argument('--name',required=True)
    audit.add_argument('--output-root',type=Path,default=Path('artifacts'))
    for operation in ['status','save','freeze','reveal','revise','explain']:
        audit_stage=commands.add_parser('behaviour-audit-'+operation,help='Saved single-reviewer audit workflow')
        audit_stage.add_argument('--config',type=Path,required=True)
        audit_stage.add_argument('--output-root',type=Path,default=Path('artifacts'))
        if operation in ['save','revise']:audit_stage.add_argument('--labels',type=Path,required=True)
        if operation=='revise':
            audit_stage.add_argument('--revision',required=True)
            audit_stage.add_argument('--reason',required=True)
        if operation=='explain':
            audit_stage.add_argument('--annotations',type=Path,required=True)
            audit_stage.add_argument('--revision',required=True)
            audit_stage.add_argument('--source-revision')
    transitions=commands.add_parser('behaviour-transitions',help='CPU paired answer transitions and intervals')
    transitions.add_argument('--config',type=Path,required=True)
    transitions.add_argument('--name',required=True)
    transitions.add_argument('--output-root',type=Path,default=Path('artifacts'))
    transitions.add_argument('--max-new-units',type=int)
    sampling=commands.add_parser('behaviour-sampling',help='CPU saved eight-draw GRPO reliability')
    sampling.add_argument('--config',type=Path,required=True)
    sampling.add_argument('--sampled-source-index',type=int,required=True)
    sampling.add_argument('--name',required=True)
    sampling.add_argument('--output-root',type=Path,default=Path('artifacts'))
    sampling.add_argument('--max-new-units',type=int)
    for operation in ('prepare','score','verify'):
        behaviour=commands.add_parser('behaviour-'+operation,help='CPU saved-output behavioural breakdown')
        behaviour.add_argument('--output-root',type=Path,default=Path('artifacts'))
        if operation=='prepare':
            behaviour.add_argument('--selection',type=Path,required=True)
            behaviour.add_argument('--name',required=True)
        else:behaviour.add_argument('--config',type=Path,required=True)
        if operation=='score':behaviour.add_argument('--max-new-units',type=int)
    for operation in ('prepare', 'nulls-run', 'nulls-verify', 'run', 'verify'):
        analysis = commands.add_parser('traces-analysis-' + operation,
            help='CPU automatic trace relevance with saved nulls and paired document draws')
        analysis.add_argument('--output-root', type=Path, required=True)
        if operation == 'prepare':
            analysis.add_argument('--readouts', type=Path, required=True)
            analysis.add_argument('--name', required=True)
        else:
            analysis.add_argument('--config', type=Path, required=True)
        if operation == 'nulls-run':
            analysis.add_argument('--max-new-chunks', type=int)
        if operation == 'run':
            analysis.add_argument('--max-new-units', type=int)
    assets=commands.add_parser('traces-readout-assets',help='CPU extraction of pinned norm/head; no decoder inference')
    assets.add_argument('--source','--capture',dest='source',type=Path,required=True)
    assets.add_argument('--output-root',type=Path,required=True)
    assets.add_argument('--name',required=True)
    for operation in ('propose','prepare','run','verify'):
        readout=commands.add_parser('traces-readouts-'+operation,help='Gated saved-vector readout workflow')
        readout.add_argument('--output-root',type=Path,required=True)
        if operation in ['propose','prepare']:
            for field in ['capture','references']:readout.add_argument('--'+field,type=Path,required=True)
            readout.add_argument('--name',required=True)
        else:readout.add_argument('--config',type=Path,required=True)
        if operation=='prepare':
            readout.add_argument('--assets',type=Path,required=True)
            readout.add_argument('--review',type=Path,required=True)
        if operation=='run':readout.add_argument('--max-new-units',type=int)
    for operation in ('propose','freeze','run','verify'):
        production=commands.add_parser('traces-production-'+operation,help='Reviewed Pilot 5 durable capture workflow')
        production.add_argument('--output-root',type=Path,required=True)
        if operation=='propose':
            production.add_argument('--profile',type=Path,required=True)
            production.add_argument('--name',required=True)
        elif operation=='freeze':
            production.add_argument('--proposal',type=Path,required=True)
            production.add_argument('--review',type=Path,required=True)
        else:production.add_argument('--config',type=Path,required=True)
        if operation=='run':production.add_argument('--staging-root',type=Path)
    for operation in ('prepare', 'diagnose', 'verify', 'approve'):
        qualification = commands.add_parser('traces-lens-qualification-' + operation,
            help='CPU JLens diagnostics from four saved profile documents; explicit review required')
        qualification.add_argument('--output-root', type=Path, required=True)
        if operation == 'prepare':
            for field in ('profile', 'inspection', 'assets'):
                qualification.add_argument('--' + field, type=Path, required=True)
            qualification.add_argument('--name', required=True)
            qualification.add_argument('--bootstrap-draws', type=int, required=True)
            qualification.add_argument('--seed', type=int, required=True)
        else:
            qualification.add_argument('--config', type=Path, required=True)
        if operation == 'diagnose':
            qualification.add_argument('--max-new-units', type=int)
        if operation == 'approve':
            qualification.add_argument('--review', type=Path, required=True)
    for operation in ('prepare', 'diagnose', 'verify', 'approve'):
        capture_q = commands.add_parser('traces-capture-qualification-' + operation,
            help='CPU capture acceptance from saved zero/profile evidence; explicit limits required')
        capture_q.add_argument('--output-root', type=Path, required=True)
        if operation == 'prepare':
            for field in ('profile', 'assets'):
                capture_q.add_argument('--' + field, type=Path, required=True)
            capture_q.add_argument('--name', required=True)
            capture_q.add_argument('--batch-size', type=int, required=True)
            capture_q.add_argument('--storage-dtype', choices=['float16', 'float32'], required=True)
            capture_q.add_argument('--bootstrap-draws', type=int, required=True)
            capture_q.add_argument('--seed', type=int, required=True)
        else:
            capture_q.add_argument('--config', type=Path, required=True)
        if operation == 'diagnose':
            capture_q.add_argument('--max-new-units', type=int)
        if operation == 'approve':
            capture_q.add_argument('--review', type=Path, required=True)
    lens=commands.add_parser('traces-lens-inspect',help='CPU inspection of the pinned community lens; never fits')
    lens.add_argument('--output-root',type=Path,required=True)
    lens.add_argument('--name',required=True)
    lens.add_argument('--artifact-dir',type=Path)
    for operation in ('prepare','run','verify'):
        profile=commands.add_parser('traces-profile-'+operation,help='Approved bounded Pilot 5 endpoint cost profile')
        profile.add_argument('--output-root',type=Path,required=True)
        if operation=='prepare':
            profile.add_argument('--diagnostic',type=Path,required=True)
            profile.add_argument('--random-config',type=Path,required=True)
            profile.add_argument('--name',required=True)
        else:profile.add_argument('--config',type=Path,required=True)
    for operation in ('prepare','diagnose','verify'):
        capture=commands.add_parser('traces-capture-'+operation,help='Bounded Pilot 5 capture instrument diagnostic')
        capture.add_argument('--output-root',type=Path,required=True)
        if operation=='prepare':
            for field in ('corpus','sft-config','grpo-config'):capture.add_argument('--'+field,type=Path,required=True)
            capture.add_argument('--name',required=True)
        else:capture.add_argument('--config',type=Path,required=True)
    for operation in ('prepare','verify','review'):
        reference=commands.add_parser('traces-reference-'+operation,help='CPU frozen task-reference workflow')
        reference.add_argument('--output-root',type=Path,required=True)
        if operation=='prepare':
            reference.add_argument('--corpus',type=Path,required=True)
            reference.add_argument('--sft-config',type=Path,required=True)
            reference.add_argument('--grpo-config',type=Path,required=True)
            reference.add_argument('--name',required=True)
        else:reference.add_argument('--config',type=Path,required=True)
        if operation=='review':
            reference.add_argument('--reviewer',required=True)
            reference.add_argument('--notes',required=True)
            reference.add_argument('--exposure-status',choices=['not-seen','seen-before-freeze','unknown'],required=True)
    for operation in ('exposure', 'inspect', 'prepare'):
        review = commands.add_parser('traces-review-' + operation, help='CPU prereveal review history')
        review.add_argument('--name', required=True)
        review.add_argument('--output-root', type=Path, required=True)
        if operation == 'exposure':
            review.add_argument('--event', type=Path, required=True)
        if operation == 'prepare':
            review.add_argument('--config', type=Path, required=True)
    corpus_prepare = commands.add_parser('traces-corpus-prepare', help='CPU preparation of frozen Pilot 5 raw inputs')
    corpus_prepare.add_argument('--source-config', type=Path, required=True)
    corpus_prepare.add_argument('--review', type=Path, required=True)
    corpus_prepare.add_argument('--name', required=True)
    corpus_prepare.add_argument('--output-root', type=Path, default=Path('artifacts'))
    corpus_verify = commands.add_parser('traces-corpus-verify', help='verify saved Pilot 5 corpus without downloads')
    corpus_verify.add_argument('--config', type=Path, required=True)
    corpus_verify.add_argument('--output-root', type=Path, default=Path('artifacts'))
    final_report = commands.add_parser('grpo-final-report', help='verify sources and publish the paired-arm handoff')
    final_verify = commands.add_parser('grpo-final-verify', help='CPU-only final report and download inventory verification')
    final_export = commands.add_parser('grpo-final-export', help='stream verified report and raw inventory into a tar bundle')
    final_export.add_argument('--archive-name', required=True)
    for command in (final_report, final_verify, final_export):
        command.add_argument('--selection', type=Path, required=True)
        command.add_argument('--name', required=True)
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
    for command in (final_report, final_verify, final_export, prepare, run, fork, revise, report, rescore, sft_prepare, sft_train, sft_eval, sft_compare, sft_benchmark, score_audit, activation_prepare, activation_audit, activation_validate, activation_profile, activation_freeze, activation_recover, activation_measure, activation_verify, activation_report):
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
        if args.command.startswith('stage0-distance-'):
            from pilot_eval import stage0_distance as workflow
            operation=args.command.removeprefix('stage0-distance-')
            if operation=='prepare':result=workflow.prepare(args.rollouts,args.output_root,args.name,context_chunk=args.context_chunk,workspace_bytes=args.workspace_bytes)
            elif operation=='runtime':result=workflow.runtime(args.config,args.output_root,dependencies=dependencies)
            elif operation=='freeze':result=workflow.freeze(args.config,args.output_root,args.review)
            elif operation=='run':result=workflow.run(args.config,args.output_root,dependencies=dependencies,approval_path=args.approval,max_new_units=args.max_new_units,progress=not args.no_progress)
            elif operation=='recover-writer':result=workflow.recover_writer(args.config,args.output_root,args.owner_token,args.review_notes)
            elif operation in ['report','verify']:
                from pilot_eval import stage0_distance_report
                result=getattr(stage0_distance_report,operation)(args.config,args.output_root)
            else:result=getattr(workflow,operation.replace('-','_'))(args.config,args.output_root)
            if isinstance(result,dict):result={k:v for k,v in result.items() if k not in ['per_problem','top_gold_tokens']}
            print(str(result) if isinstance(result,Path) else json.dumps(result,indent=2))
        elif args.command.startswith('stage0-rollouts-'):
            from pilot_eval import stage0_rollouts as workflow
            operation=args.command.removeprefix('stage0-rollouts-')
            if operation=='audit':result=workflow.audit(args.audit,args.baseline,args.output_root,args.name)
            elif operation=='prepare':result=workflow.prepare(args.audit,args.baseline,args.weights_dir,args.output_root,args.name,dependencies=dependencies)
            elif operation=='runtime':result=workflow.runtime(args.config,args.output_root,dependencies=dependencies)
            elif operation in ['profile','run']:result=getattr(workflow,operation)(args.config,args.output_root,dependencies=dependencies,approval_path=args.approval,max_new_units=args.max_new_units,progress=not args.no_progress)
            elif operation=='freeze':result=workflow.freeze(args.config,args.output_root,args.review)
            elif operation=='recover-writer':result=workflow.recover_writer(args.config,args.output_root,args.phase,args.owner_token,args.review_notes)
            else:result=getattr(workflow,operation.replace('-','_'))(args.config,args.output_root)
            if isinstance(result,dict):result={k:v for k,v in result.items() if k not in ['parents','legacy','historical','training']}
            print(str(result) if isinstance(result,Path) else json.dumps(result,indent=2))
        elif args.command.startswith('stage0-intruder-edits-'):
            from pilot_eval import stage0_intruder_edits as workflow
            operation=args.command.removeprefix('stage0-intruder-edits-')
            if operation=='prepare':result=workflow.prepare(args.edits,args.geometry,args.output_root,args.name)
            elif operation=='run':result=workflow.run(args.config,args.output_root,dependencies=dependencies,max_new_units=args.max_new_units,progress=not args.no_progress)
            elif operation=='recover-writer':result=workflow.recover_writer(args.config,args.output_root,args.owner_token,args.review_notes)
            else:result=getattr(workflow,operation)(args.config,args.output_root)
            if isinstance(result,dict) and 'matching' in result:
                result={k:v for k,v in result.items() if k not in ['matching','variants']}
            print(str(result) if isinstance(result,Path) else json.dumps(result,indent=2))
        elif args.command.startswith('stage0-intruders-'):
            from pilot_eval import stage0_intruders as workflow
            operation=args.command.removeprefix('stage0-intruders-')
            if operation=='prepare':result=workflow.prepare(args.spectral,args.output_root,args.name)
            elif operation=='run':result=workflow.run(args.config,args.output_root,dependencies=dependencies,max_new_units=args.max_new_units,progress=not args.no_progress)
            elif operation=='recover-writer':result=workflow.recover_writer(args.config,args.output_root,args.owner_token,args.review_notes)
            else:result=getattr(workflow,operation)(args.config,args.output_root)
            if isinstance(result,dict) and 'module_results' in result:
                result={k:v for k,v in result.items() if k!='module_results'}
            print(str(result) if isinstance(result,Path) else json.dumps(result,indent=2))
        elif args.command.startswith('stage0-findings-'):
            from pilot_eval import stage0_findings as workflow
            operation=args.command.removeprefix('stage0-findings-')
            if operation=='prepare':result=workflow.prepare(args.evaluation,args.drift,args.output_root,args.name)
            elif operation=='run':result=workflow.run(args.config,args.output_root,max_new_units=args.max_new_units)
            elif operation.startswith('audit-'):
                from pilot_eval import stage0_findings_review as review
                action=operation.removeprefix('audit-')
                if action=='prepare':result=review.prepare(args.config,args.output_root,reuse_audit=args.reuse_audit,reuse_revision=args.reuse_revision,reuse_assisted_report=args.reuse_assisted_report)
                elif action=='save':result=review.save(args.config,args.output_root,args.labels)
                else:result=getattr(review,action)(args.config,args.output_root)
            else:result=getattr(workflow,operation)(args.config,args.output_root)
            if isinstance(result,dict) and 'contrasts' in result:
                result={k:v for k,v in result.items() if k!='contrasts'}
                result['detail']='All contrast tables, raw pairs and plots are saved in the findings directory.'
            print(str(result) if isinstance(result,Path) else json.dumps(result,indent=2))
        elif args.command.startswith('stage0-drift-'):
            from pilot_eval.stage0_drift import prepare,run,verify,status,recover_writer
            operation=args.command.removeprefix('stage0-drift-')
            if operation=='prepare':
                result=prepare(args.execution,args.inputs,args.output_root,args.name,batch_size=args.batch_size,
                    batches_per_unit=args.batches_per_unit,context_chunk=args.context_chunk,workspace_bytes=args.workspace_bytes)
            elif operation=='run':
                result=run(args.config,args.output_root,dependencies=dependencies,approval_path=args.approval,
                    max_new_units=args.max_new_units,progress=not args.no_progress)
            elif operation=='recover-writer':result=recover_writer(args.config,args.output_root,args.owner_token,args.review_notes)
            else:result=(verify if operation=='verify' else status)(args.config,args.output_root)
            if isinstance(result,dict) and 'per_document' in result:
                result={k:v for k,v in result.items() if k not in ['per_document','cells']}
                result['detail']='Full document/module results are saved under the canonical result directory.'
            print(str(result) if isinstance(result,Path) else json.dumps(result,indent=2))
        elif args.command.startswith('stage0-evaluation-'):
            from pilot_eval.stage0_evaluation import prepare,run,verify,status,recover_writer
            operation=args.command.removeprefix('stage0-evaluation-')
            if operation=='prepare':
                result=prepare(args.execution,args.output_root,batch_size=args.batch_size,batches_per_unit=args.batches_per_unit)
            elif operation=='run':
                result=run(args.config,args.output_root,dependencies=dependencies,
                    max_new_units=args.max_new_units,approval_path=args.approval,progress=not args.no_progress)
            elif operation=='recover-writer':
                result=recover_writer(args.config,args.output_root,args.owner_token,args.review_notes)
            else:
                result=(verify if operation=='verify' else status)(args.config,args.output_root)
            print(str(result) if isinstance(result,Path) else json.dumps(result,indent=2))
        elif args.command == 'stage0-budget-settle':
            from pilot_eval.stage0_budget import settle
            result=settle(args.output_root.resolve(),args.reservation_id)
            print(json.dumps(result,indent=2))
        elif args.command.startswith('stage0-qualification-'):
            from pilot_eval.stage0_qualification import prepare, runtime, run, verify, approve, freeze, status, recover_writer
            operation = args.command.removeprefix('stage0-qualification-')
            if operation == 'prepare':
                result = prepare(args.edits, args.cohort, args.output_root, args.name, catalogue_path=args.catalogue)
            elif operation == 'runtime':
                result = runtime(args.config, args.output_root, dependencies=dependencies)
            elif operation == 'run':
                result = run(args.config, args.output_root, args.approval,
                             dependencies=dependencies, max_new_units=args.max_new_units)
            elif operation == 'verify':
                result = verify(args.config, args.output_root)
            elif operation == 'status':
                result = status(args.config, args.output_root)
            elif operation == 'recover-writer':
                result = recover_writer(args.config, args.output_root, args.owner_token, args.review_notes)
            elif operation == 'approve':
                result = approve(args.config, args.output_root, args.review)
            else:
                result = freeze(args.config, args.output_root, args.review)
            print(result if isinstance(result, (str, Path)) else json.dumps(result, indent=2))
        elif args.command.startswith('stage0-edits-'):
            from pilot_eval.stage0_edits import prepare,run,verify
            if args.command=='stage0-edits-prepare':
                print(prepare(args.spectral,args.output_root,args.name))
            else:
                result=(run(args.config,args.output_root,max_new_units=args.max_new_units)
                        if args.command=='stage0-edits-run' else verify(args.config,args.output_root))
                print(json.dumps(result,indent=2))
        elif args.command.startswith('stage0-spectral-'):
            from pilot_eval.stage0_spectral import prepare, run, verify, report
            if args.command == 'stage0-spectral-prepare':
                print(prepare(args.audit, args.weights_dir, args.output_root, args.name, dependencies=dependencies))
            else:
                if args.command == 'stage0-spectral-run':
                    result=run(args.config,args.output_root,dependencies=dependencies,max_new_units=args.max_new_units)
                elif args.command == 'stage0-spectral-report':
                    result=report(args.config,args.output_root,args.name,args.review)
                else:
                    result=verify(args.config,args.output_root)
                print(json.dumps({key:value for key,value in result.items() if key!='module_results'},indent=2))
        elif args.command.startswith('stage0-'):
            from pilot_eval.stage0_workflow import prepare, audit, status, record_budget, recover_writer, console_summary
            if args.command == 'stage0-prepare':
                print(prepare(args.selection, args.output_root, args.name))
            else:
                if args.command == 'stage0-audit':
                    result = audit(args.config, args.output_root, max_new_units=args.max_new_units)
                elif args.command == 'stage0-budget-record':
                    result = record_budget(args.config, args.output_root, args.observation)
                elif args.command == 'stage0-recover-writer':
                    result = recover_writer(args.config, args.output_root, args.owner_token,
                                            args.review_notes, product=args.product)
                else:
                    result = status(args.config, args.output_root)
                print(json.dumps(console_summary(result), indent=2))
        elif args.command=='behaviour-findings':
            from pilot_eval.behaviour_findings import build_findings
            result=build_findings(args.config,args.transitions,args.sampling,args.output_root,args.name,
                                  audit_path=args.audit_config,assisted_report=args.assisted_report,
                                  max_new_units=args.max_new_units)
            print(json.dumps(result,indent=2))
        elif args.command=='behaviour-audit-prepare':
            from pilot_eval.behaviour_audit import prepare_audit
            print(prepare_audit(args.transitions,args.output_root,args.name))
        elif args.command.startswith('behaviour-audit-'):
            from pilot_eval.behaviour_audit import audit_status,save_labels,freeze_labels,reveal_audit,revise_labels,explain_audit
            operation=args.command.removeprefix('behaviour-audit-')
            if operation=='save':result=save_labels(args.config,args.output_root,args.labels)
            elif operation=='revise':result=revise_labels(args.config,args.output_root,args.labels,args.revision,args.reason)
            elif operation=='explain':result=explain_audit(args.config,args.output_root,args.annotations,args.revision,source_revision=args.source_revision)
            else:result={'status':audit_status,'freeze':freeze_labels,'reveal':reveal_audit}[operation](args.config,args.output_root)
            print(json.dumps(result,indent=2))
        elif args.command=='behaviour-transitions':
            from pilot_eval.behaviour_transitions import report_transitions
            result=report_transitions(args.config,args.output_root,args.name,max_new_units=args.max_new_units)
            print(json.dumps({'execution_kind':result['execution_kind'],'report':str(args.output_root/'reports'/args.name/'results/report.md')},indent=2))
        elif args.command=='behaviour-sampling':
            from pilot_eval.behaviour_sampling import report_sampling
            result=report_sampling(args.config,args.output_root,args.name,args.sampled_source_index,
                                   max_new_units=args.max_new_units)
            print(json.dumps({'state':result.get('state','completed'), 'report':str(args.output_root/'reports'/args.name/'results/report.md')},indent=2))
        elif args.command.startswith('behaviour-'):
            from pilot_eval.behaviour_breakdown import prepare_behaviour,score_behaviour,verify_behaviour
            if args.command=='behaviour-prepare':print(prepare_behaviour(args.selection,args.output_root,args.name))
            elif args.command=='behaviour-score':print(json.dumps(score_behaviour(args.config,args.output_root,args.max_new_units),indent=2))
            else:print(json.dumps(verify_behaviour(args.config,args.output_root),indent=2))
        elif args.command=='traces-readout-assets':
            from pilot_eval.trace_readout_assets import prepare_assets
            print(prepare_assets(args.source,args.output_root,args.name))
        elif args.command.startswith('traces-readouts-'):
            from pilot_eval.trace_readouts import propose_readouts,prepare_readouts,run_readouts,verify_readouts
            if args.command=='traces-readouts-propose':
                print(propose_readouts(args.capture,args.references,args.output_root,args.name))
            elif args.command=='traces-readouts-prepare':
                print(prepare_readouts(args.capture,args.assets,args.references,args.review,args.output_root,args.name))
            else:
                result=(run_readouts(args.config,args.output_root,max_new_units=args.max_new_units)
                        if args.command=='traces-readouts-run' else verify_readouts(args.config,args.output_root))
                print(json.dumps(result,indent=2))
        elif args.command.startswith('traces-production-'):
            from pilot_eval.trace_production_plan import propose_capture,freeze_capture
            from pilot_eval.trace_production import run_capture,verify_capture
            if args.command=='traces-production-propose':
                print(propose_capture(args.profile,args.output_root,args.name))
            elif args.command=='traces-production-freeze':
                print(freeze_capture(args.proposal,args.review,args.output_root))
            else:
                result=(run_capture(args.config,args.output_root,staging_root=args.staging_root,dependencies=dependencies)
                        if args.command=='traces-production-run' else verify_capture(args.config,args.output_root))
                print(json.dumps(result,indent=2))
        elif args.command.startswith('traces-lens-qualification-'):
            from pilot_eval.trace_lens_qualification import prepare, diagnose, verify, approve, print_summary
            operation = args.command.removeprefix('traces-lens-qualification-')
            if operation == 'prepare':
                result = prepare(args.profile, args.inspection, args.assets, args.output_root,
                    args.name, bootstrap_draws=args.bootstrap_draws, seed=args.seed)
            elif operation == 'diagnose':
                result = diagnose(args.config, args.output_root, max_new_units=args.max_new_units)
            elif operation == 'verify':
                result = verify(args.config, args.output_root)
            else:
                result = approve(args.config, args.review, args.output_root)
            print_summary(result)
        elif args.command.startswith('traces-analysis-'):
            from pilot_eval.trace_analysis import prepare_analysis, run_nulls, verify_nulls, run_analysis, verify_analysis
            operation = args.command.removeprefix('traces-analysis-')
            if operation == 'prepare':
                result = prepare_analysis(args.readouts, args.output_root, args.name)
            elif operation == 'nulls-run':
                result = run_nulls(args.config, args.output_root, max_new_chunks=args.max_new_chunks)
            elif operation == 'nulls-verify':
                result = verify_nulls(args.config, args.output_root)
            elif operation == 'run':
                result = run_analysis(args.config, args.output_root, max_new_units=args.max_new_units)
            else:
                result = verify_analysis(args.config, args.output_root)
            print(result if isinstance(result, Path) else json.dumps(result, indent=2))
        elif args.command=='traces-lens-inspect':
            from pilot_eval.trace_community_lens import inspect_community
            path=inspect_community(args.output_root,args.name,artifact_dir=args.artifact_dir)
            print(path.read_text())
            print('Saved lens status:',path)
        elif args.command.startswith('traces-profile-'):
            from pilot_eval.trace_profile import prepare_profile,run_profile,verify_profile
            if args.command=='traces-profile-prepare':
                print(prepare_profile(args.diagnostic,args.random_config,args.output_root,args.name))
            else:
                result=(run_profile(args.config,args.output_root,dependencies=dependencies)
                        if args.command=='traces-profile-run' else verify_profile(args.config,args.output_root))
                print(json.dumps(result,indent=2))
        elif args.command.startswith('traces-capture-qualification-'):
            from pilot_eval.trace_capture_qualification import prepare, diagnose, verify, approve
            operation = args.command.removeprefix('traces-capture-qualification-')
            if operation == 'prepare':
                result = prepare(args.profile, args.assets, args.output_root, args.name,
                    batch_size=args.batch_size, storage_dtype=args.storage_dtype,
                    bootstrap_draws=args.bootstrap_draws, seed=args.seed)
            elif operation == 'diagnose':
                result = diagnose(args.config, args.output_root, max_new_units=args.max_new_units)
            elif operation == 'verify':
                result = verify(args.config, args.output_root)
            else:
                result = approve(args.config, args.review, args.output_root)
            if isinstance(result, Path):
                print(result)
            else:
                summary = {**result, 'measurements': [
                    {k:v for k,v in row.items() if k != 'bootstrap_draws'}
                    for row in result.get('measurements', [])]}
                print(json.dumps(summary, indent=2))
        elif args.command.startswith('traces-capture-'):
            from pilot_eval.trace_capture_workflow import prepare_diagnostic,run_diagnostic,verify_diagnostic
            if args.command=='traces-capture-prepare':
                print(prepare_diagnostic(args.corpus,args.sft_config,args.grpo_config,args.output_root,args.name))
            else:
                result=(run_diagnostic(args.config,args.output_root,dependencies=dependencies)
                        if args.command=='traces-capture-diagnose' else verify_diagnostic(args.config,args.output_root))
                print(json.dumps(result,indent=2))
        elif args.command.startswith('traces-reference-'):
            from pilot_eval.trace_reference_workflow import prepare_reference_run,verify_reference_run
            if args.command=='traces-reference-prepare':
                print(prepare_reference_run(args.corpus,args.sft_config,args.grpo_config,args.output_root,args.name,dependencies=dependencies))
            elif args.command=='traces-reference-verify':
                print(json.dumps(verify_reference_run(args.config,args.output_root),indent=2))
            else:
                from pilot_eval.trace_corpus import _inside
                from pilot_eval.trace_vocabulary import approve_references
                path=_inside(args.output_root.resolve(),args.config)
                verify_reference_run(path,args.output_root)
                receipt=approve_references(path,reviewer=args.reviewer,notes=args.notes,exposure_status=args.exposure_status)
                from pilot_eval.run import _write_state
                _write_state(path.parent.parent,'frozen',3,3)
                print(json.dumps(receipt,indent=2))
        elif args.command.startswith('traces-review-'):
            from pilot_eval.trace_review import record_exposure, inspect_review, prepare_review
            if args.command == 'traces-review-exposure':
                result = record_exposure(args.output_root, args.name, json.loads(args.event.read_text()))
            elif args.command == 'traces-review-prepare':
                result = prepare_review(args.config, args.output_root, args.name)
            else:
                result = inspect_review(args.output_root, args.name)
            print(json.dumps(result, indent=2))
        elif args.command == 'traces-corpus-prepare':
            from pilot_eval.trace_corpus import prepare_corpus
            print(prepare_corpus(args.source_config, args.output_root, args.name,
                review=json.loads(args.review.read_text()), dependencies=dependencies))
        elif args.command == 'traces-corpus-verify':
            from pilot_eval.trace_corpus import load_corpus
            config, rows = load_corpus(args.config, args.output_root)
            print(json.dumps({'corpus_complete': True, 'examples': len(rows),
                'counts': config['counts'], 'content_tokens': config['content_tokens'],
                'tokenizer': config['tokenizer'], 'selected_new_indices': config['selected_new_indices']}, indent=2))
        elif args.command.startswith('tokens-'):
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
        elif args.command == 'grpo-final-export':
            from pilot_eval.grpo_handoff import export_handoff
            result = export_handoff(args.selection, args.output_root, args.name, args.archive_name)
            print(json.dumps({key: result[key] for key in ['archive_path','archive_bytes']}))
        elif args.command in ['grpo-final-report', 'grpo-final-verify']:
            from pilot_eval.grpo_handoff import report_handoff, verify_handoff
            operation = report_handoff if args.command == 'grpo-final-report' else verify_handoff
            result = operation(args.selection, args.output_root, args.name)
            print(json.dumps({'report_complete': result['report_complete'],
                              'download_files': len(result['download_inventory'])}))
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
