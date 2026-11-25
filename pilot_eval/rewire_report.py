"""Sparse rewiring ticket 18: production report and catalog registration.

Reads only sealed, hash-verified outputs of one finished chain run, from a directory that holds ``c-attribution/``
(the volume copy ``$VOL/c-attribution/sparse-rewiring-<label>`` or a restored HF export):

- the sealed launch record (``c-attribution/launch/<hash>/launch_record.json``);
- qualification and references (``sparse-rewiring/qualification/summary``, ``sparse-rewiring/references/summary``);
- per arm (``sparse-rewiring/runs/<run_id>``): ``arm_evidence`` (plan, step/validation/settle logs, hardening),
  every planned checkpoint (``rewire_checkpoint.verify_arm``), the final frozen survivor export, the greedy
  validation unit at each checkpoint and cohort150 at the final step (D12);
- optional analyses (tickets 10-13), summarized when sealed under ``sparse-rewiring/analysis/``:
  ``rank-comparison/rank_comparison.json``, ``standalone-<arm>/standalone_analysis.json``,
  ``attribution/overlaps.json`` and ``dynamics/`` figures. Missing ones are listed, never invented.

``build`` seals ``sparse-rewiring/report/report.json`` and writes ``report.md`` from the JSON as read back through
the serializer (ticket 18: round trip before publishing). ``register`` adds the two arms and their checkpoints to
``catalog/arms.yaml``, one response set per sealed greedy unit (validation at each checkpoint, references,
cohort150 at the final step), the validation cohort ``gsm8k-train-val128`` if absent, and the new prior uses of both
cohorts. Registration never overwrites an entry and edits YAML text in place (comments survive).

    python -m pilot_eval.rewire_report build --root $VOL/c-attribution/sparse-rewiring-production
    python -m pilot_eval.rewire_report register --root <same> --location sl-0 \\
        --prefix c-attribution/sparse-rewiring-production --catalog catalog
    python -m pilot_eval.catalog verify --root sl-0=$VOL
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from pilot_eval import attribution_units as au

SCHEMA = 'sparse-rewiring/report-v1'
VAL_COHORT = 'gsm8k-train-val128'
TEST_COHORT = 'gsm8k-test-cohort150'
VALIDATION_SOURCE = 'validation/sparse-rewiring-ticket04-v1/validation.exclusions.json'   # local artifacts root
ANALYSES = {
    'rank_comparison': 'rank-comparison/rank_comparison.json',
    'attribution': 'attribution/overlaps.json',
}


class ReportError(ValueError):
    pass


def _j(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sealed(directory: Path) -> Path:
    au.verify_run(directory)
    return directory


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paths(root: str | Path) -> dict[str, Path]:
    from pilot_eval import rewire_chain as rc
    sub = Path(root)
    rw = sub / rc.REWIRE_SUBTREE
    return dict(sub=sub / 'c-attribution', rw=rw, runs=rw / 'runs', report=rw / 'report', analysis=rw / 'analysis')


def launch_record(root: str | Path) -> dict[str, Any]:
    """The run's sealed launch record. A rebind reseals it under a new hash; every copy must name the same run."""
    from pilot_eval import c_attribution_chain as cc
    launch = Path(root) / cc.LAUNCH_DIR
    records = [_j(_sealed(d) / 'launch_record.json') for d in sorted(launch.iterdir()) if d.is_dir()] \
        if launch.is_dir() else []
    if not records:
        raise ReportError(f'no sealed launch record under {launch}')
    strip = lambda r: {k: v for k, v in r.items() if k not in ('pod_id', 'volume_identity', 'inputs')}
    if any(strip(r) != strip(records[0]) for r in records[1:]):
        raise ReportError('the sealed launch records disagree beyond the Pod binding')
    return records[0]


def _greedy(unit: Path) -> dict[str, Any]:
    _sealed(unit)
    scores = _j(unit / 'scores.json')
    return {k: scores[k] for k in ('questions', 'correct', 'accuracy', 'invalid', 'capped', 'tokens_mean',
                                   'tokens_median', 'exploratory')} | {'unit': unit.as_posix()}


def _rel(path: Path, root: Path) -> str:
    return Path(path).relative_to(root).as_posix()


def arm_summary(root: Path, record: Mapping[str, Any], arm: str) -> dict[str, Any]:
    from pilot_eval import rewire_chain as rc
    from pilot_eval import rewire_standalone as rs
    spec = record['arms'][arm]
    arm_dir = paths(root)['runs'] / spec['run_id']
    plan = _j(_sealed(arm_dir / rc.ARM_EVIDENCE_DIR) / 'plan.json')
    art = rs.load_arm_artifacts(arm_dir, plan)   # verify_arm: every planned checkpoint under the sealed plan
    steps = {int(r['step']): r for r in art['step_rows']}
    final = int(record['training']['steps'])
    if sorted(steps) != list(range(0, final + 1)):
        raise ReportError(f'{arm}: steps.jsonl does not hold every step 0..{final}')
    val_ce = {int(r['step']): float(r['val_ce']) for r in art['val_ce_rows'] if 'val_ce' in r}
    checkpoints = {}
    for step in plan['checkpoints']:
        row = steps[int(step)]
        greedy = _greedy(arm_dir / 'eval' / str(step) / 'greedy-val')
        greedy['unit'] = _rel(Path(greedy['unit']), root)
        checkpoints[str(step)] = dict(expected_active=row['expected_active'], nonzero_z_det=row['nonzero_z_det'],
                                      binary_survivors=row['binary_survivors'], ce_loss=row['ce_loss'],
                                      val_ce=val_ce.get(int(step)), greedy_val=greedy)
    cohort = {}
    for step in record['evaluation']['cohort150']['steps']:
        unit = arm_dir / 'eval' / str(step) / 'greedy-cohort150'
        cohort[str(step)] = _greedy(unit) | {'unit': _rel(unit, root)}
    export = art['final_export']
    per_layer = [0] * int(record['model']['layers'])
    for s in export['survivors']:
        per_layer[int(s['layer'])] += 1
    return dict(run_id=spec['run_id'], rank=int(spec['rank']), read_lr=float(spec['read_lr']),
                plan_sha256=plan['plan_sha256'], code_commit=art['arm_seal']['code_commit'],
                runtime_sha256=art['arm_seal']['runtime_sha256'],
                survivors=dict(count=len(export['survivors']), per_layer=per_layer),
                hardening=art['hardening'], checkpoints=checkpoints, cohort150=cohort,
                settle=art['settle_rows'][-1] if art['settle_rows'] else None,
                val_ce_series=[[s, val_ce[s]] for s in sorted(val_ce)])


def _analyses(root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    base = paths(root)['analysis']
    out, missing = {}, []
    names = dict(ANALYSES, **{f'standalone:{a}': f'standalone-{a}/standalone_analysis.json' for a in record['arms']})
    for name, rel in sorted(names.items()):
        path = base / rel
        try:
            _sealed(path.parent)
        except Exception as exc:   # absent or broken: listed, never summarized
            missing.append(f'{name}: {type(exc).__name__}' if path.parent.exists() else name)
            continue
        out[name] = dict(file=_rel(path, root), sha256=_sha(path), body=_j(path))
    figures = sorted(p for p in (base / 'dynamics').glob('*.png')) if (base / 'dynamics').is_dir() else []
    if figures:
        out['dynamics_figures'] = [_rel(p, root) for p in figures]
    else:
        missing.append('dynamics figures')
    return dict(present=out, missing=missing)


def build(root: str | Path) -> dict[str, Any]:
    """Assemble, seal and render the report. Refuses an unfinished or tampered run."""
    from pilot_eval import c_attribution_chain as cc
    root = Path(root)
    p = paths(root)
    record = launch_record(root)
    refs = _j(_sealed(p['rw'] / 'references' / 'summary') / 'references.json')
    qual = _j(_sealed(p['rw'] / 'qualification' / 'summary') / 'qualification.json')
    references = {}
    for name, ref in refs.items():
        unit = p['rw'] / 'references' / f"{name}-step-{ref['step']}" / 'greedy-val'
        references[name] = dict(step=ref['step'], val_ce=ref['val_ce'], greedy_val=_greedy(unit) |
                                {'unit': _rel(unit, root)})
    arms = {arm: arm_summary(root, record, arm) for arm in record['arms']}
    report = dict(schema=SCHEMA, run_id=f"sparse-rewiring-{record['label']}", mode=record['mode'],
                  label=record['label'], launch_record_sha256=cc.record_sha256(record),
                  code_commit=record['code_commit'], final_step=int(record['training']['steps']),
                  freeze_after_step=int(record['training']['freeze_after_step']),
                  l0_lambda=float(record['objective']['l0_lambda']), hardware=record['hardware'],
                  qualification=dict(decision=(qual.get('signoff') or {}).get('decision', qual.get('decision')),
                                     estimates={a: (d or {}).get('estimates') for a, d in
                                                (qual.get('details') or {}).items()}),
                  references=references, arms=arms, analyses=_analyses(root, record),
                  labels=dict(cohort150='exploratory, evaluation only (D12), final step only',
                              validation='GSM8K train, held out from training (ticket 04)',
                              accuracy='greedy, deterministic gates, gsm8k-flexible-v3'))
    au.seal_run(p['report'], {'report.json': report})
    loaded = _j(_sealed(p['report']) / 'report.json')
    if loaded != json.loads(json.dumps(report, default=au._json_default)):
        raise ReportError('report.json did not round-trip through the serializer')
    (p['report'].parent / 'report.md').write_text(markdown(loaded), encoding='utf-8')
    return loaded


def _f(x: Any, d: int = 4) -> str:
    return '—' if x is None else f'{x:.{d}f}' if isinstance(x, float) else str(x)


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    return ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)] + \
        ['| ' + ' | '.join(_f(c) for c in r) + ' |' for r in rows]


def markdown(r: Mapping[str, Any]) -> str:
    """The report page, built only from ``report.json`` as read back."""
    out = [f"# Sparse rewiring on GSM8K: {r['label']} ({r['mode']})", '',
           f"Run `{r['run_id']}`, code `{r['code_commit'][:12]}`, launch record `{r['launch_record_sha256'][:16]}`. "
           f"{r['final_step']} steps, masks frozen after step {r['freeze_after_step']}, λ {r['l0_lambda']}, "
           f"{r['hardware']['count']}× {r['hardware']['gpu']}. Qualification: {r['qualification']['decision']}.", '',
           'Exploratory pilot, single seed. Accuracy is greedy with deterministic gates; cohort150 is evaluation '
           'only and scored once, at the final step (D12).', '', '## Final results', '']
    rows = [[name, ref['step'], '—', ref['val_ce'], f"{ref['greedy_val']['correct']}/{ref['greedy_val']['questions']}",
             '—'] for name, ref in r['references'].items()]
    final = str(r['final_step'])
    for arm, a in r['arms'].items():
        c, c150 = a['checkpoints'][final], a['cohort150'].get(final)
        rows.append([f"{arm} (rank {a['rank']}, LR {a['read_lr']:g})", r['final_step'], a['survivors']['count'],
                     c['val_ce'], f"{c['greedy_val']['correct']}/{c['greedy_val']['questions']}",
                     f"{c150['correct']}/{c150['questions']}" if c150 else '—'])
    out += _table(['Arm', 'Step', 'Survivors', 'Validation CE', 'Validation correct', 'Cohort150 correct'], rows)
    for arm, a in r['arms'].items():
        out += ['', f"## {arm}", '', f"Plan `{a['plan_sha256'][:16]}`. Survivors per layer: "
                f"{', '.join(str(n) for n in a['survivors']['per_layer'])}.", '']
        h = a['hardening'] or {}
        if h:
            out += [f"Hardening at step {h.get('freeze_step')}: expected-on {_f(h.get('pre_expected_active'), 1)}, "
                    f"{h.get('pre_nonzero')} deterministic survivors → {h.get('post_binary')} binary; validation CE "
                    f"{_f(h.get('val_ce_before'))} → {_f(h.get('val_ce_after'))}.", '']
        out += _table(['Step', 'Expected-on', 'z_det > 0', 'Train CE', 'Validation CE', 'Correct', 'Capped',
                       'Invalid', 'Mean tokens'],
                      [[s, c['expected_active'], c['nonzero_z_det'], c['ce_loss'], c['val_ce'],
                        f"{c['greedy_val']['correct']}/{c['greedy_val']['questions']}", c['greedy_val']['capped'],
                        c['greedy_val']['invalid'], c['greedy_val']['tokens_mean']]
                       for s, c in sorted(a['checkpoints'].items(), key=lambda kv: int(kv[0]))])
    an = r['analyses']
    out += ['', '## Analyses (tickets 10–13)', '']
    out += [f"- {name}: `{v['file']}` (sha256 `{v['sha256'][:16]}`)" for name, v in sorted(an['present'].items())
            if name != 'dynamics_figures']
    out += [f"- figure: `{f}`" for f in an['present'].get('dynamics_figures', [])]
    out += [f"- not available: {m}" for m in an['missing']]
    return '\n'.join(out) + '\n'


# --- catalog ---------------------------------------------------------------------------------------

def _append_block(text: str, key: str, block: str) -> str:
    """Append YAML ``block`` (already indented) at the end of the top-level mapping ``key``."""
    lines = text.splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if l.rstrip() == f'{key}:')
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip() and not lines[i][0].isspace()
                and not lines[i].startswith('#')), len(lines))
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    body = ''.join(lines[:end])
    if not body.endswith('\n'):
        body += '\n'
    return body + '\n' + block + ''.join(lines[end:])


def _indent(value: Mapping[str, Any], spaces: int) -> str:
    import yaml
    text = yaml.safe_dump(dict(value), sort_keys=False, width=110, allow_unicode=True)
    return ''.join(' ' * spaces + l + '\n' for l in text.splitlines())


def _entry(rid, *, arm, step, cohort, coverage, unit, root, location, prefix, cap, eos, note):
    path = Path(unit) / 'responses.jsonl'
    full = Path(root) / path
    return dict(id=rid, kind='evaluation', arm=arm, checkpoint_step=step, cohort=cohort, cohort_coverage=coverage,
                question_field='question_id', draws_per_question=1,
                policy=dict(mode='greedy', repetition_penalty=1.0, max_new_tokens=int(cap), eos_token_id=int(eos)),
                scorer=dict(name='gsm8k-flexible-v3'),
                selection=dict(correctness_filtered=False, gold_access=False, all_groups_retained=True,
                               best_of_n=False),
                c_suitability='greedy-only', replay_of=None,
                files=[dict(location=location, path=(Path(prefix) / path).as_posix(), sha256=_sha(full),
                            rows_total=sum(1 for l in full.read_text().splitlines() if l.strip()))],
                evidence=dict(status='verified', checked=['sealed evaluation unit (hash-verified)', 'file sha256',
                                                          'row count']),
                notes=note)


def register(root: str | Path, catalog_dir: str | Path, *, location: str, prefix: str = '') -> dict[str, Any]:
    import yaml
    from pilot_eval import catalog as cat
    from pilot_eval import rewire_checkpoint as rckpt
    root, catalog_dir = Path(root), Path(catalog_dir)
    p = paths(root)
    report = _j(_sealed(p['report']) / 'report.json')
    record = launch_record(root)
    if report['launch_record_sha256'] != __import__('pilot_eval.c_attribution_chain', fromlist=['x']).record_sha256(record):
        raise ReportError('report.json was built from another launch record; rebuild it')
    label, cap, eos = record['label'], record['evaluation']['cap'], record['evaluation']['eos_token_id']
    tag = f'sparse-rewiring-{label}'
    loc = lambda rel: (Path(prefix) / rel).as_posix()

    arms_path, cohorts_path = catalog_dir / 'arms.yaml', catalog_dir / 'cohorts.yaml'
    arms_text, cohorts_text = arms_path.read_text(), cohorts_path.read_text()
    known_arms = yaml.safe_load(arms_text)['arms']
    cohorts = yaml.safe_load(cohorts_text)['cohorts']
    written = dict(arms=[], responses=[], cohorts=[], prior_uses={})

    for arm, a in report['arms'].items():
        if arm in known_arms:
            raise ReportError(f'arm {arm} is already in arms.yaml; registration never overwrites')
        arm_dir = p['runs'] / a['run_id']
        ckpts = {}
        for step in a['checkpoints']:
            f = rckpt.checkpoint_dir(arm_dir, int(step)) / rckpt.STATE_FILE
            ckpts[int(step)] = dict(location=location, path=loc(_rel(f, root)), sha256=_sha(f))
        body = dict(aliases=[a['run_id']],
                    description=(f"Sparse gold-SFT rewiring, rank {a['rank']} read changes with hard-concrete L0 "
                                 f"gates (λ {report['l0_lambda']}), read LR {a['read_lr']:g}, "
                                 f"{report['final_step']} steps on gsm8k-train-512 gold texts ({tag})."),
                    training_data='gold-sft.training-data',
                    training_run=loc(_rel(arm_dir, root)),
                    checkpoints=ckpts,
                    notes=('Checkpoint sha256 = rewire_state.safetensors bytes (read factors and log-alpha only, '
                           'D24), not a LoRA adapter.'))
        arms_text = _append_block(arms_text, 'arms', f'  {arm}:\n' + _indent(body, 4))
        written['arms'].append(arm)

    entries = []
    for name, ref in report['references'].items():
        entries.append(_entry(f"{name}.s{ref['step']}.greedy.val128.{tag}", arm=name,
                              step=ref['step'] if name != 'untuned' else 0, cohort=VAL_COHORT, coverage='full',
                              unit=ref['greedy_val']['unit'], root=root, location=location, prefix=prefix, cap=cap,
                              eos=eos, note=f'{tag}: reference before training (validation, ticket 04 set).'))
    for arm, a in report['arms'].items():
        for step, c in a['checkpoints'].items():
            entries.append(_entry(f'{arm}.s{step}.greedy.val128.{tag}', arm=arm, step=int(step), cohort=VAL_COHORT,
                                  coverage='full', unit=c['greedy_val']['unit'], root=root, location=location,
                                  prefix=prefix, cap=cap, eos=eos,
                                  note=f'{tag}: deterministic gates; validation at a planned checkpoint.'))
        for step, c in a['cohort150'].items():
            entries.append(_entry(f'{arm}.s{step}.greedy.test150.{tag}', arm=arm, step=int(step), cohort=TEST_COHORT,
                                  coverage='full', unit=c['unit'], root=root, location=location, prefix=prefix,
                                  cap=cap, eos=eos, note=f'{tag}: deterministic gates; exploratory (D12), final step.'))
    for e in entries:
        if (catalog_dir / 'responses' / f"{e['id']}.yaml").exists():
            raise ReportError(f"{e['id']} is already registered; registration never overwrites")

    val_ids = None
    if VAL_COHORT not in cohorts:
        first = next(iter(report['references'].values()))
        unit = root / first['greedy_val']['unit']
        val_ids = _j(unit / 'config.json')['question_ids']
        block = dict(description='Ticket 04 frozen GSM8K train validation set for sparse rewiring (128 questions, '
                                 'separate from cohort150 and from gsm8k-train-512).',
                     split='train', dataset='openai/gsm8k main, revision 740312add88f781978c0658806c59bc2815b9866',
                     size=len(val_ids),
                     source=dict(location='local', path=VALIDATION_SOURCE, json_path=['ids']),
                     ids_sha256=cat.ids_digest(val_ids),
                     prior_uses=['sparse rewiring ticket 04: frozen (2026-11-24)'],
                     held_out_from=['gsm8k-train-512', TEST_COHORT])
        cohorts_text = _append_block(cohorts_text, 'cohorts', f'  {VAL_COHORT}:\n' + _indent(block, 4))
        written['cohorts'].append(VAL_COHORT)

    from pilot_eval.c_attribution_report import _append_prior_uses
    arms_txt = ', '.join(f"{arm} (rank {a['rank']})" for arm, a in report['arms'].items())
    uses = {
        VAL_COHORT: [f"{tag} ({report['mode']}): qualification checks, references (untuned, gold-sft step 64), "
                     f"validation CE every {record['cadences']['validation_ce_every']} steps and greedy at every "
                     f"checkpoint for {arms_txt}; read during training, so not held out from these arms"],
        TEST_COHORT: [f"{tag} ({report['mode']}): greedy evaluation (deterministic gates) of {arms_txt} at step "
                      f"{report['final_step']}; all 150 questions; exploratory (D12)"],
    }
    if report['mode'] != 'production':
        uses.pop(TEST_COHORT)
    for cohort, new in uses.items():
        current = (yaml.safe_load(cohorts_text)['cohorts'][cohort].get('prior_uses') or [])
        new = [u for u in new if u not in current]
        if new:
            cohorts_text = _append_prior_uses(cohorts_text, cohort, new)
            if yaml.safe_load(cohorts_text)['cohorts'][cohort]['prior_uses'] != current + new:
                raise ReportError(f'cohorts.yaml did not round-trip after appending prior uses to {cohort}')
        written['prior_uses'][cohort] = new

    if yaml.safe_load(arms_text)['arms'].keys() - known_arms.keys() != set(written['arms']):
        raise ReportError('arms.yaml did not round-trip after appending the arms')
    arms_path.write_text(arms_text)
    cohorts_path.write_text(cohorts_text)
    for e in entries:
        (catalog_dir / 'responses' / f"{e['id']}.yaml").write_text(yaml.safe_dump(e, sort_keys=False, width=110))
        written['responses'].append(e['id'])
    return written


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--root', required=True, help='the run directory that holds c-attribution/')
    r = sub.add_parser('register')
    r.add_argument('--root', required=True)
    r.add_argument('--location', default='sl-0')
    r.add_argument('--prefix', default='', help="root's path relative to the location root")
    r.add_argument('--catalog', default='catalog')
    args = parser.parse_args(argv)
    if args.command == 'build':
        report = build(args.root)
        print(json.dumps(dict(report=str(paths(args.root)['report'] / 'report.json'),
                              missing_analyses=report['analyses']['missing']), indent=2))
        return 0
    print(json.dumps(register(args.root, args.catalog, location=args.location, prefix=args.prefix), indent=2))
    print('next: python -m pilot_eval.catalog verify --root <location>=<path>; commit only when it reports OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
