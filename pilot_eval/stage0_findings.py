"""CPU primary findings through frozen public workflows and sealed comparisons."""
import csv
import io
import json
from pathlib import Path

import numpy as np

from pilot_eval.behaviour_transitions import _bootstrap
from pilot_eval.run import _write_json
from pilot_eval.sft import safe_name
from pilot_eval.stage0_drift import _load as load_drift, _verify as verify_drift
from pilot_eval.stage0_evaluation import _load as load_evaluation, _verify as verify_evaluation, _jsonl
from pilot_eval.stage0_findings_math import CONTRASTS, STATISTICS, PREDICTIONS, LIMITS, behaviour, drift
from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_workflow import _writer, _state, _now
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash, _save_frozen


def _implementation():
    folder = Path(__file__).parent
    return {n: file_hash(folder/n) for n in ['stage0_findings.py', 'stage0_findings_math.py',
        'stage0_findings_render.py', 'stage0_findings_review.py', 'behaviour_audit.py',
        'behaviour_audit_ui.py', 'behaviour_transitions.py', 'behaviour_findings.py', 'behaviour_assisted_report.py']}


def _parents(evaluation_path, drift_path, root):
    evaluation, _, edir = load_evaluation(evaluation_path, root)
    measurement, ddir = load_drift(drift_path, root)
    if (evaluation['execution_sha256'] != measurement['execution_sha256']
            or evaluation['runtime'] != measurement['runtime']
            or evaluation['execution_kind'] != measurement['execution_kind']
            or any(evaluation['adapter_files'][v] != measurement['adapter_files'][v]
                   for v in evaluation['variants'])):
        raise ValueError('Behavior/KL join requires matching frozen execution, runtime, inference batch and adapters.')
    if evaluation['batch_size']!=measurement['batch_size']:
        proof=evaluation.get('batch_agreement')
        if measurement['batch_size']!=1 or not proof:
            raise ValueError('Different schedules require reviewed generation batch agreement and singleton internals.')
        from pilot_eval.stage0_batching import verify
        batch=verify(inside(root,proof['plan_path']),root)
        if (file_hash(inside(root,proof['plan_path']))!=proof['plan_sha256']
                or file_hash(inside(root,proof['receipt_path']))!=proof['receipt_sha256']
                or not batch['eligible_for_review'] or evaluation['batch_size'] not in batch['qualified_sizes']):
            raise ValueError('Batch schedule join evidence changed or is ineligible.')
    return evaluation, measurement, edir, ddir


def prepare(evaluation_path, drift_path, output_root, name):
    root = Path(output_root).resolve(); safe_name(name)
    ep, dp = inside(root, evaluation_path), inside(root, drift_path)
    evaluation, measurement, edir, ddir = _parents(ep, dp, root)
    contrasts,extension=_family(evaluation,root)
    path = root/'plans'/name/'stage0.findings.json'
    directory = root/f'runs/stage0/{name}/findings'
    with _writer(directory):
        if path.exists():
            config, _ = _load(path, root)
            if config['evaluation_path'] != str(ep.relative_to(root)) or config['drift_path'] != str(dp.relative_to(root)):
                raise ValueError('Analysis name already binds another collection.')
            return path
        started = any(any((d/'units').glob('*/complete.json')) or any((d/'attempts').glob('*/start.json'))
                      or (d/'writer.json').exists() for d in [edir, ddir])
        config = dict(schema_version=1, protocol='stage0-primary-findings-v1', name=name,
            run_path=str(directory.relative_to(root)), evaluation_path=str(ep.relative_to(root)),
            evaluation_sha256=file_hash(ep), drift_path=str(dp.relative_to(root)), drift_sha256=file_hash(dp),
            execution_sha256=evaluation['execution_sha256'], runtime=evaluation['runtime'],
            execution_kind=evaluation['execution_kind'], contrasts=contrasts, statistics=STATISTICS,
            inference_schedules=dict(behaviour_batch_size=evaluation['batch_size'],internals_batch_size=measurement['batch_size'],
                batch_agreement=evaluation.get('batch_agreement')),
            noninferiority_margin=-.05, hypotheses=PREDICTIONS, limitations=LIMITS,
            registration=dict(recorded_at=_now(), status='after-collection-start; exploratory' if started else 'before-collection',
                n150_risk='Non-inferiority may be inconclusive with 150 paired items.',
                observed_collection_units=sum(len(list((d/'units').glob('*/complete.json'))) for d in [edir, ddir])),
            implementation=_implementation(),
            optional_work='not required; absent products do not block primary findings',
            conditional_v6='not included; applicability has not been inferred from missing evidence')
        config.update(extension)
        _save_frozen(path, config); _save_frozen(directory/'config.json', config)
        _save_frozen(directory/'meta/run_manifest.json', dict(config_sha256=_hash(config),
            model_calls=0, resources='CPU only', total_units=len(config['contrasts'])))
        _state(directory, 'prepared', 0, len(config['contrasts']))
    return path


def _load(path, root):
    config = json.loads(inside(root, path).read_text())
    if (config.get('protocol') != 'stage0-primary-findings-v1' or config.get('implementation') != _implementation()
            or config.get('statistics') != STATISTICS or config.get('contrasts') != _expected_contrasts(config,root)
            or config.get('noninferiority_margin') != -.05):
        raise ValueError('Analysis code/statistics changed; retain evidence and use a new named plan.')
    directory = inside(root, config['run_path'])
    if (json.loads((directory/'config.json').read_text()) != config
            or json.loads((directory/'meta/run_manifest.json').read_text())['config_sha256'] != _hash(config)):
        raise ValueError('Frozen analysis plan changed.')
    for label in ['evaluation', 'drift']:
        if file_hash(inside(root, config[label+'_path'])) != config[label+'_sha256']:
            raise ValueError('Analysis source plan changed.')
    evaluation, _, _, _ = _parents(inside(root, config['evaluation_path']), inside(root, config['drift_path']), root)
    if evaluation['runtime'] != config['runtime'] or evaluation['execution_sha256'] != config['execution_sha256']:
        raise ValueError('Paired analysis provenance changed.')
    return config, directory


def _sources(config, root):
    evaluation, measurement, edir, ddir = _parents(inside(root, config['evaluation_path']), inside(root, config['drift_path']), root)
    verify_evaluation(evaluation, edir, root)
    measured = verify_drift(measurement, ddir, root)
    rows = [json.loads(line) for line in (edir/'results/responses.jsonl').read_text().splitlines()]
    seals = {label: dict(path=str((d/'complete.json').relative_to(root)), sha256=file_hash(d/'complete.json'))
             for label, d in [('evaluation', edir), ('drift', ddir)]}
    return rows, measured, seals


def _groups(rows, key):
    groups = {}
    for row in rows:
        group = key(row); identifier = row.get('item_id', row.get('id'))
        if identifier in groups.setdefault(group, {}):
            raise ValueError('Duplicate item/document in a paired analysis group.')
        groups[group][identifier] = row
    return groups


def _comparison(contrast, rows, measurement, directory):
    groups = _groups(rows, lambda r: r['variant'])
    reference, edited = groups[contrast['reference']], groups[contrast['edited']]
    ids = sorted(reference)
    if any(reference[i]['prompt'] != edited[i]['prompt'] or reference[i]['gold'] != edited[i]['gold'] for i in ids):
        raise ValueError('Paired prompts/gold differ between interventions.')
    indices, path = _bootstrap(directory, ids)
    summary = dict(**contrast, behaviour=behaviour(reference, edited, ids, indices), behaviour_bootstrap=path, drift=[])
    paired_items = []
    for i in ids:
        a, b = reference[i], edited[i]
        paired_items.append(dict(item_id=i, reference_record_id=a['record_id'], edited_record_id=b['record_id'],
            reference_flexible=int(a['flexible']['correct']), edited_flexible=int(b['flexible']['correct']),
            reference_strict=int(a['strict']['correct']), edited_strict=int(b['strict']['correct']),
            reference_tokens=a['token_count'], edited_tokens=b['token_count'],
            reference_capped=a['capped'], edited_capped=b['capped']))
    docs = _groups(measurement['per_document'], lambda r: (r['variant'], r['corpus'], r['view']))
    paired_documents = []
    cells = sorted({(r['corpus'], r['view']) for r in measurement['per_document']})
    for corpus, view in cells:
        a, b = docs[contrast['reference'], corpus, view], docs[contrast['edited'], corpus, view]
        ids = sorted(a); indices, path = _bootstrap(directory, ids)
        cell = drift(a, b, ids, indices, measurement['modules'])
        cell.update(corpus=corpus, view=view, bootstrap=path)
        summary['drift'].append(cell)
        for i in ids:
            paired_documents.append(dict(id=i, corpus=corpus, view=view, reference=a[i], edited=b[i]))
    return summary, paired_items, paired_documents


def _unit(directory, contrast):
    folder = directory/'units'/contrast['id']
    if not (folder/'complete.json').exists():
        return None
    seal = json.loads((folder/'complete.json').read_text())
    if seal.get('contrast') != contrast or set(seal.get('files', {})) != {'result.json', 'paired-items.jsonl', 'paired-documents.jsonl'}:
        raise ValueError('Comparison seal identity mismatch.')
    for relative, digest in seal['files'].items():
        if file_hash(inside(folder, relative)) != digest:
            raise ValueError('Sealed comparison payload changed.')
    return json.loads((folder/'result.json').read_text())


def _tables(directory, result):
    rows = []
    for c in result['contrasts']:
        for scorer in ['flexible', 'strict']:
            value = c['behaviour'][scorer]
            rows.append(dict(contrast=c['id'], family=c['family'], scorer=scorer, n=c['behaviour']['n'],
                estimate=value['difference']['estimate'], ci_low=value['difference']['interval_95'][0],
                ci_high=value['difference']['interval_95'][1], gains=value['gains'], losses=value['losses'],
                noninferiority=value['noninferiority'], mcnemar_p=value['mcnemar_exact_two_sided']))
    def write(name, values):
        buffer = io.StringIO(); writer = csv.DictWriter(buffer, fieldnames=list(values[0]))
        writer.writeheader(); writer.writerows(values)
        (directory/'results'/name).write_text(buffer.getvalue())
    write('behaviour.csv', rows)
    write('length.csv', [dict(contrast=c['id'], estimate=c['behaviour']['length']['median_paired_change']['estimate'],
        ci_low=c['behaviour']['length']['median_paired_change']['interval_95'][0],
        ci_high=c['behaviour']['length']['median_paired_change']['interval_95'][1],
        shorter_share=c['behaviour']['length']['shorter_share']) for c in result['contrasts']])
    write('kl.csv', [dict(contrast=c['id'], corpus=v['corpus'], view=v['view'], n=v['n'],
        estimate=v['kl']['difference']['estimate'], ci_low=v['kl']['difference']['interval_95'][0],
        ci_high=v['kl']['difference']['interval_95'][1],
        reference_token_mean=v['kl']['reference_token_mean'], edited_token_mean=v['kl']['edited_token_mean'])
        for c in result['contrasts'] for v in c['drift']])
    write('writes.csv', [dict(contrast=c['id'], corpus=v['corpus'], view=v['view'], **m)
        for c in result['contrasts'] for v in c['drift'] for m in v['writes']['module_mean_ratio_changes']])


def run(path, output_root, *, max_new_units=None):
    root = Path(output_root).resolve(); config, directory = _load(path, root)
    if max_new_units is not None and (type(max_new_units) is not int or max_new_units < 0):
        raise ValueError('Invalid new-comparison limit.')
    with _writer(directory):
        if (directory/'complete.json').exists():
            return _verify(config, directory, root)
        rows, measurement, seals = _sources(config, root)
        _save_frozen(directory/'inputs/source-seals.json', seals)
        _save_frozen(directory/'inputs/analysis-runtime.json', dict(numpy_version=np.__version__, resource='CPU statistics'))
        _jsonl(directory/'inputs/responses.jsonl', rows)
        _jsonl(directory/'inputs/drift-documents.jsonl', measurement['per_document'])
        summaries = []; new = 0
        try:
            for contrast in config['contrasts']:
                value = _unit(directory, contrast)
                if value is None:
                    if max_new_units is not None and new >= max_new_units:
                        return _state(directory, 'incomplete', len(summaries), len(config['contrasts']))
                    value, items, docs = _comparison(contrast, rows, measurement, directory)
                    folder = directory/'units'/contrast['id']
                    _write_json(folder/'result.json', value)
                    _jsonl(folder/'paired-items.jsonl', items); _jsonl(folder/'paired-documents.jsonl', docs)
                    _write_json(folder/'complete.json', dict(contrast=contrast,
                        files={p: file_hash(folder/p) for p in ['result.json', 'paired-items.jsonl', 'paired-documents.jsonl']}))
                    new += 1
                summaries.append(value)
                _state(directory, 'running', len(summaries), len(config['contrasts']))
                print(f'Stage 0 findings: {len(summaries)}/{len(config["contrasts"])} comparisons sealed; {contrast["id"]}.', flush=True)
            result = dict(schema_version=1, mode='stage0-conditional-intruder-findings' if config.get('analysis_kind') else 'stage0-primary-findings', analysis_complete=True,
                real_gpu_measurement=config['execution_kind'] == 'real-cuda', execution_kind=config['execution_kind'],
                registration=config['registration'], statistics=config['statistics'], contrasts=summaries,
                numerical_resolution=measurement.get('numerical_resolution'),
                review=dict(state='not-prepared', scope='raw automatic scores; targeted review pending'),
                source_seals=seals, analysis_runtime=dict(numpy_version=np.__version__, resource='CPU statistics'),
                limitations=LIMITS, optional_work=config['optional_work'], conditional_v6=config['conditional_v6'])
            if config.get('analysis_kind'):result['catalogue_scope']=config['catalogue_scope']
            _write_json(directory/'results/results.json', result)
            pairs = []
            for contrast in config['contrasts']:
                for line in (directory/'units'/contrast['id']/'paired-items.jsonl').read_text().splitlines():
                    pairs.append(dict(contrast=contrast['id'], **json.loads(line)))
            _jsonl(directory/'results/paired-items.jsonl', pairs)
            _tables(directory, result)
            from pilot_eval.stage0_findings_render import render
            render(config, result, directory)
            files = {str(p.relative_to(directory)): file_hash(p)
                for sub in ['inputs', 'units', 'results', 'plots', 'bootstrap'] for p in (directory/sub).rglob('*') if p.is_file()}
            files['report.html'] = file_hash(directory/'report.html')
            _write_json(directory/'complete.json', dict(config_sha256=_hash(config), files=files))
            _state(directory, 'complete', len(config['contrasts']), len(config['contrasts']), human_review='pending')
            return result
        except (Exception, KeyboardInterrupt) as exc:
            _state(directory, 'failed', len(summaries), len(config['contrasts']), error=str(exc))
            log = directory/'logs/errors.jsonl'; log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('a') as stream:
                stream.write(json.dumps(dict(error=str(exc), type=type(exc).__name__))+'\n')
            raise



def _verified_files(config, directory, root):
    rows, measurement, seals = _sources(config, root)
    marker = json.loads((directory/'complete.json').read_text())
    required = {'report.html', 'results/results.json', 'results/behaviour.csv', 'results/length.csv',
        'results/kl.csv', 'results/writes.csv', 'results/paired-items.jsonl', 'inputs/source-seals.json',
        'inputs/responses.jsonl', 'inputs/drift-documents.jsonl', 'inputs/analysis-runtime.json'}
    if marker.get('config_sha256') != _hash(config) or not required <= marker.get('files', {}).keys():
        raise ValueError('Primary report completion identity differs.')
    for relative, digest in marker['files'].items():
        if file_hash(inside(directory, relative)) != digest:
            raise ValueError('Primary report payload changed: '+relative)
    if json.loads((directory/'inputs/source-seals.json').read_text()) != seals:
        raise ValueError('Collected source completion evidence changed.')
    read = lambda name: [json.loads(line) for line in (directory/name).read_text().splitlines()]
    if read('inputs/responses.jsonl') != rows or read('inputs/drift-documents.jsonl') != measurement['per_document']:
        raise ValueError('Saved analysis input snapshots do not reconcile with the verified collections.')
    runtime = json.loads((directory/'inputs/analysis-runtime.json').read_text())
    if runtime['numpy_version'] != np.__version__:
        raise ValueError('CPU statistics NumPy version differs from this saved analysis; use its recorded version.')
    result = json.loads((directory/'results/results.json').read_text())
    if result.get('analysis_runtime') != runtime: raise ValueError('Saved statistics runtime identity changed.')
    if (result.get('execution_kind') != config['execution_kind']
            or result.get('real_gpu_measurement') != (config['execution_kind'] == 'real-cuda')
            or result.get('analysis_complete') is not True or result.get('registration') != config['registration']
            or result.get('limitations') != LIMITS):
        raise ValueError('Analysis execution/registration/claim scope differs from the frozen plan.')
    if config.get('analysis_kind') and (result.get('mode')!='stage0-conditional-intruder-findings' or result.get('catalogue_scope')!=config['catalogue_scope']):
        raise ValueError('Conditional extension claim scope changed.')
    return result, rows, measurement, seals


def _verify(config, directory, root):
    result, rows, measurement, seals = _verified_files(config, directory, root)
    values = []
    for contrast in config['contrasts']:
        saved = _unit(directory, contrast)
        if saved is None:
            raise ValueError('Missing declared comparison.')
        expected, items, docs = _comparison(contrast, rows, measurement, directory)
        folder = directory/'units'/contrast['id']
        read = lambda name: [json.loads(line) for line in (folder/name).read_text().splitlines()]
        if saved != expected or read('paired-items.jsonl') != items or read('paired-documents.jsonl') != docs:
            raise ValueError('Paired outcomes/bootstrap intervals do not reconcile with source records.')
        values.append(saved)
    if (result.get('numerical_resolution')!=measurement.get('numerical_resolution')
            or result['contrasts'] != values or result['source_seals'] != seals or result['statistics'] != STATISTICS):
        raise ValueError('Primary aggregate disagrees with saved comparisons.')
    return result


def verify(path, output_root):
    root = Path(output_root).resolve(); config, directory = _load(path, root)
    return _verify(config, directory, root)


def status(path, output_root):
    root = Path(output_root).resolve(); config, directory = _load(path, root)
    completed = sum(_unit(directory, c) is not None for c in config['contrasts'])
    complete = (directory/'complete.json').exists()
    if complete: _verify(config, directory, root)
    return dict(analysis_complete=complete, completed_units=completed, total_units=len(config['contrasts']),
        execution_kind=config['execution_kind'], registration=config['registration'], report=str(directory/'report.html'))


def report(path, output_root):
    """Publish an immutable review-status snapshot; preserve the original raw report."""
    root = Path(output_root).resolve(); config, directory = _load(path, root)
    raw = _verify(config, directory, root)
    from pilot_eval.stage0_findings_review import overlay
    review, scores, evidence = overlay(path, root)
    identity = dict(raw_report_sha256=file_hash(directory/'complete.json'), review=review,
                    evidence_sha256=_hash(evidence))
    target = directory/'reports'/('review-'+_hash(identity)[:20])
    with _writer(target):
        if (target/'complete.json').exists():
            seal = json.loads((target/'complete.json').read_text())
            if seal.get('identity') != identity: raise ValueError('Review report identity changed.')
            for p, digest in seal['files'].items():
                if file_hash(inside(target, p)) != digest: raise ValueError('Review report payload changed.')
            return dict(report=str(target/'report.html'), review=review, reused=True)
        result = json.loads(json.dumps(raw)); result['review'] = review; result['evidence_base'] = '../../'
        if scores:
            rows = [json.loads(line) for line in (directory/'inputs/responses.jsonl').read_text().splitlines()]
            groups = _groups(rows, lambda r: r['variant'])
            for c in result['contrasts']:
                before, after = groups[c['reference']], groups[c['edited']]
                ids = sorted(before); indices, _ = _bootstrap(directory, ids)
                c['behaviour'] = behaviour(before, after, ids, indices, overlay=scores)
        _save_frozen(target/'config.json', dict(**config, review_snapshot=identity))
        _write_json(target/'results/results.json', result)
        _jsonl(target/'results/review-evidence.jsonl', evidence)
        import shutil
        shutil.copyfile(directory/'results/paired-items.jsonl', target/'results/paired-items.jsonl')
        _tables(target, result)
        from pilot_eval.stage0_findings_render import render
        render(config, result, target)
        files = {str(p.relative_to(target)): file_hash(p) for p in target.rglob('*')
                 if p.is_file() and 'writer.json' != p.name}
        _write_json(target/'complete.json', dict(identity=identity, files=files))
        _state(target, 'complete', 1, 1, human_review=review['state'])
    return dict(report=str(target/'report.html'), review=review, original_raw_report=str(directory/'report.html'))


def _family(evaluation,root):
    if evaluation.get('collection_kind')!='conditional-intruders':return CONTRASTS,{}
    from pilot_eval.stage0_conditional import binding,contrasts
    frozen=binding(evaluation['mixed_catalogue_path'],root)
    return contrasts(evaluation['variants']),dict(analysis_kind='conditional-intruders',
        variant_definitions=evaluation['variants'],catalogue_scope=frozen['catalogue_scope'],
        hypotheses=['Halving flagged directions may change behavioral outcomes and fixed-context drift; compare each arm against its original adapter and matched controls.',
                    'Report full, matched and original-pair ratio sensitivity as distinct comparisons; no automatic mechanism claim.'],
        conditional_v6='Separately identified conditional extension; full/matched/ratio scopes remain distinct',
        optional_work='Primary findings are independently runnable; this extension is not a primary-report prerequisite')


def _expected_contrasts(config,root):
    if config.get('analysis_kind')!='conditional-intruders':return CONTRASTS
    evaluation,_,_=load_evaluation(inside(root,config['evaluation_path']),root)
    family,extension=_family(evaluation,root)
    if any(config.get(k)!=v for k,v in extension.items()):raise ValueError('Conditional findings family or scope changed.')
    return family
