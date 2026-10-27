"""Targeted single-reviewer intended-answer audit for the Stage 0 catalogue."""
import hashlib
import json
import secrets
from pathlib import Path

from pilot_eval.behaviour_audit import (_labels, _draft, _export_draft, _frozen, _seal, _check_seal,
                                        _json_bytes, _atomic_json, LIMITS)
from pilot_eval.behaviour_audit_ui import review_page
from pilot_eval.behaviour_breakdown import _extract_last_number
from pilot_eval.run import _write_json
from pilot_eval.scoring import _numeric_value
from pilot_eval.stage0_findings_math import bin_
from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_workflow import _writer, _now
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash, _save_frozen

PACKET_FILES = ['config.json', 'reviewer/packet.json', 'reviewer/review.html', 'private/mapping.json',
                'private/selection.json', 'private/reused.json', 'private/preparation.json']


def _parent(path, root):
    from pilot_eval.stage0_findings import _load, _verify
    config, directory = _load(path, root)
    _verify(config, directory, root)
    rows = [json.loads(line) for line in (directory/'inputs/responses.jsonl').read_text().splitlines()]
    return config, directory, rows


def _value(score):
    return _numeric_value(score['extracted']) if score['status'] == 'valid' else None


def selection(rows, contrasts):
    """Every answer-changing/invalid or extraction-suspect case; strict-only format is automatic."""
    groups = {}
    reasons = {}
    for row in rows:
        groups.setdefault(row['variant'], {})[row['item_id']] = row
        last = _extract_last_number(row['text'], row['capped'])
        why = []
        if row['flexible']['status'] != 'valid':
            why.append('flexible_unextractable')
        if (row['flexible']['status'] == 'valid') != (last['status'] == 'valid') or _value(row['flexible']) != _value(last):
            why.append('last_number_diagnostic_disagrees')
        if why: reasons.setdefault(row['record_id'], set()).update(why)
    for contrast in contrasts:
        for item, a in groups[contrast['reference']].items():
            b = groups[contrast['edited']][item]
            if _value(a['flexible']) != _value(b['flexible']) or bin_(a['flexible']) != bin_(b['flexible']):
                for row in [a, b]:
                    reasons.setdefault(row['record_id'], set()).add('answer_change:'+contrast['id'])
    return {k: sorted(v) for k, v in sorted(reasons.items())}


def _historical(root, path, revision):
    if path is None: return [], None
    from pilot_eval.behaviour_audit import _load, _frozen, _revealed_source
    config, directory = _load(inside(root, path), root)
    _frozen(config, directory)
    result, digest = _revealed_source(directory, revision)
    rows = []
    for row in result['responses']:
        rows.append(dict(question=row['question'], response=row['response'], label=row['label'],
            reviewer=result['reviewer'], source=dict(config_path=str(inside(root, path).relative_to(root)),
                revision=revision, result_sha256=digest, response_id=row['response_id'],
                post_reveal_exposed=result['post_reveal_revision'])))
    return rows, dict(config_path=str(inside(root, path).relative_to(root)), revision=revision, result_sha256=digest)


def identical_label(row, candidates):
    matches = [c for c in candidates if c['question'] == row['prompt'] and c['response'] == row['text']]
    if not matches: return None
    labels = [_labels([dict(c['label'], response_id='same')], ['same'], complete=True)[0] for c in matches]
    if len({_hash(dict(status=l['status'], answer=l['answer'])) for l in labels}) != 1:
        raise ValueError('Conflicting confirmed labels on identical response text; resolve the source explicitly.')
    return matches[0]


def prepare(path, output_root, *, reuse_audit=None, reuse_revision=None):
    root = Path(output_root).resolve(); config, directory, rows = _parent(path, root)
    audit = directory/'audit'; reasons = selection(rows, config['contrasts'])
    candidates, history = _historical(root, reuse_audit, reuse_revision)
    with _writer(audit):
        if (audit/'packet.complete.json').exists():
            _load(config, directory, root)
            saved = json.loads((audit/'config.json').read_text())
            if saved['historical_source'] != history:
                raise ValueError('Audit already binds another historical label source.')
            return status(path, root)
        source = dict(findings_sha256=file_hash(directory/'complete.json'), historical_source=history)
        groups = {}; reused = []
        for row in rows:
            if row['record_id'] not in reasons: continue
            confirmed = identical_label(row, candidates)
            if confirmed is not None:
                reused.append(dict(record_id=row['record_id'], prompt=row['prompt'], text=row['text'],
                    label=confirmed['label'], reviewer=confirmed['reviewer'], source=confirmed['source']))
                continue
            key = _hash([row['prompt'], row['text']])
            groups.setdefault(key, []).append(row)
        construction = audit/'private/preparation.json'
        if construction.exists():
            checkpoint = json.loads(construction.read_text())
            if checkpoint['source'] != source or checkpoint['reasons'] != reasons or checkpoint['reused'] != reused:
                raise ValueError('Audit preparation sources changed.')
            packet, mapping = checkpoint['packet'], checkpoint['mapping']
        else:
            packet = []; mapping = []
            for members in groups.values():
                response_id = 'r_'+secrets.token_hex(12)
                packet.append(dict(response_id=response_id, question=members[0]['prompt'], response=members[0]['text']))
                mapping.append(dict(response_id=response_id, record_ids=[r['record_id'] for r in members]))
            secrets.SystemRandom().shuffle(packet)
            _save_frozen(construction, dict(source=source, reasons=reasons, reused=reused, packet=packet, mapping=mapping))
        packet_hash = hashlib.sha256(_json_bytes(packet)).hexdigest()
        audit_config = dict(packet_sha256=packet_hash, findings_sha256=source['findings_sha256'], historical_source=history)
        _save_frozen(audit/'config.json', audit_config)
        _save_frozen(audit/'private/selection.json', reasons)
        _save_frozen(audit/'private/reused.json', reused)
        _save_frozen(audit/'private/mapping.json', mapping)
        _save_frozen(audit/'reviewer/packet.json', packet)
        (audit/'reviewer/review.html').write_bytes(review_page(packet, packet_hash))
        _seal(audit, 'packet.complete.json', PACKET_FILES)
        _write_json(audit/'meta/status.json', dict(state='awaiting-review', selected_records=len(reasons),
            unique_responses=len(packet), reused_records=len(reused), recorded_at=_now()))
    return status(path, root)


def _load(config, directory, root):
    audit = directory/'audit'
    _check_seal(audit, 'packet.complete.json', PACKET_FILES)
    value = json.loads((audit/'config.json').read_text())
    if value['findings_sha256'] != file_hash(directory/'complete.json'):
        raise ValueError('Audit source report changed.')
    if value['historical_source'] is not None:
        history = value['historical_source']
        _, current = _historical(root, history['config_path'], history['revision'])
        if current != history: raise ValueError('Confirmed historical label source changed.')
    return value, audit


def save(path, output_root, labels_path):
    root = Path(output_root).resolve(); config, directory, _ = _parent(path, root)
    value, audit = _load(config, directory, root)
    patch = json.loads(Path(labels_path).read_text())
    if (not isinstance(patch, dict) or set(patch)-{'packet_sha256', 'reviewer', 'base_revision', 'revision', 'labels'}
            or patch.get('packet_sha256') != value['packet_sha256']
            or not isinstance(patch.get('reviewer'), str) or not patch['reviewer'].strip()
            or type(patch.get('base_revision')) is not int
            or ('revision' in patch and patch['revision'] != patch['base_revision'])):
        raise ValueError('Labels require the exact packet, reviewer and saved base revision.')
    with _writer(audit):
        if (audit/'labels/frozen.complete.json').exists():
            raise ValueError('Labels are frozen; preserve them and start an explicitly exposed new audit for corrections.')
        current, ids, previous = _draft(value, audit)
        digest = _hash(patch)
        if current.get('update_sha256') == digest:
            _export_draft(audit, current); return dict(revision=current['revision'], reused=True)
        if patch['base_revision'] != current['revision']:
            raise ValueError('Stale draft; import the saved reviewer/draft.json before continuing.')
        if current['reviewer'] is not None and current['reviewer'] != patch['reviewer']:
            raise ValueError('Single-reviewer identity changed.')
        updates = _labels(patch.get('labels'), ids)
        merged = {r['response_id']: r for r in current['labels']}
        merged.update({r['response_id']: r for r in updates})
        revision = current['revision']+1
        saved = dict(packet_sha256=value['packet_sha256'], reviewer=patch['reviewer'], revision=revision,
            previous_sha256=previous, update_sha256=digest, labels=[merged[i] for i in ids])
        relative = f'drafts/{revision:06d}.json'
        _save_frozen(audit/relative, saved); _seal(audit, f'drafts/{revision:06d}.complete.json', [relative])
        _export_draft(audit, saved)
    return status(path, root)


def freeze(path, output_root):
    root = Path(output_root).resolve(); config, directory, _ = _parent(path, root)
    value, audit = _load(config, directory, root)
    with _writer(audit):
        if (audit/'labels/frozen.complete.json').exists():
            _frozen(value, audit); return dict(state='labels-frozen')
        draft, ids, digest = _draft(value, audit)
        labels = _labels(draft['labels'], ids, complete=True)
        if ids and draft['reviewer'] is None: raise ValueError('No human reviewer has saved labels.')
        frozen = dict(packet_sha256=value['packet_sha256'], reviewer=draft['reviewer'],
            draft_revision=draft['revision'], draft_sha256=digest, labels=labels, label_scope='before-reveal')
        _save_frozen(audit/'labels/frozen.json', frozen)
        _seal(audit, 'labels/frozen.complete.json', ['labels/frozen.json'])
    return dict(state='labels-frozen', unique_responses=len(ids))


def status(path, output_root):
    root = Path(output_root).resolve(); config, directory, _ = _parent(path, root)
    if not (directory/'audit/packet.complete.json').exists():
        return dict(state='not-prepared', scope=LIMITS)
    value, audit = _load(config, directory, root)
    draft, ids, _ = _draft(value, audit)
    _export_draft(audit, draft)
    frozen = (audit/'labels/frozen.complete.json').exists()
    if frozen: _frozen(value, audit)
    counts = {s: sum(r['status'] == s for r in draft['labels']) for s in ['numeric', 'absent', 'ambiguous', 'pending']}
    return dict(state='labels-frozen' if frozen else 'awaiting-review', revision=draft['revision'],
        unique_responses=len(ids), label_counts=counts, selected_records=len(json.loads((audit/'private/selection.json').read_text())),
        reused_records=len(json.loads((audit/'private/reused.json').read_text())), scope=LIMITS,
        reviewer_page=str(audit/'reviewer/review.html'), saved_draft=str(audit/'reviewer/draft.json'))


def overlay(path, output_root):
    """Only confirmed numeric answers change scores; caps retain the original zero rule."""
    root = Path(output_root).resolve(); config, directory, rows = _parent(path, root)
    summary = status(path, root)
    if summary['state'] == 'not-prepared': return summary, {}, []
    value, audit = _load(config, directory, root)
    by_id = {r['record_id']: r for r in rows}; confirmed = []
    for row in json.loads((audit/'private/reused.json').read_text()):
        current = by_id[row['record_id']]
        if current['text'] != row['text'] or current['prompt'] != row['prompt']:
            raise ValueError('Confirmed label response text no longer matches exactly.')
        confirmed.append(dict(record_id=row['record_id'], label=row['label'], origin='identical-text historical reuse', source=row['source']))
    if summary['state'] == 'labels-frozen':
        labels = {r['response_id']: r for r in _frozen(value, audit)['labels']}
        for group in json.loads((audit/'private/mapping.json').read_text()):
            for record_id in group['record_ids']:
                confirmed.append(dict(record_id=record_id, label=labels[group['response_id']], origin='new frozen intended-answer review',
                    source=dict(response_id=group['response_id'], packet_sha256=value['packet_sha256'],
                                frozen_sha256=file_hash(audit/'labels/frozen.json'))))
    scores = {}; evidence = []
    for item in confirmed:
        row = by_id[item['record_id']]; label = item['label']; numeric = label['status'] == 'numeric'
        intended_correct = (_numeric_value(label['answer']) == _numeric_value(row['gold'].rsplit('####', 1)[-1].strip())) if numeric else None
        if numeric: scores[row['record_id']] = int(intended_correct and not row['capped'])
        evidence.append(dict(**item, intended_answer_correct=intended_correct, capped=row['capped'],
            applied=bool(numeric), overlay_correct=scores.get(row['record_id']), original_correct=row['flexible']['correct'],
            question=row['prompt'], response=row['text'], gold=row['gold']))
    summary['numeric_overlay_records'] = len(scores)
    summary['nonnumeric_scope'] = 'Absent/ambiguous labels are reported; raw scores retained rather than inventing a numeric correction.'
    return summary, scores, evidence


def reveal(path, output_root):
    root = Path(output_root).resolve(); config, directory, _ = _parent(path, root)
    value, audit = _load(config, directory, root)
    _frozen(value, audit)
    summary, scores, evidence = overlay(path, root)
    result = dict(review=summary, confirmed_evidence=evidence, numeric_overlay=scores, reasoning_validated=False)
    with _writer(audit):
        _save_frozen(audit/'reveal/results.json', result)
        _seal(audit, 'reveal/complete.json', ['reveal/results.json'])
    return dict(state='revealed', evidence_path=str(audit/'reveal/results.json'), numeric_overlay_records=len(scores))
