"""Gold-independent intended-answer proposals; never writes human label journals.

Run with ``python -m pilot_eval.behaviour_assistance --help``. Inputs and outputs
are explicit; source hashes bind a separate assisted-review artifact.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
from datetime import datetime, timezone

from pilot_eval.scoring import _NUMBER, _numeric_value

TOKEN = re.compile(r'(?<![\w.])(' + _NUMBER + r')(?![\w.]\d)')


def _single(text):
    matches = list(TOKEN.finditer(text))
    try:
        values = {_numeric_value(m.group(1)) for m in matches}
    except (ValueError, ZeroDivisionError):
        return None
    return matches[-1].group(1) if len(values) == 1 else None


def propose_answer(response):
    """Use only response text, not questions, gold, scorer outcomes or arithmetic."""
    text = response.strip()
    result = dict(answer=None, rule='unresolved', evidence=text[-500:], human_confirmed=False)
    # Repetition/malformed calculation tails do not establish an intended answer.
    if re.search(r'(?:>>\s*\d+){3,}', text[-500:]):
        return result
    markers = list(re.finditer(r'####\s*(' + _NUMBER + r')\s*[.!]?\s*$', text))
    if markers:
        m = markers[-1]
        return dict(result, answer=m.group(1), rule='terminal_marker', evidence=m.group(0))
    phrases = list(re.finditer(r'final answer\s*:|(?:the\s+)?answer is\s*:?', text, re.I))
    if phrases:
        region = text[phrases[-1].start():]
        answer = _single(region)
        return dict(result, answer=answer, rule='final_phrase' if answer else 'unresolved', evidence=region)
    boxes = list(re.finditer(r'\\boxed\{(' + _NUMBER + r')\}', text))
    if boxes and not TOKEN.search(text[boxes[-1].end():]):
        m = boxes[-1]
        return dict(result, answer=m.group(1), rule='terminal_box', evidence=text[max(0, m.start()-100):])
    region = re.split(r'\n\s*\n', text)[-1]
    if re.search(r'\b(therefore|thus|so|in total|altogether)\b', region, re.I):
        answer = _single(region)
        return dict(result, answer=answer, rule='conclusion_single_value' if answer else 'unresolved', evidence=region)
    return result


def build_assistance(packet, backup, comparisons):
    labels = {row['response_id']: row for row in backup['labels']}
    ids = {row['response_id'] for row in packet}
    if len(labels) != len(backup['labels']) or set(labels) != ids:
        raise ValueError('backup IDs do not match packet')
    rows = []
    for index, item in enumerate(packet, 1):
        label = labels[item['response_id']]
        proposal = None
        reasons = []
        if label['status'] == 'numeric':
            if label.get('answer'):
                _numeric_value(label['answer'])
                proposal = dict(answer=label['answer'], rule='existing_human_transcription',
                                evidence=label.get('evidence', ''), human_confirmed=True)
            else:
                proposal = propose_answer(item['response'])
            if proposal['answer'] is None:
                reasons.append('no_unique_textual_proposal')
            for ref in comparisons.get(item['response_id'], []):
                if ref['capped']:
                    reasons.append('capped_response')
                if proposal['answer'] is None:
                    continue
                for scorer, score in ref['scores'].items():
                    # Strict rejection mainly measures required formatting; flag an
                    # actual strict numeric disagreement, not every missing ####.
                    if score['status'] != 'valid':
                        if scorer != 'strict':
                            reasons.append(scorer + '_unextractable')
                    elif _numeric_value(score['extracted']) != _numeric_value(proposal['answer']):
                        reasons.append(scorer + '_disagreement')
        rows.append(dict(index=index, response_id=item['response_id'], question=item['question'],
                         response=item['response'], human_label=label, proposal=proposal,
                         review_reasons=sorted(set(reasons))))
    return dict(schema_version=1, packet_sha256=backup['packet_sha256'],
                reviewer=backup['reviewer'], base_revision=backup['base_revision'],
                scope='assisted proposals; not frozen human labels or a completed human accuracy audit',
                proposal_inputs='response text only; scorer extractions used afterward for triage', rows=rows)


def page(result):
    esc = lambda x: html.escape(str(x))
    flagged = [row for row in result['rows'] if row['review_reasons']]
    unflagged = [row for row in result['rows'] if row['proposal'] and not row['proposal']['human_confirmed'] and not row['review_reasons']]
    def card(row):
        proposal = row['proposal'] or {}
        return (f'<details><summary>Response {row["index"]} · proposed {esc(proposal.get("answer") or "unresolved")} · '
                f'{esc(", ".join(row["review_reasons"]) or "no automatic flag")}</summary>'
                f'<p><b>Proposal rule:</b> {esc(proposal.get("rule"))}</p>'
                f'<p><b>Evidence:</b> {esc(proposal.get("evidence", ""))}</p>'
                f'<h3>Question</h3><pre>{esc(row["question"])}</pre>'
                f'<h3>Response</h3><pre>{esc(row["response"])}</pre>'
                f'<p><b>Original human classification:</b> {esc(row["human_label"]["status"])}</p></details>')
    return ('<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Assisted answer review</title><style>body{max-width:1050px;margin:40px auto;padding:0 24px;background:#f7f8f4;color:#203c42;font:16px/1.6 system-ui}'
            'h1,h2{line-height:1.2}details{background:white;border:1px solid #ccd8d2;border-radius:8px;padding:16px;margin:12px 0}'
            'summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}.note{background:#e5eee9;padding:18px}</style>'
            '<h1>Intended answers: assisted triage</h1><p class="note">Your original classifications are preserved. '
            'These numbers are automatic proposals, not additional human labels. No gold answers, methods or scorer verdicts are displayed. '
            'No automatic flag does not guarantee correctness. This page does not save or freeze labels.</p>'
            f'<p>{len(result["rows"])} original classifications · {len(flagged)} flagged cases · {len(unflagged)} unconfirmed proposals without flags.</p>'
            '<h2>Review these cases first</h2><p>Open each case and identify its intended answer. You can send corrections using the response number. '
            'Scorer disagreements are triage signals, not proof that the proposal is right.</p>'
            + ''.join(card(row) for row in flagged)
            + '<h2>Other automatic proposals</h2><p>Available for inspection; remain labelled automatic unless explicitly confirmed.</p>'
            + ''.join(card(row) for row in unflagged) + '</html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-config', type=Path, required=True)
    parser.add_argument('--backup', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        raise ValueError('name must be a single directory name')
    root = args.output_root.resolve()
    config = json.loads(args.audit_config.read_text())
    directory = (root / config['run_path']).resolve()
    if not directory.is_relative_to(root):
        raise ValueError('audit path outside artifact root')
    packet_path = directory / 'reviewer/packet.json'
    mapping_path = directory / 'private/mapping.json'
    backup = json.loads(args.backup.read_text())
    if backup['packet_sha256'] != config['packet_sha256']:
        raise ValueError('wrong response packet')
    if hashlib.sha256(packet_path.read_bytes()).hexdigest() != config['packet_sha256']:
        raise ValueError('response packet bytes changed')
    packet = json.loads(packet_path.read_text())
    mapping = json.loads(mapping_path.read_text())
    # Do not pass gold, correctness, method or checkpoint to the proposal/triage builder.
    comparisons = {row['response_id']: [dict(capped=ref['record']['capped'], scores={
        name: {key: score.get(key) for key in ('status', 'extracted')}
        for name, score in ref['record']['scores'].items()}) for ref in row['references']] for row in mapping}
    result = build_assistance(packet, backup, comparisons)
    sources = [args.audit_config, args.backup, packet_path, mapping_path, Path(__file__)]
    manifest = dict(schema_version=1, name=args.name, mode='cpu-saved-response-assistance',
                    sources={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    human_labels_modified=False, reveal_performed=False)
    target = directory / 'assistance' / args.name
    outputs = {'config.json': json.dumps(manifest, indent=2)+'\n',
               'results.json': json.dumps(result, indent=2)+'\n', 'review.html': page(result)}
    target.mkdir(parents=True, exist_ok=True)
    for filename, text in outputs.items():
        path = target / filename
        if path.exists() and path.read_text() != text:
            raise ValueError('existing assistance differs; use a new name')
    for filename, text in outputs.items():
        (target / filename).write_text(text)
    status = dict(state='awaiting-targeted-review', updated_at=datetime.now(timezone.utc).isoformat(),
                  classifications=len(result['rows']), flagged=sum(bool(r['review_reasons']) for r in result['rows']),
                  unconfirmed_proposals=sum(bool(r['proposal']) and not r['proposal']['human_confirmed'] for r in result['rows']),
                  files={f:hashlib.sha256((target/f).read_bytes()).hexdigest() for f in outputs})
    (target / 'status.json').write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps(dict(directory=str(target), **status), indent=2))


if __name__ == '__main__':
    main()
