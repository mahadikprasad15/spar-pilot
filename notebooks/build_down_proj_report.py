"""Parameterized self-contained HTML report builder for Sparse Rewiring down_proj interpretation.

Supports two modes:
  1. ``--mode gold`` (default): builds the 18-survivor Gold SFT (R0 vs R1) standalone HTML report
     from ``--cache`` (default: ``notebooks/cache/down_proj_decoded.pt``), preserving full Gold
     regression compatibility.
  2. ``--mode k2`` (comparative Gold vs Sampling SFT k = 2): works with any K2 survivor count
     (including 1–3 survivors) and no Gold-specific rank fields (``r0_only``/``r1_only``/``in_r0``).
     Renders:
       - Experiment & arm metadata (lambda regime, l0_lambda, seeds, survivor count)
       - Exact Gold vs K2 neuron overlap (shared IDs, Jaccard, layer breakdown)
       - R-Lens cross-cosine similarity matrix & subspace principal angles (degrees + cosines)
       - 18-token GSM8K / style / EOS comparison probe across all 4 lenses (R-Lens, J++, J-Lens, Logit)
       - Activation shift sign (``Δa_j``, active pole ``+v`` vs ``-v``, top contexts) & top-k decoded tokens
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

NOTEBOOKS_DIR = Path(__file__).resolve().parent
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

from _gold_down_proj_report import CACHE_PATH, OUT_PATHS, build_gold_report  # noqa: E402
import decode_down_proj as ddp  # noqa: E402

LENS_LABELS = {
    'rlens': 'R-Lens',
    'jpp': 'J++',
    'jlens': 'J-Lens',
    'logit_lens': 'Logit Lens',
}


def _orthonormal_row_basis(X: torch.Tensor) -> torch.Tensor:
    """Rank-revealing orthonormal basis for the row-space of ``X [n, d]`` via SVD.

    Unlike thin QR, drops zero/near-zero singular directions caused by duplicate or linearly
    dependent rows so principal angles reflect the true subspace dimensions.
    """
    Xf = X.detach().float()
    if Xf.shape[0] == 0:
        return torch.zeros((Xf.shape[1], 0), dtype=torch.float32)
    U, s, _ = torch.linalg.svd(Xf.T, full_matrices=False)
    if s.numel() == 0 or float(s[0].item()) == 0.0:
        return torch.zeros((Xf.shape[1], 0), dtype=torch.float32)
    tol = max(Xf.shape) * torch.finfo(torch.float32).eps * float(s[0].item())
    rank = int((s > tol).sum().item())
    return U[:, :rank]


def compute_principal_angles(A: torch.Tensor, B: torch.Tensor) -> dict[str, Any]:
    """Compute exact subspace principal angles between row-spaces of ``A [n_a, d]`` and ``B [n_b, d]``.

    Uses a rank-revealing SVD basis so duplicate or linearly dependent rows never inject spurious
    zero-angle directions. Handles small survivor sets (including ``n_a = 1, 2, 3``) cleanly.
    """
    Af = A.detach().float()
    Bf = B.detach().float()
    if Af.ndim != 2 or Bf.ndim != 2 or Af.shape[1] != Bf.shape[1]:
        raise ValueError(f'incompatible subspace shapes: {tuple(Af.shape)} vs {tuple(Bf.shape)}')
    Qa = _orthonormal_row_basis(Af)
    Qb = _orthonormal_row_basis(Bf)
    if Qa.shape[1] == 0 or Qb.shape[1] == 0:
        return {
            'rank_a': int(Qa.shape[1]), 'rank_b': int(Qb.shape[1]),
            'cosines': [], 'angles_deg': [], 'mean_cos': 0.0, 'max_cos': 0.0, 'min_angle_deg': 90.0,
        }
    svals = torch.linalg.svdvals(Qa.T @ Qb).clamp(0.0, 1.0)
    cosines = [float(x) for x in svals.tolist()]
    angles = [float(math.degrees(math.acos(max(-1.0, min(1.0, c))))) for c in cosines]
    return {
        'rank_a': int(Qa.shape[1]),
        'rank_b': int(Qb.shape[1]),
        'cosines': [round(c, 4) for c in cosines],
        'angles_deg': [round(a, 2) for a in angles],
        'mean_cos': round(float(np.mean(cosines)), 4),
        'max_cos': round(float(max(cosines)), 4),
        'min_angle_deg': round(float(min(angles)), 2),
    }


def _cosine_cross(A: torch.Tensor, B: torch.Tensor) -> np.ndarray:
    An = A.detach().float()
    Bn = B.detach().float()
    An = An / (An.norm(dim=-1, keepdim=True) + 1e-8)
    Bn = Bn / (Bn.norm(dim=-1, keepdim=True) + 1e-8)
    return (An @ Bn.T).cpu().numpy()


def _merge_shifts(survivors: list[dict[str, Any]], shifts_path: str | Path | None) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    if shifts_path is not None:
        raw = json.loads(Path(shifts_path).read_text(encoding='utf-8'))
        rows = raw['survivors'] if isinstance(raw, dict) and 'survivors' in raw else raw
        for r in rows:
            if 'activation_shift' in r:
                by_key[(int(r['layer']), int(r['idx']))] = dict(r['activation_shift'])
    out: list[dict[str, Any]] = []
    for s in survivors:
        rec = dict(s)
        key = (int(rec['layer']), int(rec['idx']))
        if 'activation_shift' not in rec and key in by_key:
            rec['activation_shift'] = by_key[key]
        if 'activation_shift' in rec and 'active_pole' not in rec['activation_shift']:
            md = float(rec['activation_shift'].get('mean_delta', 0.0))
            rec['activation_shift']['active_pole'] = '+v' if md >= 0.0 else '-v'
        out.append(rec)
    return out


def _fmt_tok(s: str) -> str:
    return repr(s)[1:-1] if any(ch in s for ch in ('\n', '\r', '\t')) else s


def build_comparative_report(
    k2_cache_path: str | Path,
    *,
    gold_cache_path: str | Path | None = None,
    out_paths: Sequence[str | Path],
    k2_shifts_path: str | Path | None = None,
    title: str = 'Sampling SFT (k = 2) vs Gold SFT — Sparse Rewiring down_proj Interpretation',
    experiment_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a self-contained comparative Gold-vs-K2 HTML report for any K2 survivor count (including 1–3)."""
    k2_data = torch.load(str(k2_cache_path), map_location='cpu', weights_only=False)
    k2_surv = _merge_shifts(list(k2_data['survivors']), k2_shifts_path)
    if not k2_surv:
        raise ValueError(f'{k2_cache_path}: no K2 survivors in cache')
    lens_names: list[str] = list(k2_data.get('lens_names') or ['logit_lens', 'jlens', 'jpp', 'rlens'])
    k2_logits = {ln: k2_data['logits_fp16'][ln].float() for ln in lens_names}
    k2_trans = {ln: k2_data['transported_u'][ln].float() for ln in lens_names}
    k2_id_to_str: list[str] = list(k2_data['id_to_str'])
    k2_topk = k2_data['top_k_records']
    k2_probe = k2_data.get('probe') or ddp.compute_probe_zscores(k2_logits, k2_id_to_str)

    gold_data = None
    overlap_summary: dict[str, Any] = {'gold_loaded': False}
    rlens_cross: list[list[float]] = []
    principal_angles: dict[str, Any] = {}
    gold_labels: list[str] = []
    gold_probe_mean: dict[str, list[float]] = {}

    if gold_cache_path is not None and Path(gold_cache_path).is_file():
        gold_data = torch.load(str(gold_cache_path), map_location='cpu', weights_only=False)
        for rev_key in ('model_id', 'model_revision', 'lens_repo', 'lens_revision'):
            kv, gv = k2_data.get(rev_key), gold_data.get(rev_key)
            if kv is not None and gv is not None and kv != gv:
                raise ValueError(
                    f'cannot compare caches with different {rev_key}: k2={kv!r} vs gold={gv!r}'
                )
        gold_surv = list(gold_data['survivors'])
        gold_labels = [s.get('label', f"L{s['layer']} #{s['idx']}") for s in gold_surv]
        k2_set = {(int(s['layer']), int(s['idx'])) for s in k2_surv}
        gold_set = {(int(s['layer']), int(s['idx'])) for s in gold_surv}
        shared = sorted(k2_set & gold_set)
        union = k2_set | gold_set
        overlap_summary = {
            'gold_loaded': True,
            'k2_count': len(k2_set),
            'gold_count': len(gold_set),
            'shared_count': len(shared),
            'jaccard': round(len(shared) / max(len(union), 1), 4),
            'shared_neurons': [f'L{L} #{idx}' for L, idx in shared],
            'k2_only_neurons': [f'L{L} #{idx}' for L, idx in sorted(k2_set - gold_set)],
        }
        primary_lens = 'rlens' if 'rlens' in k2_trans and 'rlens' in gold_data['transported_u'] else lens_names[-1]
        rlens_cross = np.round(
            _cosine_cross(k2_trans[primary_lens], gold_data['transported_u'][primary_lens].float()), 3
        ).tolist()
        principal_angles = {
            'rlens_transported': compute_principal_angles(
                k2_trans[primary_lens], gold_data['transported_u'][primary_lens].float()
            ),
            'rlens_logits': compute_principal_angles(
                k2_logits[primary_lens], gold_data['logits_fp16'][primary_lens].float()
            ),
        }
        gold_logits = {ln: gold_data['logits_fp16'][ln].float() for ln in lens_names if ln in gold_data['logits_fp16']}
        gold_probe = gold_data.get('probe') or ddp.compute_probe_zscores(gold_logits, list(gold_data['id_to_str']))
        gold_col_idx = {c['name']: i for i, c in enumerate(gold_probe['columns'])}
        for ln in lens_names:
            if ln in gold_probe['zscores']:
                gmat = np.asarray(gold_probe['zscores'][ln], dtype=np.float64)
                means = []
                for col in k2_probe['columns']:
                    gi = gold_col_idx.get(col['name'])
                    means.append(round(float(gmat[:, gi].mean()), 3) if gi is not None else 0.0)
                gold_probe_mean[ln] = means

    meta = dict(k2_data.get('experiment_meta') or {})
    if experiment_meta:
        meta.update(dict(experiment_meta))
    meta.setdefault('experiment', 'Sampling SFT (k = 2) Sparse Rewiring')
    meta.setdefault('survivor_count', len(k2_surv))

    # Build neuron payload for interactive JS rendering
    neuron_cards = []
    for i, s in enumerate(k2_surv):
        shift = s.get('activation_shift') or {}
        per_lens_toks = {}
        for ln in lens_names:
            rec = k2_topk[ln][i]
            per_lens_toks[ln] = {
                'pos': [[_fmt_tok(t), round(float(v), 2)] for _, t, v in rec['pos'][:10]],
                'neg': [[_fmt_tok(t), round(float(v), 2)] for _, t, v in rec['neg'][:10]],
            }
        neuron_cards.append({
            'label': s.get('label', f"L{s['layer']} #{s['idx']}"),
            'layer': int(s['layer']),
            'idx': int(s['idx']),
            'group': str(s.get('group', 'k2')),
            'log_alpha': round(float(s['log_alpha']), 4) if 'log_alpha' in s else None,
            'z_det': round(float(s['z_det']), 4) if 'z_det' in s else None,
            'v_norm': round(float(s.get('v_norm', 0.0)), 4),
            'activation_shift': {
                'mean_delta': round(float(shift['mean_delta']), 5) if 'mean_delta' in shift else None,
                'mean_abs_delta': round(float(shift['mean_abs_delta']), 5) if 'mean_abs_delta' in shift else None,
                'frac_pos': round(float(shift['frac_pos']), 4) if 'frac_pos' in shift else None,
                'active_pole': shift.get('active_pole'),
                'top_pos_contexts': shift.get('top_pos_contexts', [])[:3],
                'top_neg_contexts': shift.get('top_neg_contexts', [])[:3],
            } if shift else None,
            'tokens': per_lens_toks,
        })

    payload = {
        'title': title,
        'meta': meta,
        'lens_names': lens_names,
        'lens_labels': LENS_LABELS,
        'neurons': neuron_cards,
        'overlap': overlap_summary,
        'gold_labels': gold_labels,
        'rlens_cross_cos': rlens_cross,
        'principal_angles': principal_angles,
        'probe_columns': k2_probe['columns'],
        'k2_probe': k2_probe['zscores'],
        'gold_probe_mean': gold_probe_mean,
    }
    payload_json = json.dumps(payload, ensure_ascii=False).replace('</', '<\\/')

    meta_badges = ''.join(
        f'<span class="badge"><strong>{html.escape(str(k))}:</strong> {html.escape(str(v))}</span>'
        for k, v in meta.items()
    )
    lens_buttons = ''.join(
        f'<button class="tab-btn{" active" if ln == ("rlens" if "rlens" in lens_names else lens_names[-1]) else ""}" '
        f'data-lens="{html.escape(ln)}">{html.escape(LENS_LABELS.get(ln, ln))}</button>'
        for ln in lens_names
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{html.escape(title)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #1f2937; background: #f8fafc; margin: 0; padding: 24px; line-height: 1.5; }}
  .wrap {{ max-width: 1180px; margin: 0 auto; }}
  h1 {{ font-size: 1.65rem; margin: 0 0 8px; color: #0f172a; }}
  h2 {{ font-size: 1.2rem; margin: 28px 0 10px; border-bottom: 2px solid #e2e8f0; padding-bottom: 4px; color: #0f172a; }}
  .badges {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px; }}
  .badge {{ background: #eef2ff; color: #312e81; padding: 4px 10px; border-radius: 999px; font-size: 0.84rem; }}
  .card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px 18px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }}
  .tabs {{ display: flex; gap: 6px; margin-bottom: 12px; }}
  .tab-btn {{ border: 1px solid #cbd5e1; background: #fff; color: #334155; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 0.85rem; }}
  .tab-btn.active {{ background: #1d4ed8; color: #fff; border-color: #1d4ed8; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
  th, td {{ border: 1px solid #e2e8f0; padding: 6px 8px; text-align: left; vertical-align: top; }}
  th {{ background: #f1f5f9; font-weight: 600; }}
  .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.8rem; }}
  .pos {{ color: #166534; font-weight: 600; }}
  .neg {{ color: #991b1b; font-weight: 600; }}
  .scroll-x {{ overflow-x: auto; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>{html.escape(title)}</h1>
  <div class="badges">{meta_badges}</div>

  <section class="card" id="sec-overlap">
    <h2>1. Gold vs K2 Survivor Overlap &amp; R-Lens Principal Angles</h2>
    <div id="overlap-body"></div>
  </section>

  <section class="card" id="sec-neurons">
    <h2>2. K2 Surviving Neurons — Activation Shift Sign (Δa<sub>j</sub>) &amp; Multi-Lens Decoded Tokens</h2>
    <div class="tabs" id="lens-tabs">{lens_buttons}</div>
    <div class="scroll-x" id="neurons-table"></div>
  </section>

  <section class="card" id="sec-cos">
    <h2>3. R-Lens Cosine Similarity Matrix (K2 Survivors × Gold Survivors)</h2>
    <div class="scroll-x" id="cos-table"></div>
  </section>

  <section class="card" id="sec-probe">
    <h2>4. Targeted 18-Token GSM8K / Style / EOS Comparison Probe (z-scores)</h2>
    <div class="scroll-x" id="probe-table"></div>
  </section>
</div>
<script>
const DATA = {payload_json};
let CURRENT_LENS = DATA.lens_names.includes("rlens") ? "rlens" : DATA.lens_names[DATA.lens_names.length - 1];

function esc(s) {{
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}}

function renderOverlap() {{
  const ov = DATA.overlap;
  const host = document.getElementById("overlap-body");
  if (!ov || !ov.gold_loaded) {{
    host.innerHTML = `<p>No Gold reference cache supplied. K2 survivors: <strong>${{DATA.neurons.length}}</strong>.</p>`;
    return;
  }}
  const paT = DATA.principal_angles.rlens_transported || {{}};
  const paL = DATA.principal_angles.rlens_logits || {{}};
  host.innerHTML = `
    <table>
      <thead><tr><th>Metric</th><th>Value</th><th>Details</th></tr></thead>
      <tbody>
        <tr><td>Exact Neuron Overlap</td><td class="num"><strong>${{ov.shared_count}} / ${{ov.k2_count}}</strong> (Jaccard ${{ov.jaccard}})</td>
            <td class="mono">Shared: ${{ov.shared_neurons.length ? esc(ov.shared_neurons.join(", ")) : "none"}} | K2-only: ${{ov.k2_only_neurons.length ? esc(ov.k2_only_neurons.join(", ")) : "none"}}</td></tr>
        <tr><td>R-Lens Transported Subspace Principal Angles</td><td class="num">min ${{paT.min_angle_deg}}° (max cos ${{paT.max_cos}})</td>
            <td class="mono">angles°: [${{(paT.angles_deg || []).join(", ")}}] | cosines: [${{(paT.cosines || []).join(", ")}}]</td></tr>
        <tr><td>R-Lens Logit Subspace Principal Angles</td><td class="num">min ${{paL.min_angle_deg}}° (max cos ${{paL.max_cos}})</td>
            <td class="mono">angles°: [${{(paL.angles_deg || []).join(", ")}}] | cosines: [${{(paL.cosines || []).join(", ")}}]</td></tr>
      </tbody>
    </table>`;
}}

function renderNeurons() {{
  const host = document.getElementById("neurons-table");
  let rows = "";
  for (const n of DATA.neurons) {{
    const sh = n.activation_shift;
    const poleHtml = sh
      ? `<span class="${{sh.active_pole === '+v' ? 'pos' : 'neg'}}">${{esc(sh.active_pole)}}</span><br>` +
        `<span class="mono">mean Δa=${{sh.mean_delta >= 0 ? '+' : ''}}${{sh.mean_delta}} | |Δa|=${{sh.mean_abs_delta}} | +frac=${{sh.frac_pos}}</span>`
      : `<span class="mono">not provided</span>`;
    const toks = (n.tokens && n.tokens[CURRENT_LENS]) || {{pos: [], neg: []}};
    const posStr = toks.pos.slice(0, 6).map(([t, v]) => `${{esc(t)}} (${{v}})`).join(", ");
    const negStr = toks.neg.slice(0, 6).map(([t, v]) => `${{esc(t)}} (${{v}})`).join(", ");
    rows += `<tr>
      <td><strong>${{esc(n.label)}}</strong><br><span class="mono">${{esc(n.group)}} | logα=${{n.log_alpha ?? "—"}}</span></td>
      <td>${{poleHtml}}</td>
      <td class="mono pos">${{posStr}}</td>
      <td class="mono neg">${{negStr}}</td>
    </tr>`;
  }}
  host.innerHTML = `<table><thead><tr><th>Survivor</th><th>Activation Shift Sign (Δa<sub>j</sub>)</th><th>+v Top Tokens (${{esc(DATA.lens_labels[CURRENT_LENS] || CURRENT_LENS)}})</th><th>−v Top Tokens</th></tr></thead><tbody>${{rows}}</tbody></table>`;
}}

function renderCosMatrix() {{
  const host = document.getElementById("cos-table");
  if (!DATA.rlens_cross_cos || !DATA.rlens_cross_cos.length) {{
    host.innerHTML = "<p>No Gold cache supplied for R-Lens cosine matrix.</p>";
    return;
  }}
  let hdr = "<tr><th>K2 \\\\ Gold</th>" + DATA.gold_labels.map(l => `<th class="mono">${{esc(l)}}</th>`).join("") + "</tr>";
  let body = "";
  DATA.rlens_cross_cos.forEach((row, r) => {{
    body += `<tr><th class="mono">${{esc(DATA.neurons[r].label)}}</th>` +
      row.map(v => `<td class="num mono">${{v >= 0 ? "+" : ""}}${{v.toFixed(3)}}</td>`).join("") + "</tr>";
  }});
  host.innerHTML = `<table><thead>${{hdr}}</thead><tbody>${{body}}</tbody></table>`;
}}

function renderProbe() {{
  const host = document.getElementById("probe-table");
  const cols = DATA.probe_columns || [];
  const mat = (DATA.k2_probe && DATA.k2_probe[CURRENT_LENS]) || [];
  if (!cols.length || !mat.length) {{
    host.innerHTML = "<p>No probe tokens matched this vocabulary.</p>";
    return;
  }}
  let hdr = "<tr><th>Row (" + esc(DATA.lens_labels[CURRENT_LENS] || CURRENT_LENS) + ")</th>" +
    cols.map(c => `<th class="mono">${{esc(c.name)}}</th>`).join("") + "</tr>";
  let body = "";
  mat.forEach((row, r) => {{
    body += `<tr><th class="mono">${{esc(DATA.neurons[r].label)}}</th>` +
      row.map(v => `<td class="num mono">${{v >= 0 ? "+" : ""}}${{Number(v).toFixed(2)}}</td>`).join("") + "</tr>";
  }});
  const gmean = DATA.gold_probe_mean && DATA.gold_probe_mean[CURRENT_LENS];
  if (gmean && gmean.length === cols.length) {{
    body += `<tr><th class="mono">Gold Mean (n=${{DATA.gold_labels.length}})</th>` +
      gmean.map(v => `<td class="num mono">${{v >= 0 ? "+" : ""}}${{Number(v).toFixed(2)}}</td>`).join("") + "</tr>";
  }}
  host.innerHTML = `<table><thead>${{hdr}}</thead><tbody>${{body}}</tbody></table>`;
}}

document.querySelectorAll("#lens-tabs .tab-btn").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll("#lens-tabs .tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    CURRENT_LENS = btn.dataset.lens;
    renderNeurons();
    renderProbe();
  }});
}});

renderOverlap();
renderNeurons();
renderCosMatrix();
renderProbe();
</script>
</body>
</html>
"""

    written: list[Path] = []
    for p in out_paths:
        out_p = Path(p)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(html_doc, encoding='utf-8')
        written.append(out_p)
    return {'written': [str(p) for p in written], 'payload': payload}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Build Gold or comparative Gold-vs-K2 down_proj HTML report')
    parser.add_argument('--mode', choices=['gold', 'k2'], default='gold')
    parser.add_argument('--cache', default=CACHE_PATH, help='decoded down_proj .pt cache path')
    parser.add_argument('--gold-cache', default=None, help='Gold decoded .pt cache for comparative k2 mode')
    parser.add_argument('--k2-shifts', default=None, help='optional extract_survivor_shifts.py JSON output')
    parser.add_argument('--out', nargs='+', default=None, help='one or more output HTML paths')
    parser.add_argument('--title', default='Sampling SFT (k = 2) vs Gold SFT — Sparse Rewiring down_proj Interpretation')
    parser.add_argument('--arm-meta', default=None, help='optional JSON string or path with experiment/arm metadata')
    args = parser.parse_args(argv)

    if args.mode == 'gold':
        out_paths = args.out or OUT_PATHS
        written = build_gold_report(cache_path=args.cache, out_paths=out_paths)
        for p in written:
            print(f'Wrote {p} ({p.stat().st_size / 1024:.1f} KB)')
        return 0

    if not args.out:
        raise SystemExit('--out is required for --mode k2')
    meta = None
    if args.arm_meta:
        mp = Path(args.arm_meta)
        meta = json.loads(mp.read_text(encoding='utf-8')) if mp.is_file() else json.loads(args.arm_meta)
    res = build_comparative_report(
        args.cache,
        gold_cache_path=args.gold_cache,
        out_paths=args.out,
        k2_shifts_path=args.k2_shifts,
        title=args.title,
        experiment_meta=meta,
    )
    for p in res['written']:
        print(f'Wrote {p} ({os.path.getsize(p) / 1024:.1f} KB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
