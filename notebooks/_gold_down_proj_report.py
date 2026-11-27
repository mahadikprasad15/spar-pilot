"""Build self-contained HTML report for Sparse Rewiring down_proj interpretation.

Reads cache/down_proj_decoded.pt and writes identical standalone HTML reports to:
  - /Users/prasadmahadik/.gemini/antigravity/scratch/spar-pilot/docs/sparse-rewiring-down-proj-report.html
  - /Users/prasadmahadik/.gemini/antigravity/scratch/spar-pilot/notebooks/sparse-rewiring-down-proj-report.html
"""

import html
import json
import os
import math
import numpy as np
from pathlib import Path
import torch

BASE_DIR = "/Users/prasadmahadik/.gemini/antigravity/scratch/spar-pilot"
CACHE_PATH = os.path.join(BASE_DIR, "notebooks/cache/down_proj_decoded.pt")
OUT_PATHS = [
    os.path.join(BASE_DIR, "docs/sparse-rewiring-down-proj-report.html"),
    os.path.join(BASE_DIR, "notebooks/sparse-rewiring-down-proj-report.html"),
]


def build_gold_report(cache_path=CACHE_PATH, out_paths=OUT_PATHS):
    CACHE_PATH = str(cache_path)
    OUT_PATHS = [str(p) for p in out_paths]
    data = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    SURVIVORS = data["survivors"]
    LENS_NAMES = data["lens_names"]  # ['logit_lens', 'jlens', 'jpp', 'rlens']
    LENS_LABELS = {
        "rlens": "R-Lens",
        "jpp": "J++",
        "jlens": "J-Lens",
        "logit_lens": "Logit Lens",
    }
    ID_TO_STR = data["id_to_str"]
    RAW_V = data["raw_v"].float()
    TRANS_U = {k: v.float() for k, v in data["transported_u"].items()}
    U_RAW_NORMS = {k: v.float() for k, v in data["u_raw_norms"].items()}
    LOGITS = {k: v.float() for k, v in data["logits_fp16"].items()}
    TOP_K_RECS = data["top_k_records"]
    LAYER_STATS = data["layer_norm_stats"]

    N = len(SURVIVORS)
    LABELS = [s["label"] for s in SURVIVORS]
    GROUPS = [s["group"] for s in SURVIVORS]
    IDX_SHARED = [i for i, s in enumerate(SURVIVORS) if s["group"] == "shared"]
    IDX_R0_ONLY = [i for i, s in enumerate(SURVIVORS) if s["group"] == "r0_only"]
    IDX_R1_ONLY = [i for i, s in enumerate(SURVIVORS) if s["group"] == "r1_only"]
    IDX_R0_ALL = [i for i, s in enumerate(SURVIVORS) if s["in_r0"]]
    IDX_R1_ALL = [i for i, s in enumerate(SURVIVORS) if s["in_r1"]]

    # Curated human-readable functional role summaries grounded in top-k + targeted probe
    ROLE_SUMMARIES = {
        "L11 #8235": ("Inline continuation/spacing & plain arithmetic vs. sentence-ending breaks",
                      "Promotes inline spacing, '=', '<<', digits, and 'So' vs. sentence-ending line breaks ('？\\n\\n', '）\\n\\n'), '\\[', and '<|endoftext|>'."),
        "L16 #5569": ("Markdown/LaTeX colon openers vs. <|endoftext|> & newlines",
                      "Separates markdown/colon openers ('):\\n\\n', ':**', '[*', '\\[') on +v from '<|endoftext|>' (−7.8σ) and line breaks ('”\\n', '\\n\\n\\n') on −v."),
        "L17 #6230": ("Truth / assertion / inequality (≠, ≥, True) vs. malformed / breaks",
                      "Promotes equality/inequality & certainty ('=', '>>', '≠', '≥', '确实', '(True') vs. malformed/disorder tokens ('紊乱', ' malformed', '?\\n\\n\\n')."),
        "L17 #7054": ("LaTeX & markdown heading-newline delimiters vs. ellipsis",
                      "Promotes LaTeX/markdown delimiters ('##\\n\\n', '\\[', ';\\n\\n', ':\\n\\n', '**') vs. ellipsis continuation ('…”', '…)', '…and') and '\\n'."),
        "L19 #5606": ("Master markup / underscore delimiter switch vs. prose & EOS",
                      "Unanimous across all 4 lenses on +v ('__', '____', '______') plus +2.9σ to +8.2σ on EVERY markup token ('**', '\\[', '####', '###', '<<', '>>', 'boxed') vs. plain prose, contractions ('d/'m/'ll), and '<|endoftext|>'."),
        "L3 #574":   ("Math / school word-problem domain",
                      "Only visible via J++ & R-Lens (cos = 0.97!): decodes +v to '数学', 'Maths', '老师' (teacher), 'rozwiązania' (solutions), and '\\[' (+3.1σ); invisible in Logit/J-Lens."),
        "L4 #1398":  ("Structured workflow / enterprise jargon vs. prose/misc",
                      "Promotes workflow/tech tokens ('SharePoint', 'Workflow', 'PowerShell', '客户需求') vs. prose/misc ('essay', 'capitalist'); mild positive load on '####'/'>>'."),
        "L6 #1362":  ("Structured tags vs. trailing ellipsis / CoT / newlines",
                      "Promotes '-BEGIN', 'shipments', '-import' while suppressing '\\n\\n' (−6.6σ), '\\n' (−5.3σ), '<|endoftext|>' (−3.6σ), 'boxed' (−2.8σ), 'Therefore' (−2.6σ), and ':\\n\\n'."),
        "L12 #6760": ("Paragraph spacing & '####' vs. single-newline sentence endings",
                      "Cross-layer substitute for R1's L6 #5466 (R-Lens cos +0.66): whitespace / '\\n\\n' (+14.8σ) / '####' (+3.0σ) vs. single-newline endings ('.\\n', '!\\n', '?\\n', −3.1σ) and '<|endoftext|>' (−5.4σ)."),
        "L18 #4890": ("Sentence-ending newline specialist vs. continuation",
                      "Extreme newline spike ('.\\n', '。\\n', '?\\n', '!\\n'; '\\n' +15.2σ, '\\n\\n' +8.6σ) vs. ellipsis ('...)', '...]'). LL/J/R agree 0.77–0.94 (note: J++ sign-flipped at L18, cos −0.32)."),
        "L27 #4914": ("Spelled-out English number words (twenty..ninety) on −v",
                      "Directly before lm_head: −v promotes English number words ('nineteen', 'eighty', 'twenty', 'seventy', 'forty', 'sixty') and digits (−2.1σ) vs. NBSP / '**'."),
        "L27 #7926": ("ASCII single digits (1, 2, 3, 4, 9) & chat EOS (<|im_end|>)",
                      "Promotes ASCII digits ('3','4','2','9','1'; +3.1σ) and '<|endoftext|>' (+2.7σ); only strong hit on Qwen's chat stop '<|im_end|>' (−2.6σ)."),
        "L2 #4005":  ("Closing paren / equation-newline vs. URLs & markdown/LaTeX",
                      "Early-layer switch: promotes ').\\n\\n', '=\\n\\n', '\\n\\n' (+3.2σ) while suppressing URLs, backslash-newlines, and all 4 markdown/LaTeX markers ('**','###','\\[','boxed' −2.1σ to −2.4σ)."),
        "L6 #5466":  ("Paragraph double-newlines (\\n\\n) vs. single-newline endings (.\\n)",
                      "Cross-layer substitute for R0's L12 #6760 (R-Lens cos +0.66): '\\n\\n' (+14.9σ), '####' (+4.1σ), '\\[' (+4.3σ) vs. single-newline endings ('.\\n', '!)\\n', '\\n' −8.0σ)."),
        "L14 #3925": ("Inline whitespace/code vs. trailing ellipsis, markdown & EOS",
                      "Cross-layer match to R0's L6 #1362 (cos +0.39): strongly negative on '\\n\\n' (−7.3σ), '<|endoftext|>' (−6.0σ), '**' (−4.8σ), '\\n' (−4.8σ), '###' (−3.0σ), and 'boxed' (−2.7σ)."),
        "L26 #8751": ("GSM8K word-problem / completion cues (Left, Done)",
                      "Promotes 'Left', 'left', 'Done', 'Shown', '<<' (+2.5σ) — classic GSM8K phrasing ('How many are left?'). LL/J/R agree 0.99 (note: J++ attenuates L26, gain 0.26)."),
        "L27 #1683": ("Short 2–3 letter lowercase tokens / abbreviations",
                      "Promotes short lowercase fragments ('yg','hb','hf','stm','nf','stk'); no strong formatting or digit probe hit (moderate R-Lens cos +0.40 with L3 #574)."),
        "L27 #6214": ("All 10 ASCII digits 0–9 cleanly on −v",
                      "Directly before lm_head: every single top-10 token on −v is a digit '0'–'9' (−11.8σ across all 10 digits) plus '$' (−2.5σ)."),
    }

    def cosine_matrix(X: torch.Tensor) -> np.ndarray:
        Xn = X / (X.norm(dim=-1, keepdim=True) + 1e-8)
        return (Xn @ Xn.T).numpy()

    COS_RAW = cosine_matrix(RAW_V)
    COS_LOGITS = {ln: cosine_matrix(LOGITS[ln]) for ln in LENS_NAMES}

    def participation_ratio(X: torch.Tensor) -> float:
        s = torch.linalg.svdvals(X.float())
        lam = s ** 2
        return float((lam.sum() ** 2) / (lam.pow(2).sum() + 1e-12))

    def dirs_for_90pct(X: torch.Tensor) -> int:
        s = torch.linalg.svdvals(X.float())
        cum = torch.cumsum(s ** 2, dim=0) / (s ** 2).sum()
        return int((cum < 0.90).sum().item()) + 1

    def subspace_overlap(X1: torch.Tensor, X2: torch.Tensor) -> dict:
        Q1, _ = torch.linalg.qr(X1.T.float())
        Q2, _ = torch.linalg.qr(X2.T.float())
        sv = torch.linalg.svdvals(Q1.T @ Q2).clamp(0.0, 1.0)
        return {
            "mean_cos": float(sv.mean()),
            "r2": float((sv ** 2).mean()),
            "top3": [round(float(x), 3) for x in sv[:3]],
        }

    # Cross-lens agreement per neuron
    def lcos(a: str, b: str, i: int) -> float:
        return float(torch.nn.functional.cosine_similarity(LOGITS[a][i], LOGITS[b][i], dim=0))

    def topk_jaccard(a: str, b: str, i: int, k: int = 20, pole: str = "pos") -> float:
        s1 = {tid for tid, _, _ in TOP_K_RECS[a][i][pole][:k]}
        s2 = {tid for tid, _, _ in TOP_K_RECS[b][i][pole][:k]}
        return len(s1 & s2) / max(len(s1 | s2), 1)

    AGREEMENT = []
    for i, s in enumerate(SURVIVORS):
        AGREEMENT.append({
            "i": i,
            "label": s["label"],
            "layer": s["layer"],
            "group": s["group"],
            "ll_j": round(lcos("logit_lens", "jlens", i), 3),
            "ll_r": round(lcos("logit_lens", "rlens", i), 3),
            "j_jpp": round(lcos("jlens", "jpp", i), 3),
            "j_r": round(lcos("jlens", "rlens", i), 3),
            "jpp_r": round(lcos("jpp", "rlens", i), 3),
            "jacc20_jr": round(topk_jaccard("jlens", "rlens", i, 20, "pos"), 2),
            "jacc20_jppr": round(topk_jaccard("jpp", "rlens", i, 20, "pos"), 2),
        })

    # Targeted GSM8K / style / EOS probe across all 4 lenses
    str_to_id = {}
    for tid, tok_str in enumerate(ID_TO_STR):
        if tok_str not in str_to_id:
            str_to_id[tok_str] = tid

    PROBE_GROUPS = [
        ("####",          ["####", " ####"],           "gsm8k"),
        ("<<",            ["<<"],                      "gsm8k"),
        (">>",            [">>"],                      "gsm8k"),
        ("digits 0–9",    [str(d) for d in range(10)], "gsm8k"),
        ("=",             ["=", " ="],                 "gsm8k"),
        ("$",             ["$", " $"],                 "gsm8k"),
        ("\\n",           ["\n"],                      "gsm8k"),
        ("\\n\\n",        ["\n\n"],                    "gsm8k"),
        ("** (bold)",     ["**", " **"],               "style"),
        ("### (hdr)",     ["###", "##"],               "style"),
        ("\\[ (LaTeX)",   ["\\[", "\\("],              "style"),
        ("boxed",         ["boxed"],                   "style"),
        ("Step",          ["Step", " Step"],           "style"),
        ("Therefore",     ["Therefore", " Therefore"], "style"),
        ("So",            ["So", " So"],               "style"),
        ("Final/Ans",     ["Answer", " Answer", "Final", " Final"], "style"),
        ("<|im_end|>",    ["<|im_end|>"],              "eos"),
        ("<|endoftext|>", ["<|endoftext|>"],           "eos"),
    ]

    PROBE_RESOLVED = []
    for disp, cands, cat in PROBE_GROUPS:
        tids = [str_to_id[c] for c in cands if c in str_to_id]
        PROBE_RESOLVED.append({"name": disp, "cands": cands, "tids": tids, "cat": cat})

    PROBE_MATS = {}
    for ln in LENS_NAMES:
        L = LOGITS[ln]
        Z = (L - L.mean(dim=-1, keepdim=True)) / (L.std(dim=-1, keepdim=True) + 1e-8)
        mat = np.zeros((N, len(PROBE_RESOLVED)), dtype=np.float32)
        for col_i, item in enumerate(PROBE_RESOLVED):
            mat[:, col_i] = Z[:, item["tids"]].mean(dim=-1).numpy()
        PROBE_MATS[ln] = np.round(mat, 2).tolist()

    # Closest opposite-arm match in R-Lens logit space
    CLOSEST_OTHER = {}
    for i, s in enumerate(SURVIVORS):
        if s["group"] == "shared":
            CLOSEST_OTHER[i] = None
        else:
            other = IDX_R1_ALL if s["group"] == "r0_only" else IDX_R0_ALL
            best_j = max(other, key=lambda j: abs(COS_LOGITS["rlens"][i, j]))
            CLOSEST_OTHER[i] = {
                "label": LABELS[best_j],
                "group": GROUPS[best_j],
                "cos_r": round(float(COS_LOGITS["rlens"][i, best_j]), 2),
                "cos_raw": round(float(COS_RAW[i, best_j]), 2),
            }

    # Token display helper
    def disp_tok(tok_str: str) -> str:
        # Make invisible/whitespace characters unmistakable inside token pills
        if tok_str == " ":
            return "' '"
        r = repr(tok_str)
        # Strip outer quotes only if normal readable word/symbol; keep escapes visible
        inner = r[1:-1]
        return inner

    # Build JSON data blob for interactive JS tables & heatmaps
    NEURON_JSON = []
    for i, s in enumerate(SURVIVORS):
        L = s["layer"]
        vn = s["v_norm"]
        tokens_by_lens = {}
        for ln in LENS_NAMES:
            std_val = max(TOP_K_RECS[ln][i]["std"], 1e-8)
            tokens_by_lens[ln] = {
                "pos": [[disp_tok(t), round(v / std_val, 1)] for _, t, v in TOP_K_RECS[ln][i]["pos"][:8]],
                "neg": [[disp_tok(t), round(v / std_val, 1)] for _, t, v in TOP_K_RECS[ln][i]["neg"][:8]],
            }
        role_short, role_detail = ROLE_SUMMARIES[s["label"]]
        NEURON_JSON.append({
            "i": i,
            "label": s["label"],
            "layer": L,
            "idx": s["idx"],
            "group": s["group"],
            "in_r0": s["in_r0"],
            "in_r1": s["in_r1"],
            "v_norm": round(vn, 3),
            "layer_med": round(LAYER_STATS[L]["median"], 3),
            "norm_pct": round(s["v_norm_pct"], 1),
            "gains": {
                "jlens": round(float(U_RAW_NORMS["jlens"][i]) / vn, 2),
                "jpp": round(float(U_RAW_NORMS["jpp"][i]) / vn, 2),
                "rlens": round(float(U_RAW_NORMS["rlens"][i]) / vn, 2),
            },
            "ag": AGREEMENT[i],
            "closest_other": CLOSEST_OTHER[i],
            "role_short": role_short,
            "role_detail": role_detail,
            "tokens": tokens_by_lens,
        })

    # Build SVG Figure 0: Survivor Layer Map (echoing Figure 6 of the companion PDF)
    def build_layer_map_svg() -> str:
        W, H = 820, 195
        pad_l, pad_r = 56, 24
        mid_y = 98
        bw, bh, gap = 18, 22, 3
        step_x = (W - pad_l - pad_r) / 28.0
        parts = [
            f'<svg viewBox="0 0 {W} {H}" class="svg-fig" role="img" aria-label="Surviving neurons per layer">',
            f'<text x="{W/2}" y="18" text-anchor="middle" class="svg-sub">layer (0 = first, 27 = last); each block is one surviving neuron</text>',
            f'<line x1="{pad_l - 6}" y1="{mid_y}" x2="{W - pad_r + 6}" y2="{mid_y}" stroke="#9ca3af" stroke-width="1.2"/>',
            f'<text x="8" y="{mid_y - 46}" class="svg-lbl" fill="#b43c12">Rank 0</text>',
            f'<text x="8" y="{mid_y + 56}" class="svg-lbl" fill="#1d4ed8">Rank 1</text>',
        ]
        # Group neurons per layer for R0 (above) and R1 (below)
        r0_by_L = {L: [] for L in range(28)}
        r1_by_L = {L: [] for L in range(28)}
        for s in SURVIVORS:
            if s["in_r0"]:
                r0_by_L[s["layer"]].append(s)
            if s["in_r1"]:
                r1_by_L[s["layer"]].append(s)
        for L in range(28):
            cx = pad_l + (L + 0.5) * step_x
            if L % 3 == 0 or L == 27:
                parts.append(f'<text x="{cx:.1f}" y="{H - 12}" text-anchor="middle" class="svg-tick">{L}</text>')
            for k, s in enumerate(r0_by_L[L]):
                fill = "#15803d" if s["group"] == "shared" else "#b43c12"
                y = mid_y - 4 - (k + 1) * bh - k * gap
                tip = html.escape(f"{s['label']} ({s['group']}) — {ROLE_SUMMARIES[s['label']][0]}")
                parts.append(f'<rect x="{cx - bw/2:.1f}" y="{y:.1f}" width="{bw}" height="{bh}" rx="3" fill="{fill}"><title>{tip}</title></rect>')
            for k, s in enumerate(r1_by_L[L]):
                fill = "#15803d" if s["group"] == "shared" else "#1d4ed8"
                y = mid_y + 4 + k * (bh + gap)
                tip = html.escape(f"{s['label']} ({s['group']}) — {ROLE_SUMMARIES[s['label']][0]}")
                parts.append(f'<rect x="{cx - bw/2:.1f}" y="{y:.1f}" width="{bw}" height="{bh}" rx="3" fill="{fill}"><title>{tip}</title></rect>')
        # Legend top-right
        lx = W - 265
        for idx_l, (col, txt) in enumerate([("#15803d", "Shared (5)"), ("#b43c12", "Rank 0 only (7)"), ("#1d4ed8", "Rank 1 only (6)")]):
            x = lx + idx_l * 88
            parts.append(f'<rect x="{x}" y="27" width="10" height="10" rx="2" fill="{col}"/>')
            parts.append(f'<text x="{x + 14}" y="36" class="svg-legend">{txt}</text>')
        parts.append("</svg>")
        return "\n".join(parts)

    # Build SVG Figure 1: Cross-Lens Agreement by Layer Depth
    def build_agreement_svg() -> str:
        ordered = sorted(AGREEMENT, key=lambda r: (r["layer"], r["label"]))
        W, H = 840, 315
        pl, pr, pt, pb = 54, 24, 42, 74
        pw, ph = W - pl - pr, H - pt - pb
        y_min, y_max = -0.45, 1.08

        def sx(k):
            return pl + (k / (len(ordered) - 1)) * pw
        def sy(v):
            return pt + (y_max - v) / (y_max - y_min) * ph

        parts = [f'<svg viewBox="0 0 {W} {H}" class="svg-fig" role="img" aria-label="Cross-lens agreement by layer depth">']
        # Grid lines
        for yv in [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            yy = sy(yv)
            stroke = "#9ca3af" if yv == 0.0 else "#e5e7eb"
            sw = "1.1" if yv == 0.0 else "0.8"
            parts.append(f'<line x1="{pl}" y1="{yy:.1f}" x2="{W - pr}" y2="{yy:.1f}" stroke="{stroke}" stroke-width="{sw}"/>')
            parts.append(f'<text x="{pl - 8}" y="{yy + 3.5:.1f}" text-anchor="end" class="svg-tick">{yv:.1f}</text>')

        series = [
            ("ll_r",  "Logit Lens vs R-Lens", "#6b7280", "5,4", 1.8),
            ("j_r",   "J-Lens vs R-Lens",     "#b43c12", "",    2.2),
            ("jpp_r", "J++ vs R-Lens",        "#1d4ed8", "",    2.2),
        ]
        # Clean non-overlapping top legend
        for s_i, (_, name, col, dash, _) in enumerate(series):
            lx = pl + s_i * 190
            da = f' stroke-dasharray="{dash}"' if dash else ""
            parts.append(f'<line x1="{lx}" y1="18" x2="{lx + 24}" y2="18" stroke="{col}" stroke-width="2.4"{da}/>')
            parts.append(f'<circle cx="{lx + 12}" cy="18" r="3.5" fill="{col}"/>')
            parts.append(f'<text x="{lx + 30}" y="22" class="svg-legend">{name}</text>')

        # Plot lines & interactive circles
        for key, name, col, dash, sw in series:
            pts = " ".join(f"{sx(k):.1f},{sy(r[key]):.1f}" for k, r in enumerate(ordered))
            da = f' stroke-dasharray="{dash}"' if dash else ""
            parts.append(f'<polyline fill="none" stroke="{col}" stroke-width="{sw}"{da} points="{pts}"/>')
            for k, r in enumerate(ordered):
                cx, cy = sx(k), sy(r[key])
                tip = html.escape(f"{r['label']} ({r['group']}): {name} = {r[key]:+.3f}")
                parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.0" fill="{col}" stroke="#fff" stroke-width="1"><title>{tip}</title></circle>')

        # Callout annotations (cleanly placed)
        # 1. L3 #574 (k=1): J++ & R agree 0.969 while LL=0.130
        k_l3 = next(k for k, r in enumerate(ordered) if r["label"] == "L3 #574")
        parts.append(f'<text x="{sx(k_l3) + 8:.1f}" y="{sy(0.97) - 6:.1f}" class="svg-annot" fill="#1d4ed8">L3 #574: J++↔R = 0.97</text>')
        # 2. L18 #4890: J++ sign flip (-0.315)
        k_l18 = next(k for k, r in enumerate(ordered) if r["label"] == "L18 #4890")
        parts.append(f'<text x="{sx(k_l18) - 10:.1f}" y="{sy(-0.315) + 15:.1f}" text-anchor="middle" class="svg-annot" fill="#1d4ed8">J++ sign flip (−0.32)</text>')
        # 3. L26 #8751: J++ drop (0.774, gain 0.26)
        k_l26 = next(k for k, r in enumerate(ordered) if r["label"] == "L26 #8751")
        parts.append(f'<text x="{sx(k_l26) - 6:.1f}" y="{sy(0.774) + 16:.1f}" text-anchor="end" class="svg-annot" fill="#1d4ed8">J++ L26 atten.</text>')

        # X-axis neuron labels colored by group
        grp_col = {"shared": "#15803d", "r0_only": "#b43c12", "r1_only": "#1d4ed8"}
        for k, r in enumerate(ordered):
            x = sx(k)
            y = H - pb + 12
            parts.append(
                f'<text x="{x:.1f}" y="{y:.1f}" transform="rotate(-42 {x:.1f} {y:.1f})" '
                f'text-anchor="end" class="svg-tick" fill="{grp_col[r["group"]]}" font-weight="600">{r["label"]}</text>'
            )

        parts.append(f'<text x="14" y="{pt + ph/2:.1f}" transform="rotate(-90 14 {pt + ph/2:.1f})" text-anchor="middle" class="svg-axis">Vocab Logit Cosine Similarity</text>')
        parts.append("</svg>")
        return "\n".join(parts)

    # Precompute arm-level subspace metrics table rows
    ARM_SUBSPACE_ROWS = []
    for ln in LENS_NAMES:
        X_r0 = TRANS_U[ln][IDX_R0_ALL]
        X_r1 = TRANS_U[ln][IDX_R1_ALL]
        ov_all = subspace_overlap(X_r0, X_r1)
        ov_only = subspace_overlap(TRANS_U[ln][IDX_R0_ONLY], TRANS_U[ln][IDX_R1_ONLY])
        ARM_SUBSPACE_ROWS.append({
            "lens": LENS_LABELS[ln],
            "r0_pr": f"{participation_ratio(X_r0):.2f} ({dirs_for_90pct(X_r0)}/12 for 90%)",
            "r1_pr": f"{participation_ratio(X_r1):.2f} ({dirs_for_90pct(X_r1)}/11 for 90%)",
            "all_cos": f"{ov_all['mean_cos']:.3f}",
            "only_cos": f"{ov_only['mean_cos']:.3f}",
            "only_top3": ", ".join(f"{x:.2f}" for x in ov_only["top3"]),
        })

    # Key pair callouts
    label_to_i = {l: i for i, l in enumerate(LABELS)}
    KEY_PAIRS = [
        ("L12 #6760", "L6 #5466",  "Cross-layer substitute (R0 ↔ R1)", "Both separate '\\n\\n' (+14.8σ/+14.9σ) & '####' from '\\n'."),
        ("L11 #8235", "L3 #574",   "Anti-aligned (Shared ↔ R0)",       "L3 #574 boosts '？\\n\\n' & '\\[' (+3.1σ); L11 #8235 suppresses both."),
        ("L3 #574",   "L27 #1683", "Cross-arm (R0 ↔ R1)",              "Moderate alignment in R-Lens (+0.40) and J++ (+0.38)."),
        ("L6 #1362",  "L14 #3925", "Cross-layer substitute (R0 ↔ R1)", "Both strongly suppress '\\n\\n', '\\n', 'boxed', 'Therefore', and '<|endoftext|>'."),
        ("L19 #5606", "L14 #3925", "Anti-aligned (Shared ↔ R1)",       "L19 #5606 promotes '**'/'###'/'boxed' (+8.2σ/+4.1σ/+2.9σ); L14 #3925 suppresses all three."),
        ("L17 #6230", "L17 #7054", "Shared L17 pair (same layer)",     "Completely orthogonal (cos +0.04) despite Rank 1 forcing a shared input read direction."),
        ("L6 #1362",  "L6 #5466",  "Cross-arm L6 (same layer)",        "Orthogonal in L6 (−0.03): R0 suppresses CoT/newlines; R1 splits '\\n\\n' vs '\\n'."),
        ("L27 #7926", "L27 #6214", "Cross-arm L27 digits (same layer)","Low full-vocab cos (−0.09) because 10 digits are 0.006% of vocab, yet BOTH target digits 0–9 (+3.1σ vs −11.8σ)."),
        ("L27 #7926", "L27 #1683", "Cross-arm L27 (same layer)",       "Mild negative full-vocab overlap (−0.24)."),
    ]
    KEY_PAIR_ROWS_HTML = []
    for a, b, tag, note in KEY_PAIRS:
        ia, ib = label_to_i[a], label_to_i[b]
        r_cos = COS_LOGITS["rlens"][ia, ib]
        cls = "num-pos" if r_cos >= 0.3 else ("num-neg" if r_cos <= -0.3 else "")
        KEY_PAIR_ROWS_HTML.append(
            f"<tr>"
            f"<td class='mono'><strong>{a}</strong> ↔ <strong>{b}</strong></td>"
            f"<td>{html.escape(tag)}</td>"
            f"<td class='mono num'>{COS_RAW[ia, ib]:+.2f}</td>"
            f"<td class='mono num'>{COS_LOGITS['jlens'][ia, ib]:+.2f}</td>"
            f"<td class='mono num'>{COS_LOGITS['jpp'][ia, ib]:+.2f}</td>"
            f"<td class='mono num {cls}'><strong>{r_cos:+.2f}</strong></td>"
            f"<td class='muted-sm'>{html.escape(note)}</td>"
            f"</tr>"
        )

    # Norm summary table rows
    NORM_ROWS_HTML = []
    for n in NEURON_JSON:
        badge_cls = {"shared": "b-shared", "r0_only": "b-r0", "r1_only": "b-r1"}[n["group"]]
        badge_txt = {"shared": "Shared", "r0_only": "Rank 0", "r1_only": "Rank 1"}[n["group"]]
        pct_style = "font-weight:700;color:#b43c12;" if n["norm_pct"] <= 10.5 else ""
        jpp_note = " (atten.)" if n["layer"] == 26 else ""
        NORM_ROWS_HTML.append(
            f"<tr>"
            f"<td><span class='badge {badge_cls}'>{badge_txt}</span></td>"
            f"<td class='mono'><strong>{n['label']}</strong></td>"
            f"<td class='mono num'>{n['v_norm']:.3f}</td>"
            f"<td class='mono num muted'>{n['layer_med']:.3f}</td>"
            f"<td class='mono num' style='{pct_style}'>{n['norm_pct']:.1f}%</td>"
            f"<td class='mono num'>{n['gains']['jlens']:.2f}×</td>"
            f"<td class='mono num'>{n['gains']['jpp']:.2f}×{jpp_note}</td>"
            f"<td class='mono num'>{n['gains']['rlens']:.2f}×</td>"
            f"</tr>"
        )

    # Functional Role Taxonomy table
    TAXONOMY = [
        ("Master markup & delimiter switch",   ["L19 #5606"],                     "** (+8.2σ), \\[ (+6.0σ), #### (+5.7σ), ### (+4.1σ), << (+2.9σ), boxed (+2.9σ)"),
        ("Plain-text GSM8K arithmetic vs. LaTeX", ["L11 #8235"],                  "= (+3.8σ), << (+2.9σ), So (+2.6σ), digits (+2.3σ) vs. \\[ (−1.9σ), EOS (−2.3σ)"),
        ("Markdown/LaTeX vs. EOS stopping",    ["L16 #5569", "L17 #7054"],        "L16: ** (+3.0σ), \\[ (+2.5σ) vs. <|endoftext|> (−7.8σ). L17: \\[ (+4.4σ), $ (+2.3σ)"),
        ("Paragraph breaks ('\\n\\n') & '####' vs. '\\n'", ["L12 #6760", "L6 #5466"], "Cross-layer substitutes (cos +0.66): both +14.8σ/+14.9σ on \\n\\n, +3–4σ on ####, −3 to −8σ on \\n"),
        ("Single-newline & plain-paragraph controls", ["L18 #4890", "L2 #4005"],  "L18: \\n (+15.2σ), \\n\\n (+8.6σ). L2: \\n\\n (+3.2σ) while suppressing **, ###, \\[, boxed (−2.1σ to −2.4σ)"),
        ("CoT / boxed / newline / EOS suppressors", ["L6 #1362", "L14 #3925"],    "Cross-layer pair (cos +0.39): both strongly negative on \\n\\n, \\n, boxed, Therefore, <|endoftext|>"),
        ("Final-layer digit & number knobs",   ["L27 #7926", "L27 #4914", "L27 #6214"], "L27 #6214: all 10 digits (−11.8σ). L27 #7926: digits (+3.1σ) & <|im_end|> (−2.6σ). L27 #4914: 'twenty'..'ninety'"),
        ("Domain / math-problem / assertion cues", ["L3 #574", "L17 #6230", "L26 #8751"], "L3: 'Maths'/'数学'/'老师' (J++/R). L17: '=' (+2.4σ), '≠', '≥', '(True'. L26: 'Left', 'Done', '<<' (+2.5σ)"),
        ("Unclear / weak formatting alignment", ["L4 #1398", "L27 #1683"],        "L4: workflow/tech jargon. L27 #1683: short lowercase fragments (no strong formatting/digit z-score)"),
    ]

    def badge_for_label(lbl: str) -> str:
        g = GROUPS[label_to_i[lbl]]
        cls = {"shared": "b-shared", "r0_only": "b-r0", "r1_only": "b-r1"}[g]
        return f"<span class='badge {cls} mono'>{lbl}</span>"

    TAXONOMY_HTML = "\n".join(
        f"<tr>"
        f"<td><strong>{html.escape(role)}</strong></td>"
        f"<td class='badge-wrap'>{' '.join(badge_for_label(x) for x in ns)}</td>"
        f"<td class='muted-sm'>{html.escape(ev)}</td>"
        f"</tr>"
        for role, ns, ev in TAXONOMY
    )

    # Full-arm comparison side-by-side rows (sorted by layer)
    def build_arm_table_html(idxs):
        rows = []
        for i in sorted(idxs, key=lambda idx: (SURVIVORS[idx]["layer"], SURVIVORS[idx]["idx"])):
            n = NEURON_JSON[i]
            cls = {"shared": "b-shared", "r0_only": "b-r0", "r1_only": "b-r1"}[n["group"]]
            st = "Shared" if n["group"] == "shared" else ("R0 only" if n["group"] == "r0_only" else "R1 only")
            top_pos = " ".join(f"<span class='tok tok-pos'>{html.escape(t)}</span>" for t, _ in n["tokens"]["rlens"]["pos"][:5])
            top_neg = " ".join(f"<span class='tok tok-neg'>{html.escape(t)}</span>" for t, _ in n["tokens"]["rlens"]["neg"][:5])
            rows.append(
                f"<tr>"
                f"<td class='mono nowrap'><strong>{n['label']}</strong><br><span class='badge {cls}'>{st}</span></td>"
                f"<td class='muted-sm'><strong>{html.escape(n['role_short'])}</strong></td>"
                f"<td><div class='tok-row'>{top_pos}</div><div class='tok-row' style='margin-top:4px'>{top_neg}</div></td>"
                f"</tr>"
            )
        return "\n".join(rows)

    # JSON bundle for JS interactivity (safely escape '</')
    JS_BUNDLE = {
        "neurons": NEURON_JSON,
        "labels": LABELS,
        "groups": GROUPS,
        "idx_shared": IDX_SHARED,
        "idx_r0_only": IDX_R0_ONLY,
        "idx_r1_only": IDX_R1_ONLY,
        "idx_r0_all": IDX_R0_ALL,
        "idx_r1_all": IDX_R1_ALL,
        "lens_names": ["rlens", "jpp", "jlens", "logit_lens"],
        "lens_labels": LENS_LABELS,
        "cos_mats": {
            "rlens": np.round(COS_LOGITS["rlens"], 3).tolist(),
            "jpp": np.round(COS_LOGITS["jpp"], 3).tolist(),
            "jlens": np.round(COS_LOGITS["jlens"], 3).tolist(),
            "raw": np.round(COS_RAW, 3).tolist(),
        },
        "probe_cols": [{"name": p["name"], "cat": p["cat"]} for p in PROBE_RESOLVED],
        "probe_mats": PROBE_MATS,
    }
    JS_BUNDLE_STR = json.dumps(JS_BUNDLE, ensure_ascii=False).replace("</", "<\\/")

    HTML_DOC = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>What do the surviving neurons write? · Sparse Rewiring down_proj Interpretation</title>
    <style>
      :root {{
        --bg: #ffffff;
        --surface: #f9fafb;
        --border: #e5e7eb;
        --border-strong: #d1d5db;
        --text: #111827;
        --text-2: #374151;
        --muted: #6b7280;
        --shared: #15803d;
        --shared-bg: #ecfdf5;
        --shared-bd: #bbf7d0;
        --r0: #b43c12;
        --r0-bg: #fff7ed;
        --r0-bd: #fed7aa;
        --r1: #1d4ed8;
        --r1-bg: #eff6ff;
        --r1-bd: #bfdbfe;
        --serif: Georgia, "Iowan Old Style", "Palatino Linotype", "Source Serif 4", serif;
        --sans: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        --mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Monaco, Consolas, monospace;
      }}
      * {{ box-sizing: border-box; }}
      html {{ scroll-behavior: smooth; }}
      body {{
        margin: 0;
        padding: 40px 20px 80px;
        background: var(--bg);
        color: var(--text);
        font-family: var(--sans);
        font-size: 14.5px;
        line-height: 1.62;
        -webkit-font-smoothing: antialiased;
      }}
      .page {{
        max-width: 1120px;
        margin: 0 auto;
      }}
      .eyebrow {{
        font-size: 11px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        font-weight: 600;
        margin-bottom: 10px;
      }}
      h1 {{
        font-family: var(--serif);
        font-size: 36px;
        line-height: 1.15;
        letter-spacing: -0.015em;
        margin: 0 0 14px;
        font-weight: 700;
        color: #0f172a;
      }}
      .lead {{
        font-size: 16px;
        line-height: 1.6;
        color: var(--text-2);
        margin: 0 0 24px;
      }}
      h2 {{
        font-family: var(--serif);
        font-size: 22px;
        font-weight: 700;
        margin: 44px 0 14px;
        padding-top: 22px;
        border-top: 1px solid var(--border);
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
      }}
      h3 {{
        font-size: 15px;
        font-weight: 700;
        margin: 22px 0 10px;
        color: var(--text);
      }}
      p {{ margin: 0 0 14px; color: var(--text-2); }}
      .mono {{ font-family: var(--mono); }}
      .nowrap {{ white-space: nowrap; }}
      .muted {{ color: var(--muted); }}
      .muted-sm {{ color: var(--text-2); font-size: 13px; line-height: 1.48; }}
      .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
      .num-pos {{ color: #991b1b; }}
      .num-neg {{ color: #1e40af; }}

      /* Cards */
      .card {{
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 18px 22px;
        margin: 16px 0;
      }}
      .summary-card ol {{
        margin: 0 0 14px;
        padding-left: 20px;
      }}
      .summary-card li {{
        margin-bottom: 10px;
        color: var(--text-2);
      }}
      .summary-card li strong {{ color: var(--text); }}
      .pills {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
      }}
      .pill {{
        display: inline-block;
        font-size: 12px;
        color: var(--muted);
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 3px 11px;
      }}

      /* Formula box */
      .formula-box {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 16px 20px;
        font-family: var(--mono);
        font-size: 13px;
        line-height: 1.85;
        overflow-x: auto;
        margin: 16px 0;
      }}
      .c-g {{ color: var(--shared); font-weight: 600; }}
      .c-r {{ color: var(--r0); font-weight: 600; }}
      .c-b {{ color: var(--r1); font-weight: 600; }}

      /* Grids */
      .grid-2 {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin: 16px 0; }}
      .grid-3 {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin: 16px 0; }}
      .grid-4 {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 16px 0; }}
      @media (max-width: 720px) {{
        .grid-2, .grid-3, .grid-4 {{ grid-template-columns: 1fr; }}
      }}
      .mini-card {{
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 14px 16px;
      }}
      .mini-card h4 {{
        margin: 0 0 6px;
        font-size: 13.5px;
        font-weight: 700;
        color: var(--text);
      }}
      .mini-card p {{ margin: 0; font-size: 12.8px; line-height: 1.5; color: var(--muted); }}
      .stat-card {{
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 16px 18px;
      }}
      .stat-val {{
        font-family: var(--mono);
        font-size: 26px;
        font-weight: 700;
        line-height: 1.15;
        margin-bottom: 6px;
        color: var(--text);
      }}
      .stat-desc {{ font-size: 12.8px; color: var(--muted); line-height: 1.45; }}

      /* Pipeline strip */
      .pipeline {{
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 10px;
        margin: 16px 0;
      }}
      @media (max-width: 760px) {{ .pipeline {{ grid-template-columns: 1fr; }} }}
      .pipe-step {{
        border: 1px solid var(--border);
        border-radius: 9px;
        padding: 11px 12px;
        background: var(--surface);
        position: relative;
      }}
      .pipe-num {{
        font-family: var(--mono);
        font-size: 11px;
        font-weight: 700;
        color: var(--muted);
        text-transform: uppercase;
        margin-bottom: 4px;
      }}
      .pipe-title {{ font-size: 13px; font-weight: 700; color: var(--text); margin-bottom: 4px; }}
      .pipe-body {{ font-size: 12px; color: var(--muted); line-height: 1.42; }}

      /* Tables */
      .table-wrap {{
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow-x: auto;
        margin: 16px 0;
        background: var(--bg);
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
      }}
      th {{
        background: var(--surface);
        text-align: left;
        font-weight: 600;
        color: var(--text-2);
        padding: 10px 12px;
        border-bottom: 1px solid var(--border);
        font-size: 12.2px;
      }}
      td {{
        padding: 9px 12px;
        border-bottom: 1px solid #f3f4f6;
        vertical-align: top;
      }}
      tr:last-child td {{ border-bottom: none; }}

      /* Badges & Token Chips */
      .badge {{
        display: inline-block;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 999px;
        border: 1px solid transparent;
        white-space: nowrap;
      }}
      .b-shared {{ color: var(--shared); background: var(--shared-bg); border-color: var(--shared-bd); }}
      .b-r0     {{ color: var(--r0);     background: var(--r0-bg);     border-color: var(--r0-bd); }}
      .b-r1     {{ color: var(--r1);     background: var(--r1-bg);     border-color: var(--r1-bd); }}
      .badge-wrap {{ display: flex; flex-wrap: wrap; gap: 5px; }}

      .tok-row {{ display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }}
      .tok {{
        font-family: var(--mono);
        font-size: 11.5px;
        padding: 1.5px 6px;
        border-radius: 5px;
        border: 1px solid var(--border);
        background: var(--surface);
        color: var(--text);
        white-space: pre;
      }}
      .tok-pos {{ background: #fef2f2; border-color: #fecaca; color: #991b1b; }}
      .tok-neg {{ background: #eff6ff; border-color: #bfdbfe; color: #1e40af; }}
      .pole-lbl {{
        font-family: var(--mono);
        font-size: 10.5px;
        font-weight: 700;
        width: 24px;
        display: inline-block;
        flex-shrink: 0;
      }}
      .pole-pos {{ color: #991b1b; }}
      .pole-neg {{ color: #1e40af; }}
      .lens-sub {{
        display: grid;
        grid-template-columns: 74px 1fr;
        gap: 6px;
        padding: 5px 0;
        border-bottom: 1px dashed #e5e7eb;
      }}
      .lens-sub:last-child {{ border-bottom: none; padding-bottom: 0; }}
      .lens-tag {{ font-family: var(--mono); font-size: 11px; font-weight: 700; color: var(--muted); padding-top: 2px; }}

      /* Interactive Tabs */
      .tab-bar {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 6px;
        margin: 12px 0 14px;
      }}
      .tab-label {{ font-size: 12px; color: var(--muted); font-weight: 600; margin-right: 4px; }}
      .tab-btn {{
        font-family: var(--sans);
        font-size: 12.2px;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 999px;
        border: 1px solid var(--border-strong);
        background: #fff;
        color: var(--text-2);
        cursor: pointer;
        transition: all 0.12s ease;
      }}
      .tab-btn:hover {{ background: var(--surface); }}
      .tab-btn.active {{
        background: #111827;
        color: #ffffff;
        border-color: #111827;
      }}

      /* Figure Cards & SVG */
      .fig-card {{
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 16px 18px 14px;
        margin: 18px 0;
        background: #fff;
      }}
      .fig-caption {{
        font-size: 13px;
        color: var(--muted);
        margin-top: 10px;
        line-height: 1.5;
      }}
      .fig-caption strong {{ color: var(--text); }}
      .svg-fig {{ width: 100%; height: auto; display: block; }}
      .svg-sub {{ font-family: var(--mono); font-size: 11px; fill: #6b7280; }}
      .svg-lbl {{ font-family: var(--mono); font-size: 11px; font-weight: 700; }}
      .svg-tick {{ font-family: var(--mono); font-size: 10px; fill: #4b5563; }}
      .svg-legend {{ font-family: var(--sans); font-size: 11.5px; fill: #374151; font-weight: 600; }}
      .svg-axis {{ font-family: var(--sans); font-size: 11px; fill: #4b5563; font-weight: 600; }}
      .svg-annot {{ font-family: var(--mono); font-size: 10px; font-weight: 700; }}

      /* Heatmaps */
      .hm-wrap {{ overflow-x: auto; }}
      .hm-grid {{
        border-collapse: collapse;
        font-family: var(--mono);
        font-size: 10px;
        margin: 0 auto;
      }}
      .hm-grid th {{
        background: transparent;
        border: none;
        padding: 3px 4px;
        font-size: 10px;
        font-weight: 600;
        color: var(--text-2);
        white-space: nowrap;
      }}
      .hm-grid th.col-rot {{
        height: 82px;
        vertical-align: bottom;
        padding-bottom: 6px;
      }}
      .hm-grid th.col-rot > div {{
        transform: rotate(-48deg);
        transform-origin: bottom left;
        width: 24px;
        margin-left: 12px;
      }}
      .hm-cell {{
        width: 34px;
        height: 24px;
        text-align: center;
        vertical-align: middle;
        padding: 0;
        border: 1px solid rgba(255,255,255,0.45);
        font-size: 9.5px;
        cursor: default;
        transition: outline 0.08s ease;
      }}
      .hm-cell:hover {{ outline: 2px solid #111827; z-index: 2; position: relative; }}
      .hm-div-r td, .hm-div-r th {{ border-bottom: 2px solid #111827 !important; }}
      .hm-div-c {{ border-right: 2px solid #111827 !important; }}
      .hm-dash-c {{ border-right: 1.5px dashed #4b5563 !important; }}
      .hm-colorbar {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        margin-top: 10px;
        font-family: var(--mono);
        font-size: 11px;
        color: var(--muted);
      }}
      .hm-grad {{
        width: 220px;
        height: 10px;
        border-radius: 5px;
        border: 1px solid var(--border);
        background: linear-gradient(90deg, #1e3a8a 0%, #60a5fa 25%, #f9fafb 50%, #f87171 75%, #991b1b 100%);
      }}

      /* Callout note */
      .callout {{
        border-left: 3px solid var(--shared);
        background: var(--surface);
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 14px 0;
        font-size: 13.5px;
        color: var(--text-2);
      }}
      .callout strong {{ color: var(--text); }}
      .meaning-card {{
        border: 1px solid #bfdbfe;
        background: #f0f7ff;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 18px 0;
      }}
      .meaning-card h3 {{ margin: 0 0 8px; font-size: 15.5px; color: #1e3a8a; }}
      .meaning-card ul {{ margin: 8px 0 0; padding-left: 20px; }}
      .meaning-card li {{ margin: 6px 0; font-size: 13.8px; color: #1f2937; }}
      .takeaway-list {{ counter-reset: tk; list-style: none; padding: 0; margin: 14px 0; }}
      .takeaway-list li {{
        counter-increment: tk;
        position: relative;
        padding: 12px 16px 12px 46px;
        margin: 10px 0;
        border: 1px solid var(--border);
        border-radius: 9px;
        background: #fff;
        font-size: 13.6px;
        color: var(--text-2);
      }}
      .takeaway-list li::before {{
        content: counter(tk);
        position: absolute;
        left: 14px;
        top: 12px;
        width: 22px;
        height: 22px;
        border-radius: 50%;
        background: #111827;
        color: #fff;
        font-family: var(--mono);
        font-size: 11.5px;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
      }}
      .takeaway-list li strong {{ color: var(--text); }}
      .norm-low {{ color: #15803d; font-weight: 700; }}
      .ag-pill {{
        display: inline-block;
        padding: 2px 7px;
        border-radius: 6px;
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        font-family: var(--mono);
        font-size: 11.5px;
        color: #111827;
      }}
      .ag-warn {{ background: #fef2f2; border-color: #fecaca; color: #991b1b; }}
      .pole-col-pos {{ background: #fefafa; }}
      .pole-col-neg {{ background: #f8fbff; }}

      /* Next-step cards */
      .todo-card {{
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
        background: #fff;
      }}
      .todo-card h4 {{ margin: 0 0 5px; font-size: 14px; color: var(--text); }}
      .todo-card p {{ margin: 0; font-size: 13.2px; color: var(--text-2); }}

      /* Floating tooltip */
      #tooltip {{
        position: fixed;
        display: none;
        pointer-events: none;
        background: #111827;
        color: #f9fafb;
        padding: 8px 11px;
        border-radius: 7px;
        font-family: var(--mono);
        font-size: 11.5px;
        line-height: 1.5;
        box-shadow: 0 8px 24px rgba(0,0,0,0.22);
        z-index: 9999;
        max-width: 340px;
      }}

      /* Print-to-PDF polish */
      @media print {{
        body {{ padding: 0; font-size: 12px; }}
        .page {{ max-width: 100%; }}
        .tab-bar {{ display: none !important; }}
        .card, .fig-card, .stat-card, .todo-card, .table-wrap, .mini-card {{ break-inside: avoid; }}
        h2 {{ break-after: avoid; margin-top: 26px; padding-top: 14px; }}
      }}
    </style>
    </head>
    <body>
    <div class="page">

      <div class="eyebrow">SPARSE REWIRING · DOWN_PROJ INTERPRETATION · 26 NOV 2026</div>
      <h1>What do the surviving neurons write?</h1>
      <p class="lead">
        Because sparse L0 rewiring froze every MLP neuron’s output column (<span class="mono">down_proj</span>), the 18 surviving neurons across Rank 0 and Rank 1 could only steer <span class="mono">Qwen2.5-1.5B-Instruct</span> along pre-existing output directions. We decode all 18 frozen vectors using Logit Lens and three Jacobian-lens variants (<span class="mono">J-Lens</span>, <span class="mono">J++</span>, <span class="mono">R-Lens</span>) to see what concepts and tokens they align with.
      </p>

      <div class="meaning-card">
        <h3>What these results mean in plain English (correlational findings &amp; hypothesis)</h3>
        <p style="margin:0 0 8px;font-size:13.8px;color:#1f2937">
          Static <span class="mono">down_proj</span> decoding measures <em>what tokens each frozen output direction aligns with</em> (not its active firing sign or causal impact). Combined with the observed model-level metrics (<strong>90%+ validation CE recovery</strong> alongside an <strong>accuracy drop from 80.5% → 67%</strong>), the decoded vectors are consistent with your <strong>Section 7 style-and-length-compression hypothesis</strong> — pending the deferred activation-shift (<span class="mono">Δa_j</span>) and ablation tests in Section 8:
        </p>
        <ul>
          <li><strong>The 5 shared neurons (L11–L19) load strongly on formatting &amp; stopping tokens:</strong> <span class="mono">L16 #5569</span> sits between markdown/colon structure (<span class="mono">:**</span>, <span class="mono">:\\n\\n</span>, <span class="mono">[*</span>) on <span class="mono">+v</span> and <span class="mono">&lt;|endoftext|&gt;</span> (stopping, −7.8σ) on <span class="mono">−v</span>; <span class="mono">L17 #7054</span> aligns with LaTeX/heading delimiters (<span class="mono">##\\n\\n</span>, <span class="mono">\\[</span>, <span class="mono">:\\n\\n</span>); <span class="mono">L19 #5606</span> separates underscore/markup tokens (+2.9σ to +8.2σ on every delimiter) from plain English prose &amp; contractions (<span class="mono">'m</span>, <span class="mono">'d</span>, <span class="mono">'ll</span>).</li>
          <li><strong>Both arms independently kept digit/number directions right before the output (L27):</strong> Rank 0 kept <span class="mono">L27 #7926</span> (digits 1, 2, 3, 4, 9 on <span class="mono">+v</span>) + <span class="mono">L27 #4914</span> (spelled-out English numbers <em>twenty..ninety</em> on <span class="mono">−v</span>); Rank 1 kept <span class="mono">L27 #6214</span> (all 10 ASCII digits 0–9 cleanly on <span class="mono">−v</span>) + <span class="mono">L26 #8751</span> (<em>Left</em>, <em>Done</em>).</li>
          <li><strong>Output-space overlap exceeds 5/18:</strong> Even where the two arms picked different neuron indices, some align closely across layers — most clearly Rank 0’s <span class="mono">L12 #6760</span> ↔ Rank 1’s <span class="mono">L6 #5466</span> (<strong>R-Lens cos = +0.66</strong>, both loading on paragraph breaks vs. single-newline endings).</li>
          <li><strong>Proposed explanation for the CE vs. accuracy tradeoff (hypothesis, not yet causally tested):</strong> <span class="mono">Qwen2.5-1.5B-Instruct</span> naturally writes long markdown/LaTeX explanations (~220 tokens), whereas GSM8K gold answers are terse (~130 tokens). Re-weighting roughly a dozen pre-existing formatting, stopping, and digit directions could plausibly align output style with terse gold targets (large CE reduction) while shortening reasoning chains (accuracy drop). Whether these neurons actually fire with the predicted sign and cause the accuracy drop remains to be tested via <span class="mono">Δa_j</span> measurements and leave-one-out ablations (Section 8).</li>
        </ul>
      </div>

      <h3>Five Big Takeaways (including Geometry &amp; Cross-Lens Checks)</h3>
      <ol class="takeaway-list">
        <li><strong>Consistent with the Section 7 style/formatting hypothesis (correlational evidence pending causal checks):</strong> Four of the five shared neurons (<span class="mono">L11 #8235</span>, <span class="mono">L16 #5569</span>, <span class="mono">L17 #7054</span>, <span class="mono">L19 #5606</span>) associate heavily with markdown/LaTeX delimiters, EOS (<span class="mono">&lt;|endoftext|&gt;</span>), newlines, and prose/contractions, while both arms independently retained L27 digit/number directions (<span class="mono">L27 #7926</span>, <span class="mono">L27 #4914</span>, <span class="mono">L27 #6214</span>) and <span class="mono">L26 #8751</span> (<em>Left/Done</em>).</li>
        <li><strong>Candidate cross-layer substitutes between the two arms (high output-space alignment):</strong> The single strongest pair across all 18 neurons is Rank 0’s <span class="mono">L12 #6760</span> ↔ Rank 1’s <span class="mono">L6 #5466</span> (<span class="mono">raw +0.40</span>, <span class="mono">J-Lens +0.67</span>, <span class="mono">R-Lens +0.66</span>, <span class="mono">J++ +0.45</span>). Excluding the 5 shared neurons completely, the 7 Rank-0-only and 6 Rank-1-only subspaces still share 2 strong common directions in R-Lens space (top principal-angle cosines <strong>0.78</strong> and <strong>0.58</strong>, vs. only 0.40 and 0.18 in raw Logit Lens); whether these cosine-aligned pairs act as functional substitutes in forward passes remains to be tested via <span class="mono">Δa_j</span> and ablations. Conversely, L27 pairs have low full-vocab cosine (−0.24 to +0.14) even though <span class="mono">L27 #7926</span> and <span class="mono">L27 #6214</span> both hit digits 0–9 in their top-10 — because 10 digits are only 0.006% of the 152k vocab, showing why both token tables and heatmaps are needed.</li>
        <li><strong>Shared L17 pair (<span class="mono">#6230</span> vs. <span class="mono">#7054</span>) is completely orthogonal (<span class="mono">raw +0.009</span>, <span class="mono">R-Lens +0.042</span>):</strong> Neither duplicates nor push-pull opposites — <span class="mono">#7054</span> writes markdown/LaTeX delimiters (<span class="mono">##\\n\\n</span>, <span class="mono">\\[</span>, <span class="mono">:\\n\\n</span>) while <span class="mono">#6230</span> writes assertion/inequality tokens (<span class="mono">≠</span>, <span class="mono">≥</span>, <span class="mono">确实</span>, <span class="mono">(True</span>). Keeping both under Rank 1’s shared input read vector <span class="mono">a</span> suggests both are triggered by the same GSM8K context while steering orthogonal output axes.</li>
        <li><strong>No high-norm / massive-activation bias — in fact the opposite:</strong> All four L27 survivors (<span class="mono">1.3%–10.5%</span> norm %ile), plus shared <span class="mono">L11 #8235</span> (<span class="mono">1.5%</span>) and <span class="mono">L19 #5606</span> (<span class="mono">1.2%</span>), sit in the bottom 1–10% of their layer’s <span class="mono">down_proj</span> norm distribution (medians: <strong>18.9%</strong> Rank-0-only, <strong>8.6%</strong> Rank-1-only).</li>
        <li><strong>Cross-lens behavior (<span class="mono">R-Lens</span> was the cleanest overall):</strong> Logit Lens degrades heavily at L2–L6 (<span class="mono">cos(LL, R) = 0.13–0.28</span>) before converging by L26–L27 (<span class="mono">0.99–1.00</span>). Early-layer win: on <span class="mono">L3 #574</span>, both <span class="mono">J++</span> and <span class="mono">R-Lens</span> agree at <strong>cos = 0.97</strong> and cleanly surface math/teacher/solutions (<span class="mono">数学</span>, <span class="mono">Maths</span>, <span class="mono">老师</span>, <span class="mono">rozwiązania</span>), whereas plain J-Lens and Logit Lens showed noise. Two <span class="mono">J++</span> caveats to watch: a sign flip on <span class="mono">L18 #4890</span> (<span class="mono">cos(J++, R) = −0.32</span> while LL/J/R agree at 0.77–0.94) and heavy attenuation on <span class="mono">L26 #8751</span> (<span class="mono">gain = 0.26</span> while LL/J/R agree at 0.99).</li>
      </ol>

      <div class="card summary-card">
        <div class="pills" style="margin-top:0">
          <span class="pill">Frozen down_proj only — active sign (Δact) measured separately on GPU</span>
          <span class="pill">Pre-fitted Wikitext lenses (n = 64 prompts)</span>
          <span class="pill">L27 has no downstream blocks (J = I)</span>
          <span class="pill">25 MB local decoded cache (zero 3 GB model stored)</span>
        </div>
      </div>

      <!-- SECTION 1 -->
      <h2>1 · Setup &amp; Method</h2>
      <p>
        In the production sparse-rewiring run, only the input read rows (<span class="mono">gate_proj</span>, <span class="mono">up_proj</span>) of each MLP neuron were allowed to change; the write matrix <span class="mono">down_proj</span> remained frozen at its base <span class="mono">Qwen2.5-1.5B-Instruct</span> weights. Whenever surviving neuron <em>j</em> in layer <em>L</em> changes its activation by <span class="mono">Δa_j(x)</span>, it injects a scalar multiple of the fixed vector <span class="mono">v_j = W_down^(L)[:, j] ∈ ℝ^1536</span> into the post-block residual stream.
      </p>

      <div class="formula-box">
        v_j        = W_down^(L)[:, j]                                  <span class="muted"># frozen output column in ℝ^1536</span><br>
        u_j        = <span class="c-b">J_lens^(L)</span> · v_j                                  <span class="muted"># linear transport L → L27 (identity for Logit Lens &amp; L27)</span><br>
        u_readout  = (u_j / RMS(u_j)) ⊙ <span class="c-g">γ_final</span>                        <span class="muted"># unit-RMS scale + final RMSNorm gain</span><br>
        logits_j   = <span class="c-r">W_U</span> · u_readout  −  mean(W_U · u_readout)          <span class="muted"># project to 151,936 vocab tokens, mean-center</span><br>
        z_j(w)     = logits_j(w) / std(logits_j)                       <span class="muted"># standardized vocab z-score (+v promoted, −v suppressed)</span>
      </div>

      <div class="grid-4">
        <div class="mini-card">
          <h4>Logit Lens</h4>
          <p>No transport (<span class="mono">J = I</span>). Direct projection through final RMSNorm gain &amp; <span class="mono">W_U</span>. Exact at L27, noisy in early layers.</p>
        </div>
        <div class="mini-card">
          <h4>J-Lens (<span class="mono">jlens</span>)</h4>
          <p>Corpus-averaged input–output Jacobian <span class="mono">E[∂h_27 / ∂h_L]</span> over Wikitext. Transports intermediate vectors to final-layer coordinates.</p>
        </div>
        <div class="mini-card">
          <h4>J++ Lens (<span class="mono">jpp</span>)</h4>
          <p>Jacobian-filtered variant (Ayonrinde &amp; Lindsey, 2026) designed to remove early-layer noise and improve workspace coherence.</p>
        </div>
        <div class="mini-card">
          <h4>R-Lens (<span class="mono">rlens</span>)</h4>
          <p>LRP-modified backward pass (Blank, Bhatia &amp; Nanda, 2026). Cleanest early-layer fidelity and highest cross-depth consistency here.</p>
        </div>
      </div>

      <div class="fig-card">
        {build_layer_map_svg()}
        <div class="fig-caption"><strong>Figure 0. Surviving neurons by layer (reference map).</strong> Rank 0 (12) above the axis, Rank 1 (11) below. All 5 shared neurons (green) sit in L11–L19; early layers (L2–L6) and final layers (L26–L27) have zero index overlap yet both arms pick neurons there. Hover any block for its functional role.</div>
      </div>

      <div class="table-wrap">
        <table>
          <thead><tr><th style="width:20%">Setting</th><th style="width:42%">Value</th><th>Why</th></tr></thead>
          <tbody>
            <tr><td><strong>Base weights</strong></td><td class="mono">Qwen/Qwen2.5-1.5B-Instruct (BF16 → FP32)</td><td>Identical frozen <span class="mono">down_proj</span>, <span class="mono">model.norm.weight</span>, and tied <span class="mono">embed_tokens / lm_head</span> weights used in sparse rewiring.</td></tr>
            <tr><td><strong>Pre-fitted lenses</strong></td><td class="mono">neuronpedia/jacobian-lens<br>qwen2.5-1.5b-it/{{jlens,jpp,rlens}}/Salesforce-wikitext/</td><td>Covers all 27 source layers <span class="mono">L0–L26</span> (<span class="mono">capture_point = block_output</span>, <span class="mono">target_layer = 27</span>, FP16, 64 Wikitext prompts). At L27, <span class="mono">J = I</span>.</td></tr>
            <tr><td><strong>Tensor extraction</strong></td><td>HTTP byte-range streaming of 15 tensors (~824 MB temp) → 25.05 MB local cache</td><td>Avoids downloading the full 3.09 GB model; temporary downloads &amp; 382 MB lenses deleted after caching.</td></tr>
            <tr><td><strong>Orientation check</strong></td><td class="mono">J @ e vs. J.T @ e on token embeddings (L2, L11, L17, L25)</td><td>Verified <span class="mono">J @ v</span> wins across all 3 lenses: <span class="mono">jlens</span> +13.44 vs +11.40, <span class="mono">jpp</span> +18.75 vs +18.41, <span class="mono">rlens</span> +18.94 vs +18.38.</td></tr>
            <tr><td><strong>Both poles (+v / −v)</strong></td><td>Top-30 promoted (<span class="mono">+v</span>) &amp; Top-30 suppressed (<span class="mono">−v</span>)</td><td>SwiGLU activations are signed and rewiring can suppress a neuron; both directions must be inspected until <span class="mono">Δact</span> signs are measured.</td></tr>
            <tr><td><strong>Vocab centering</strong></td><td>Subtract mean across all 151,936 tokens &amp; z-score</td><td>Removes shared frequency/anisotropy offset in Qwen2.5’s tied unembedding matrix and puts all layers/lenses on a comparable σ scale.</td></tr>
          </tbody>
        </table>
      </div>

      <!-- SECTION 2 -->
      <h2>2 · Pipeline &amp; Diagnostics</h2>

      <div class="pipeline">
        <div class="pipe-step">
          <div class="pipe-num">Step 1</div>
          <div class="pipe-title">Registry</div>
          <div class="pipe-body">Load the 18 survivor <span class="mono">(L, idx)</span> pairs grouped into Shared (5), R0-only (7), R1-only (6).</div>
        </div>
        <div class="pipe-step">
          <div class="pipe-num">Step 2</div>
          <div class="pipe-title">Stream Tensors</div>
          <div class="pipe-body">HTTP Range-fetch only <span class="mono">norm</span>, <span class="mono">embed_tokens</span>, and 13 <span class="mono">down_proj</span> layers + 3 lenses.</div>
        </div>
        <div class="pipe-step">
          <div class="pipe-num">Step 3</div>
          <div class="pipe-title">4-Lens Decode</div>
          <div class="pipe-body">Compute <span class="mono">u = J_L v</span>, apply <span class="mono">γ</span> &amp; <span class="mono">W_U</span>, save 25 MB cache for &lt;2 s iteration.</div>
        </div>
        <div class="pipe-step">
          <div class="pipe-num">Step 4</div>
          <div class="pipe-title">Geometry</div>
          <div class="pipe-body">Measure lens agreement, 18×18 cosine matrices, output PR, and subspace overlap.</div>
        </div>
        <div class="pipe-step">
          <div class="pipe-num">Step 5</div>
          <div class="pipe-title">GSM8K Probe</div>
          <div class="pipe-body">Score all 18 neurons on 18 targeted GSM8K, markdown/LaTeX, and EOS tokens.</div>
        </div>
      </div>

      <div class="fig-card">
        {build_agreement_svg()}
        <div class="fig-caption"><strong>Figure 1. Cross-lens agreement across layer depth (L2 → L27).</strong> Logit Lens agrees poorly with R-Lens in early layers (0.13–0.28) before converging at L26–L27 (0.99–1.00). In early layers L2–L6, <strong>J++ and R-Lens agree closely (0.83–0.97)</strong>, cleanly decoding <span class="mono">L3 #574</span> as math/teacher/solutions where plain J-Lens and Logit Lens miss it. Two J++ caveats appear: a sign flip at <span class="mono">L18 #4890</span> (−0.32, where LL/J/R all agree at 0.77–0.94) and heavy attenuation at L26 (gain 0.26×, where LL/J/R agree at 0.99). <strong>R-Lens is the most consistent single lens across all depths.</strong></div>
      </div>

      <h3>Down-projection norms &amp; transport gains</h3>
      <div class="grid-3">
        <div class="stat-card">
          <div class="stat-val" style="color:var(--r0)">1.3% – 10.5%</div>
          <div class="stat-desc">Norm percentile of all four <strong>L27 survivors</strong>—in the bottom 10% of L27’s 8,960 neurons.</div>
        </div>
        <div class="stat-card">
          <div class="stat-val" style="color:var(--shared)">1.2% / 1.5%</div>
          <div class="stat-desc">Norm percentiles of shared <span class="mono">L19 #5606</span> and <span class="mono">L11 #8235</span>. Rules out high-norm outlier bias.</div>
        </div>
        <div class="stat-card">
          <div class="stat-val" style="color:var(--r1)">18.9% / 8.6%</div>
          <div class="stat-desc">Median norm percentile for Rank-0-only vs. Rank-1-only (Shared median: 50.9%).</div>
        </div>
      </div>

      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Group</th><th>Neuron</th><th class="num">‖v‖₂</th><th class="num">Layer med.</th>
              <th class="num">Norm %ile</th><th class="num">J-Lens gain</th><th class="num">J++ gain</th><th class="num">R-Lens gain</th>
            </tr>
          </thead>
          <tbody>
            {"".join(NORM_ROWS_HTML)}
          </tbody>
        </table>
      </div>

      <!-- INTERACTIVE LENS SWITCHER FOR SECTIONS 3-5 -->
      <div class="tab-bar" id="global-lens-bar">
        <span class="tab-label">Token readout lens (Sections 3–5):</span>
        <button class="tab-btn active" data-lens="rlens">R-Lens (recommended)</button>
        <button class="tab-btn" data-lens="jpp">J++</button>
        <button class="tab-btn" data-lens="jlens">J-Lens</button>
        <button class="tab-btn" data-lens="logit_lens">Logit Lens</button>
        <button class="tab-btn" data-lens="all">All 4 Lenses</button>
      </div>

      <!-- SECTION 3 -->
      <h2>3 · Shared Neurons (5) <span class="badge b-shared">In both Rank 0 &amp; Rank 1 · L11–L19</span></h2>
      <p>
        Every shared survivor lies in the middle-late control band (L11–L19). Red chips show top promoted tokens (<span class="mono">+v</span>); blue chips show top suppressed / negative-activation tokens (<span class="mono">−v</span>). Hover any token chip for its exact vocab z-score.
      </p>
      <div class="table-wrap" id="tbl-shared"></div>

      <div class="callout">
        <strong>Orthogonal L17 pair (<span class="mono">#6230</span> vs. <span class="mono">#7054</span>):</strong> Both arms kept both L17 neurons even though Rank 1 forced them to share a single input read direction <span class="mono">a</span>. Yet their output vectors are completely orthogonal (<span class="mono">raw cos = +0.009</span>, <span class="mono">R-Lens cos = +0.042</span>): <span class="mono">#7054</span> writes LaTeX/heading delimiters (<span class="mono">\\[</span>, <span class="mono">##\\n\\n</span>, <span class="mono">;\\n\\n</span>) while <span class="mono">#6230</span> writes equality/assertion cues (<span class="mono">=</span>, <span class="mono">≠</span>, <span class="mono">≥</span>, <span class="mono">(True</span>)—two independent outputs triggered by one shared context.
      </div>

      <!-- SECTION 4 -->
      <h2>4 · Rank 0 Only Neurons (7) <span class="badge b-r0">Unconstrained Δ_j only</span></h2>
      <p>
        Includes three early-layer neurons (L3, L4, L6), two mid-layer formatting neurons (L12, L18), and two L27 number/digit neurons. Notice how <span class="mono">L3 #574</span> decodes cleanly to math/school word problems under <span class="mono">R-Lens</span> and <span class="mono">J++</span> (switch to Logit Lens or J-Lens above to see it disappear into noise).
      </p>
      <div class="table-wrap" id="tbl-r0-only"></div>

      <!-- SECTION 5 -->
      <h2>5 · Rank 1 Only Neurons (6) <span class="badge b-r1">Layer-shared read direction only</span></h2>
      <p>
        Mirrors Rank 0’s layer coverage with zero index overlap: early formatting (<span class="mono">L2 #4005</span>, <span class="mono">L6 #5466</span>), mid-layer CoT/markdown suppression (<span class="mono">L14 #3925</span>), GSM8K phrasing (<span class="mono">L26 #8751</span>: <span class="mono">Left</span>, <span class="mono">Done</span>), and a pure 0–9 digit neuron (<span class="mono">L27 #6214</span>).
      </p>
      <div class="table-wrap" id="tbl-r1-only"></div>

      <!-- SECTION 6 -->
      <h2>6 · Rank 0 (12) vs. Rank 1 (11): Geometry &amp; Substitutability</h2>
      <p>
        To test whether the non-overlapping neurons represent genuinely different solutions or redundant substitutes, we measure output-space effective rank (mirroring Section 6 of the training report), subspace principal angles, and pairwise 18×18 cosine similarities across all neurons.
      </p>

      <div class="grid-3">
        <div class="stat-card">
          <div class="stat-val" style="color:var(--r0)">8.05 – 9.69</div>
          <div class="stat-desc">Rank 0 output participation ratio (of 12) in J-Lens / R-Lens—more compressed than its <strong>11.6 / 12</strong> input PR.</div>
        </div>
        <div class="stat-card">
          <div class="stat-val" style="color:var(--shared)">+0.66</div>
          <div class="stat-desc">Strongest pair overall: cross-layer substitute <span class="mono">L12 #6760</span> (R0) ↔ <span class="mono">L6 #5466</span> (R1) in R-Lens (+0.67 J-Lens, +0.40 raw).</div>
        </div>
        <div class="stat-card">
          <div class="stat-val" style="color:var(--r1)">0.78, 0.58</div>
          <div class="stat-desc">Top-2 principal-angle cosines between <strong>rank-only</strong> subspaces (excluding the 5 shared neurons) in R-Lens space.</div>
        </div>
      </div>

      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Lens space</th><th>Rank 0 output PR (/12)</th><th>Rank 1 output PR (/11)</th>
              <th class="num">All R0↔R1 mean cos(θ)</th><th class="num">Rank-only mean cos(θ)</th><th>Rank-only top-3 cos(θ)</th>
            </tr>
          </thead>
          <tbody>
            {"".join(
                f"<tr><td><strong>{r['lens']}</strong></td><td class='mono'>{r['r0_pr']}</td><td class='mono'>{r['r1_pr']}</td>"
                f"<td class='mono num'>{r['all_cos']}</td><td class='mono num'>{r['only_cos']}</td><td class='mono'>[{r['only_top3']}]</td></tr>"
                for r in ARM_SUBSPACE_ROWS
            )}
          </tbody>
        </table>
      </div>

      <div class="fig-card">
        <div class="tab-bar" id="cos-tab-bar" style="margin-top:0">
          <span class="tab-label">Similarity space:</span>
          <button class="tab-btn active" data-cos="rlens">R-Lens Logit Space (cross-layer)</button>
          <button class="tab-btn" data-cos="jpp">J++ Logit Space</button>
          <button class="tab-btn" data-cos="jlens">J-Lens Logit Space</button>
          <button class="tab-btn" data-cos="raw">Raw down_proj (same-layer)</button>
        </div>
        <div class="hm-wrap" id="cos-heatmap-host"></div>
        <div class="hm-colorbar"><span>−1.0</span><div class="hm-grad"></div><span>+1.0</span></div>
        <div class="fig-caption"><strong>Figure 2. 18×18 pairwise cosine similarity matrix.</strong> Ordered <span style="color:var(--shared);font-weight:600">Shared (0–4)</span> | <span style="color:var(--r0);font-weight:600">Rank 0 only (5–11)</span> | <span style="color:var(--r1);font-weight:600">Rank 1 only (12–17)</span>. Off-diagonal values with <span class="mono">|cos| ≥ 0.25</span> are annotated. Hover any cell to compare raw, J-Lens, J++, and R-Lens cosines simultaneously.</div>
      </div>

      <h3>Key same-layer &amp; cross-layer neuron pairs</h3>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Pair</th><th>Relationship</th><th class="num">Raw</th><th class="num">J-Lens</th>
              <th class="num">J++</th><th class="num">R-Lens</th><th>Interpretation</th>
            </tr>
          </thead>
          <tbody>{"".join(KEY_PAIR_ROWS_HTML)}</tbody>
        </table>
      </div>

      <h3>Full-arm layer-by-layer view (Rank 0’s 12 vs. Rank 1’s 11)</h3>
      <div class="grid-2">
        <div class="table-wrap" style="margin:0">
          <table>
            <thead><tr><th colspan="3" style="color:var(--r0)">Rank 0 — Full Set (12 neurons · 94.4% CE gain)</th></tr></thead>
            <tbody>{build_arm_table_html(IDX_R0_ALL)}</tbody>
          </table>
        </div>
        <div class="table-wrap" style="margin:0">
          <table>
            <thead><tr><th colspan="3" style="color:var(--r1)">Rank 1 — Full Set (11 neurons · 91.1% CE gain)</th></tr></thead>
            <tbody>{build_arm_table_html(IDX_R1_ALL)}</tbody>
          </table>
        </div>
      </div>

      <!-- SECTION 7 -->
      <h2>7 · Targeted GSM8K &amp; Style Token Probe</h2>
      <div class="callout" style="border-left-color:#1d4ed8">
        <strong>What “probing specific GSM8K tokens” means (in plain English):</strong> Top-10 / bottom-10 lists search across all <strong>151,936 tokens</strong> in Qwen2.5’s vocabulary — which contains thousands of Chinese, code, and rare whitespace tokens. A neuron can still strongly boost or suppress a key GSM8K token like <span class="mono">####</span>, <span class="mono">&lt;&lt;</span>, <span class="mono">&gt;&gt;</span>, <span class="mono">**</span> (bold), <span class="mono">\\[</span> / <span class="mono">boxed</span> (LaTeX), or <span class="mono">&lt;|im_end|&gt;</span> (stop token) even if that token ranks #30 instead of #1. Probing pulls the exact standardized z-score for 18 hand-picked GSM8K, CoT/markdown, and EOS tokens across all 18 neurons so we can see directly: <em>which neuron pushes <span class="mono">####</span>? which one kills <span class="mono">**</span> / LaTeX? which one triggers stopping?</em>
      </div>
      <p>
        Single-token columns have zero aggregation; multi-variant columns (e.g., <span class="mono">digits 0–9</span> or <span class="mono">["####", " ####"]</span>) average the exact z-scores of their surface variants.
      </p>

      <div class="fig-card">
        <div class="tab-bar" id="probe-tab-bar" style="margin-top:0">
          <span class="tab-label">Probe lens:</span>
          <button class="tab-btn active" data-probe="rlens">R-Lens (default)</button>
          <button class="tab-btn" data-probe="jpp">J++</button>
          <button class="tab-btn" data-probe="jlens">J-Lens</button>
          <button class="tab-btn" data-probe="logit_lens">Logit Lens</button>
        </div>
        <div class="hm-wrap" id="probe-heatmap-host"></div>
        <div class="hm-colorbar"><span>−4.5σ (suppress)</span><div class="hm-grad"></div><span>+4.5σ (promote)</span></div>
        <div class="fig-caption"><strong>Figure 3. Targeted GSM8K gold-format vs. Qwen CoT/Markdown/LaTeX vs. EOS probe.</strong> Solid horizontal lines separate Shared / Rank-0-only / Rank-1-only; dashed vertical lines separate GSM8K gold format, Qwen CoT/markup, and EOS tokens. Cells with <span class="mono">|z| ≥ 1.5σ</span> are annotated. Hover any cell for exact z-scores across all 4 lenses.</div>
      </div>

      <h3>Functional role taxonomy across all 18 survivors</h3>
      <div class="table-wrap">
        <table>
          <thead><tr><th style="width:27%">Functional role</th><th style="width:25%">Neurons</th><th>Key R-Lens / J++ evidence</th></tr></thead>
          <tbody>{TAXONOMY_HTML}</tbody>
        </table>
      </div>

      <!-- SECTION 8 -->
      <h2>8 · Scope Caveats &amp; Not Done Yet</h2>
      <p>
        While the 3σ–15σ alignment with formatting, newlines, LaTeX, EOS, and digits is consistent with the style-compression hypothesis for the 90%+ CE recovery and −14 pt accuracy drop, static <span class="mono">down_proj</span> decoding alone cannot establish causality or rule out reasoning effects: several neurons (<span class="mono">L3 #574</span>, <span class="mono">L17 #6230</span>, <span class="mono">L4 #1398</span>) align with domain/assertion cues, a single 1,536-dim vector can be polysemantic, and the rewired <span class="mono">gate_proj</span>/<span class="mono">up_proj</span> inputs still determine <em>when</em> each output fires.
      </p>

      <div class="todo-card">
        <h4>1. Activation shifts &amp; active pole sign (<span class="mono">Δa_j</span> on <span class="mono">val128</span>, GPU / <span class="mono">sl-0</span>)</h4>
        <p>Measure <span class="mono">mean(Δa_j)</span>, positive/negative fraction, and top firing contexts for both arms. Critical for neurons like <span class="mono">L19 #5606</span>, which carries both unwanted markdown (<span class="mono">**</span> +8.2σ, <span class="mono">\\[</span> +6.0σ) and wanted <span class="mono">####</span> (+5.7σ) on <span class="mono">+v</span>—it may flip from negative during CoT to positive at answer termination.</p>
      </div>
      <div class="todo-card">
        <h4>2. Leave-one-out causal ablations (GPU / <span class="mono">sl-0</span>)</h4>
        <p>Turn off one survivor at a time and measure <span class="mono">ΔVal-CE</span>, <span class="mono">Δlength</span>, and <span class="mono">Δaccuracy</span>. Tests directly whether ablating 2–3 formatting/EOS switches (<span class="mono">L16 #5569</span>, <span class="mono">L19 #5606</span>, <span class="mono">L12 #6760</span>) recovers untuned accuracy (~80%).</p>
      </div>
      <div class="todo-card">
        <h4>3. In-distribution GSM8K finite-difference JVP (~1 min on A100)</h4>
        <p>Compute <span class="mono">(logits(h_L + εv) − logits(h_L − εv)) / 2ε</span> directly on <span class="mono">val128</span> completions to verify that Wikitext-fitted lenses match downstream Jacobian transport on GSM8K prompts.</p>
      </div>
      <div class="todo-card">
        <h4>4. L17–L19 SAE monosemanticity check (<span class="mono">HuggingAnalist/sae-qwen2.5-1.5B-res</span>)</h4>
        <p>Project shared <span class="mono">L17 #6230</span>, <span class="mono">L17 #7054</span>, <span class="mono">L19 #5606</span>, and <span class="mono">L18 #4890</span> onto the public residual-stream SAEs covering L17–L19 to check whether each surviving neuron aligns with a single sparse feature or a superposition.</p>
      </div>

      <!-- SECTION 9 -->
      <h2>9 · Provenance</h2>
      <div class="table-wrap">
        <table>
          <tbody>
            <tr><td style="width:24%"><strong>Base model</strong></td><td class="mono">Qwen/Qwen2.5-1.5B-Instruct (28 layers, d_model = 1536, d_mlp = 8960, tied W_U)</td></tr>
            <tr><td><strong>Pre-fitted lenses</strong></td><td class="mono">neuronpedia/jacobian-lens :: qwen2.5-1.5b-it/{{jlens,jpp,rlens}}/Salesforce-wikitext/ (source: bubskee/dolos-qwen25-workspace-lenses@4fdc5766)</td></tr>
            <tr><td><strong>Companion training run</strong></td><td class="mono">sparse-gold-sft-r0-1 (plan 7efb84879b260e66) &amp; sparse-gold-sft-r1-1 (plan 714f68e4ae2ff512)</td></tr>
            <tr><td><strong>Notebook &amp; cache</strong></td><td class="mono">notebooks/sparse-rewiring-down-proj-jlens.ipynb · notebooks/cache/down_proj_decoded.pt (25.05 MB)</td></tr>
            <tr><td><strong>Report outputs</strong></td><td class="mono">docs/sparse-rewiring-down-proj-report.html · notebooks/sparse-rewiring-down-proj-report.html</td></tr>
            <tr><td><strong>Generator script</strong></td><td class="mono">notebooks/build_down_proj_report.py (reproducible directly from the 25 MB cache)</td></tr>
          </tbody>
        </table>
      </div>

    </div>

    <div id="tooltip"></div>

    <script>
    const DATA = {JS_BUNDLE_STR};

    function esc(s) {{
      return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }}

    function tokenChips(list, poleClass) {{
      return list.map(([t, z]) => {{
        const sign = z >= 0 ? "+" : "";
        return `<span class="tok ${{poleClass}}" title="z = ${{sign}}${{z.toFixed(1)}}σ">${{esc(t)}}</span>`;
      }}).join("");
    }}

    function renderGroupTable(hostId, indices, lensMode, showOther) {{
      const host = document.getElementById(hostId);
      const lensTag = lensMode === "all" ? "All 4 lenses" : DATA.lens_labels[lensMode];
      const otherLabel = hostId === "tbl-r0-only" ? "Closest R1 (R-Lens)" : "Closest R0 (R-Lens)";
      const col5Hdr = showOther
        ? `<th style="width:13%">${{otherLabel}}<br><span class="muted" style="font-weight:400">+ cos(J,R / J++,R)</span></th>`
        : `<th style="width:12%">Lens agreement<br><span class="muted" style="font-weight:400">cos(J,R / J++,R)</span></th>`;
      let html = `<table><thead><tr>
        <th style="width:10%">Neuron</th>
        <th class="num" style="width:7%">Norm %ile</th>
        <th class="pole-col-pos" style="width:23%"><span class="pole-pos">+v Promoted</span><br><span class="muted" style="font-weight:400">(${{lensTag}})</span></th>
        <th class="pole-col-neg" style="width:23%"><span class="pole-neg">−v Suppressed / Neg act</span><br><span class="muted" style="font-weight:400">(${{lensTag}})</span></th>
        ${{col5Hdr}}
        <th style="width:24%">Clear reading</th>
      </tr></thead><tbody>`;

      for (const i of indices) {{
        const n = DATA.neurons[i];
        const agPair = `${{n.ag.j_r.toFixed(2)}} / ${{n.ag.jpp_r.toFixed(2)}}`;
        const agClass = (n.ag.jpp_r < 0 || n.ag.j_r < 0.35) ? "ag-pill ag-warn" : "ag-pill";
        const normClass = n.norm_pct <= 11.0 ? "norm-low" : "";

        let posHtml = "";
        let negHtml = "";
        if (lensMode === "all") {{
          posHtml = DATA.lens_names.map(ln => {{
            const tk = n.tokens[ln];
            return `<div class="lens-sub"><div class="lens-tag">${{DATA.lens_labels[ln]}}</div><div class="tok-row">${{tokenChips(tk.pos.slice(0, 5), "tok-pos")}}</div></div>`;
          }}).join("");
          negHtml = DATA.lens_names.map(ln => {{
            const tk = n.tokens[ln];
            return `<div class="lens-sub"><div class="lens-tag">${{DATA.lens_labels[ln]}}</div><div class="tok-row">${{tokenChips(tk.neg.slice(0, 5), "tok-neg")}}</div></div>`;
          }}).join("");
        }} else {{
          const tk = n.tokens[lensMode];
          posHtml = `<div class="tok-row">${{tokenChips(tk.pos.slice(0, 7), "tok-pos")}}</div>`;
          negHtml = `<div class="tok-row">${{tokenChips(tk.neg.slice(0, 7), "tok-neg")}}</div>`;
        }}

        let col5Td = "";
        if (showOther) {{
          if (n.closest_other) {{
            const c = n.closest_other;
            const cc = c.cos_r >= 0.35 ? "num-pos" : (c.cos_r <= -0.35 ? "num-neg" : "");
            col5Td = `<td class="mono" style="font-size:12px">
              <strong>${{esc(c.label)}}</strong> <span class="${{cc}}">(${{c.cos_r >= 0 ? "+" : ""}}${{c.cos_r.toFixed(2)}})</span><br>
              <span class="muted" style="font-size:10.5px">raw: ${{c.cos_raw >= 0 ? "+" : ""}}${{c.cos_raw.toFixed(2)}}</span><br>
              <span class="${{agClass}}" style="margin-top:4px" title="cos(J-Lens, R-Lens) / cos(J++, R-Lens)">${{agPair}}</span>
            </td>`;
          }} else {{
            col5Td = `<td class="muted">—</td>`;
          }}
        }} else {{
          col5Td = `<td class="mono nowrap">
            <span class="${{agClass}}" title="cos(J-Lens, R-Lens) / cos(J++, R-Lens)">${{agPair}}</span><br>
            <span class="muted" style="font-size:10.5px">LL↔R ${{n.ag.ll_r.toFixed(2)}}</span>
          </td>`;
        }}

        html += `<tr>
          <td class="mono nowrap"><strong>${{esc(n.label)}}</strong></td>
          <td class="mono num nowrap ${{normClass}}">${{n.norm_pct.toFixed(1)}}%</td>
          <td class="pole-col-pos">${{posHtml}}</td>
          <td class="pole-col-neg">${{negHtml}}</td>
          ${{col5Td}}
          <td class="muted-sm"><strong>${{esc(n.role_short)}}</strong><br><span style="font-size:12px;color:#4b5563">${{esc(n.role_detail)}}</span></td>
        </tr>`;
      }}
      html += `</tbody></table>`;
      host.innerHTML = html;
    }}

    function rdbuColor(val, vmax) {{
      const t = Math.max(-1, Math.min(1, val / vmax));
      let r, g, b;
      if (t >= 0) {{
        // White (#f9fafb) -> Rust/Red (#991b1b)
        r = Math.round(249 + (153 - 249) * t);
        g = Math.round(250 + (27  - 250) * t);
        b = Math.round(251 + (27  - 251) * t);
      }} else {{
        const u = -t;
        // White (#f9fafb) -> Cobalt/Navy (#1e3a8a)
        r = Math.round(249 + (30  - 249) * u);
        g = Math.round(250 + (58  - 250) * u);
        b = Math.round(251 + (138 - 251) * u);
      }}
      return `rgb(${{r}},${{g}},${{b}})`;
    }}

    function grpColor(g) {{
      return g === "shared" ? "#15803d" : (g === "r0_only" ? "#b43c12" : "#1d4ed8");
    }}

    // Tooltip helper
    const tooltip = document.getElementById("tooltip");
    function bindTooltips(container) {{
      container.querySelectorAll("[data-tip]").forEach(el => {{
        el.addEventListener("mouseenter", e => {{
          tooltip.innerHTML = el.getAttribute("data-tip");
          tooltip.style.display = "block";
        }});
        el.addEventListener("mousemove", e => {{
          const x = Math.min(window.innerWidth - 355, e.clientX + 14);
          const y = Math.min(window.innerHeight - 130, e.clientY + 14);
          tooltip.style.left = x + "px";
          tooltip.style.top = y + "px";
        }});
        el.addEventListener("mouseleave", () => {{ tooltip.style.display = "none"; }});
      }});
    }}

    function renderCosHeatmap(space) {{
      const host = document.getElementById("cos-heatmap-host");
      const M = DATA.cos_mats[space];
      let html = `<table class="hm-grid"><thead><tr><th></th>`;
      for (let c = 0; c < DATA.labels.length; c++) {{
        const divC = (c === 4 || c === 11) ? " hm-div-c" : "";
        html += `<th class="col-rot${{divC}}"><div style="color:${{grpColor(DATA.groups[c])}}">${{esc(DATA.labels[c])}}</div></th>`;
      }}
      html += `</tr></thead><tbody>`;
      for (let r = 0; r < DATA.labels.length; r++) {{
        const divR = (r === 4 || r === 11) ? " class='hm-div-r'" : "";
        html += `<tr${{divR}}><th style="text-align:right;padding-right:7px;color:${{grpColor(DATA.groups[r])}}">${{esc(DATA.labels[r])}}</th>`;
        for (let c = 0; c < DATA.labels.length; c++) {{
          const v = M[r][c];
          const bg = rdbuColor(v, 1.0);
          const fg = Math.abs(v) > 0.55 ? "#ffffff" : "#111827";
          const txt = (r !== c && Math.abs(v) >= 0.25) ? v.toFixed(2) : "";
          const divC = (c === 4 || c === 11) ? " hm-div-c" : "";
          const tip = esc(
            `${{DATA.labels[r]}} ↔ ${{DATA.labels[c]}}\\n` +
            `R-Lens: ${{DATA.cos_mats.rlens[r][c] >= 0 ? "+" : ""}}${{DATA.cos_mats.rlens[r][c].toFixed(3)}}  |  ` +
            `J++: ${{DATA.cos_mats.jpp[r][c] >= 0 ? "+" : ""}}${{DATA.cos_mats.jpp[r][c].toFixed(3)}}\\n` +
            `J-Lens: ${{DATA.cos_mats.jlens[r][c] >= 0 ? "+" : ""}}${{DATA.cos_mats.jlens[r][c].toFixed(3)}}  |  ` +
            `Raw: ${{DATA.cos_mats.raw[r][c] >= 0 ? "+" : ""}}${{DATA.cos_mats.raw[r][c].toFixed(3)}}`
          ).replace(/\\n/g, "<br>");
          html += `<td class="hm-cell${{divC}}" style="background:${{bg}};color:${{fg}}" data-tip="${{tip}}">${{txt}}</td>`;
        }}
        html += `</tr>`;
      }}
      html += `</tbody></table>`;
      host.innerHTML = html;
      bindTooltips(host);
    }}

    function renderProbeHeatmap(lens) {{
      const host = document.getElementById("probe-heatmap-host");
      const M = DATA.probe_mats[lens];
      const cols = DATA.probe_cols;
      let html = `<table class="hm-grid"><thead><tr><th></th>`;
      for (let c = 0; c < cols.length; c++) {{
        const dash = (c === 7 || c === 15) ? " hm-dash-c" : "";
        html += `<th class="col-rot${{dash}}"><div>${{esc(cols[c].name)}}</div></th>`;
      }}
      html += `</tr></thead><tbody>`;
      for (let r = 0; r < DATA.labels.length; r++) {{
        const divR = (r === 4 || r === 11) ? " class='hm-div-r'" : "";
        const gLbl = DATA.groups[r] === "shared" ? "shared" : (DATA.groups[r] === "r0_only" ? "R0" : "R1");
        html += `<tr${{divR}}><th style="text-align:right;padding-right:7px;color:${{grpColor(DATA.groups[r])}}">${{esc(DATA.labels[r])}} <span style="opacity:.65">[${{gLbl}}]</span></th>`;
        for (let c = 0; c < cols.length; c++) {{
          const v = M[r][c];
          const bg = rdbuColor(v, 4.5);
          const fg = Math.abs(v) > 2.5 ? "#ffffff" : "#111827";
          const fw = Math.abs(v) >= 2.5 ? "700" : "400";
          const txt = Math.abs(v) >= 1.5 ? (v >= 0 ? "+" : "") + v.toFixed(1) : "";
          const dash = (c === 7 || c === 15) ? " hm-dash-c" : "";
          const tip = esc(
            `${{DATA.labels[r]}}  ×  ${{cols[c].name}}\\n` +
            `R-Lens: ${{DATA.probe_mats.rlens[r][c] >= 0 ? "+" : ""}}${{DATA.probe_mats.rlens[r][c].toFixed(2)}}σ  |  ` +
            `J++: ${{DATA.probe_mats.jpp[r][c] >= 0 ? "+" : ""}}${{DATA.probe_mats.jpp[r][c].toFixed(2)}}σ\\n` +
            `J-Lens: ${{DATA.probe_mats.jlens[r][c] >= 0 ? "+" : ""}}${{DATA.probe_mats.jlens[r][c].toFixed(2)}}σ  |  ` +
            `Logit: ${{DATA.probe_mats.logit_lens[r][c] >= 0 ? "+" : ""}}${{DATA.probe_mats.logit_lens[r][c].toFixed(2)}}σ`
          ).replace(/\\n/g, "<br>");
          html += `<td class="hm-cell${{dash}}" style="background:${{bg}};color:${{fg}};font-weight:${{fw}};width:38px" data-tip="${{tip}}">${{txt}}</td>`;
        }}
        html += `</tr>`;
      }}
      html += `</tbody></table>`;
      host.innerHTML = html;
      bindTooltips(host);
    }}

    function renderAllNeuronTables(lensMode) {{
      renderGroupTable("tbl-shared",  DATA.idx_shared,  lensMode, false);
      renderGroupTable("tbl-r0-only", DATA.idx_r0_only, lensMode, true);
      renderGroupTable("tbl-r1-only", DATA.idx_r1_only, lensMode, true);
    }}

    // Wire up tabs
    document.querySelectorAll("#global-lens-bar .tab-btn").forEach(btn => {{
      btn.addEventListener("click", () => {{
        document.querySelectorAll("#global-lens-bar .tab-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        renderAllNeuronTables(btn.dataset.lens);
      }});
    }});
    document.querySelectorAll("#cos-tab-bar .tab-btn").forEach(btn => {{
      btn.addEventListener("click", () => {{
        document.querySelectorAll("#cos-tab-bar .tab-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        renderCosHeatmap(btn.dataset.cos);
      }});
    }});
    document.querySelectorAll("#probe-tab-bar .tab-btn").forEach(btn => {{
      btn.addEventListener("click", () => {{
        document.querySelectorAll("#probe-tab-bar .tab-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        renderProbeHeatmap(btn.dataset.probe);
      }});
    }});

    // Initial render
    renderAllNeuronTables("rlens");
    renderCosHeatmap("rlens");
    renderProbeHeatmap("rlens");
    </script>
    </body>
    </html>
    """

    clean_html = HTML_DOC.rstrip() + "\n"
    for p in OUT_PATHS:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(clean_html)
    return [Path(p) for p in OUT_PATHS]
