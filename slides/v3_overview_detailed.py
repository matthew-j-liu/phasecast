"""Detailed v3 deck -> slides/v3_overview_detailed.pptx.

Reuses the base deck's helpers and unchanged slides (title, dividers, A1-A3, A6,
B1, B6) and replaces A4, A5, B2, B3, B4, B5 with concrete, mechanism-level
visuals (vignettes, a real posterior, embedding internals, attention + pooling,
the training loop).

    conda run -n alignn_fix python slides/v3_overview_detailed.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

import v3_overview as base
from v3_overview import (
    INK, BLUE, TEAL, PURPLE, ORANGE, GREEN, GRAYL, GRAYT, WHITE, GREENCHK, REDX,
    _rgb, _lighten, _text, box, flow_arrow, line, hglyph, header, caption, notes,
    new_slide, make, title_slide, divider,
    a1_unitaskers, a2_foundation, a3_ssl, a6_axes,
    pipeline, PX, PW, PY, PH,
)

MASKG = "9AA3B2"


def tok(s, x, y, w, h, sym, amt, color, masked=False):
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = _rgb(MASKG if masked else _lighten(color, 0.78))
    shp.line.color.rgb = _rgb(MASKG if masked else color)
    shp.line.width = Pt(2.5 if masked else 1.75)
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    _text(tf, ["[MASK]", ("hide amount",)] if masked else [sym, (amt,)],
          17 if masked else 18, WHITE if masked else INK, bold=True)
    return shp


def hbar(s, x, y, maxw, value, label, color, vmax=0.5):
    bw = max(0.12, maxw * value / vmax)
    bar = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(bw), Inches(0.32))
    bar.fill.solid(); bar.fill.fore_color.rgb = _rgb(color)
    bar.line.fill.background(); bar.shadow.inherit = False
    t = s.shapes.add_textbox(Inches(x + bw + 0.1), Inches(y - 0.05), Inches(4.2), Inches(0.4))
    _text(t.text_frame, f"{label}  {value:.2f}", 12, INK, align=PP_ALIGN.LEFT)


def gaussian(s, x, y, w, h, comps, color, label):
    n = 48
    top = []
    for i in range(n + 1):
        t = i / n
        v = sum(wt * math.exp(-0.5 * ((t - mu) / sig) ** 2) for mu, sig, wt in comps)
        top.append((t, v))
    mx = max(v for _, v in top) or 1.0
    base_y = y + h
    pts = [(x + t * w, base_y - (v / mx) * h) for t, v in top]
    fb = s.shapes.build_freeform(Inches(x), Inches(base_y), scale=1.0)
    seg = [(Inches(px), Inches(py)) for px, py in pts] + [(Inches(x + w), Inches(base_y))]
    fb.add_line_segments(seg, close=True)
    shp = fb.convert_to_shape()
    shp.fill.solid(); shp.fill.fore_color.rgb = _rgb(_lighten(color, 0.72))
    shp.line.color.rgb = _rgb(color); shp.line.width = Pt(2)
    shp.shadow.inherit = False
    line(s, x, base_y, x + w, base_y, color=GRAYT, width=1)
    t = s.shapes.add_textbox(Inches(x - 0.2), Inches(base_y + 0.05), Inches(w + 0.4), Inches(0.4))
    _text(t.text_frame, label, 12, INK, bold=True)


def a4_fanout(prs):
    s = new_slide(prs)
    header(s, "ACT 1  -  A4", "One encoder, many applications")
    hglyph(s, 1.7, 3.7, r=0.92)
    caption(s, 3.3, 1.45, 9.5, "NEEDS THE LEARNED LATENT", 13, TEAL, align=PP_ALIGN.LEFT)
    vg = [("element imputation", "rock missing Yb  ->  h  ->  fill 2.1 ppm"),
          ("k-NN similarity retrieval", "query rock  ->  3 nearest real rocks"),
          ("OOD / measurement-error flag", "anomaly score 0.92  ->  likely bad analysis")]
    for i, (title, ex) in enumerate(vg):
        box(s, 3.3 + i * 3.25, 1.8, 3.0, 1.7, [title, "", (ex,)], TEAL, size=12)
        line(s, 2.62, 3.7, 3.3 + i * 3.25, 2.65, TEAL, 1.5)
    caption(s, 3.3, 3.95, 9.5, "COMPUTABLE FROM FEATURES  -  encoder does them for free",
            13, GRAYT, align=PP_ALIGN.LEFT)
    for i, (title, ex) in enumerate([("tectonic-setting class", "discrimination diagrams already do this"),
                                     ("petrogenetic ratios", "La/Yb, Sr/Y ... just division when present")]):
        box(s, 3.3 + i * 3.25, 4.35, 3.0, 1.1, [title, (ex,)], GRAYT, lit=False, size=12)
        line(s, 2.62, 3.85, 3.3 + i * 3.25, 4.9, GRAYL, 1.2)
    box(s, 9.8, 4.35, 2.95, 1.1, ["+ supervised heads", ("ModeHead, CompHead, CondHead...",)],
        GREEN, size=12)
    caption(s, 0.6, 6.35, 12.2, "The distinctive value is the shared learned geometry "
            "(distance, anomaly, graceful missingness) -- and doing all of it from one model.",
            13, INK, align=PP_ALIGN.LEFT)
    notes(s, "Ranked by how representation-dependent each is. Tier 1 needs the latent; tier 2 "
             "is computable from features and falls out for free (illustrating breadth, not "
             "distinctiveness). The encoder's edge is the shared geometry + the aggregate.")


def a5_inverse(prs):
    s = new_slide(prs)
    header(s, "ACT 1  -  A5", "The headline: an inverse, probabilistic emulator")
    box(s, 0.55, 2.9, 2.5, 1.7, ["basalt bulk", "", ("SiO2 50  Al2O3 16",), ("MgO 7.5  FeO 11",),
        ("Sr 400 ppm ...",)], BLUE, size=12)
    flow_arrow(s, 3.15, 3.55, 0.5, 0.4)
    hglyph(s, 4.5, 3.75, r=0.72)
    flow_arrow(s, 5.35, 3.55, 0.5, 0.4)
    # ModeHead panel
    box(s, 6.0, 1.55, 6.9, 2.25, [""], GREEN, size=10)
    caption(s, 6.2, 1.6, 6.5, "ModeHead  -  modal mineralogy (Dirichlet)", 12, GREEN, align=PP_ALIGN.LEFT)
    modes = [("plagioclase", 0.42), ("clinopyroxene", 0.28), ("olivine", 0.16),
             ("orthopyroxene", 0.08), ("oxides", 0.04)]
    for i, (m, v) in enumerate(modes):
        hbar(s, 6.25, 2.05 + i * 0.34, 2.3, v, m, GREEN)
    # CondHead panel
    box(s, 6.0, 4.0, 6.9, 2.55, [""], PURPLE, size=10)
    caption(s, 6.2, 4.05, 6.5, "CondHead  -  formation conditions (per-axis GMM)", 12, PURPLE, align=PP_ALIGN.LEFT)
    gaussian(s, 6.35, 4.65, 1.9, 1.05, [(0.5, 0.16, 1.0)], BLUE, "T  1140 +/- 35 C")
    gaussian(s, 8.5, 4.65, 1.9, 1.05, [(0.5, 0.18, 1.0)], TEAL, "P  4.8 +/- 1.9 kbar")
    gaussian(s, 10.65, 4.65, 1.95, 1.05, [(0.32, 0.1, 1.0), (0.72, 0.12, 0.8)], ORANGE,
             "H2O  bimodal: ~1 or ~3 wt%")
    caption(s, 0.55, 6.75, 12.3, "MAGEMin: you pick T, P, H2O  ->  one assemblage.   "
            "v3: bulk in  ->  the whole posterior (note H2O is multimodal).", 13, INK, align=PP_ALIGN.LEFT)
    notes(s, "The product slide, made concrete. ModeHead emits a Dirichlet over minerals; "
             "CondHead a per-axis GMM. H2O drawn bimodal to make 'multimodal posterior' literal "
             "-- two plausible hydration regimes consistent with one bulk. This is the integral "
             "MAGEMin doesn't natively do.")


def b1_tokens(prs):
    s = new_slide(prs)
    header(s, "ACT 2 - HOW IT WORKS  -  B1", "A rock is a bag of (element, amount) tokens")
    pipeline(s, 0)
    toks = [("Si", "0.50", BLUE), ("Al", "0.16", BLUE), ("Ca", "0.10", BLUE),
            ("Mg", "0.075", BLUE), ("Sr", "4e-4", TEAL), ("Yb", "2e-6", TEAL)]
    tw = 1.5
    x0 = (13.333 - (len(toks) * tw + (len(toks) - 1) * 0.22)) / 2
    for i, (sym, amt, col) in enumerate(toks):
        tok(s, x0 + i * (tw + 0.22), 4.95, tw, 1.0, sym, amt, col)
    caption(s, 0.55, 6.1, 12.2, "9 majors (wt%/100, blue)  +  13 traces (ppm/1e6, teal); "
            "absent = padding.  No T, P, or H2O ever enters.", 12, GRAYT)
    notes(s, "Input = up to 22 (element id, amount) pairs. That's the entire model input.")


def b2_embed(prs):
    s = new_slide(prs)
    header(s, "ACT 2  -  B2", "Each token becomes a chemistry-aware vector")
    tok(s, 0.7, 3.15, 1.6, 1.2, "Si", "0.50", BLUE)
    line(s, 2.3, 3.45, 3.2, 2.55, GRAYT, 1.4)
    line(s, 2.3, 4.05, 3.2, 4.75, GRAYT, 1.4)
    box(s, 3.2, 2.15, 2.5, 0.85, ["atomic number 14"], PURPLE, size=12)
    flow_arrow(s, 5.8, 2.35, 0.45, 0.4)
    box(s, 6.35, 1.95, 4.4, 1.25, ["mat2vec lookup", ("200-d element vector,",),
        ("pretrained on materials-science text",)], PURPLE, size=12)
    box(s, 3.2, 4.4, 2.5, 0.85, ["amount  0.50"], TEAL, size=12)
    flow_arrow(s, 5.8, 4.6, 0.45, 0.4)
    box(s, 6.35, 4.2, 4.4, 1.05, ["linear projection", ("amount -> vector",)], TEAL, size=12)
    box(s, 11.0, 3.0, 1.9, 1.5, ["+", ("token", "vector",), ("256-d",)], ORANGE, size=18)
    line(s, 10.75, 2.55, 11.0, 3.4, GRAYT, 1.4)
    line(s, 10.75, 4.7, 11.0, 4.1, GRAYT, 1.4)
    caption(s, 0.55, 6.2, 12.2, "mat2vec is a cross-field import (materials informatics, CrabNet) "
            "-- a learned chemistry vector per element, scaled by how much is present.", 12, INK)
    notes(s, "Opens the 'embed' black box: element id -> mat2vec chemistry vector, amount -> "
             "projection, summed into the per-token vector.")


def b3_encoder(prs):
    s = new_slide(prs)
    header(s, "ACT 2  -  B3", "The encoder mixes elements, then pools to one h")
    syms = ["Si", "Al", "Mg", "Fe"]
    gx, gy, c = 1.1, 2.2, 0.62
    caption(s, gx, gy - 0.5, 4, "self-attention: every token attends to every token", 12, PURPLE, align=PP_ALIGN.LEFT)
    for i, a in enumerate(syms):
        tb = s.shapes.add_textbox(Inches(gx - 0.55), Inches(gy + 0.05 + i * c), Inches(0.5), Inches(0.4))
        _text(tb.text_frame, a, 11, INK, bold=True)
        tb2 = s.shapes.add_textbox(Inches(gx + i * c), Inches(gy - 0.4), Inches(0.6), Inches(0.3))
        _text(tb2.text_frame, a, 11, INK, bold=True)
    for i in range(len(syms)):
        for j in range(len(syms)):
            w = 0.85 if i == j else 0.30 + 0.12 * ((i + j) % 4)
            cell = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(gx + j * c), Inches(gy + i * c), Inches(c - 0.06), Inches(c - 0.06))
            cell.fill.solid(); cell.fill.fore_color.rgb = _rgb(_lighten(PURPLE, 1 - w))
            cell.line.color.rgb = _rgb(WHITE); cell.line.width = Pt(1)
            cell.shadow.inherit = False
    flow_arrow(s, 4.9, 3.5, 0.7, 0.45)
    box(s, 5.8, 2.95, 3.4, 1.5, ["masked-mean pool", ("average the real tokens", ), ("(skip padding)",)], TEAL, size=14)
    flow_arrow(s, 9.4, 3.5, 0.7, 0.45)
    hglyph(s, 11.3, 3.7, r=0.85)
    caption(s, 0.55, 6.2, 12.2, "attention lets elements interact; pooling collapses the 22 "
            "contextualized vectors into a single 256-d fingerprint h.", 12, INK)
    notes(s, "Makes 'self-attention' and 'mean-pool' literal: a 4x4 attention heatmap (every "
             "token attends to every token, diagonal strongest) then averaging the non-padding "
             "tokens into h.")


def b4_train(prs):
    s = new_slide(prs)
    header(s, "ACT 2  -  B4", "How the encoder learned: fill in the blank")
    caption(s, 0.55, 1.15, 12.2, "like predicting a missing word from its sentence -- but the "
            "'words' are elements and the blank is an amount", 13, GRAYT, align=PP_ALIGN.LEFT)
    toks = [("Si", "0.50"), ("Al", "0.16"), ("Ca", "0.10"), ("Mg", "0.075"), ("Fe", "0.11"), ("Na", "0.025"), ("Sr", "4e-4")]
    mi, tw = 3, 1.5
    x0 = (13.333 - (len(toks) * tw + (len(toks) - 1) * 0.22)) / 2
    xs = [x0 + i * (tw + 0.22) for i in range(len(toks))]
    for i, (sym, amt) in enumerate(toks):
        tok(s, xs[i], 1.95, tw, 1.05, sym, amt, TEAL if i == 6 else BLUE, masked=(i == mi))
    enc_y = 3.75
    for i in range(len(toks)):
        cx = xs[i] + tw / 2
        line(s, cx, 3.0, cx, enc_y, color=(MASKG if i == mi else GRAYL), width=1.4)
    box(s, x0, enc_y, len(toks) * tw + (len(toks) - 1) * 0.22, 0.95,
        ["Transformer encoder", ("being trained -- weights updated each step (not frozen)",)], PURPLE, size=15)
    mcx = xs[mi] + tw / 2
    flow_arrow(s, mcx - 0.16, enc_y + 1.0, 0.32, 0.38)
    box(s, 2.3, 5.35, 3.0, 1.0, ["encoder predicts", ("Mg amount ~ 0.071",)], ORANGE, size=13)
    box(s, 5.6, 5.35, 1.9, 1.0, ["truth", ("0.075",)], GRAYT, lit=False, size=13)
    flow_arrow(s, 5.35, 5.65, 0.28, 0.4, lit=True)
    flow_arrow(s, 7.55, 5.65, 0.28, 0.4, lit=True)
    box(s, 7.9, 5.35, 2.9, 1.0, ["log-space", "MSE loss"], REDX, size=14)
    line(s, 10.8, 5.85, 12.7, 5.85, REDX, 2.5)
    line(s, 12.7, 5.85, 12.7, enc_y + 0.5, REDX, 2.5)
    flow_arrow(s, 12.15, enc_y + 0.32, 0.55, 0.34, lit=True)
    a = s.shapes.add_shape(MSO_SHAPE.LEFT_ARROW, Inches(12.15), Inches(enc_y + 0.32), Inches(0.55), Inches(0.34))
    a.fill.solid(); a.fill.fore_color.rgb = _rgb(REDX); a.line.fill.background(); a.shadow.inherit = False
    cap = s.shapes.add_textbox(Inches(9.6), Inches(4.5), Inches(3.4), Inches(0.6))
    _text(cap.text_frame, ["backpropagation:", "updates encoder weights"], 12, REDX, bold=True)
    notes(s, "Masked-element modelling: hide one element's amount, predict it from the rest, "
             "backprop the error into the encoder. Encoder is being TRAINED here (frozen is the "
             "downstream state). Loss is log-space so traces get supervised. Repeat over 474k rocks.")


def b5_heads(prs):
    s = new_slide(prs)
    header(s, "ACT 2  -  B5", "Heads ride on the same frozen h")
    pipeline(s, 3, hide_after=3)
    heads = [("ModeHead", "mineral modes (see A5)"), ("CompHead", "per-mineral composition"),
             ("CondHead", "T, P, H2O (see A5)"), ("ClassHead", "rock-class odds"), ("G_sys", "free energy")]
    for i, (name, shape) in enumerate(heads):
        y = 1.25 + i * 1.05
        box(s, 9.7, y, 3.1, 0.9, [name, (shape,)], GREEN, size=13)
        line(s, 8.98, 3.78, 9.7, y + 0.45, GREEN, 1.5)
    caption(s, 0.55, 5.6, 8.5, "now frozen -- swap or add heads without retraining the encoder; "
            "the actual outputs are on A5", 12, INK)
    notes(s, "Structural view of the heads on the now-frozen h. Outputs themselves are drawn on "
             "A5 (don't redraw the distributions twice).")


def b6_full(prs):
    s = new_slide(prs)
    header(s, "ACT 2  -  B6", "One shared backbone, many outputs")
    pipeline(s, 4)
    box(s, 8.7, 0.95, 2.6, 0.85, ["imputation"], TEAL, size=12)
    box(s, 8.7, 5.7, 2.6, 0.85, ["k-NN retrieval"], TEAL, size=12)
    cy = PY + PH / 2
    line(s, PX[3] + PW / 2, cy - 0.78, 9.9, 1.8, TEAL, 1.5)
    line(s, PX[3] + PW / 2, cy + 0.78, 9.9, 5.7, TEAL, 1.5)
    caption(s, 0.55, 5.05, 8.5, "bulk -> tokens -> embed -> encoder -> h -> {heads / apps}  --  "
            "microseconds per rock", 13, INK, align=PP_ALIGN.LEFT)
    notes(s, "Whole pipeline lit: h feeds the supervised heads and the training-free apps alike.")


if __name__ == "__main__":
    prs = make()
    title_slide(prs)
    divider(prs, "ACT 1", "Why a foundation model -- and what it unlocks")
    a1_unitaskers(prs); a2_foundation(prs); a3_ssl(prs)
    a4_fanout(prs); a5_inverse(prs); a6_axes(prs)
    divider(prs, "ACT 2", "How the transformer works")
    b1_tokens(prs); b2_embed(prs); b3_encoder(prs); b4_train(prs); b5_heads(prs); b6_full(prs)
    out = Path(__file__).resolve().parent / "v3_overview_detailed.pptx"
    prs.save(out)
    print(f"wrote {out}  ({len(prs.slides._sldIdLst)} slides)")
