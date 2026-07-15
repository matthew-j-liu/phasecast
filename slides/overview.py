"""Living model/pipeline overview deck -> slides/overview.pptx.

The deck's sole job is to visualize the current canonical model and the
pipeline/workflow that trains and uses it -- components, data flow at training
and inference, and how they compose. It deliberately excludes positioning,
comparisons to other work, and value framing (the owner layers that on
separately). Grows and retires sections as the architecture does.

    conda run -n alignn_fix python slides/overview.py
"""

import math
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

INK = "1A2238"
BLUE = "2E5BFF"
TEAL = "12B5A5"
PURPLE = "6C5CE7"
ORANGE = "E8883A"
GREEN = "2A9D8F"
GRAYL = "C8CDD6"
GRAYT = "8A93A3"
WHITE = "FFFFFF"
REDX = "C0392B"
MASKG = "9AA3B2"

W, H = Inches(13.333), Inches(7.5)


def _rgb(h):
    return RGBColor.from_string(h)


def _lighten(h, amt=0.85):
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * amt) for c in (r, g, b))
    return f"{r:02X}{g:02X}{b:02X}"


def _text(tf, lines, size, color, bold=False, align=PP_ALIGN.CENTER):
    tf.word_wrap = True
    for i, ln in enumerate(lines if isinstance(lines, list) else [lines]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        sub = isinstance(ln, tuple)
        run = p.add_run()
        run.text = ln[0] if sub else ln
        run.font.size = Pt(size - 3 if sub else size)
        run.font.bold = bold and not sub
        run.font.color.rgb = _rgb(GRAYT if sub else color)


def box(s, x, y, w, h, lines, color, lit=True, size=13):
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = _rgb(_lighten(color, 0.86) if lit else WHITE)
    shp.line.color.rgb = _rgb(color if lit else GRAYL)
    shp.line.width = Pt(2 if lit else 1)
    shp.shadow.inherit = False
    shp.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    _text(shp.text_frame, lines, size, INK if lit else GRAYT, bold=True)
    return shp


def flow_arrow(s, x, y, w, h, lit=True, shape=MSO_SHAPE.RIGHT_ARROW):
    a = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    a.fill.solid(); a.fill.fore_color.rgb = _rgb(INK if lit else GRAYL)
    a.line.fill.background(); a.shadow.inherit = False
    return a


def line(s, x1, y1, x2, y2, color=GRAYT, width=1.5):
    c = s.shapes.add_connector(2, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = _rgb(color); c.line.width = Pt(width)
    c.shadow.inherit = False
    return c


def hglyph(s, cx, cy, lit=True, r=0.85):
    shp = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - r), Inches(cy - r), Inches(2 * r), Inches(2 * r))
    shp.fill.solid(); shp.fill.fore_color.rgb = _rgb(ORANGE if lit else WHITE)
    shp.line.color.rgb = _rgb(ORANGE if lit else GRAYL); shp.line.width = Pt(2.5 if lit else 1)
    shp.shadow.inherit = False
    shp.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    _text(shp.text_frame, ["h", ("256-d latent",)], 22, WHITE if lit else GRAYT, bold=True)
    return shp


def header(s, kicker, title):
    k = s.shapes.add_textbox(Inches(0.55), Inches(0.3), Inches(12), Inches(0.4))
    _text(k.text_frame, kicker, 13, BLUE, bold=True, align=PP_ALIGN.LEFT)
    t = s.shapes.add_textbox(Inches(0.55), Inches(0.66), Inches(12.2), Inches(0.8))
    _text(t.text_frame, title, 26, INK, bold=True, align=PP_ALIGN.LEFT)


def caption(s, x, y, w, lines, size=12, color=GRAYT, align=PP_ALIGN.CENTER):
    t = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.8))
    _text(t.text_frame, lines, size, color, align=align)


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


def new_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def make():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    return prs


def tok(s, x, y, w, h, sym, amt, color, masked=False):
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = _rgb(MASKG if masked else _lighten(color, 0.78))
    shp.line.color.rgb = _rgb(MASKG if masked else color); shp.line.width = Pt(2.5 if masked else 1.75)
    shp.shadow.inherit = False
    shp.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    _text(shp.text_frame, ["[MASK]", ("hide amount",)] if masked else [sym, (amt,)],
          17 if masked else 18, WHITE if masked else INK, bold=True)
    return shp


def hbar(s, x, y, maxw, value, label, color, vmax=0.5, ci=None, presence=None):
    bw = max(0.12, maxw * value / vmax)
    bar = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(bw), Inches(0.32))
    bar.fill.solid()
    # fade the bar when the phase may genuinely be absent, so uncertain
    # presence reads at a glance instead of only in the p number
    faded = presence is not None and presence < 0.75
    bar.fill.fore_color.rgb = _rgb(_lighten(color, 0.6) if faded else color)
    bar.line.fill.background(); bar.shadow.inherit = False
    # ci is the credible-interval half-width in value units: draw a whisker
    # centered on the bar tip with end caps, so a bar reads as a posterior
    # mean with spread rather than a point estimate.
    if ci is not None:
        cy = y + 0.16
        half = maxw * ci / vmax
        lo, hi = max(x + 0.02, x + bw - half), x + bw + half
        line(s, lo, cy, hi, cy, color=INK, width=1.25)
        line(s, lo, cy - 0.07, lo, cy + 0.07, color=INK, width=1.25)
        line(s, hi, cy - 0.07, hi, cy + 0.07, color=INK, width=1.25)
        x_text = hi + 0.1
    else:
        x_text = x + bw + 0.1
    label_text = f"{label}  {value:.2f}"
    if presence is not None:
        label_text += f"  |  p {presence:.2f}"
    t = s.shapes.add_textbox(Inches(x_text), Inches(y - 0.05), Inches(4.6), Inches(0.4))
    _text(t.text_frame, label_text, 12, INK, align=PP_ALIGN.LEFT)


def gaussian(s, x, y, w, h, comps, color, label):
    n = 48
    top = [(i / n, sum(wt * math.exp(-0.5 * ((i / n - mu) / sig) ** 2) for mu, sig, wt in comps))
           for i in range(n + 1)]
    mx = max(v for _, v in top) or 1.0
    by = y + h
    pts = [(x + t * w, by - (v / mx) * h) for t, v in top]
    fb = s.shapes.build_freeform(Inches(x), Inches(by), scale=1.0)
    fb.add_line_segments([(Inches(px), Inches(py)) for px, py in pts] + [(Inches(x + w), Inches(by))], close=True)
    shp = fb.convert_to_shape()
    shp.fill.solid(); shp.fill.fore_color.rgb = _rgb(_lighten(color, 0.72))
    shp.line.color.rgb = _rgb(color); shp.line.width = Pt(2); shp.shadow.inherit = False
    line(s, x, by, x + w, by, color=GRAYT, width=1)
    t = s.shapes.add_textbox(Inches(x - 0.2), Inches(by + 0.05), Inches(w + 0.4), Inches(0.4))
    _text(t.text_frame, label, 12, INK, bold=True)


STAGES = [(["input tokens", ("22 elements",)], BLUE), (["embed", ("mat2vec + amount",)], TEAL),
          (["transformer", "encoder"], PURPLE), (None, ORANGE), (["heads", ("parallel",)], GREEN)]
PX = [0.55, 3.05, 5.6, 8.2, 10.75]
PW, PY, PH = 2.0, 3.0, 1.55


def pipeline(s, lit_through, hide_after=None):
    cy = PY + PH / 2
    for i, (lines, color) in enumerate(STAGES):
        if hide_after is not None and i > hide_after:
            continue
        lit = i <= lit_through
        if i == 3:
            hglyph(s, PX[i] + PW / 2, cy, lit=lit, r=0.78)
        else:
            box(s, PX[i], PY, PW, PH, lines, color, lit=lit, size=12)
        if i > 0:
            pr = PX[i - 1] + PW
            flow_arrow(s, pr + 0.05, cy - 0.2, PX[i] - pr - 0.1, 0.4, lit=(i <= lit_through))


def s_scenarios(prs):
    s = new_slide(prs)
    header(s, "HEADS", "The coherent answer: ranked scenarios")
    caption(s, 0.55, 1.15, 12.2, "the marginal table can't show WHICH minerals trade off; "
            "scenarios can -- sample the posterior, cluster by assemblage, rank by weight",
            13, GRAYT, align=PP_ALIGN.LEFT)
    box(s, 0.7, 1.95, 5.9, 3.6, [
        "Scenario A -- deep & wet . weight ~45%",
        ("P 6-9 kbar   H2O 4-6 wt%   T ~950-990 C",), "",
        ("phenocrysts (wt% of cargo, sums to 100):",),
        ("plagioclase 44 +/- 4    clinopyroxene 26 +/- 3",),
        ("amphibole 18 +/- 4      Fe-Ti oxides 12 +/- 2",)], GREEN, size=14)
    box(s, 6.9, 1.95, 5.9, 3.6, [
        "Scenario B -- shallow & dry . weight ~35%",
        ("P 1-3 kbar   H2O 1-2.5 wt%   T ~1000-1040 C",), "",
        ("phenocrysts (wt% of cargo, sums to 100):",),
        ("plagioclase 48 +/- 4    clinopyroxene 24 +/- 3",),
        ("orthopyroxene 16 +/- 3  Fe-Ti oxides 12 +/- 2",)], TEAL, size=14)
    caption(s, 0.7, 5.7, 12.1, "+ intermediate scenario (amp + opx coexist, ~12%) + tail (~8%) -- "
            "nothing silently dropped; weights are prior-weighted plausibility, reweightable",
            12, INK, align=PP_ALIGN.LEFT)
    caption(s, 0.7, 6.35, 12.1, "per-phase composition (An#, Mg#) is bulk-controlled -- one shared "
            "band across scenarios, not re-predicted per scenario. Each scenario is a falsifiable "
            "hypothesis: amphibole in thin section selects A, and with it A's conditions.",
            12, GRAYT, align=PP_ALIGN.LEFT)
    notes(s, "View 2 from heads.md, modernized to the locked corpus design: crystals-renormalized "
             "(no residual-melt row -- melt-fraction head dropped), composition shared across "
             "scenarios (CompHead reads h only), no inference-time mass-balance checkmark (closure "
             "is teacher-forced at training; the inference residual as OOD signal is an open "
             "rethink). Marginal view = projection of these scenarios; scenarios are the primary "
             "object. Open mechanism detail for #10/#13: coupling conditions to assemblage in a "
             "sampled scenario requires joint sampling across heads. Numbers illustrative.")


def s_title(prs):
    s = new_slide(prs)
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(2.6), W, Inches(1.9))
    band.fill.solid(); band.fill.fore_color.rgb = _rgb(INK)
    band.line.fill.background(); band.shadow.inherit = False
    t = s.shapes.add_textbox(Inches(0.8), Inches(2.85), Inches(11.7), Inches(0.9))
    _text(t.text_frame, "CrabNet-Mineral v3", 40, WHITE, bold=True, align=PP_ALIGN.LEFT)
    st = s.shapes.add_textbox(Inches(0.83), Inches(3.8), Inches(11.7), Inches(0.6))
    _text(st.text_frame, "model & pipeline overview", 22, ORANGE, align=PP_ALIGN.LEFT)
    notes(s, "Living deck: how the current canonical model and its training/inference pipeline "
             "work. Regenerated on every major architecture change.")


def s_corpora(prs):
    s = new_slide(prs)
    header(s, "DATA", "Three corpora")
    box(s, 0.6, 2.0, 3.9, 2.6, ["SSL pretrain", "", ("data/bulks/gard2019.csv",),
        ("~167k igneous bulks",), ("inputs only (no labels)",)], TEAL, size=14)
    box(s, 4.7, 2.0, 3.9, 2.6, ["supervised", "", ("equilibrium corpus (MAGEMin)",),
        ("volcanic bulks x 48 (P, H2O) draws",), ("phenocryst-regime labels",)], GREEN, size=14)
    box(s, 8.8, 2.0, 3.9, 2.6, ["held-out eval", "", ("730 real volcanic modes",),
        ("+ mineralML grains (~88k)",), ("never trained on",)], PURPLE, size=14)
    caption(s, 0.6, 4.9, 3.9, "trains the encoder", 12, TEAL)
    caption(s, 4.7, 4.9, 3.9, "trains the heads (next two slides)", 12, GREEN)
    caption(s, 8.8, 4.9, 3.9, "evaluation only", 12, PURPLE)
    caption(s, 0.6, 5.7, 12.1, "inputs everywhere are the same 22-token cation+trace vector; "
            "the corpora differ in what supervision (if any) they attach.", 13, INK, align=PP_ALIGN.LEFT)
    notes(s, "Three corpora. Pretrain (Gard) is inputs-only and trains the encoder; the "
             "equilibrium corpus (MAGEMin cooling, phenocryst-regime endpoint -- generation "
             "pipeline on the next slide) carries the labels that train the heads; the real "
             "volcanic benchmark and mineralML grains are held out for validation.")


def s_labelgen(prs):
    s = new_slide(prs)
    header(s, "DATA", "How the supervised labels are generated")
    bx, bw, bh, by = [0.55, 3.15, 5.75, 8.35, 10.95], 2.35, 1.9, 1.55
    steps = [
        (["volcanic core", ("gard2019 -> 324k bulks",), ("stratified: 10k train",),
          ("+ 500 val/test",)], TEAL, "build_corpus_bulk_tables.py"),
        (["sample conditions", ("48 (P, H2O) draws/bulk",), ("solubility-bounded LHS",),
          ("density q stored per row",)], BLUE, "cnmineral.data.condition_proposal"),
        (["equilibrium cooling", ("MAGEMin, 1200 -> 600 C",), ("stop below 30% melt",),
          ("chunked + resumable",)], PURPLE, "generate_mineralogy_corpus.py"),
        (["corpus tables", ("draws / ladder / rungs",), ("full labels at",),
          ("60/50/40/30% melt",)], ORANGE, "parquet + manifest per chunk"),
        (["training labels", ("crystals-renormalized",), ("solids; presence + modes",),
          ("rebuild, never a regen",)], GREEN, "training-time transform"),
    ]
    for i, (lines, color, script) in enumerate(steps):
        box(s, bx[i], by, bw, bh, lines, color, size=12)
        caption(s, bx[i] - 0.15, by + bh + 0.06, bw + 0.3, script, 10, GRAYT)
        if i > 0:
            pr = bx[i - 1] + bw
            flow_arrow(s, pr + 0.04, by + bh / 2 - 0.18, bx[i] - pr - 0.08, 0.36)

    # bounded (P, H2O) support: N&L 2002 rhyolite solubility (850 C, values from
    # condition_proposal) capped at the 8 wt% melt-inclusion ceiling
    px, pyb, pw, ph = 1.2, 6.7, 3.6, 1.75
    bound = [(0.5, 2.60), (1.0, 3.88), (1.5, 4.93), (2.0, 5.92),
             (2.5, 6.90), (3.0, 7.89), (3.05, 8.0), (12.0, 8.0)]
    to_xy = lambda p, w: (px + (p - 0.5) / 11.5 * pw, pyb - w / 8.0 * ph)
    frame = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(px), Inches(pyb - ph),
                               Inches(pw), Inches(ph))
    frame.fill.background(); frame.line.color.rgb = _rgb(GRAYL)
    frame.line.width = Pt(1); frame.shadow.inherit = False
    fb = s.shapes.build_freeform(Inches(px), Inches(pyb), scale=1.0)
    pts = [to_xy(p, w) for p, w in bound] + [(px + pw, pyb)]
    fb.add_line_segments([(Inches(a), Inches(b)) for a, b in pts], close=True)
    shp = fb.convert_to_shape()
    shp.fill.solid(); shp.fill.fore_color.rgb = _rgb(_lighten(BLUE, 0.72))
    shp.line.color.rgb = _rgb(BLUE); shp.line.width = Pt(2); shp.shadow.inherit = False
    caption(s, px - 0.2, pyb + 0.02, pw + 0.4, "P  0.5 -> 12 kbar", 10, GRAYT)
    t = s.shapes.add_textbox(Inches(px - 0.62), Inches(pyb - ph - 0.05), Inches(0.6), Inches(0.4))
    _text(t.text_frame, ["H2O", ("0-8",)], 10, GRAYT)
    caption(s, px + 0.2, pyb - ph - 0.4, pw, "sampled (P, H2O) region", 11, BLUE)
    caption(s, 5.4, 5.95, 7.4,
            "water is bounded by what melt can dissolve at each pressure (Newman & "
            "Lowenstern 2002 rhyolite curve, capped at 8 wt%) -- the old flat box "
            "(whole rectangle) sampled impossible wet-shallow corners. q is the "
            "sampling density; inference reweights scenarios by p/q, so the corpus "
            "itself stays prior-agnostic.", 12, INK, align=PP_ALIGN.LEFT)
    notes(s, "The Track B label factory, left to right: stratified volcanic bulk tables; "
             "solubility-bounded Latin-Hypercube (P, H2O) draws with the proposal density "
             "stored per row; chunked MAGEMin equilibrium cooling stopped below 30% residual "
             "melt (fresh Julia worker per chunk, atomic parquet + manifest, rerun-to-resume); "
             "three tables with full labels at the four candidate rungs; crystals-renormalized "
             "training labels rebuilt at train time (presence cutoff is a rebuild, not a regen).")


def s_regen_status(prs):
    s = new_slide(prs)
    header(s, "DATA", "Corpus regen: where we are")
    steps = [
        ("1. bounded sampler", ["landed + tested", ("condition_proposal",)], True, False),
        ("2. orchestrator", ["landed + smoke-run", ("generate_mineralogy_corpus",)], True, False),
        ("3. water A/B", ["switch holds;", ("biotite gap is",),
                          ("prior-side, not coverage",)], True, False),
        ("4. pilot + rung pick", ["first 5 train chunks", ("health report +",),
                                  ("rung_prescreen.py -> owner",)], False, True),
        ("5. production", ["10k x 48 + 500 x 128", ("~2 days, pausable",),
                           ("rerun command = resume",)], False, False),
    ]
    bx, bw, bh, by = [0.55, 3.15, 5.75, 8.35, 10.95], 2.35, 2.1, 2.3
    for i, (name, lines, landed, is_next) in enumerate(steps):
        color = GREEN if landed else (ORANGE if is_next else GRAYT)
        b = box(s, bx[i], by, bw, bh, [name, ""] + lines, color, lit=landed or is_next, size=12)
        tag = "LANDED" if landed else ("NEXT" if is_next else "queued")
        caption(s, bx[i], by - 0.42, bw, tag, 11, GREEN if landed else (ORANGE if is_next else GRAYT))
        if i > 0:
            pr = bx[i - 1] + bw
            flow_arrow(s, pr + 0.04, by + bh / 2 - 0.18, bx[i] - pr - 0.08, 0.36,
                       lit=landed or is_next)
    caption(s, 0.55, 5.2, 12.2, "gates between stages: the A/B re-checks the amp/opx "
            "multimodality magnitude and the biotite presence gap under bounded water; the "
            "pilot pre-screen scores each rung against the real benchmark (presence per SiO2 "
            "band, then rank of proportions) and the owner picks the training rung before "
            "production launches.", 13, INK, align=PP_ALIGN.LEFT)
    caption(s, 0.55, 6.35, 12.2, "spec: docs/corpus_regen_spec.md (all design decisions locked "
            "2026-07-14; what remains is measurement and generation).", 12, GRAYT, align=PP_ALIGN.LEFT)
    notes(s, "Status strip for the corpus regen. Stages 1-2 are code, landed and tested. "
             "Stage 3-5 are runs the owner executes in his own terminal; each has a decision "
             "gate before the next. Update this slide as stages complete.")


def s_architecture(prs):
    s = new_slide(prs)
    header(s, "ARCHITECTURE", "At a glance")
    pipeline(s, 4)
    box(s, 8.7, 0.95, 2.6, 0.8, ["imputation / retrieval"], TEAL, size=11)
    box(s, 8.7, 5.75, 2.6, 0.8, ["OOD / similarity"], TEAL, size=11)
    cy = PY + PH / 2
    line(s, PX[3] + PW / 2, cy - 0.78, 9.9, 1.75, TEAL, 1.4)
    line(s, PX[3] + PW / 2, cy + 0.78, 9.9, 5.75, TEAL, 1.4)
    caption(s, 0.55, 5.0, 8.0, "one shared encoder -> latent h -> parallel heads (supervised) "
            "and training-free encoder apps", 13, INK, align=PP_ALIGN.LEFT)
    notes(s, "The map before the detailed walk-through: tokens -> embed -> encoder -> h -> "
             "{heads, encoder apps}. Following slides zoom into each block.")


def s_inputs(prs):
    s = new_slide(prs)
    header(s, "INPUTS", "A rock is a bag of (element, amount) tokens")
    toks = [("Si", "0.50", BLUE), ("Al", "0.16", BLUE), ("Ca", "0.10", BLUE),
            ("Mg", "0.075", BLUE), ("Sr", "4e-4", TEAL), ("Yb", "2e-6", TEAL)]
    tw = 1.5
    x0 = (13.333 - (len(toks) * tw + (len(toks) - 1) * 0.22)) / 2
    for i, (sym, amt, col) in enumerate(toks):
        tok(s, x0 + i * (tw + 0.22), 3.2, tw, 1.1, sym, amt, col)
    caption(s, 0.55, 4.7, 12.2, "9 majors (wt%/100, blue)  +  13 traces (ppm/1e6, teal), "
            "22 token slots; absent = padding (src=0)", 13, INK)
    caption(s, 0.55, 5.4, 12.2, "no T, P, or H2O ever enters the forward graph", 12, GRAYT)
    notes(s, "Input = up to 22 (element id, amount) pairs. Padding is src==0. The conditions "
             "(T,P,H2O) are CondHead targets, never inputs.")


def s_embed(prs):
    s = new_slide(prs)
    header(s, "ENCODER", "Each token becomes a chemistry-aware vector")
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
    caption(s, 0.55, 6.2, 12.2, "mat2vec gives a learned chemistry vector per element "
            "(materials-informatics embedding); summed with the projected amount.", 12, INK)
    notes(s, "Opens the embed step: element id -> mat2vec vector, amount -> projection, summed "
             "into the per-token vector.")


def s_encoder(prs):
    s = new_slide(prs)
    header(s, "ENCODER", "Attention mixes elements, then pool to one h")
    syms = ["Si", "Al", "Mg", "Fe"]
    gx, gy, c = 1.1, 2.2, 0.62
    caption(s, gx, gy - 0.5, 5, "self-attention: every token attends to every token", 12, PURPLE, align=PP_ALIGN.LEFT)
    for i, a in enumerate(syms):
        tb = s.shapes.add_textbox(Inches(gx - 0.55), Inches(gy + 0.05 + i * c), Inches(0.5), Inches(0.4))
        _text(tb.text_frame, a, 11, INK, bold=True)
        tb2 = s.shapes.add_textbox(Inches(gx + i * c), Inches(gy - 0.4), Inches(0.6), Inches(0.3))
        _text(tb2.text_frame, a, 11, INK, bold=True)
    for i in range(len(syms)):
        for j in range(len(syms)):
            wt = 0.85 if i == j else 0.30 + 0.12 * ((i + j) % 4)
            cell = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(gx + j * c), Inches(gy + i * c), Inches(c - 0.06), Inches(c - 0.06))
            cell.fill.solid(); cell.fill.fore_color.rgb = _rgb(_lighten(PURPLE, 1 - wt))
            cell.line.color.rgb = _rgb(WHITE); cell.line.width = Pt(1); cell.shadow.inherit = False
    flow_arrow(s, 4.9, 3.5, 0.7, 0.45)
    box(s, 5.8, 2.95, 3.4, 1.5, ["masked-mean pool", ("average the real tokens",), ("(skip padding)",)], TEAL, size=14)
    flow_arrow(s, 9.4, 3.5, 0.7, 0.45)
    hglyph(s, 11.3, 3.7, r=0.85)
    caption(s, 0.55, 6.2, 12.2, "attention lets elements interact; pooling collapses the 22 "
            "contextualized vectors into one 256-d fingerprint h.", 12, INK)
    notes(s, "Self-attention (4x4 heatmap, diagonal strongest) then masked-mean pool over "
             "non-padding tokens into h.")


def s_ssl(prs):
    s = new_slide(prs)
    header(s, "TRAINING", "SSL pretraining: fill in the blank")
    caption(s, 0.55, 1.15, 12.2, "mask one element's amount, predict it from the rest -- "
            "the encoder learns which elements co-vary", 13, GRAYT, align=PP_ALIGN.LEFT)
    toks = [("Si", "0.50"), ("Al", "0.16"), ("Ca", "0.10"), ("Mg", "0.075"), ("Fe", "0.11"), ("Na", "0.025"), ("Sr", "4e-4")]
    mi, tw = 3, 1.5
    x0 = (13.333 - (len(toks) * tw + (len(toks) - 1) * 0.22)) / 2
    xs = [x0 + i * (tw + 0.22) for i in range(len(toks))]
    for i, (sym, amt) in enumerate(toks):
        tok(s, xs[i], 1.95, tw, 1.05, sym, amt, TEAL if i == 6 else BLUE, masked=(i == mi))
    ey = 3.75
    for i in range(len(toks)):
        cx = xs[i] + tw / 2
        line(s, cx, 3.0, cx, ey, color=(MASKG if i == mi else GRAYL), width=1.4)
    box(s, x0, ey, len(toks) * tw + (len(toks) - 1) * 0.22, 0.95,
        ["transformer encoder", ("being trained -- weights updated each step (not frozen)",)], PURPLE, size=15)
    mcx = xs[mi] + tw / 2
    flow_arrow(s, mcx - 0.16, ey + 1.0, 0.32, 0.38, shape=MSO_SHAPE.DOWN_ARROW)
    box(s, 2.3, 5.35, 3.0, 1.0, ["encoder predicts", ("Mg amount ~ 0.071",)], ORANGE, size=13)
    box(s, 5.6, 5.35, 1.9, 1.0, ["truth", ("0.075",)], GRAYT, lit=False, size=13)
    flow_arrow(s, 5.35, 5.65, 0.28, 0.4)
    flow_arrow(s, 7.55, 5.65, 0.28, 0.4)
    box(s, 7.9, 5.35, 2.9, 1.0, ["log-space", "MSE loss"], REDX, size=14)
    line(s, 10.8, 5.85, 12.7, 5.85, REDX, 2.5)
    line(s, 12.7, 5.85, 12.7, ey + 0.5, REDX, 2.5)
    a = s.shapes.add_shape(MSO_SHAPE.LEFT_ARROW, Inches(12.15), Inches(ey + 0.32), Inches(0.55), Inches(0.34))
    a.fill.solid(); a.fill.fore_color.rgb = _rgb(REDX); a.line.fill.background(); a.shadow.inherit = False
    cap = s.shapes.add_textbox(Inches(9.6), Inches(4.5), Inches(3.4), Inches(0.6))
    _text(cap.text_frame, ["backpropagation:", "updates encoder weights"], 12, REDX, bold=True)
    box(s, 1.6, 6.5, 4.6, 0.85,
        ["zⱼ = (log₁₀ fⱼ − μⱼ) / σⱼ",
         ("target: per-element z-scored log fraction; μⱼ, σⱼ from the train split only",)],
        REDX, lit=False, size=15)
    box(s, 6.9, 6.5, 4.6, 0.85,
        ["L = 1/|M| Σⱼ∈M (ŷⱼ − zⱼ)²",
         ("plain MSE between prediction ŷ and z, at the masked positions M only",)],
        REDX, lit=False, size=15)
    caption(s, 0.55, 6.55, 1.0, "the loss,\nexplicitly:", 11, REDX, align=PP_ALIGN.LEFT)
    notes(s, "Masked-element modelling. Encoder is TRAINED here (frozen is the downstream state). "
             "Loss is per-element standardized log-space MSE so traces get supervised (fracs span "
             "~8 decades; linear MSE leaves traces unsupervised). mu/sigma are per-element log10-frac "
             "stats from the train split (pretrain_objective.compute_loss_stats), stored as head "
             "buffers. Over ~167k rocks.")


def s_workflow(prs):
    s = new_slide(prs)
    header(s, "TRAINING", "Workflow: pretrain, freeze, train heads")
    box(s, 0.7, 2.6, 3.3, 1.7, ["1. SSL pretrain", "", ("Gard bulks, masked-element",),
        ("-> encoder",)], TEAL, size=14)
    flow_arrow(s, 4.1, 3.25, 0.7, 0.45)
    box(s, 5.0, 2.6, 3.0, 1.7, ["2. freeze", "encoder", ("weights fixed",)], PURPLE, size=14)
    flow_arrow(s, 8.1, 3.25, 0.7, 0.45)
    box(s, 9.0, 2.6, 3.6, 1.7, ["3. train heads", "", ("equilibrium corpus labels",),
        ("encoder untouched",)], GREEN, size=14)
    caption(s, 0.7, 4.6, 11.9, "the encoder is the durable contribution: heads can be "
            "re-supervised against any label source without retraining it.", 13, INK, align=PP_ALIGN.LEFT)
    caption(s, 0.7, 5.4, 11.9, "evaluation: frozen-latent probes + the held-out mineralML grains.",
            12, GRAYT, align=PP_ALIGN.LEFT)
    notes(s, "The three-stage training pipeline. Stage 1 is the SSL loop from the prior slide; "
             "stage 2 freezes; stage 3 trains the heads on the labeled equilibrium corpus.")


def s_heads(prs):
    s = new_slide(prs)
    header(s, "HEADS", "Parallel heads on the frozen h")
    pipeline(s, 3, hide_after=3)
    heads = [("ModeHead", "modal mineralogy"), ("CompHead", "per-mineral composition"),
             ("CondHead", "T, P, H2O"), ("ClassHead", "rock class (aux)"), ("G_sys", "free energy (aux)")]
    for i, (name, role) in enumerate(heads):
        y = 1.25 + i * 1.05
        box(s, 9.7, y, 3.1, 0.9, [name, (role,)], GREEN, size=13)
        line(s, 8.98, 3.78, 9.7, y + 0.45, GREEN, 1.5)
    caption(s, 0.55, 5.6, 8.5, "encoder now frozen; each head is a small network reading the "
            "same h (outputs on the next slide)", 12, INK)
    notes(s, "Structural view: which heads exist and that they all read the same frozen h. "
             "Outputs are drawn on the next slide.")


def s_outputs(prs):
    s = new_slide(prs)
    header(s, "HEADS", "What the heads emit (distributions, not point estimates)")
    box(s, 0.55, 2.9, 2.5, 1.7, ["example bulk", "", ("SiO2 50  Al2O3 16",), ("MgO 7.5  FeO 11",),
        ("Sr 400 ppm ...",)], BLUE, size=12)
    flow_arrow(s, 3.15, 3.55, 0.5, 0.4)
    hglyph(s, 4.5, 3.75, r=0.72)
    flow_arrow(s, 5.35, 3.55, 0.5, 0.4)
    box(s, 6.0, 1.55, 6.9, 2.25, [""], GREEN)
    caption(s, 6.2, 1.6, 6.5, "ModeHead -> per phase: p(present) x amount-when-present "
            "(hurdle Dirichlet; crystals renormalized to 1, groundmass/melt excluded)",
            12, GREEN, align=PP_ALIGN.LEFT)
    mode_bars = [("plagioclase", 0.45, 0.05, 1.00), ("clinopyroxene", 0.27, 0.04, 0.97),
                 ("amphibole", 0.18, 0.06, 0.41), ("orthopyroxene", 0.10, 0.04, 0.63),
                 ("magnetite", 0.04, 0.02, 0.95)]
    for i, (m, v, ci, p) in enumerate(mode_bars):
        hbar(s, 6.25, 2.05 + i * 0.34, 2.3, v, m, GREEN, ci=ci, presence=p)
    box(s, 6.0, 4.0, 6.9, 2.55, [""], PURPLE)
    caption(s, 6.2, 4.05, 6.5,
            "CondHead -> storage conditions (per-axis GMM; P and H2O are sampled axes, "
            "T is the derived phenocryst-window endpoint temperature)", 12, PURPLE,
            align=PP_ALIGN.LEFT)
    gaussian(s, 6.35, 4.85, 1.9, 1.05, [(0.5, 0.16, 1.0)], BLUE, "T  1010 +/- 40 C")
    gaussian(s, 8.5, 4.85, 1.9, 1.05, [(0.5, 0.18, 1.0)], TEAL, "P  4.8 +/- 1.9 kbar")
    gaussian(s, 10.65, 4.85, 1.95, 1.05, [(0.32, 0.1, 1.0), (0.72, 0.12, 0.8)], ORANGE, "H2O  multimodal")
    caption(s, 0.55, 6.75, 12.3, "values illustrative -- every phase carries both numbers; faded "
            "bars = presence genuinely uncertain (here the amphibole/orthopyroxene either-or: "
            "wet-deep grows amp, dry-shallow grows opx). This marginal view is a projection; the "
            "primary output is the ranked-scenario view on the next slide.", 12, GRAYT,
            align=PP_ALIGN.LEFT)
    notes(s, "Output FORM, not measured results (heads train in issue #10). ModeHead = hurdle "
             "Dirichlet over the crystals-renormalized solid assemblage (liq/fl dropped, solids "
             "sum to 1 -- matches real itemized-phenocrysts + lumped-groundmass granularity). "
             "EVERY phase gets a presence Bernoulli AND a conditional amount +/- CI; near-certain "
             "phases just have p~1. Amp 0.41 / opx 0.63 anti-correlated is the flagship "
             "multimodality (deep-wet-amp vs shallow-dry-opx), so their marginal amounts are "
             "conditional on presence, not rock-wide means. CondHead = per-axis GMM; P and H2O "
             "were sampled at generation, T is derived (cooling endpoint at the phenocryst "
             "window), hence the lower illustrative T. Second read-out view: top-K scenarios; "
             "weights labeled prior-weighted plausibility, reweightable p/q. Mass-balance "
             "closure teacher-forces the label melt fraction at training. Numbers illustrative.")


if __name__ == "__main__":
    prs = make()
    s_title(prs)
    s_corpora(prs); s_labelgen(prs); s_regen_status(prs); s_architecture(prs)
    s_inputs(prs); s_embed(prs); s_encoder(prs)
    s_ssl(prs); s_workflow(prs)
    s_heads(prs); s_outputs(prs); s_scenarios(prs)
    out = Path(__file__).resolve().parent / "overview.pptx"
    prs.save(out)
    print(f"wrote {out}  ({len(prs.slides._sldIdLst)} slides)")
