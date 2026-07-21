"""Living model/pipeline overview deck -> slides/overview.pptx.

The deck's sole job is to visualize the current canonical model and the
pipeline/workflow that trains and uses it -- components, data flow at training
and inference, and how they compose. It deliberately excludes positioning,
comparisons to other work, and value framing (the owner layers that on
separately). Grows and retires sections as the architecture does.

Every ML component (each head; encoder changes) gets a standard two-slide
report: an anatomy slide (`head_anatomy_slide` template) and a per-head
eval-semantics slide (bespoke layout, worked example). Together they must
cover the 8-item checklist:

  1. interface -- input/output tensors, where it sits, and what one training
     example is (row semantics)
  2. architecture diagram -- layers, widths, activations, branches
  3. parameters -- exact per-block counts + total, share of the frozen encoder
  4. objective -- explicit formula per term, what each term's minimum means,
     masking / teacher-forcing semantics, composition weights
  5. hyperparameters -- every config dial with current value, plus what is
     fixed in code (activation, optimizer, clipping)
  6. ground truth -- label construction, split, and what a prediction means
     against it
  7. headline eval -- metric, what good looks like, artifact pointer,
     split identity (val roster vs sealed test)
  8. known limits -- 2-3 bullets, pointers to heads.md / ADRs for rationale

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


def phw_region(s, px, pyb, pw, ph):
    """Bounded (P, H2O) sampling region, drawn with its baseline at `pyb`.

    N&L 2002 rhyolite solubility (850 C, values from condition_proposal)
    capped at the 8 wt% melt-inclusion ceiling.
    """
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

    phw_region(s, 1.2, 6.7, 3.6, 1.75)
    caption(s, 1.4, 6.7 - 1.75 - 0.4, 3.6, "sampled (P, H2O) region", 11, BLUE)
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


def s_one_to_many(prs):
    s = new_slide(prs)
    header(s, "DATA", "One rock, 96 labels: the one-to-many mapping")
    caption(s, 0.4, 1.3, 2.5, "ONE input rock", 11, BLUE, align=PP_ALIGN.LEFT)
    caption(s, 3.5, 1.3, 2.4, "48 sampled (P, H2O) paths", 11, PURPLE, align=PP_ALIGN.LEFT)
    caption(s, 6.2, 1.3, 1.6, "x 2 rungs", 11, ORANGE, align=PP_ALIGN.LEFT)
    caption(s, 8.3, 1.3, 4.45, "96 training labels (solid wt%, crystals-renormalized)",
            11, GREEN, align=PP_ALIGN.LEFT)
    box(s, 0.55, 3.3, 2.0, 1.1, ["one gard2019 rock",
        ("SiO2 58 . Al2O3 17 . MgO 3.5 ...",)], BLUE, size=12)
    draws = [
        (1.6, ["(P₁, H₂O₁)", ("9 kbar . 5.5 wt% -- deep, wet",)]),
        (3.2, ["(P₂, H₂O₂)", ("2 kbar . 1.5 wt% -- shallow, dry",)]),
        (5.3, ["(P₄₈, H₂O₄₈)", ("5 kbar . 3 wt% -- intermediate",)]),
    ]
    labels = [
        "plag 44 . cpx 26 . amp 19 . ox 11",
        "plag 40 . cpx 25 . amp 24 . ox 11",
        "plag 49 . cpx 26 . opx 14 . ox 11   (no amp)",
        "plag 46 . cpx 25 . opx 18 . ox 11   (no amp)",
        "plag 45 . cpx 24 . amp 10 . opx 10 . ox 11",
        "plag 43 . cpx 23 . amp 12 . opx 11 . ox 11",
    ]
    k = 0
    for dy, dlines in draws:
        c = dy + 0.4
        line(s, 2.55, 3.85, 3.6, c, GRAYT, 1.4)
        box(s, 3.6, dy, 2.1, 0.8, dlines, PURPLE, size=11)
        for rc in (c - 0.4, c + 0.4):
            line(s, 5.7, c, 6.3, rc, GRAYT, 1.2)
            box(s, 6.3, rc - 0.25, 1.35, 0.5,
                ["50% melt" if rc < c else "30% melt"], ORANGE, size=10)
            line(s, 7.65, rc, 8.3, rc, GRAYT, 1.2)
            box(s, 8.3, rc - 0.25, 4.45, 0.5, [labels[k]], GREEN, size=10)
            k += 1
    for x, w in ((3.6, 2.1), (6.3, 1.35), (8.3, 4.45)):
        caption(s, x, 4.3, w, ". . .", 14, GRAYT)
    caption(s, 0.55, 5.05, 3.2, "each path drawn from the bounded (P, H2O) region:",
            10, GRAYT, align=PP_ALIGN.LEFT)
    phw_region(s, 0.95, 6.85, 2.4, 1.3)
    caption(s, 3.7, 6.35, 9.1, "why this shape matters for the presence head: it only ever "
            "sees the left box, so all 96 rows share one input and it emits ONE p per mineral "
            "against 96 different truths -- amp grows on the wet-deep paths, opx on the "
            "dry-shallow. BCE's optimum is the fraction of paths growing the mineral, so an "
            "intermediate p is the multimodal signal, not indecision.",
            11, INK, align=PP_ALIGN.LEFT)
    notes(s, "The one-to-many mapping that everything downstream leans on: one gard bulk -> "
             "48 solubility-bounded (P, H2O) draws -> each read out at two melt-fraction "
             "rungs (50/30) -> 96 label rows sharing one input vector. Assemblage numbers "
             "illustrative; amp-vs-opx flips with wetness/depth (the flagship multimodality). "
             "This is the visual justification for fractional presence targets (within-bulk "
             "frequency), the bulk-level split rule, and the per-bulk latent cache in "
             "training. Same-bulk-at-both-rungs is intentional (rungs LOCKED 50+30).")


def s_regen_status(prs):
    s = new_slide(prs)
    header(s, "DATA", "Corpus regen: where we are")
    steps = [
        ("1. bounded sampler", ["landed + tested", ("condition_proposal",)], True, False),
        ("2. orchestrator", ["landed + smoke-run", ("generate_mineralogy_corpus",)], True, False),
        ("3. water A/B", ["switch holds;", ("biotite gap is",),
                          ("prior-side, not coverage",)], True, False),
        ("4. pilot + rung pick", ["passed; rungs locked", ("50 + 30 (owner,",),
                                  ("rung_prescreen.py)",)], True, False),
        ("5. production", ["both tiers complete", ("948k train rows;",),
                           ("valtest 125k, sealed",), ("221/278 val/test roster",)], True, False),
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
    caption(s, 0.55, 5.2, 12.2, "all gates passed: the water A/B held under bounded "
            "sampling; the pilot pre-screen locked rungs 50 + 30; production completed "
            "2026-07-17 (train 99.2% point convergence, valtest 97.7% usable rows after "
            "dropping one solver-poisoned bulk). The 499 valtest bulks divide 16:20 into "
            "a committed val/test roster (valtest_split.csv); test stays sealed.",
            13, INK, align=PP_ALIGN.LEFT)
    caption(s, 0.55, 6.35, 12.2, "spec: docs/corpus_regen_spec.md (all design decisions locked "
            "2026-07-14; generation complete 2026-07-17).", 12, GRAYT, align=PP_ALIGN.LEFT)
    notes(s, "Status strip for the corpus regen, now fully landed -- the slide is a record, "
             "not a plan. Candidate to retire or fold into the corpora slide at the next "
             "deck pass (owner call).")


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
            "same h (anatomy on the next slide, outputs after)", 12, INK)
    notes(s, "Structural view: which heads exist and that they all read the same frozen h. "
             "Head internals on the next slide; outputs after.")


def head_anatomy_slide(prs, *, title, intro, input_note, trunk, branches,
                       params_line, dials_line, loss_boxes, notes_text):
    """Standard anatomy slide for one head (checklist items 1-5).

    `trunk` = (lines, color); `branches` = list of {box, out, color} rendered
    top to bottom (2 or 3 supported); `loss_boxes` = list of (lines, color)
    strips along the bottom.
    """
    s = new_slide(prs)
    header(s, "HEADS", title)
    caption(s, 0.55, 1.15, 12.2, intro, 13, GRAYT, align=PP_ALIGN.LEFT)
    hglyph(s, 1.6, 3.85, r=0.75)
    caption(s, 0.5, 4.85, 2.2, input_note, 10, GRAYT)
    flow_arrow(s, 2.5, 3.65, 0.5, 0.4)
    box(s, 3.1, 3.05, 2.9, 1.6, trunk[0], trunk[1], size=13)
    n = len(branches)
    bh = 1.55 if n == 2 else 1.15
    step = (5.75 - 2.0 - bh) / (n - 1)
    for i, br in enumerate(branches):
        y = 2.0 + i * step
        line(s, 6.0, 3.85, 6.75, y + bh / 2, GRAYT, 1.5)
        box(s, 6.75, y, 3.3, bh, br["box"], br["color"], size=13)
        flow_arrow(s, 10.15, y + bh / 2 - 0.18, 0.4, 0.35)
        box(s, 10.65, y + 0.05, 2.25, bh - 0.1, br["out"], br["color"], lit=False, size=12)
    caption(s, 0.55, 5.85, 12.2, params_line, 11, INK, align=PP_ALIGN.LEFT)
    caption(s, 0.55, 6.2, 12.2, dials_line, 10, GRAYT, align=PP_ALIGN.LEFT)
    k = len(loss_boxes)
    w = (10.9 - 0.3 * (k - 1)) / k
    for i, (lines_, color) in enumerate(loss_boxes):
        box(s, 1.6 + i * (w + 0.3), 6.6, w, 0.85, lines_, color, lit=False,
            size=14 if k <= 2 else 12)
    caption(s, 0.5, 6.65, 1.05, "the loss,\nexplicitly:", 11, REDX, align=PP_ALIGN.LEFT)
    notes(s, notes_text)
    return s


def s_mode_head_anatomy(prs):
    head_anatomy_slide(
        prs,
        title="Inside ModeHead: a small MLP on the frozen h",
        intro="ModeHead, the first head trained end-to-end -- two clean factors of "
              "the hurdle likelihood P(modal) = P(present) x P(modal | present)",
        input_note="from the frozen encoder (3.23M params, not trained here)",
        trunk=(["shared trunk", ("3 x [Linear -> 256]",), ("GELU + dropout each",),
                ("197,376 params",)], PURPLE),
        branches=[
            {"box": ["presence branch", ("Linear 256 -> 21, sigmoid",),
                     ("p_i = P(phase i present)",), ("5,397 params",)],
             "out": ["21 Bernoullis", ("which minerals",), ("BCE loss",)],
             "color": GREEN},
            {"box": ["magnitude branch", ("Linear 256 -> 21, softplus",),
                     ("alpha_i, Dirichlet concentration",), ("5,397 params",)],
             "out": ["cond. Dirichlet", ("how much of each,",),
                     ("on the present set (NLL)",)],
             "color": ORANGE},
        ],
        params_line="208,170 trainable params (trunk 197,376 + 2 x 5,397) -- ~6.4% of the "
                    "frozen encoder; the 256 x 3 trunk is the joint-HPO winner (untuned "
                    "128 x 1 gave up 0.105 nats of assemblage NLL).",
        dials_line="dials (configs/heads/joint_heads.yaml): mode_hidden 256 . mode_layers 3 . "
                   "dropout 0.348 . magnitude_weight 0.25 . alpha_eps 1e-6 . lr 2.15e-4 . "
                   "weight_decay 1.04e-5 . batch 1024 . epochs 120 / per-head patience 10 -- "
                   "fixed in code: GELU, AdamW, grad-clip 1.0",
        loss_boxes=[
            (["L_presence = −Σᵢ [ yᵢ ln pᵢ + (1−yᵢ) ln(1−pᵢ) ]",
              ("BCE, all 21 phases x every row; minimized when "
               "p = the rock's true presence frequency",)], GREEN),
            (["L_magnitude = −log Dir(mₛ | αₛ)",
              ("Dirichlet NLL scored only on the present set S -- "
               "absent phases belong to the Bernoullis",)], ORANGE),
            (["L = L_presence + λ_mag · L_magnitude",
              ("trained jointly: one backward pass; each branch gets only its "
               "own term's gradient, the shared trunk gets the SUM of both -- "
               "∇L(trunk) = ∇L_pres + λ_mag ∇L_mag",)], PURPLE),
        ],
        notes_text="Anatomy of the trained ModeHead (checkpoints/joint_heads_seed0): shared trunk "
             "3 stacked Linear(->256)+GELU+dropout blocks, then two parallel linear branches -- "
             "presence logits (sigmoid -> per-phase Bernoulli, BCE) and concentrations (softplus + eps "
             "-> conditional Dirichlet, masked NLL on the true support). Param counts are "
             "pulled from the checkpoint (mode_hidden 256, mode_layers 3, the joint-HPO "
             "winner; fANOVA put architecture near zero, the gain came with the tuned "
             "environment). Loss boxes mirror the SSL slide: BCE is a proper "
             "scoring rule (per-rock minimum = within-bulk presence frequency, which is why "
             "intermediate p is signal); the Dirichlet NLL is masked to the label's present "
             "set (sub-composition property keeps it a proper density); λ_mag is the "
             "loss-weighting coefficient (config key magnitude_weight, 0.25 tuned) -- "
             "λ chosen over w to avoid clashing with layer weights and to match the "
             "λ_balance / λ_comp family. 'Trained jointly' is literal chain-rule math: L is "
             "one scalar, so one backward pass and one AdamW step update all params at once. "
             "Each branch's weights appear in only its own term (the other term's gradient "
             "is exactly zero there); the trunk appears in both, so its gradient is the sum "
             "-- λ_mag is the exchange rate between the two error signals shaping the shared "
             "features. NOT alternating epochs, NOT separate optimizers, NOT staged "
             "pretraining of one branch. The hurdle split exists because one full-simplex "
             "Dirichlet cannot be simultaneously confident about absences and honest about "
             "present-phase spread (shared concentration). CompHead/CondHead follow the same "
             "small-MLP-emitting-distribution-parameters pattern; their two-slide reports "
             "follow ModeHead's.")


def s_mode_head_reliability(prs):
    s = new_slide(prs)
    header(s, "HEADS", "Is 70% honest? One bin of the reliability check")
    caption(s, 0.55, 1.15, 12.2, "the presence head's headline eval: pool every prediction "
            "near a probability, then ask how often the mineral was really there",
            13, GRAYT, align=PP_ALIGN.LEFT)
    rocks = [("rock A (andesite)", "p_cpx = 0.71", "256 rows: cpx in 187"),
             ("rock B (bas. andesite)", "p_cpx = 0.74", "256 rows: cpx in 163"),
             ("rock C (andesite)", "p_cpx = 0.77", "256 rows: cpx in 197")]
    caption(s, 0.55, 1.75, 2.9, "22-token chemistry -> one p per rock", 11, BLUE)
    caption(s, 4.05, 1.75, 3.2, "truth: rows where cpx > 0.1 wt% of the solids", 11, GREEN)
    for i, (name, pred, truth) in enumerate(rocks):
        y = 2.1 + i * 1.05
        box(s, 0.55, y, 2.9, 0.85, [name, (pred,)], BLUE, size=12)
        flow_arrow(s, 3.55, y + 0.25, 0.4, 0.35)
        box(s, 4.05, y, 3.2, 0.85, [truth], GREEN, size=12)
        line(s, 7.25, y + 0.42, 7.85, 3.7, GRAYT, 1.5)
    box(s, 7.85, 2.75, 2.6, 1.9, ["predictions binned:", ("p in [0.70, 0.80): 3 predictions,",),
        ("x 256 labels each = 768 pairs",), ("mean p = 0.740  .  actually",),
        ("present: 547 / 768 = 0.712",)], ORANGE, size=13)
    px, py, pw, ph = 10.8, 1.95, 2.3, 2.3
    frame = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(px), Inches(py), Inches(pw), Inches(ph))
    frame.fill.background(); frame.line.color.rgb = _rgb(GRAYL)
    frame.line.width = Pt(1); frame.shadow.inherit = False
    line(s, px, py + ph, px + pw, py, GRAYT, 1.2)
    dot = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(px + 0.740 * pw - 0.09),
                             Inches(py + (1 - 0.712) * ph - 0.09), Inches(0.18), Inches(0.18))
    dot.fill.solid(); dot.fill.fore_color.rgb = _rgb(ORANGE)
    dot.line.fill.background(); dot.shadow.inherit = False
    ghost = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(px + 0.740 * pw - 0.08),
                               Inches(py + (1 - 0.469) * ph - 0.08), Inches(0.16), Inches(0.16))
    ghost.fill.solid(); ghost.fill.fore_color.rgb = _rgb(GRAYL)
    ghost.line.fill.background(); ghost.shadow.inherit = False
    line(s, 8.95, 4.65, px + 0.740 * pw, py + (1 - 0.712) * ph + 0.12, ORANGE, 1.5)
    caption(s, px - 0.3, py + ph + 0.05, pw + 0.6, "predicted p ->", 10, GRAYT)
    caption(s, px - 0.3, py - 0.45, pw + 0.6, "on the diagonal = calibrated", 11, ORANGE)
    caption(s, px - 0.3, py + ph + 0.42, pw + 0.6,
            "gray dot: overconfident (if truth were 360/768)", 9, GRAYT)
    caption(s, 0.55, 5.15, 12.2, '"actually present" is the pooled hit rate over the rows whose '
            "prediction landed in the bin -- never a corpus-wide rate, never one rock alone. "
            "Per-rock misses can cancel inside a bin (B deserved 0.64); the p-vs-frequency "
            "scatter is the stricter per-rock test, and the head is scored on both.",
            11, INK, align=PP_ALIGN.LEFT)
    caption(s, 0.55, 5.9, 12.2, "3 rocks shown; the real figure pools the val-roster rocks "
            "only -- 218 rocks x ~255 rows = 55,576 (p, truth) pairs per mineral panel, "
            "partitioned across the 10 bins (the gray bars). Training rocks are excluded "
            "because a head that memorized them would grade itself calibrated. And since "
            "all ~255 rows of a rock share one prediction, a panel holds only 218 distinct "
            "p values, each stamped ~255x -- the effective sample size is rocks, not rows. "
            "The population is the fixed val roster of the sealed valtest tier (128 "
            "draws/rock at rungs 50+30), so panels are comparable across retrains.",
            11, INK, align=PP_ALIGN.LEFT)
    caption(s, 0.55, 6.95, 12.2, "BCE's per-rock minimum is the true fraction, so an honest "
            "head lands on the diagonal by construction. current limits (tuned model): "
            "intermediate-frequency presence is the miss mode (garnet recall 0.58 in "
            "f = [0.5, 0.9) and 0.0 elsewhere; opx 0.25 / ru 0.08 / ol 0.08 overall); "
            "spinel is the worst-calibrated panel (ECE 0.037); amphibole magnitude spread "
            "stays too narrow (next slide). full report: results/mode_head/<tag>/",
            10, GRAYT, align=PP_ALIGN.LEFT)
    notes(s, "Worked example for the reliability diagram in results/mode_head/<tag>/"
             "reliability.png. Numbers illustrative but to scale: three rocks predicted in "
             "[0.70, 0.80) -- a real bin edge of the 10 equal-width bins the code uses -- "
             "contribute 3 predictions x 256 labels = 768 (p, truth) pairs -- 3 predictions, "
             "not 768; each rock's single p is scored against its 256 labels. "
             "x = count-weighted mean p (0.740), "
             "y = pooled fraction of pairs with cpx present (547/768 = 0.712). ECE on the "
             "real panels is the count-weighted |x - y| over 10 such bins. The gray ghost "
             "dot shows the overconfident failure mode. Real population: the sealed valtest "
             "tier's val roster (221 assigned rocks, 218 contributing 55,576 rows at 128 "
             "draws x rungs 50/30, minus attrition). Fixed roster: panel numbers move only "
             "when the head retrains. Pair with figure 2 (p vs within-bulk "
             "frequency): reliability is aggregate honesty, the scatter is per-rock "
             "sharpness; a base-rate predictor passes the first and fails the second.")


def s_mode_head_dirichlet(prs):
    s = new_slide(prs)
    header(s, "HEADS", "If it's there, how much? One row through the Dirichlet")
    caption(s, 0.55, 1.15, 12.2, "the magnitude branch's eval: slice the predicted "
            "concentrations to the row's true support, then meet the truth at the "
            "scoring step -- are the means right, and is the claimed spread honest "
            "(PIT)?", 13, GRAYT, align=PP_ALIGN.LEFT)
    box(s, 0.55, 1.95, 2.8, 1.5, ["head output for bulk X",
        ("alpha for all 21 phases,",),
        ("computed from the latent --",),
        ("truth is not an input",)], ORANGE, size=11)
    flow_arrow(s, 3.45, 2.55, 0.35, 0.3)
    box(s, 3.85, 1.95, 2.8, 1.5, ["alpha sliced to the support",
        ("plag 12.0    cpx 7.0",),
        ("amp 4.5      ilm 1.5",),
        ("alpha0 = sum = 25 (confidence)",)], ORANGE, size=11)
    flow_arrow(s, 6.75, 2.55, 0.35, 0.3)
    box(s, 7.15, 1.95, 2.8, 1.5, ["predicted mean +/- spread",
        ("plag 0.48 +/- 0.10",),
        ("cpx 0.28 +/- 0.09",),
        ("amp 0.18 +/- 0.08",),
        ("ilm 0.06 +/- 0.05",)], PURPLE, size=11)
    box(s, 0.55, 3.75, 2.8, 1.3, ["one row's truth",
        ("bulk X . draw 17 . rung 50",),
        ("present: plag cpx amp ilm",),
        ("modes: .44  .26  .19  .11",)], GREEN, size=11)
    flow_arrow(s, 3.45, 4.25, 0.35, 0.3)
    box(s, 3.85, 3.75, 2.8, 1.3, ["scoring -- truth meets prediction",
        ("eval: PIT of each true mode",),
        ("train: masked Dirichlet NLL",)], BLUE, size=11)
    line(s, 2.9, 3.75, 4.3, 3.45, GREEN, 1.5)
    caption(s, 0.6, 3.47, 2.6, "support only (which phases)", 9, GREEN,
            align=PP_ALIGN.LEFT)
    line(s, 8.55, 3.45, 6.65, 4.3, PURPLE, 1.5)
    box(s, 8.0, 3.75, 4.7, 1.3, ["L_magnitude = −log Dir(mₛ | αₛ), expanded:",
        "= Σ lnΓ(αᵢ) − lnΓ(α₀) − Σ (αᵢ−1) ln mᵢ   (sums over S)",
        ("mᵢ true mode . αᵢ predicted concentration . α₀ = Σ αᵢ . "
         "Γ extends factorial",),
        ("Γ terms: normalizer -- the cost of confidence, grows with α₀",),
        ("(αᵢ−1) ln mᵢ: shape term -- the only place truth meets prediction",),
        ("a density, not a probability -- well-scored rows go negative",)],
        BLUE, lit=False, size=11)
    gx, gy, gw, gh = 11.05, 3.6, 2.0, 1.15
    gaussian(s, gx, gy - gh, gw, gh, [(0.48, 0.10, 1.0)], PURPLE,
             "plag marginal: Beta(12, 13)")
    tx = gx + 0.44 * gw
    line(s, tx, gy, tx, gy - gh - 0.15, GREEN, 2.0)
    caption(s, gx - 0.25, gy - gh - 0.55, gw + 0.5, "truth 0.44 -> PIT 0.34", 10, GREEN)
    caption(s, 0.55, 5.2, 12.2, "each present phase's marginal is a Beta distribution; "
            "PIT = where the truth lands inside it (its CDF value). Over many rows a flat "
            "PIT histogram means the spread is honest; a U means overconfident (truth keeps "
            "escaping the claimed range), a hump means underconfident.",
            11, INK, align=PP_ALIGN.LEFT)
    caption(s, 0.55, 6.0, 12.2, "the truth plays two roles, never as a model input: its "
            "support slices alpha (teacher-forced at train/eval, sampled from the presence "
            "branch at inference; a Dirichlet sub-composition is itself a Dirichlet, so the "
            "sliced score is a proper density), and its modes appear only at the scoring "
            "step.", 11, INK, align=PP_ALIGN.LEFT)
    caption(s, 0.55, 6.85, 12.2, "population: only truly-present (row, phase) pairs are "
            "scored -- ~41k pairs for plag down to ~240 for andalusite; single-phase rows "
            "are excluded (nothing to size). current limits (tuned model): amphibole stays "
            "the flagship miss -- 65% of its (rock, support) PIT groups are mean-shifted "
            "(within-bulk bimodality collapses to the mean; mixture fix staged, #25) and "
            "tuning did not move its magnitude error; plag/afs/q are majority-shifted too "
            "(the support-conditioning signal, heads.md). full report: "
            "results/mode_head/<tag>/ (magnitude_parity, magnitude_pit).",
            10, GRAYT, align=PP_ALIGN.LEFT)
    notes(s, "Eval-semantics slide for the magnitude branch, mirroring the presence "
             "branch's reliability worked example. Laid out as two tracks converging "
             "at a scoring node: the prediction track never sees the truth's modes; "
             "the truth contributes only its support to the slice (teacher-forced) "
             "and meets the prediction at scoring. Numbers illustrative but to scale: "
             "alpha sliced to the true 4-phase support gives Dirichlet means "
             "alpha_i/alpha0 and Beta marginals Beta(alpha_i, alpha0 - alpha_i); plag "
             "Beta(12,13) has mean 0.48, sd ~0.10, and truth 0.44 lands at CDF 0.34. "
             "alpha0 is the single shared confidence knob per row -- the reason per-phase "
             "spreads can't be tuned independently (see alpha_concentration.png). PIT "
             "histograms are the magnitude analog of reliability diagrams; parity plots "
             "are the point-estimate view. Scored by masked Dirichlet NLL (anatomy "
             "slide's L_magnitude). Worked numbers for the expansion box: this row "
             "scores -4.34 nats (normalizer -28.37 + shape +24.03); scaling alpha x4 "
             "improves to -5.26 when truth is near the peak but explodes to +17.9 when "
             "truth swaps to the other hump (plag/amp values exchanged) -- the mechanism "
             "that pins amphibole's alpha0 low and its mean in the middle.")


def s_comp_head_anatomy(prs):
    head_anatomy_slide(
        prs,
        title="Inside CompHead: one oxide Dirichlet per mineral",
        intro="CompHead emits a Dirichlet over the 11 MAGEMin oxides for each of the 21 "
              "phases -- composition is bulk-controlled, so it reads h only (never the "
              "support or conditions)",
        input_note="from the frozen encoder (3.23M params, not trained here)",
        trunk=(["own trunk", ("Linear 256 -> 128",), ("GELU + dropout",),
                ("32,896 params",)], PURPLE),
        branches=[
            {"box": ["mean branch", ("Linear 128 -> 231 (21x11)",),
                     ("softmax per phase: m_p, the oxide mix",), ("29,799 params",)],
             "out": ["21 simplex means", ("what each mineral",), ("is made of",)],
             "color": TEAL},
            {"box": ["concentration branch", ("Linear 128 -> 21, softplus + eps",),
                     ("c_p = one certainty scalar per phase",), ("2,709 params",)],
             "out": ["21 certainties", ("alpha_p = c_p · m_p",),
                     ("(mean invariant to c_p)",)], "color": ORANGE},
        ],
        params_line="65,404 trainable params (trunk 32,896 + mean 29,799 + concentration "
                    "2,709) -- ~2.0% of the frozen encoder; the largest head, and still "
                    "trivially retrainable.",
        dials_line="dials (configs/heads/joint_heads.yaml): comp_hidden 128 . comp_layers 1 "
                   "(mini-sweep verdict: capacity above 128 x 1 hurts comp) . dropout 0.348 . "
                   "alpha_eps 1e-6 . comp_eps 1e-4 (hard-zero clip on targets) . "
                   "lambda_comp 1.0 . lambda_balance 54.5 -- fixed in code: GELU, AdamW, "
                   "per-head grad-clip 1.0",
        loss_boxes=[
            (["alpha_p = c_p · softmax(u_p)",
              ("mean x shared-concentration factorization: c_p scales certainty, "
               "never the predicted composition",)], TEAL),
            (["L_comp = −mean_{p ∈ S} log Dir(x_p | alpha_p)",
              ("Dirichlet NLL only where the label marks phase p present; "
               "targets eps-clipped to full support, renormalized",)], ORANGE),
            (["L_closure = ‖(1−F_liq−F_fl)·Σᵢ mᵢCᵢ + F_liq·C_liq + F_fl·C_fl − B‖²",
              ("mass balance: ModeHead's modes m x CompHead's comps C must rebuild "
               "the input bulk B; melt/fluid side teacher-forced from the corpus",)], REDX),
        ],
        notes_text="Anatomy of CompHead (trains in the joint trainer, "
             "checkpoints/joint_heads_seed0). Interface: h [256] in, alpha [21, 11] out; "
             "one training example is one (bulk, draw, rung) row; targets are per-phase "
             "oxide mass fractions from the rungs table (same-token instances merged by "
             "mass-weighted mean, feldspar split into plag/afs BEFORE averaging so the "
             "solvus pair never mixes), eps-clipped at comp_eps and renormalized. "
             "Param counts are exact (instantiated at d_model 256, hidden 128). The "
             "shared-concentration factorization is the locked heads.md parameterization: "
             "one c_p per phase blurs per-oxide error bars toward each other -- accepted "
             "because composition spreads are 10-30x tighter than modal spreads (the "
             "reason CompHead is bulk-controlled and unconditioned). L_comp's minimum: "
             "the Dirichlet matching the per-bulk composition distribution of each phase. "
             "L_closure is the one loss term coupling two heads (mode x comp product); "
             "squared-L2 is a constraint penalty, not a head likelihood, so the no-MSE "
             "rule for heads doesn't apply. Known limits: hard-zero oxides are clipped "
             "(a q row is 'SiO2 + 10 eps-floors'); ~0.3% of corpus rows don't self-close "
             "(melt immiscibility, dropped-phase mass, worst 5.2 wt%) and set an "
             "unattainable closure floor there; endmember bonus heads deferred "
             "(heads.md).")


def s_comp_head_pit(prs):
    s = new_slide(prs)
    header(s, "HEADS", "Is the recipe right? One phase through the oxide Dirichlet")
    caption(s, 0.55, 1.15, 12.2, "CompHead's eval: slice the per-phase Dirichlets to the "
            "row's true support, then score each present phase's 11-oxide simplex -- right "
            "mix (parity), honest error bars (PIT), and mass balance as the physics "
            "self-check", 13, GRAYT, align=PP_ALIGN.LEFT)
    box(s, 0.55, 1.95, 2.8, 1.5, ["head output for bulk X",
        ("alpha [21 phases x 11 oxides],",),
        ("computed from the latent --",),
        ("truth is not an input",)], ORANGE, size=11)
    flow_arrow(s, 3.45, 2.55, 0.35, 0.3)
    box(s, 3.85, 1.95, 2.8, 1.5, ["one present phase: plag",
        ("alpha0 = 780 (confidence)",),
        ("mean = alpha / alpha0:",),
        ("SiO2 .55  Al2O3 .28  CaO .12 ...",)], ORANGE, size=11)
    flow_arrow(s, 6.75, 2.55, 0.35, 0.3)
    box(s, 7.15, 1.95, 2.8, 1.5, ["predicted mix +/- spread",
        ("SiO2 0.55 +/- 0.018",),
        ("Al2O3 0.28 +/- 0.016",),
        ("CaO 0.12 +/- 0.012",)], PURPLE, size=11)
    box(s, 0.55, 3.75, 2.8, 1.3, ["one row's truth",
        ("bulk X . draw 17 . rung 50",),
        ("plag measured (mass-frac):",),
        (".553  .285  .118  .041 ...",)], GREEN, size=11)
    flow_arrow(s, 3.45, 4.25, 0.35, 0.3)
    box(s, 3.85, 3.75, 2.8, 1.3, ["scoring -- truth meets prediction",
        ("eval: per-oxide Beta PIT",),
        ("train: masked comp Dirichlet NLL",)], BLUE, size=11)
    gx, gy, gw, gh = 11.05, 3.6, 2.0, 1.15
    gaussian(s, gx, gy - gh, gw, gh, [(0.55, 0.018 * 8, 1.0)], PURPLE,
             "SiO2 marginal (Beta)")
    tx = gx + 0.53 * gw
    line(s, tx, gy, tx, gy - gh - 0.15, GREEN, 2.0)
    caption(s, gx - 0.25, gy - gh - 0.55, gw + 0.5, "truth 0.553 -> PIT 0.43", 10, GREEN)
    caption(s, 0.55, 5.2, 12.2, "population: only truly-present (row, phase) pairs -- 42k "
            "for cpx down to ~240 for andalusite. headline (val roster, tuned): comp NLL "
            "-53.85 nats per present (row, phase), from -50.25 untuned; majors' PIT tails "
            "sit near the nominal 10% (cpx 0.055/0.056).", 11, INK, align=PP_ALIGN.LEFT)
    caption(s, 0.55, 5.95, 12.2, "the tuned confidence spans ~3 decades and tracks real "
            "mineralogy: near-stoichiometric phases (q, ru, sph, sill) earn alpha0 up to "
            "~30,000 while wide solid solutions (spl, ilm, cpx) stay soft -- the untuned "
            "model hedged every phase into alpha0 100-650. mass balance closes to 1.3% "
            "mean L2 with signed per-oxide residuals centered on zero; plag An# parity "
            "r = 0.93.", 11, INK, align=PP_ALIGN.LEFT)
    caption(s, 0.55, 6.85, 12.2, "current limits: hard-zero oxides ride the eps floor (a "
            "quartz row is 'SiO2 + 10 floors'); ilmenite FeO/TiO2 stays the accuracy "
            "outlier; ~0.3% of corpus rows cannot self-close and set an unattainable "
            "closure floor there. full report: results/comp_head/<tag>/",
            10, GRAYT, align=PP_ALIGN.LEFT)
    notes(s, "Eval-semantics slide for CompHead, mirroring the ModeHead magnitude worked "
             "example: two tracks converge at a scoring node; the truth contributes its "
             "support to the slice and meets the prediction only at scoring. Box numbers "
             "illustrative but to scale (a plag Dirichlet with alpha0 ~780 has SiO2 sd "
             "~0.018); headline numbers are real, from "
             "results/comp_head/joint_heads_seed0/metrics.json vs the preserved "
             "joint_heads_seed0_baseline run: NLL -50.25 -> -53.85, closure L2 mean "
             "0.0181 -> 0.0134 with the stable +SiO2/-Al2O3 signed bias collapsed to "
             "zero-centered residuals (the closure-penalty-form question, #38, closed on "
             "this evidence), An# parity mae 0.064 -> 0.060 / r 0.920 -> 0.927. The "
             "alpha0-span story is the headline fix: comp_concentration.png baseline vs "
             "tuned shows the 100-650 compression opening to 25-32,000, ordered by how "
             "constrained each mineral's chemistry really is. PIT here is per-oxide Beta "
             "marginals of the 11-simplex Dirichlet (same sub-composition property as the "
             "mode slice).")


def s_cond_head_anatomy(prs):
    head_anatomy_slide(
        prs,
        title="Inside CondHead: a support-conditioned mixture over (T, P, H2O)",
        intro="the inverse-inference head, chain rule p(support | h) x p(T, P, H2O | h, "
              "support): the presence vector joins h at the input -- teacher-forced true "
              "set at train, ModeHead-sampled at inference",
        input_note="frozen h [256] ++ presence vector [21] = 277-dim input",
        trunk=(["own trunk", ("2 x [Linear -> 512]",), ("GELU + dropout each",),
                ("404,992 params",)], PURPLE),
        branches=[
            {"box": ["weight branch", ("Linear 512 -> 9 (3 axes x K=3)",),
                     ("log-softmax over K per axis",), ("4,617 params",)],
             "out": ["mixture weights", ("which condition mode,",), ("how likely",)],
             "color": TEAL},
            {"box": ["mean branch", ("Linear 512 -> 9",),
                     ("mu_k in raw units (C, kbar, wt%)",), ("4,617 params",)],
             "out": ["component centers", ("e.g. deep-wet vs",), ("shallow-dry",)],
             "color": GREEN},
            {"box": ["spread branch", ("Linear 512 -> 9, softplus + eps",),
                     ("sigma_k, heteroscedastic",), ("4,617 params",)],
             "out": ["per-mode spreads", ("row-dependent",), ("error bars",)],
             "color": ORANGE},
        ],
        params_line="418,843 trainable params (trunk 404,992 + 3 x 4,617) -- ~13% of the "
                    "frozen encoder; the 512 x 2 trunk is the mini-sweep winner (0.29 nats "
                    "over the starved 128 x 1). Output biases seeded at the targets' "
                    "mean/std (raw-unit cold-start fix).",
        dials_line="dials (configs/heads/joint_heads.yaml): cond_n_components 3 . "
                   "cond_sigma_eps 1e-3 . cond_hidden 512 . cond_layers 2 (adopted from "
                   "the architecture mini-sweep) . dropout 0.348 . lambda_cond 0.01 -- fixed "
                   "in code: GELU, AdamW, per-head grad-clip 1.0, linear units (no log "
                   "transform of T/P/H2O)",
        loss_boxes=[
            (["L_cond = Σ_axis −log Σ_k w_k · N(t | mu_k, sigma_k²)",
              ("per-axis GMM NLL in raw units; minimized when the mixture "
               "matches each rock's condition distribution per support",)], TEAL),
            (["support in, never through the encoder",
              ("teacher-forced true set at train -> no gradient coupling to "
               "ModeHead; sampled sets at inference (exposure bias, #31)",)], GREEN),
            (["L = L_mode + λ_comp·L_comp + λ_cond·L_cond + λ_bal·L_closure",
              ("heads share no weights and each is snapshotted at its own raw-val "
               "best, so λ_cond only absorbs the raw-unit scale in the total",)], PURPLE),
        ],
        notes_text="Anatomy of CondHead (trains in the joint trainer, "
             "checkpoints/joint_heads_seed0). Interface: concat(h [256], presence [21]) "
             "in; (log_w, mu, sigma) each [3 axes, K] out; one training example is one "
             "(bulk, draw, rung) row with targets (T_end from the rung row; P_kbar, H2O "
             "from the draw). Param counts pulled from the checkpoint (cond_hidden 512, "
             "cond_layers 2, K=3): cond was the one head that wanted capacity -- starved "
             "at 128x1 under the shared tuned lr, fixed by the mini-sweep adoption. "
             "Support-conditioning is the locked chain-rule decision: independent heads "
             "assume a conditional independence the corpus violates by design -- the "
             "within-bulk label variance is driven by the shared hidden (P, H2O) path, "
             "and marginals forget which condition mode went with which support. Raw "
             "linear units are deliberate (no log transforms; negative mass is a "
             "diagnostic, not a bug -- mass_below_zero in the cond eval). init_output_bias "
             "seeds component means at target mean +/- 1 std and sigmas at the target "
             "std; without it Adam needs ~mu/lr steps to walk mu from 0 to ~900 C. "
             "lambda_cond nuance: L_cond's gradient reaches only CondHead's own weights "
             "(frozen encoder, separate trunks, teacher-forced input), and per-head "
             "clipping + Adam's per-parameter scaling make the constant nearly inert for "
             "optimization; per-head best selection (each head keeps its own raw-val "
             "best epoch) removed its old early-stopping lever too, leaving it a "
             "raw-unit scale absorber (mineralogy-first). Known limits: axes are independent mixtures "
             "(no (T, P) correlation, deferred per ADR 006); exposure bias measured, "
             "not yet mitigated (#31); T is a derived cooling endpoint, not a sampled "
             "axis, in the scenario view.")


def s_cond_head_coverage(prs):
    s = new_slide(prs)
    header(s, "HEADS", "Is the 68% band honest? Coverage, and the exposure gap")
    caption(s, 0.55, 1.15, 12.2, "CondHead's eval scores every row twice: told the TRUE "
            "mineral list (teacher-forced -- the head's own skill) and told ModeHead's "
            "SAMPLED list (the deployed pipeline). The first is calibrated; the gap to "
            "the second is the exposure bias.", 13, GRAYT, align=PP_ALIGN.LEFT)
    box(s, 0.55, 2.0, 2.9, 1.4, ["one row's input",
        ("h [256] ++ true presence [21]",),
        ("bulk X: plag cpx amp ilm",)], BLUE, size=11)
    flow_arrow(s, 3.55, 2.55, 0.35, 0.3)
    box(s, 3.95, 2.0, 3.1, 1.4, ["T-axis mixture (K=3)",
        ("deep-wet:  mu 780 C, w 0.58",),
        ("shallow-dry: mu 940 C, w 0.39",),
        ("(third component ~unused)",)], TEAL, size=11)
    gx, gy, gw, gh = 7.55, 3.3, 2.6, 1.25
    gaussian(s, gx, gy - gh, gw, gh, [(0.32, 0.09, 0.6), (0.72, 0.08, 0.4)], TEAL,
             "p(T | h, support): two condition modes")
    tx = gx + 0.38 * gw
    line(s, tx, gy, tx, gy - gh - 0.15, GREEN, 2.0)
    caption(s, gx - 0.2, gy - gh - 0.55, gw + 0.4, "truth T = 815 C, inside the 68% band",
            10, GREEN)
    box(s, 10.55, 2.0, 2.35, 1.4, ["coverage question",
        ("does the central 68% of",),
        ("the mixture catch the",),
        ("truth ~68% of rows?",)], GREEN, size=11)
    box(s, 0.55, 4.0, 5.9, 1.15, ["teacher-forced (true list): honest",
        ("68% band catches truth on 71.5% of rows (T),",),
        ("67.1% (P), 68.4% (H2O) . NLL 8.32 nats/row (was 8.60)",)], GREEN, size=12)
    box(s, 6.85, 4.0, 6.05, 1.15, ["ModeHead-sampled list: the exposure gap",
        ("coverage collapses to 26-35% . NLL +2.4 to +3.9 nats/axis",),
        ("(P_kbar hits inf rows) -- sharper under truth = more wrong under mismatch",)],
        REDX, size=12)
    caption(s, 0.55, 5.4, 12.2, "the gap is a deliberate debt: the head trains and is "
            "graded teacher-forced (mineralogy-first -- conditions are the bonus product), "
            "and tuning made the head sharper, which widened the gap. It is measured, "
            "tracked (#31, P3), and untouched by design until the mineralogy heads are "
            "done.", 11, INK, align=PP_ALIGN.LEFT)
    caption(s, 0.55, 6.15, 12.2, "population: all 55,576 val-roster rows, both scoring "
            "passes on the same rows with a fixed sampling seed. per-axis NLL (tuned, "
            "teacher-forced): T 4.97, P 1.82, H2O 1.53 -- summed 8.32.",
            11, INK, align=PP_ALIGN.LEFT)
    caption(s, 0.55, 6.9, 12.2, "current limits: axes are independent mixtures (no (T, P) "
            "correlation, deferred per ADR 006); T is a derived cooling endpoint, not a "
            "sampled axis; sampled-support P rows can blow up numerically (the inf rows "
            "are #31's starting point). full report: results/cond_head/<tag>/",
            10, GRAYT, align=PP_ALIGN.LEFT)
    notes(s, "Eval-semantics slide for CondHead. The two-pass protocol is the point: "
             "teacher-forced scoring isolates the head's conditional skill p(T,P,H2O | h, "
             "true support); sampled-support scoring measures the deployed chain with "
             "ModeHead's Bernoulli draws standing in for the truth (exposure_seed pins "
             "reproducibility). Mixture drawing illustrative but to scale for a bimodal "
             "amp/opx bulk (deep-wet vs shallow-dry is the flagship condition "
             "multimodality). Real numbers from results/cond_head/joint_heads_seed0/"
             "metrics.json vs the preserved baseline: teacher-forced NLL 8.60 -> 8.32 "
             "and coverage_68 0.65-0.69 -> 0.67-0.72 (tuning + the 512x2 trunk), while "
             "the exposure delta WIDENED (T 2.91 -> 3.94, H2O 1.53 -> 2.40, P finite -> "
             "inf on some rows) -- the sharper head is more wrong under mismatched "
             "supports. Coverage is central-interval mass via the mixture CDF (PIT), "
             "same construction as the mode/comp calibration reads.")


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
    s_corpora(prs); s_labelgen(prs); s_one_to_many(prs); s_regen_status(prs)
    s_architecture(prs)
    s_inputs(prs); s_embed(prs); s_encoder(prs)
    s_ssl(prs); s_workflow(prs)
    s_heads(prs); s_mode_head_anatomy(prs); s_mode_head_reliability(prs)
    s_mode_head_dirichlet(prs)
    s_comp_head_anatomy(prs); s_comp_head_pit(prs)
    s_cond_head_anatomy(prs); s_cond_head_coverage(prs)
    s_outputs(prs); s_scenarios(prs)
    out = Path(__file__).resolve().parent / "overview.pptx"
    prs.save(out)
    print(f"wrote {out}  ({len(prs.slides._sldIdLst)} slides)")
