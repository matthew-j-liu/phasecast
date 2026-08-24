"""Shared style for paper figures: layout presets, palette, PDF output.

Import `use(layout)` at the top of a figure script, draw, then `save(fig,
name)`. Font sizes come from the layout preset rather than from per-call
`fontsize=` arguments, so a size change lands in one place.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.transforms import Bbox

FIGDIR = Path(__file__).resolve().parents[1]
DATADIR = FIGDIR / "data"
ROOT = FIGDIR.parent

# neurips \textwidth at 10pt; a full-width figure is this wide
TEXT_WIDTH_IN = 5.5

# breathing room left above and below a trimmed figure, so a tick label's
# descender or a marker's edge is not shaved by the crop
TRIM_PAD_IN = 0.02

LAYOUTS = {
    # single is owner-ratified (2026-08-11), double 2026-08-20 (fewshot
    # curve); quad is an untested starting point that gets ratified the
    # first time a figure uses it
    "single": {
        "figsize": (TEXT_WIDTH_IN, 3.9),
        "tick": 11,
        "label": 14,
        "legend": 11,
        "title": 12,
        "annot": 11,
    },
    "double": {
        "figsize": (TEXT_WIDTH_IN, 3.1),
        "tick": 8.5,
        "label": 10,
        "legend": 8.0,
        "title": 10.5,
        "annot": 8.0,
    },
    "quad": {
        "figsize": (TEXT_WIDTH_IN, 4.8),
        "tick": 7,
        "label": 8,
        "legend": 7,
        "title": 8.5,
        "annot": 7,
    },
}

# sequential ramp for magnitude encodings (water content, density, corpus
# size). Cividis over viridis because it stays separable under deuteranopia
# and survives grayscale printing.
SEQUENTIAL = "cividis"

# cividis in 10 even steps, so a trace is picked by step number rather than by
# a fraction of the ramp. Verified against get_cmap("cividis") at
# linspace(0, 1, 10). Every step holds a visible line on white, which is why
# this is preferred over a single-hue ramp whose pale end vanishes at line
# weight (see tmp/palette_compare.py).
CIVIDIS = {
    1: "#00224E", 2: "#123570", 3: "#3B496C", 4: "#575D6D", 5: "#707173",
    6: "#8A8678", 7: "#A59C74", 8: "#C3B369", 9: "#E1CC55", 10: "#FEE838",
}

# categorical slots, fixed order, assigned by what the series IS. Validated
# for lightness, chroma, CVD separation and contrast against a white surface.
GREEN = "#15855c"
PURPLE = "#4a3aa7"
ORANGE = "#d4542a"
BLUE = "#2a78d6"
CATEGORICAL = (GREEN, PURPLE, ORANGE, BLUE)

# role aliases so a series keeps its color across figures
PRESENCE = GREEN
ABUNDANCE = PURPLE
SIMULATOR = ORANGE
RECALIBRATED = BLUE

NEUTRAL = "#6b6a66"
INK = "#0b0b0b"
GRID = "#e2e1de"

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": INK,
    "axes.grid": False,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    # closed box with ticks stepping outward, weighted so the ticks read as
    # part of the frame rather than as hairlines against it
    "axes.linewidth": 1.3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
})


def use(layout="single"):
    """Apply a layout preset's font sizes; returns the preset."""
    spec = LAYOUTS[layout]
    plt.rcParams.update({
        "figure.figsize": spec["figsize"],
        "font.size": spec["label"],
        "axes.labelsize": spec["label"],
        "axes.titlesize": spec["title"],
        "xtick.labelsize": spec["tick"],
        "ytick.labelsize": spec["tick"],
        "legend.fontsize": spec["legend"],
    })
    return spec


def frame(ax):
    """House axis convention: one minor tick between majors, none on the top
    or right spine. The box itself is matplotlib's default four spines."""
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", top=False, right=False)


def save(fig, name, png=True, trim=False):
    """Write `name`.pdf into paper/figures/, plus a gitignored png preview.

    `trim` crops the empty margin above and below the drawn content without
    moving anything inside it. The full width is kept, so a figure set at a
    fraction of the line width renders its contents at the same scale and
    only loses height.
    """
    bbox = None
    if trim:
        fig.canvas.draw()
        tight = fig.get_tightbbox(fig.canvas.get_renderer())
        bbox = Bbox.from_extents(0, tight.y0 - TRIM_PAD_IN,
                                 fig.get_figwidth(), tight.y1 + TRIM_PAD_IN)
    pdf = FIGDIR / f"{name}.pdf"
    fig.savefig(pdf, bbox_inches=bbox)
    print("wrote", pdf.relative_to(FIGDIR.parent))
    if png:
        fig.savefig(FIGDIR / f"{name}.png", dpi=200, bbox_inches=bbox)
