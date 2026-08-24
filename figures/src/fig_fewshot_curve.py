"""Three analogous two-panel figures comparing fits across labeled-rock
sizes, drawn from the CSVs written by extract_fewshot_curve.py.

`fewshot_tiers` compares the recalibration tiers, `fewshot_heads` the heads
trained on measured abundances, and `fewshot_strongest` the strongest of each
against the other -- the comparison the prose leads with. Splitting them is
deliberate: a single six-curve frame invites comparison across the families,
but that comparison is a within-replicate paired quantity, which overlaid
curves understate because the selection noise is shared. All three share
y-limits so they can still be read against each other.

Color identifies a fit rather than a family, so the third figure's two curves
are lifted from the first two unchanged. Colors are Okabe-Ito, with black for
the mixture tier and blue for the head given the latent, the two the prose
leads with. Only three colors share a frame, so the constraint is two
separable triples rather than six mutually separable colors; both triples and
the third figure's pair clear the CVD and normal-vision checks.

Uncertainty is the interquartile range across the replicates, drawn as error
bars per point: at ten replicates a 10-90 band is effectively min-max, and
shaded bands stack into indistinguishable washes.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))

import style
from style import DATADIR

# Okabe-Ito, the colorblind-safe categorical eight
BLACK = "#000000"
VERMILLION = "#D55E00"
BLUISH_GREEN = "#009E73"
BLUE = "#0072B2"
ORANGE = "#E69F00"
REDDISH_PURPLE = "#CC79A7"

# (arm, color, linestyle, marker); every fit keeps its own color and marker
# across all three figures, so no mark means two things
TIERS = (("tailed", BLACK, "-", "o"),
         ("matrix", VERMILLION, "-", "s"),
         ("vector", BLUISH_GREEN, "-", "^"))
HEADS = (("measured", BLUE, "-", "D"),
         ("measured_raw", ORANGE, "-", "v"),
         ("measured_ablated", REDDISH_PURPLE, "-", "X"))
STRONGEST = (TIERS[0], HEADS[0])

LABEL = {
    "vector": "Per-mineral category tier (23 params)",
    "matrix": "Category-coupled tier (121 params)",
    "tailed": "Mixture tier (123 params)",
    "measured": "Head given the latent",
    "measured_raw": "Head given raw geochemistry",
    "measured_ablated": "Head given no chemistry",
}
BAND_Q = (0.25, 0.75)


def aggregate(curve, key):
    """Median and interquartile spread across replicates, x = labeled rocks."""
    out = {}
    for arm, rows in curve.groupby("arm"):
        g = rows.groupby("budget")
        out[arm] = pd.DataFrame({
            "x": g.n_rocks.mean(),
            "med": g[key].median(),
            "lo": g[key].quantile(BAND_Q[0]),
            "hi": g[key].quantile(BAND_Q[1]),
        }).sort_values("x")
    return out


def limits(curve, key, scale):
    """Shared across all three figures, so they are read against each other."""
    agg = aggregate(curve, key)
    lo = min(scale * a.lo.min() for a in agg.values())
    hi = max(scale * a.hi.max() for a in agg.values())
    pad = 0.06 * (hi - lo)
    return lo - pad, hi + pad


def panel(ax, curve, key, family, ylim, scale=1.0):
    agg = aggregate(curve, key)
    for arm, color, ls, marker in family:
        a = agg[arm]
        ax.errorbar(a.x, scale * a.med,
                    yerr=[scale * (a.med - a.lo), scale * (a.hi - a.med)],
                    fmt=ls, marker=marker, color=color, ms=4.0,
                    elinewidth=1.0, capsize=2.0)
    ax.set_xscale("log")
    ticks = sorted(int(round(v)) for v in next(iter(agg.values())).x)
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Labeled rocks")
    ax.set_ylim(*ylim)
    style.frame(ax)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())


def make(name, family, curve, meta, ylims):
    spec = style.LAYOUTS["double"]
    sub = curve[curve.arm.isin([a for a, _, _, _ in family])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=spec["figsize"])
    fig.subplots_adjust(left=0.09, right=0.98, top=0.91, bottom=0.34,
                        wspace=0.5)

    panel(ax1, sub, "joint", family, ylims["joint"])
    ax1.set_ylabel("Joint NLL (nats)")
    ax1.set_title("(a) Model confidence")

    panel(ax2, sub, "share_mae", family, ylims["share_mae"], scale=100.0)
    ax2.axhline(100 * meta.zero_shot_share_mae, lw=1.0, ls="--",
                color=style.NEUTRAL)
    ax2.annotate("Zero-shot", (0.97, 100 * meta.zero_shot_share_mae),
                 xycoords=("axes fraction", "data"), ha="right", va="bottom",
                 color=style.NEUTRAL, fontsize=spec["annot"])
    ax2.set_ylabel("Abundance MAE (%)")
    ax2.set_title("(b) Model accuracy")

    # hand-built so the swatches carry line and marker without the errorbar
    # whisker, which reads as part of the symbol at this size
    fig.legend(handles=[Line2D([], [], color=color, ls=ls, marker=marker,
                               ms=4.0, label=LABEL[arm])
                        for arm, color, ls, marker in family],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncols=1,
               handlelength=2.6)

    style.save(fig, name)
    plt.close(fig)


def main():
    style.use("double")
    curve = pd.read_csv(DATADIR / "fewshot_curve.csv")
    meta = pd.read_csv(DATADIR / "fewshot_curve_meta.csv").iloc[0]
    ylims = {"joint": limits(curve, "joint", 1.0),
             "share_mae": limits(curve, "share_mae", 100.0)}
    make("fewshot_tiers", TIERS, curve, meta, ylims)
    make("fewshot_heads", HEADS, curve, meta, ylims)
    make("fewshot_strongest", STRONGEST, curve, meta, ylims)


if __name__ == "__main__":
    main()
