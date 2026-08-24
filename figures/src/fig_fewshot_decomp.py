"""The fewshot_heads comparison decomposed into its presence (Bernoulli) and
abundance (Dirichlet) branches, which sum to the joint NLL per replicate.

Laid out after fig_heads_loss (three column-width panels, shared axis labels)
rather than after the two-panel fewshot family, since all three panels share
one quantity. Head colors, markers, and labels import from fig_fewshot_curve
so each fit keeps its identity across figures.

Curves are per-head medians across replicates; the differences the prose
quotes are paired within-replicate medians, which the visual gap between
curves overstates at small fitting sizes. The caption says so.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))

import style
from style import DATADIR
from fig_fewshot_curve import BAND_Q, HEADS, LABEL

PANELS = (("bernoulli", "(a) Presence", None),
          ("dirichlet", "(b) Abundance", [-2.5, -2.0, -1.5]),
          ("joint", "(c) Joint", None))


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


def panel(ax, curve, key, title, yticks):
    agg = aggregate(curve, key)
    for arm, color, ls, marker in HEADS:
        a = agg[arm]
        ax.errorbar(a.x, a.med, yerr=[a.med - a.lo, a.hi - a.med],
                    fmt=ls, marker=marker, color=color, ms=4.0,
                    elinewidth=1.0, capsize=2.0)
    ax.set_xscale("log")
    ticks = sorted(int(round(v)) for v in next(iter(agg.values())).x)
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    if yticks:
        ax.set_yticks(yticks)
    ax.set_title(title)
    style.frame(ax)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())


def main():
    style.use("double")
    curve = pd.read_csv(DATADIR / "fewshot_curve.csv")
    curve = curve[curve.arm.isin([a for a, _, _, _ in HEADS])]

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.7))
    fig.subplots_adjust(left=0.115, right=0.98, top=0.86, bottom=0.30, wspace=0.50)
    for ax, (key, title, yticks) in zip(axes, PANELS):
        panel(ax, curve, key, title, yticks)
    axes[0].set_ylabel("NLL (nats)")
    # nudge the middle panel right: its wider (negative, decimal) y-tick labels
    # eat the a-b gap, so the b-c gap reads larger without this
    pos = axes[1].get_position()
    axes[1].set_position([pos.x0 + 0.012, pos.y0, pos.width, pos.height])
    fig.supxlabel("Labeled rocks", y=0.13)

    fig.legend(handles=[Line2D([], [], color=c, ls=ls, marker=m, ms=4.0,
                               label=LABEL[a]) for a, c, ls, m in HEADS],
               loc="lower center", bbox_to_anchor=(0.5, 0.01), ncols=3,
               fontsize=8, columnspacing=1.4, handlelength=1.8)

    style.save(fig, "fewshot_decomp")
    plt.close(fig)


if __name__ == "__main__":
    main()
