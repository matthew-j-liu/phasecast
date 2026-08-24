"""Zero-shot probability outputs at cohort level: chemistry in, presence
discrimination and teacher-forced abundance out, against 640 measured rocks.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent))

import style
from style import DATADIR

# chroma-boosted cividis (HSV s x1.5, v x1.12 from the #00224E / #E1CC55
# anchors). Dark ink means the model in both panels that show it: (b) dots
# and (c) predicted violins. #002657 near-duplicates E1's #123570 -- accepted.
PREDICTED = "#002657"
MEASURED = "#FCD911"
GRADIENT = [
    "#002657", "#002857", "#002D58", "#0D3958", "#1F4658", "#31535A",
    "#435F57", "#506952", "#5E724F", "#6C7B4A", "#7A8547", "#898F42",
    "#97983E", "#A4A03A", "#B1A933", "#BDB22D", "#CAB928", "#D6C022",
    "#E2C91C", "#F0D117", "#FCD911",
]

# matches cnmineral.eval.real_rock.CARD_SHARE_FLOOR
SHARE_FLOOR = 5e-4


def violin(ax, values, y, color, *, side="both", width=0.38, alpha=0.55):
    """Horizontal KDE with a median tick; <10 values plot as points, since a
    density over a handful of rocks is a drawing, not an estimate."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 10 or np.ptp(v) < 1e-9:
        if len(v):
            ax.plot(v, np.full(len(v), y), "o", ms=4, color=color)
        return
    grid = np.linspace(v.min(), v.max(), 256)
    dens = gaussian_kde(v)(grid)
    dens = dens / dens.max() * width
    if side in ("both", "up"):
        ax.fill_between(grid, y, y - dens, color=color, alpha=alpha, lw=0)
    if side in ("both", "down"):
        ax.fill_between(grid, y, y + dens, color=color, alpha=alpha, lw=0)
    ax.plot([np.median(v)], [y], "|", ms=10, color=color, mew=1.3)


def cohort_card(name="cohort_card"):
    style.use("quad")
    plt.rcParams.update({
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "mathtext.default": "regular",
    })

    elements = pd.read_csv(DATADIR / "cohort_card_elements.csv")
    groups = pd.read_csv(DATADIR / "cohort_card_groups.csv")
    abundance = pd.read_csv(DATADIR / "cohort_card_abundance.csv")

    fig = plt.figure(figsize=(5.5, 4.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1, 1.15],
                          wspace=0.34, hspace=0.58, left=0.10, right=0.97,
                          top=0.93, bottom=0.17)
    ax0 = fig.add_subplot(gs[:, 0])
    axa = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])

    pools = {sym: np.log10(np.clip(vals, 1e-4, None))
             for sym, vals in elements.groupby("element").cat_wtpct}
    order = sorted(pools, key=lambda s: -np.median(pools[s]))
    for i, sym in enumerate(order):
        violin(ax0, pools[sym], i, GRADIENT[i], alpha=0.7)
    ax0.set_yticks(range(len(order)), order)
    ax0.invert_yaxis()
    ax0.set_xticks([-3, -2, -1, 0, 1],
                   ["$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$10^{0}$", "$10^{1}$"])
    ax0.set_xlabel("Cation wt%")
    ax0.set_title("(a) Geochemical inputs")

    y = range(len(groups))
    axa.errorbar(groups.auroc, y,
                 xerr=[groups.auroc - groups.auroc_q25,
                       groups.auroc_q75 - groups.auroc],
                 fmt="o", ms=4.5, color=PREDICTED, ecolor=PREDICTED,
                 elinewidth=1.0, capsize=2.0)
    axa.axvline(0.5, color="black", linestyle=(0, (4, 3)), linewidth=1)
    axa.set_xlim(0.4, 1.02)
    axa.set_xticks([0.5, 0.75, 1.0], ["0.5", "0.75", "1"])
    axa.set_xlabel("Presence AUROC", fontsize=10)
    axa.set_title("(b) Presence predictions")

    floor_log = np.log10(SHARE_FLOOR)
    for i, row in groups.iterrows():
        pool = abundance[abundance.group == row.group]
        for kind, color, side in (("measured", MEASURED, "down"),
                                  ("predicted", PREDICTED, "up")):
            values = pool[pool.kind == kind].share
            violin(ax2, np.log10(np.maximum(values, SHARE_FLOOR)), i, color,
                   side=side)
        ax2.annotate(f"n={row.n_quantified}", (floor_log + 0.08, i),
                     fontsize=8, color="black", ha="left", va="center")
    ax2.set_xlim(floor_log, 0.15)
    ax2.set_xticks(np.log10([0.001, 0.01, 0.1, 1.0]),
                   ["0.1%", "1%", "10%", "100%"])
    ax2.set_xlabel("Fraction of rock's crystalline mass", fontsize=10)
    ax2.set_title("(c) Abundance predictions")
    fig.legend(handles=[
        Patch(color=PREDICTED, alpha=0.55, label="predicted"),
        Patch(color=MEASURED, alpha=0.55, label="measured"),
    ], loc="lower center", bbox_to_anchor=(0.76, 0.005), ncols=2,
        columnspacing=1.2, handlelength=1.2, handletextpad=0.4)

    for ax in (axa, ax2):
        ax.set_yticks(y, groups.group)
        ax.set_ylim(len(groups) - 0.4, -0.6)
    for ax in (ax0, axa, ax2):
        style.frame(ax)
        ax.yaxis.set_minor_locator(NullLocator())

    style.save(fig, name, trim=True)


if __name__ == "__main__":
    cohort_card()
