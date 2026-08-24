"""One held-out real rock, zero-shot over recalibrated: chemistry in, the
minerals present and their share of the crystalline mass out.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))

import style
from style import DATADIR

# same chroma-boosted cividis as the cohort card: dark ink is the model
# everywhere, yellow the measurement
PREDICTED = "#002657"
MEASURED = "#FDEA7C"
GRADIENT = [
    "#002657", "#002857", "#002D58", "#0D3958", "#1F4658", "#31535A",
    "#435F57", "#506952", "#5E724F", "#6C7B4A", "#7A8547", "#898F42",
    "#97983E", "#A4A03A", "#B1A933", "#BDB22D", "#CAB928", "#D6C022",
    "#E2C91C", "#F0D117", "#FCD911",
]


def chemistry_panel(ax, chem):
    """Cation wt% as log10 on a linear axis, as the cohort card draws it: a
    true log scale fights the house minor-tick convention."""
    chem = chem.sort_values("cat_wtpct", ascending=False).reset_index(drop=True)
    colors = [GRADIENT[int(round(i * (len(GRADIENT) - 1) / (len(chem) - 1)))]
              for i in range(len(chem))]
    base = -4.05
    ax.barh(np.arange(len(chem)),
            np.log10(np.clip(chem.cat_wtpct, 1e-4, None)) - base, left=base,
            color=colors, height=0.62, alpha=0.7)
    ax.set_yticks(np.arange(len(chem)), chem.element, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(base, 1.8)
    ax.set_xticks([-4, -2, 0], ["$10^{-4}$", "$10^{-2}$", "$10^{0}$"])


def presence_panel(ax, groups):
    y = np.arange(len(groups))
    ax.plot(groups.p_pred, y - 0.15, "o", ms=4.5, color=PREDICTED)
    ax.plot(groups.presence_measured, y + 0.15, "o", ms=4.5, color=MEASURED,
            mec=style.INK, mew=0.6)
    ax.axvline(0, color=style.INK, lw=1.0, ls=":", zorder=0)
    ax.set_xlim(-0.07, 1.05)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])


def abundance_panel(ax, groups):
    """Marginal share per group on a linear axis, so the predicted column sums
    to one and reads as one rock's mass budget.

    A group absent from most sampled assemblages has both quantiles at zero
    while its mean does not, so a mineral the model calls unlikely shows a
    point with no whisker.  Linear rather than log because a budget summing to
    100% is a composition, and a log axis has no place to put an exact zero --
    the previous floor drew clamped values where they do not sit.
    """
    for i, row in groups.reset_index(drop=True).iterrows():
        value, lo, hi = row.share_pred, row.share_lo, row.share_hi
        ax.errorbar(value, i - 0.15,
                    xerr=[[max(value - lo, 0)], [max(hi - value, 0)]],
                    fmt="o", ms=4.5, color=PREDICTED, lw=1.2, capsize=2.5)
        if np.isfinite(row.censor_bound):
            bound = row.censor_bound
            ax.plot(bound, i + 0.15, marker="|", ms=8, mew=1.2, color=style.INK)
            head = max(bound - 0.06, 0.0)
            if head < bound:
                ax.annotate("", xy=(head, i + 0.15), xytext=(bound, i + 0.15),
                            arrowprops={"arrowstyle": "->", "color": style.INK,
                                        "lw": 0.9})
            continue
        x_meas = row.measured_share if np.isfinite(row.measured_share) else 0.0
        ax.plot(x_meas, i + 0.15, "o", ms=4.5,
                color=MEASURED, mec=style.INK, mew=0.6)
    ax.axvline(0, color=style.INK, lw=1.0, ls=":", zorder=0)
    ax.set_xlim(-0.05, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])


def inference_card(name="inference_card"):
    style.use("quad")
    plt.rcParams.update({
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "mathtext.default": "regular",
    })

    chem = pd.read_csv(DATADIR / "inference_card_chemistry.csv")
    groups = pd.read_csv(DATADIR / "inference_card_groups.csv")

    fig = plt.figure(figsize=(5.5, 4.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 0.9, 1.2], wspace=0.42,
                          hspace=0.52, left=0.08, right=0.97, top=0.93,
                          bottom=0.17)
    ax0 = fig.add_subplot(gs[:, 0])
    chemistry_panel(ax0, chem)
    ax0.set_xlabel("Cation wt%")
    ax0.set_title("(a) Geochemical input")

    arms = (("zero_shot", "zero-shot", "(b)", "(c)"),
            ("recalibrated", "recalibrated", "(d)", "(e)"))
    axes = [ax0]
    for r, (arm, label, left, right) in enumerate(arms):
        rows = groups[groups.arm == arm]
        ax1, ax2 = fig.add_subplot(gs[r, 1]), fig.add_subplot(gs[r, 2])
        presence_panel(ax1, rows)
        abundance_panel(ax2, rows)
        ax1.set_title(f"{left} Presence, {label}")
        ax2.set_title(f"{right} Abundance, {label}")
        for ax in (ax1, ax2):
            ax.set_yticks(np.arange(len(rows)), rows.group)
            ax.set_ylim(len(rows) - 0.5, -0.5)
        if arm == "recalibrated":
            ax1.set_xlabel("Presence probability")
            ax2.set_xlabel("Fraction of crystalline mass")
        axes += [ax1, ax2]

    fig.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=4.5, color=PREDICTED,
               label="predicted"),
        Line2D([], [], marker="o", ls="none", ms=4.5, color=MEASURED,
               mec=style.INK, mew=0.6, label="measured"),
    ], loc="lower center", bbox_to_anchor=(0.70, 0.005), ncols=2,
        columnspacing=1.2, handlelength=1.2, handletextpad=0.4)

    for ax in axes:
        style.frame(ax)
        ax.yaxis.set_minor_locator(NullLocator())

    style.save(fig, name, trim=True)


if __name__ == "__main__":
    inference_card()
