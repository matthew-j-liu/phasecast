"""Appendix cards: composition- and conditions-head predictions, one figure
for a simulated validation composition (truth overlaid) and one for the Fig 2
rhyolite (predictions only), drawn from the CSVs written by
extract_comp_cond_card.py.
"""

import re
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

PREDICTED = "#002657"
TRUTH = "#F0D117"
COND_LABEL = {"T": "Temperature (\N{DEGREE SIGN}C)", "P_kbar": "Pressure (kbar)",
              "H2O": "Water content (wt%)"}


def oxide_label(name):
    return re.sub(r"(\d+)", r"$_{\1}$", name)


def comp_panel(ax, rows):
    y = np.arange(len(rows))
    has_truth = rows.truth_med.notna().any()
    off = 0.18 if has_truth else 0.0
    ax.errorbar(rows.pred_mean, y - off,
                xerr=[np.maximum(rows.pred_mean - rows.pred_lo, 0),
                      np.maximum(rows.pred_hi - rows.pred_mean, 0)],
                fmt="o", ms=3.5, color=PREDICTED, lw=1.0, capsize=1.8)
    if has_truth:
        ax.errorbar(rows.truth_med, y + off,
                    xerr=[np.maximum(rows.truth_med - rows.truth_lo, 0),
                          np.maximum(rows.truth_hi - rows.truth_med, 0)],
                    fmt="o", ms=3.5, color=TRUTH, mec=style.INK, mew=0.5,
                    ecolor=style.INK, lw=0.8, capsize=1.8)
    ax.set_yticks(y, [oxide_label(o) for o in rows.oxide])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 150)


def cond_panel(ax, samples, truth):
    allv = samples if truth is None else np.concatenate([truth, samples])
    bins = np.histogram_bin_edges(allv, bins=30)
    if truth is not None:
        ax.hist(truth, bins=bins, density=True, color=TRUTH, alpha=0.75,
                edgecolor="none")
    ax.hist(samples, bins=bins, density=True, color=PREDICTED,
            histtype="step", lw=1.3)
    ax.set_yticks([])


def comp_cond_card(tag, comp, cond):
    with_truth = comp.truth_med.notna().any()
    fig = plt.figure(figsize=(5.5, 5.0))
    outer = fig.add_gridspec(1, 2, wspace=0.42, width_ratios=[1, 1.1],
                             left=0.12, right=0.97, top=0.94,
                             bottom=0.2 if with_truth else 0.14)
    comp_col = outer[0].subgridspec(2, 1, hspace=0.55)
    for r, panel in enumerate("ab"):
        ax = fig.add_subplot(comp_col[r])
        rows = comp[comp.panel == panel].reset_index(drop=True)
        comp_panel(ax, rows)
        ax.set_title(f"({panel}) {rows.phase.iloc[0]} composition")
        if r == 1:
            ax.set_xlabel("Mass fraction (wt%)")
        style.frame(ax)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
    cond_col = outer[1].subgridspec(3, 1, hspace=0.75)
    for a, axis_name in enumerate(dict.fromkeys(cond.axis)):
        ax = fig.add_subplot(cond_col[a])
        rows = cond[cond.axis == axis_name]
        truth = rows.value[rows.kind == "label"].to_numpy()
        cond_panel(ax, rows.value[rows.kind == "predicted"].to_numpy(),
                   truth if len(truth) else None)
        ax.set_title(f"({'cde'[a]}) {COND_LABEL[axis_name]}")
        style.frame(ax)
        ax.tick_params(left=False)
    if with_truth:
        fig.legend(handles=[
            Line2D([], [], marker="o", ls="none", ms=4, color=PREDICTED,
                   label="predicted"),
            Line2D([], [], marker="s", ls="none", ms=6, color=TRUTH,
                   mec=style.INK, mew=0.5, label="simulated label"),
        ], loc="lower center", bbox_to_anchor=(0.5, 0.01), ncols=2)
    style.save(fig, f"comp_cond_card_{tag}")
    plt.close(fig)


def main():
    style.use("quad")
    plt.rcParams.update({
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "mathtext.default": "regular",
    })
    comp = pd.read_csv(DATADIR / "comp_cond_card_comp.csv")
    cond = pd.read_csv(DATADIR / "comp_cond_card_cond.csv")
    for tag in ("simulated", "real"):
        comp_cond_card(tag, comp[comp.card == tag], cond[cond.card == tag])


if __name__ == "__main__":
    main()
