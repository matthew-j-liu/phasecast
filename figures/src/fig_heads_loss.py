"""Per-head loss over joint-heads training -- assemblage, composition, and
conditions, train vs validation -- with each head's best (saved) epoch marked.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))

import style
from style import DATADIR

TRAIN, VAL = style.CIVIDIS[2], style.CIVIDIS[9]

PANELS = [
    ("assemblage", "(a) Assemblage", 0.5),
    ("comp", "(b) Composition", None),
    ("cond", "(c) Conditions", 0.4),
]


def heads_loss(name="heads_loss"):
    style.use("double")
    d = pd.read_csv(DATADIR / "heads_loss.csv")
    n = int(d.epoch.max())

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.7))
    fig.subplots_adjust(left=0.115, right=0.98, top=0.86, bottom=0.30, wspace=0.45)
    for i, (ax, (head, title, ymajor)) in enumerate(zip(axes, PANELS)):
        sub = d[d["head"] == head].sort_values("epoch")
        best = int(sub.best_epoch.iloc[0])
        ax.plot(sub.epoch, sub.train, color=TRAIN, label="Train")
        ax.plot(sub.epoch, sub.val, color=VAL, label="Validation")
        ax.axvline(best, color=style.NEUTRAL, ls="--", lw=1.0)
        left = i == 1
        ax.text(best + (-1.5 if left else 1.5), 0.9, f"epoch {best}",
                transform=ax.get_xaxis_transform(), color=style.NEUTRAL,
                ha="right" if left else "left", va="top", fontsize=7)
        ax.set_title(title)
        ax.set_xlim(0, n)
        ax.xaxis.set_major_locator(MultipleLocator(22))
        if ymajor:
            ax.yaxis.set_major_locator(MultipleLocator(ymajor))
        style.frame(ax)
    axes[0].set_ylabel("Loss (nats)")
    fig.supxlabel("Training epoch", y=0.16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=2, fontsize=9,
               bbox_to_anchor=(0.5, 0.01), columnspacing=1.6, handlelength=1.6)
    style.save(fig, name)


if __name__ == "__main__":
    heads_loss()
