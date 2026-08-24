"""Masked-element reconstruction loss over pretraining, train vs validation,
for the production encoder."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))

import style
from style import DATADIR

TRAIN, VAL = style.CIVIDIS[2], style.CIVIDIS[9]


def pretrain_loss(name="pretrain_loss"):
    spec = style.use("single")
    d = pd.read_csv(DATADIR / "pretrain_loss.csv")

    fig, ax = plt.subplots(figsize=spec["figsize"])
    fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.14)
    ax.plot(d.epoch, d.train, color=TRAIN, marker="o", label="Train")
    ax.plot(d.epoch, d.val, color=VAL, marker="o", label="Validation")
    ax.set_xlabel("Pretraining epoch")
    ax.set_ylabel("Masked-element reconstruction loss")
    ax.set_xlim(0.5, d.epoch.max() + 0.5)
    ax.set_ylim(0.17, 0.37)
    ax.yaxis.set_major_locator(MultipleLocator(0.04))
    style.frame(ax)
    ax.legend(loc="upper right")
    style.save(fig, name)


if __name__ == "__main__":
    pretrain_loss()
