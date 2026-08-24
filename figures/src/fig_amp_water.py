"""Amphibole is decided by conditions the input chemistry does not record.

Amphibole occurrence and abundance against the water sampled for each draw,
pooled over the validation compositions of the label corpus.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import style
from style import DATADIR


def dose_figure(name="amp_water_dose"):
    spec = style.use("single")
    occurrence, abundance = style.CIVIDIS[2], style.CIVIDIS[9]

    b = pd.read_csv(DATADIR / "amp_water_bins.csv")
    x = (b.h2o_lo + b.h2o_hi) / 2

    fig, ax = plt.subplots(figsize=spec["figsize"])
    fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.14)
    ax.plot(x, 100 * b.presence_rate, color=occurrence, marker="o",
            label="Draws where amphibole crystallizes")
    ax.plot(x, 100 * b.median_share_present, color=abundance, marker="o",
            label="Median abundance when present")
    ax.set_xlabel("Dissolved water sampled for the draw (wt%)")
    ax.set_ylabel("Occurrence and abundance (%)")
    ax.set_ylim(0, 80)
    ax.set_xlim(0, 8)
    style.frame(ax)
    ax.legend(loc="lower right")
    style.save(fig, name)


if __name__ == "__main__":
    dose_figure()
