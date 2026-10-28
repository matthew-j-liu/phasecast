"""The zero-shot corpus-size line per mineral category in the paper's
style: the last-epoch heads at one fitting depth
(`final{D}_allrock_n{N}[_seed{K}]`, `--depth D`, 224 by default) read on
the volcanic val rocks, from `results/depth_ladder.csv`. Eight
panels grouped by mineral family (framework, anhydrous mafic, hydrous,
oxides) and the mean of seven. No reference lines and no fitted slopes; the
curves are read by eye first.

  python explorations/corpus_scaling/fig_zero_shot_scaling.py --depth 224

Writes `results/gradient_steps/zero_shot_scaling_{D}.png` and `.pdf`. The looks come from
`paper/figures/src/style.py` (gitignored on main), imported at draw time.
The layout and sizes below are the locked style for data-scaling
plots (2026-10-27); the plutonic line and the per-depth copies reuse them.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cnmineral.data.phase_vocab import GROUPS

RESULTS = Path(__file__).resolve().parent / "results"
STYLE_DIR = REPO / "paper" / "figures" / "src"

# panel order groups the categories by family: framework silicates,
# anhydrous mafic, hydrous, then the oxides beside the mean
PANELS = ("fsp", "q", "px", "ol", "amp", "bi", "opq", "mean7")
LABELS = {"fsp": "Feldspar", "q": "Quartz", "px": "Pyroxene", "ol": "Olivine",
          "amp": "Amphibole", "bi": "Biotite", "opq": "Fe-Ti oxides", "mean7": "Mean of seven"}
MIN_SPAN = 3.0  # percentage points of y range per panel


def with_mean7(df, prefix):
    """Adds the seven-category mean of `{prefix}_mae_{g}` per row, the mean
    of the per-category errors as `share_slopes` reads them."""
    out = df.copy()
    out[f"{prefix}_mae_mean7"] = out[[f"{prefix}_mae_{g}" for g in GROUPS]].mean(axis=1)
    return out


def load_line(depth, csv=RESULTS / "depth_ladder.csv"):
    """Seed mean, min and max per corpus size of the last-epoch heads'
    zero-shot val share error (`val_mae_*`) at `depth` gradient steps."""
    df = with_mean7(pd.read_csv(csv), "val")
    cols = [f"val_mae_{c}" for c in PANELS]
    cells = df[(df.series == "final") & (df.depth.astype(str) == str(depth))]
    agg = cells.groupby("n_bulks")[cols].agg(["mean", "min", "max"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["seeds"] = cells.groupby("n_bulks").size()
    return agg.sort_index()


def setup_style():
    """The paper's style module with the quad preset's fonts raised for a
    figure 1.6 times the paper's text width; returns (style, spec)."""
    sys.path.insert(0, str(STYLE_DIR))
    import style

    spec = style.use("quad")
    spec = {**spec, "xtick": 10.5, "ytick": 9, "label": 14, "title": 11, "legend": 9}
    plt.rcParams.update({"xtick.labelsize": spec["xtick"], "ytick.labelsize": spec["ytick"],
                         "axes.titlesize": spec["title"], "legend.fontsize": spec["legend"]})
    return style, spec


def panel_grid(style):
    """The locked 2 x 4 frame: wider than the paper's text width, since eight
    panels each labelling ten corpus sizes do not fit in 5.5 in."""
    fig, axes = plt.subplots(2, 4, figsize=(1.6 * style.TEXT_WIDTH_IN, 4.2), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.3, hspace=0.47, wspace=0.34)
    return fig, axes


def finish_panel(ax, sizes, title, style):
    """Log x with every corpus size labelled, a minimum y span so seed noise
    in a flat category is not stretched to fill the frame, the title and the
    house frame."""
    ax.set_xscale("log")
    ax.set_xticks(sizes, [f"{n:,}" for n in sizes], rotation=90)
    ax.set_xlim(min(sizes) / 1.5, max(sizes) * 1.5)
    lo, hi = ax.get_ylim()
    if hi - lo < MIN_SPAN:
        mid = (hi + lo) / 2
        ax.set_ylim(mid - MIN_SPAN / 2, mid + MIN_SPAN / 2)
    ax.set_title(title)
    style.frame(ax)
    ax.xaxis.set_minor_locator(plt.NullLocator())


def draw(agg, depth, out=None):
    out = out or RESULTS / "gradient_steps" / f"zero_shot_scaling_{depth}"
    out.parent.mkdir(exist_ok=True)
    style, spec = setup_style()
    line, mean_line = style.CIVIDIS[2], style.CIVIDIS[9]
    sizes = [int(n) for n in agg.index]
    fig, axes = panel_grid(style)
    for i, (ax, c) in enumerate(zip(axes.ravel(), PANELS)):
        m = 100 * agg[f"val_mae_{c}_mean"]
        ax.errorbar(sizes, m, yerr=[m - 100 * agg[f"val_mae_{c}_min"], 100 * agg[f"val_mae_{c}_max"] - m],
                    fmt="-", marker="o", color=mean_line if c == "mean7" else line,
                    ms=3.5, elinewidth=0.9, capsize=1.5)
        finish_panel(ax, sizes, f"({chr(97 + i)}) {LABELS[c]}", style)
    fig.suptitle("Zero-shot predictions on validation rocks (volcanic)", y=0.995, fontsize=spec["label"] + 2)
    fig.supxlabel("Simulated compositions in the training corpus", y=0.085, fontsize=spec["label"])
    fig.supylabel("Abundance MAE (%)", x=0.012, fontsize=spec["label"])
    fig.legend(handles=[Line2D([], [], color=line, marker="o", ms=3.5,
                               label=f"Head at {depth} gradient steps; bars span three training seeds")],
               loc="lower center", bbox_to_anchor=(0.53, 0.0), handlelength=2.6)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=224, help="gradient steps of the line to draw")
    args = parser.parse_args()
    agg = load_line(args.depth)
    print(f"{args.depth} steps, sizes and seeds:", dict(agg["seeds"]))
    print("mean of seven, zero-shot by size:",
          {int(n): round(v, 4) for n, v in agg["val_mae_mean7_mean"].items()})
    print("wrote", draw(agg, args.depth))


if __name__ == "__main__":
    main()
