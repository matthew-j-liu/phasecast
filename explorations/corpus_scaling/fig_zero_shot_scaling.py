"""The zero-shot corpus-size line per mineral category in the paper's
style: the last-epoch heads at one fitting depth
(`final{D}_allrock_n{N}[_seed{K}]`, `--depth D`, 224 by default) read on a
benchmark's train and val rocks, from `results/zero_shot_line_{benchmark}.csv`
(`run_zero_shot_line.py`). Eight panels grouped by mineral family
(framework, anhydrous mafic, hydrous, oxides) and the mean of seven. No
reference lines and no fitted slopes; the curves are read by eye first. On
the plutonic benchmark a second figure draws the line per pretraining group,
and where heads on random training subsets exist a third shows seed, rock
and training-subset variation together (`zero_shot_line_{benchmark}_bands.json`)
and a fourth draws the median over the random training subsets with their
25th to 75th percentiles.

  python explorations/corpus_scaling/fig_zero_shot_scaling.py --benchmark volcanic --depth 224
  python explorations/corpus_scaling/fig_zero_shot_scaling.py --benchmark plutonic --depth 224

Writes `results/gradient_steps/zero_shot_scaling_{D}.png` and `.pdf` for the
volcanic benchmark, `zero_shot_scaling_plutonic_{D}` and
`zero_shot_scaling_plutonic_{D}_pretraining` for the plutonic one. The line
over random training subsets is the headline and goes to the repo's
`results/corpus_scaling/{basis}/zero_shot_{benchmark}_{D}_subsets`, read
teacher-forced (`teacher_forced/`, given each rock's measured mineral list)
and marginally (`marginal/`, over mineral sets the head samples itself, from
`zero_shot_line_{benchmark}_marginal.csv`), each once to 10,000 and once cut
at 5,000 (`_to5000`), where the 10,000 point, a single subset
read over three seeds, is a different kind of bar. In the cut version
every panel is scaled so its longest bar covers 45% of its height, which
keeps the spread across subsets from hiding the trend and makes the panels
read alike (2026-11-02). The looks
come from `paper/figures/src/style.py` (gitignored on main), imported at
draw time. The layout and sizes below are the locked style for data-scaling
plots (2026-10-27); every per-benchmark and per-depth copy reuses them.
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

RESULTS = Path(__file__).resolve().parent / "results"
HEADLINE = REPO / "results" / "corpus_scaling"
STYLE_DIR = REPO / "paper" / "figures" / "src"

# panel order groups the categories by family: framework silicates,
# anhydrous mafic, hydrous, then the oxides beside the mean
PANELS = ("fsp", "q", "px", "ol", "amp", "bi", "opq", "mean7")
LABELS = {"fsp": "Feldspar", "q": "Quartz", "px": "Pyroxene", "ol": "Olivine",
          "amp": "Amphibole", "bi": "Biotite", "opq": "Fe-Ti oxides", "mean7": "Mean of seven"}
MIN_SPAN = 3.0  # percentage points of y range per panel
PRETRAINING_LABELS = {"ds942_seen": "DS 942, seen in pretraining", "ds942_held_out": "DS 942, held out",
                      "pp1793": "PP 1793"}


def load_line(depth, benchmark="volcanic", group="all", csv=None):
    """Seed mean (as `_mid`, the plotted point), min and max per corpus size
    of the nested-subset heads' zero-shot share error (`mae_*`) on one rock
    group at `depth` gradient steps."""
    df = pd.read_csv(csv or RESULTS / f"zero_shot_line_{benchmark}.csv")
    cols = [f"mae_{c}" for c in PANELS]
    cells = df[(df.depth == depth) & (df.group == group) & (df.subset == 0)]
    agg = cells.groupby("n_bulks")[cols].agg([("mid", "mean"), "min", "max"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["seeds"] = cells.groupby("n_bulks").size()
    agg["n_rocks"] = cells.groupby("n_bulks")["n_rocks"].first()
    return agg.sort_index()


def load_subset_line(depth, benchmark="volcanic", csv=None):
    """Per corpus size, the median over the heads trained on random training
    subsets as `_mid` and the 25th and 75th percentiles as `_min` / `_max`;
    the median because the subsets are right-skewed at small N. The full
    corpus, a single subset, takes its nested heads' seed mean, min and max."""
    df = pd.read_csv(csv or RESULTS / f"zero_shot_line_{benchmark}.csv")
    cols = [f"mae_{c}" for c in PANELS]
    cells = df[(df.depth == depth) & (df.group == "all")]
    subsets = cells[cells.subset > 0]
    full = cells[(cells.subset == 0) & ~cells.n_bulks.isin(subsets.n_bulks)]
    by = subsets.groupby("n_bulks")[cols]
    agg = pd.concat([by.median().add_suffix("_mid"), by.quantile(0.25).add_suffix("_min"),
                     by.quantile(0.75).add_suffix("_max")], axis=1)
    whole = full.groupby("n_bulks")[cols].agg([("mid", "mean"), "min", "max"])
    whole.columns = [f"{a}_{b}" for a, b in whole.columns]
    return pd.concat([agg, whole[agg.columns]]).sort_index()


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


def plot_lines(lines, title, out, bar_share=None, note=None):
    """`lines` maps a legend label to (aggregated line, colour key, or None
    for the per-panel default); one curve per entry in every panel. With
    `bar_share`, every panel's y range is set so its longest bar covers
    `bar_share` of it, centred on the data and never below zero. A `note`
    goes on its own line under the legend, for what the bars show."""
    out.parent.mkdir(exist_ok=True)
    style, spec = setup_style()
    sizes = sorted({int(n) for agg, _ in lines.values() for n in agg.index})
    fig, axes = panel_grid(style)
    for i, (ax, c) in enumerate(zip(axes.ravel(), PANELS)):
        longest, low, high = 0.0, float("inf"), 0.0
        for agg, color in lines.values():
            color = color or (style.CIVIDIS[9] if c == "mean7" else style.CIVIDIS[2])
            ns = [int(n) for n in agg.index]
            m = 100 * agg[f"mae_{c}_mid"]
            ax.errorbar(ns, m, yerr=[m - 100 * agg[f"mae_{c}_min"], 100 * agg[f"mae_{c}_max"] - m],
                        fmt="-", marker="o", color=color, ms=3.5, elinewidth=0.9, capsize=1.5)
            longest = max(longest, 100 * (agg[f"mae_{c}_max"] - agg[f"mae_{c}_min"]).max())
            low, high = min(low, 100 * agg[f"mae_{c}_min"].min()), max(high, 100 * agg[f"mae_{c}_max"].max())
        finish_panel(ax, sizes, f"({chr(97 + i)}) {LABELS[c]}", style)
        if bar_share:
            half = max(longest / bar_share, high - low) / 2
            bottom = max((low + high) / 2 - half, 0.0)
            ax.set_ylim(bottom, bottom + 2 * half)
    fig.suptitle(title, y=0.995, fontsize=spec["label"] + 2)
    fig.supxlabel("Simulated compositions in the training corpus", y=0.1 if note else 0.085, fontsize=spec["label"])
    fig.supylabel("Abundance MAE (%)", x=0.012, fontsize=spec["label"])
    fig.legend(handles=[Line2D([], [], color=color or style.CIVIDIS[2], marker="o", ms=3.5, label=label)
                        for label, (_, color) in lines.items()],
               loc="lower center", bbox_to_anchor=(0.53, 0.022 if note else 0.0), ncol=len(lines),
               handlelength=2.6)
    if note:
        fig.text(0.53, 0.004, note, ha="center", va="bottom", fontsize=spec["legend"])
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def plot_variation(benchmark, depth, out):
    """The three sources of variation on one axis per category: the
    nested-subset line with its seed range, its unit-bootstrap
    interquartile range shaded, and each random training subset's head as a
    small hollow marker."""
    import json

    df = pd.read_csv(RESULTS / f"zero_shot_line_{benchmark}.csv")
    subsets = df[(df.depth == depth) & (df.group == "all") & (df.subset > 0)]
    bands = json.loads((RESULTS / f"zero_shot_line_{benchmark}_bands.json").read_text())[str(depth)]["all"]
    agg = load_line(depth, benchmark)
    style, spec = setup_style()
    line, rep_color = style.CIVIDIS[2], style.CIVIDIS[8]
    sizes = [int(n) for n in agg.index]
    fig, axes = panel_grid(style)
    for i, (ax, c) in enumerate(zip(axes.ravel(), PANELS)):
        lo = [100 * bands["points_interquartile"][str(n)][c][0] for n in sizes]
        hi = [100 * bands["points_interquartile"][str(n)][c][1] for n in sizes]
        ax.fill_between(sizes, lo, hi, color=line, alpha=0.15, lw=0)
        m = 100 * agg[f"mae_{c}_mid"]
        ax.errorbar(sizes, m, yerr=[m - 100 * agg[f"mae_{c}_min"], 100 * agg[f"mae_{c}_max"] - m],
                    fmt="-", marker="o", color=line, ms=3.5, elinewidth=0.9, capsize=1.5)
        ax.plot(subsets["n_bulks"], 100 * subsets[f"mae_{c}"], "o", mfc="none", mec=rep_color, ms=2.5, mew=0.6,
                alpha=0.5)
        finish_panel(ax, sizes, f"({chr(97 + i)}) {LABELS[c]}", style)
    fig.suptitle(f"Seed, rock and training-subset variation ({benchmark}, {depth} steps)", y=0.995, fontsize=spec["label"] + 2)
    fig.supxlabel("Simulated compositions in the training corpus", y=0.085, fontsize=spec["label"])
    fig.supylabel("Abundance MAE (%)", x=0.012, fontsize=spec["label"])
    handles = [Line2D([], [], color=line, marker="o", ms=3.5, label="Nested training subset; bars span three seeds"),
               plt.Rectangle((0, 0), 1, 1, color=line, alpha=0.15, label="Unit-bootstrap interquartile range"),
               Line2D([], [], ls="none", marker="o", mfc="none", mec=rep_color, ms=3.5,
                      label="Random training subsets, seed 0")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0), ncol=3, handlelength=2.6)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def plot_mean_vs_median(benchmark, depth, out):
    """The mean of seven over the random training subsets two ways, on one y
    scale: (a) median with the 25th and 75th percentiles, (b) mean with one
    standard deviation. The full corpus, a single subset, is read the same
    way over its three seeds."""
    df = pd.read_csv(RESULTS / f"zero_shot_line_{benchmark}.csv")
    cells = df[(df.depth == depth) & (df.group == "all")]
    subsets = cells[cells.subset > 0]
    full = cells[(cells.subset == 0) & ~cells.n_bulks.isin(subsets.n_bulks)]
    by = (100 * pd.concat([subsets, full]).set_index("n_bulks")["mae_mean7"]).groupby(level=0)
    sizes = [int(n) for n in by.size().index]
    style, spec = setup_style()
    color = style.CIVIDIS[2]
    fig, axes = plt.subplots(1, 2, figsize=(1.3 * style.TEXT_WIDTH_IN, 3.4), sharey=True)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.8, bottom=0.33, wspace=0.12)
    median, q25, q75 = by.median(), by.quantile(0.25), by.quantile(0.75)
    mean, sd = by.mean(), by.std()
    for ax, (y, lo, hi), title in ((axes[0], (median, q25, q75), "(a) Median, 25th to 75th percentile"),
                                   (axes[1], (mean, mean - sd, mean + sd), "(b) Mean, one standard deviation")):
        ax.errorbar(sizes, y, yerr=[y - lo, hi - y], fmt="-", marker="o", color=color,
                    ms=3.5, elinewidth=0.9, capsize=1.5)
        finish_panel(ax, sizes, title, style)
    fig.supxlabel("Simulated compositions in the training corpus", y=0.02, fontsize=spec["label"] - 2)
    fig.supylabel("Abundance MAE (%)", x=0.01, fontsize=spec["label"] - 2)
    fig.suptitle(f"Mean of seven over random training subsets ({benchmark}, {depth} steps; 10,000: three seeds)",
                 y=0.99, fontsize=spec["label"] - 3)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("volcanic", "plutonic"), default="volcanic")
    parser.add_argument("--depth", type=int, default=224, help="gradient steps of the line to draw")
    parser.add_argument("--mean-vs-median", action="store_true",
                        help="draw only `gradient_steps/mean_vs_median_{benchmark}`, the two summaries of the "
                             "subset line")
    args = parser.parse_args()
    if args.mean_vs_median:
        print("wrote", plot_mean_vs_median(args.benchmark, args.depth,
                                           RESULTS / "gradient_steps" / f"mean_vs_median_{args.benchmark}"))
        return
    agg = load_line(args.depth, args.benchmark)
    stem = "zero_shot_scaling" if args.benchmark == "volcanic" else "zero_shot_scaling_plutonic"
    print(f"{args.benchmark}, {args.depth} steps, {int(agg['n_rocks'].iloc[0])} rocks, seeds:", dict(agg["seeds"]))
    print("mean of seven, zero-shot by size:",
          {int(n): round(v, 4) for n, v in agg["mae_mean7_mid"].items()})
    title = f"Zero-shot predictions on train and validation rocks ({args.benchmark})"
    label = f"Head at {args.depth} gradient steps; bars span three training seeds"
    print("wrote", plot_lines({label: (agg, None)}, title,
                              RESULTS / "gradient_steps" / f"{stem}_{args.depth}"))
    df = pd.read_csv(RESULTS / f"zero_shot_line_{args.benchmark}.csv")
    if ((df.depth == args.depth) & (df.subset > 0)).any():
        print("wrote", plot_variation(args.benchmark, args.depth,
                                      RESULTS / "gradient_steps" / f"{stem}_{args.depth}_variation"))
        n_subsets = df.loc[(df.depth == args.depth) & (df.subset > 0), "subset"].nunique()
        label = (f"Median over {n_subsets} random training subsets, each trained for {args.depth} "
                 "gradient steps")
        bars = "Bars span the 25th to 75th percentile"
        for basis, suffix in (("teacher_forced", ""), ("marginal", "_marginal")):
            subsets = load_subset_line(args.depth, args.benchmark,
                                       csv=RESULTS / f"zero_shot_line_{args.benchmark}{suffix}.csv")
            print(f"mean of seven, {basis}, median over random training subsets:",
                  {int(n): round(v, 4) for n, v in subsets["mae_mean7_mid"].items()})
            title = (f"Zero-shot predictions over random training subsets "
                     f"({args.benchmark}, {basis.replace('_', '-')})")
            out = HEADLINE / basis / f"zero_shot_{args.benchmark}_{args.depth}_subsets"
            print("wrote", plot_lines({label: (subsets, None)}, title, out,
                                      note=f"{bars} (10,000: full corpus, three seeds)"))
            print("wrote", plot_lines({label: (subsets[subsets.index <= 5000], None)}, title,
                                      out.with_name(f"{out.name}_to5000"), bar_share=0.45, note=bars))
    if args.benchmark == "plutonic":
        style, _ = setup_style()
        colors = dict(zip(PRETRAINING_LABELS, (style.CIVIDIS[2], style.CIVIDIS[6], style.CIVIDIS[9])))
        lines = {}
        for group, name in PRETRAINING_LABELS.items():
            g = load_line(args.depth, "plutonic", group)
            lines[f"{name} ({int(g['n_rocks'].iloc[0])} rocks)"] = (g, colors[group])
        print("wrote", plot_lines(lines, f"Zero-shot by pretraining membership (plutonic, {args.depth} steps)",
                                  RESULTS / "gradient_steps" / f"{stem}_{args.depth}_pretraining"))


if __name__ == "__main__":
    main()
