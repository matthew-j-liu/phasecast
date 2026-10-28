"""How far each scaling-line head's zero-shot prediction sits from the
simulator's own prediction on the same volcanic val rocks, against corpus
size and fitting depth: the other half of the depth argument, beside the
distance to the measured rocks the per-depth figures show. Training-free:
the heads are on disk and the simulator's draw mean (pooled 50 and 30% rung
rows under the broad box, as `generator_reach/results/signal.csv` builds it)
is rebuilt from the equilibrium run on the benchmark chemistry. Same basis
on both sides: the head is read as a marginal over assemblages sampled
from its own presence probabilities (`run_signal.marginal_shares`), as the
simulator's draw mean is a marginal over draws; the teacher-forced read
(every listed mineral present) is kept beside it as the distance to the
rocks the per-depth figures use.

  python explorations/corpus_scaling/fig_head_simulator_distance.py

Writes `results/gradient_steps/head_simulator_distance.csv` (one row per
cell) and `head_simulator_distance.png` / `.pdf` in the locked layout, with
colour encoding depth (224 steps as the reference line). Exits non-zero if
the rock distances do not reproduce `depth_ladder.csv` or the simulator's
own distance to the rocks does not reproduce `signal.csv`.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cnmineral.data.phase_vocab import GROUPS
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.mode_head import conditional_mean
from cnmineral.eval.real_rock import BENCHMARK_DIR, load_benchmark
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from cnmineral.training.real_rock_recalibrator import build_group_predictions
from explorations.corpus_scaling.fig_zero_shot_scaling import (
    LABELS, PANELS, RESULTS, finish_panel, panel_grid, setup_style)
from explorations.corpus_scaling.run_depth_grid import PRODUCTION, SIZES, share_metrics, with_seeds
from explorations.generator_reach import run_read, run_signal
from explorations.generator_reach.compositions import load_compositions
from explorations.prior_search.run_prior_screen import reweighted_marginals

DEPTHS = (56, 224, 896)
OUT = RESULTS / "gradient_steps" / "head_simulator_distance"
GENERATOR_REACH = REPO / "explorations" / "generator_reach"
BROAD = GENERATOR_REACH / "work" / "broad128"
SIGNAL = GENERATOR_REACH / "results" / "signal.csv"


def simulator_marginal():
    """The simulator draw mean per evaluated volcanic rock, in the
    benchmark's masked-row order, with the per-row val selector."""
    benchmark = load_benchmark(BENCHMARK_DIR)
    mask = benchmark.clean & benchmark.has_magnitude
    comps = load_compositions(GENERATOR_REACH / "results" / "compositions.csv")
    if list(comps["key"]) != list(benchmark.bulk_ids[mask]):
        raise SystemExit("results/compositions.csv does not match the benchmark's evaluated rocks")
    magnitude, numeric = benchmark.magnitude[mask], benchmark.numeric[mask]
    points = run_read.shares_and_distances(run_read.build_points(BROAD), magnitude, numeric)
    draws = run_read.load_draws(BROAD)
    rungs = run_read.rung_rows(points, draws)
    # the broad box weights every draw equally; `signal.csv`'s `simulator` row
    marg = reweighted_marginals(rungs, draws[["bulk", "draw"]].assign(w=1.0))
    sim = marg.reindex(comps["bulk"].to_numpy())[[f"mean_{g}" for g in GROUPS]].to_numpy()
    return benchmark, mask, comps["subset"].to_numpy() == "val", sim


def head_reads(ckpt, benchmark, latents, device):
    """A head's zero-shot share over each rock's measured mineral list on
    every benchmark row, on two bases: teacher-forced (every listed mineral
    present, the production read) and marginal over sampled supports (the
    basis the simulator's draw mean is on)."""
    head = ckpt.build_mode_head().to(device)
    numeric = benchmark.numeric.astype(np.float64)
    gp = build_group_predictions(head, benchmark, latents)
    forced = conditional_mean(gp.alpha7.numpy(), numeric)
    marginal = run_signal.marginal_shares(head, latents, numeric)
    return forced, marginal


def distance_table(forced, marginal, sim, magnitude, numeric):
    """Per category and the mean of seven, for one cell: the head's marginal
    read against the simulator draw mean (`d_sim`, rows the simulator
    reaches, same basis both sides), the teacher-forced read against the
    measured rocks (`d_rock`, the per-depth figures' number) and the
    marginal read against the rocks (`d_rock_marginal`)."""
    reached = numeric & np.isfinite(sim)
    d_sim, _, _ = share_metrics(marginal, sim, reached)
    d_rock, _, _ = share_metrics(forced, magnitude, numeric)
    d_rock_marginal, _, _ = share_metrics(marginal, magnitude, numeric)
    out = {}
    for name, d in (("d_sim", d_sim), ("d_rock", d_rock), ("d_rock_marginal", d_rock_marginal)):
        out.update({f"{name}_{g}": d[j] for j, g in enumerate(GROUPS)})
        out[f"{name}_mean7"] = float(np.nanmean(d))
    return out


def cells():
    for d in DEPTHS:
        for n in SIZES:
            for seed, tag in with_seeds(f"final{d}_allrock_n{n}"):
                yield dict(depth=d, n_bulks=n, seed=seed, tag=tag)


def score():
    benchmark, mask, val, sim = simulator_marginal()
    magnitude, numeric = benchmark.magnitude[mask][val], benchmark.numeric[mask][val]
    sim_val = sim[val]
    device = pick_device()
    prod = JointHeadCheckpoint.load(REPO / "checkpoints" / PRODUCTION / "heads_best.pt")
    encoder, _ = load_encoder_from_checkpoint(prod.encoder_checkpoint)
    latents = bulk_latents(encoder.to(device), benchmark, device)
    rows = []
    for cell in cells():
        ckpt = JointHeadCheckpoint.load(REPO / "checkpoints" / cell["tag"] / "heads_best.pt")
        forced, marginal = head_reads(ckpt, benchmark, latents, device)
        rows.append({**cell, **distance_table(forced[mask][val], marginal[mask][val], sim_val, magnitude, numeric)})
        r = rows[-1]
        print(f"{r['tag']:30s} marginal to simulator {100 * r['d_sim_mean7']:5.2f}  "
              f"forced to rocks {100 * r['d_rock_mean7']:5.2f}  marginal to rocks {100 * r['d_rock_marginal_mean7']:5.2f}")
    sim_to_rock, _, _ = share_metrics(sim_val, magnitude, numeric)
    return pd.DataFrame(rows), {g: sim_to_rock[j] for j, g in enumerate(GROUPS)}


def check(df, sim_to_rock):
    ladder = pd.read_csv(RESULTS / "depth_ladder.csv").set_index("tag")
    worst = max(abs(df[f"d_rock_{g}"].to_numpy() - ladder.loc[df["tag"], f"val_mae_{g}"].to_numpy()).max()
                for g in GROUPS)
    signal = pd.read_csv(SIGNAL)
    signal = signal[signal["source"] == "simulator"].set_index("category")["mae"]
    worst_sim = max(abs(sim_to_rock[g] - signal[g]) for g in GROUPS)
    print(f"rock distances against depth_ladder.csv: max abs diff {worst:.1e}; "
          f"simulator to rocks against signal.csv: {worst_sim:.1e}")
    if worst > 1e-9 or worst_sim > 1e-9:
        raise SystemExit("the distances do not reproduce the committed reads")


def draw(df, out=OUT):
    style, spec = setup_style()
    colours = {56: style.CIVIDIS[6], 224: style.CIVIDIS[2], 896: style.CIVIDIS[9]}
    widths = {56: 1.2, 224: 1.8, 896: 1.2}
    agg = df.drop(columns=["tag", "seed"]).groupby(["depth", "n_bulks"]).agg(["mean", "min", "max"])
    sizes = sorted(df["n_bulks"].unique())
    fig, axes = panel_grid(style)
    for i, (ax, c) in enumerate(zip(axes.ravel(), PANELS)):
        for d in DEPTHS:
            sub = agg.loc[d]
            m = 100 * sub[(f"d_sim_{c}", "mean")]
            ax.errorbar(sub.index, m, yerr=[m - 100 * sub[(f"d_sim_{c}", "min")], 100 * sub[(f"d_sim_{c}", "max")] - m],
                        fmt="-", marker="o", color=colours[d], lw=widths[d], ms=3.5,
                        elinewidth=0.9, capsize=1.5, zorder=3 if d == 224 else 2)
        finish_panel(ax, sizes, f"({chr(97 + i)}) {LABELS[c]}", style)
    fig.suptitle("Head against the simulator on validation rocks (volcanic)", y=0.995,
                 fontsize=spec["label"] + 2)
    fig.supxlabel("Simulated compositions in the training corpus", y=0.085, fontsize=spec["label"])
    fig.supylabel("Distance to the simulator draw mean (%)", x=0.012, fontsize=spec["label"])
    fig.legend(handles=[Line2D([], [], color=colours[d], lw=widths[d], marker="o", ms=3.5,
                               label=f"{d} gradient steps") for d in DEPTHS]
               + [Line2D([], [], color="none", label="bars span three training seeds")],
               loc="lower center", bbox_to_anchor=(0.53, 0.0), ncols=4, handlelength=2.6)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-only", action="store_true", help="redraw from the csv without rescoring")
    args = parser.parse_args()
    if args.figures_only:
        print("wrote", draw(pd.read_csv(OUT.with_suffix(".csv"))))
        return
    df, sim_to_rock = score()
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT.with_suffix(".csv"), index=False)
    check(df, sim_to_rock)
    print("simulator to rocks, per category:", {g: round(100 * v, 2) for g, v in sim_to_rock.items()})
    print("wrote", draw(df))


if __name__ == "__main__":
    main()
