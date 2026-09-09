"""Corpus size by fitting depth, zero-shot and matrix-recalibrated, from the
cached real-rock evaluations and stamped checkpoints: the two figures every
scaling experiment is reviewed on (README, "Corpus size at fixed fitting
depth").

  python explorations/corpus_scaling/run_depth_grid.py
  python explorations/corpus_scaling/run_depth_grid.py --figures-only

Run from the repo root after `run_variant.py` has trained the cells. Rows:
the depth grid (every head 8 batches of 1,024 per epoch, stopped at a fixed
epoch count, ModeHead selected on corpus val within it), `final*` cells
(last-epoch state, true fixed depth), the corpus-val ladder (#69), and the
stop-on-real-rock rows (ModeHead selected on the train split's hurdle NLL or
share error), all with seed repeats where present. The traced-regen corpus
at 10,000 compositions is a second row. A cell is read when its checkpoint
and zero-shot run both exist, so a new corpus joins the figure by training
its cells under the tag grammar in the README. The recalibrator is fit on
the real-rock train split and read on val; test stays sealed. Writes
`results/depth_ladder.{csv,json,png}` and `depth_ladder_groups.png`;
`--figures-only` redraws from the saved csv and json.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cnmineral.data.phase_vocab import GROUPS
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.real_rock import load_benchmark, split_mask
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint, selection_series
from cnmineral.training.real_rock_recalibrator import (
    build_group_predictions, fit_recalibrator, paired_bootstrap, score_recalibrator,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
SIZES = (78, 312, 1250, 5000, 10000)
DEPTHS = (24, 56, 112, 224, 448)
PRODUCTION = "joint_heads_allrock_tuned_seed0"
LADDER = {n: f"corpus_scaling_allrock_n{n}" for n in (78, 156, 312, 625, 1250, 2500, 5000)}
LADDER[10000] = PRODUCTION
STOP_RULES = {"stop nll": ("stopnll", "real_rock_train_joint"),
              "stop mae": ("stopmae", "real_rock_train_share_mae")}
BATCH, ROWS_PER_BULK = 1024, 96
SHOW_DEPTHS = (56, 224, 448)
SHOW_RULE = ("stop mae", "stopped on real-rock train")
METRICS = ("mae_mean", "headline_nll", "auroc_mean", "recal_val_joint", "recal_share_mae",
           *(f"mae_{g}" for g in GROUPS))


def batches_per_epoch(n):
    present = {78: 78, 10000: 9933}.get(n, round(n * 0.9935))
    return math.ceil(present * ROWS_PER_BULK / BATCH)


def exists(tag):
    return ((REPO / "checkpoints" / tag / "heads_best.pt").exists()
            and (REPO / "runs" / "real_rock" / tag / "metrics.json").exists())


def with_seeds(tag):
    return [(s, t) for s, t in ((0, tag), (1, f"{tag}_seed1"), (2, f"{tag}_seed2")) if exists(t)]


def cells():
    for n in SIZES:
        for d in DEPTHS:
            base = f"steps{d}_allrock_n10000" if n == 10000 else f"depth{d}_allrock_n{n}"
            yield from (dict(series="grid", n_bulks=n, depth=d, seed=s, tag=t, rule="val")
                        for s, t in with_seeds(base))
            yield from (dict(series="final", n_bulks=n, depth=d, seed=s, tag=t, rule="final")
                        for s, t in with_seeds(f"final{d}_allrock_n{n}"))
        for name, (prefix, series) in STOP_RULES.items():
            yield from (dict(series=name, n_bulks=n, depth=name, seed=s, tag=t, rule=series)
                        for s, t in with_seeds(f"{prefix}_allrock_n{n}"))
    for d in DEPTHS:
        yield from (dict(series="traced", n_bulks=10000, depth=d, seed=s, tag=t, rule="val")
                    for s, t in with_seeds(f"depth{d}_traced_n10000"))
    for n, tag in LADDER.items():
        yield from (dict(series="corpus val", n_bulks=n, depth="corpus val", seed=s, tag=t, rule="val")
                    for s, t in with_seeds(tag))


def selected_steps(cell, monitor):
    n_epochs = len(monitor["train"])
    if cell["rule"] == "final":
        epoch = n_epochs
    elif cell["rule"] == "val":
        epoch = int(np.argmin(selection_series(monitor)["mode"])) + 1
    else:
        epoch = int(np.argmin(monitor[cell["rule"]])) + 1
    per_epoch = batches_per_epoch(cell["n_bulks"]) if cell["series"] == "corpus val" else 8
    return per_epoch * epoch


def zero_shot(tag):
    m = json.loads((REPO / "runs" / "real_rock" / tag / "metrics.json").read_text())
    g = {x["group"]: x for x in m["per_group"]}
    out = {f"mae_{k}": g[k]["magnitude_mae"] for k in GROUPS}
    out["mae_mean"] = float(np.mean([g[k]["magnitude_mae"] for k in GROUPS]))
    out["auroc_mean"] = float(np.mean([g[k]["auroc"] for k in GROUPS]))
    out["headline_nll"] = m["headline"]["joint_nll"]
    return out


def score_all():
    device = pick_device()
    benchmark = load_benchmark()
    scored = benchmark.clean & benchmark.has_magnitude
    masks = {s: split_mask(benchmark, s) & scored for s in ("train", "val")}
    val_unit = benchmark.unit[masks["val"]]
    prod = JointHeadCheckpoint.load(REPO / "checkpoints" / PRODUCTION / "heads_best.pt")
    encoder, _ = load_encoder_from_checkpoint(prod.encoder_checkpoint)
    latents = bulk_latents(encoder.to(device), benchmark, device)

    per_rock, rows = {}, []
    for cell in cells():
        ckpt = JointHeadCheckpoint.load(REPO / "checkpoints" / cell["tag"] / "heads_best.pt")
        gp = build_group_predictions(ckpt.build_mode_head().to(device), benchmark, latents)
        fit = fit_recalibrator(gp.subset(masks["train"]), tier="matrix", weight_decay=1.0)
        val = gp.subset(masks["val"])
        zero, got = score_recalibrator(None, val), score_recalibrator(fit, val)
        per_rock[cell["tag"]] = got["joint"]
        rows.append(dict(**cell, steps=selected_steps(cell, ckpt.monitor), **zero_shot(cell["tag"]),
                         zero_shot_val_joint=float(zero["joint"].mean()),
                         recal_val_joint=float(got["joint"].mean()),
                         recal_share_mae=got["share_mae_mean"]))
        r = rows[-1]
        print(f"{r['tag']:32s} steps {r['steps']:5d}  zero-shot mae {r['mae_mean']:.4f} "
              f"nll {r['headline_nll']:6.2f}  recal val {r['recal_val_joint']:.3f} "
              f"share {r['recal_share_mae']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "depth_ladder.csv", index=False)
    return df, per_rock, masks, val_unit, benchmark, latents, device, prod


def aggregate(df):
    df = df.copy()
    df["depth"] = df.depth.astype(str)
    return (df.groupby(["series", "n_bulks", "depth"], sort=False)
              .agg(**{f"{m}_{s}": (m, s) for m in METRICS for s in ("mean", "min", "max")},
                   steps=("steps", "mean"), seeds=("seed", "count"), tag=("tag", "first"))
              .reset_index())


def line(agg, series, depth):
    sub = agg[(agg.series == series) & (agg.depth == str(depth))]
    if series == "grid":
        fin = agg[(agg.series == "final") & (agg.depth == str(depth))]
        sub = pd.concat([sub[~sub.n_bulks.isin(fin.n_bulks)], fin])
    return sub.sort_values("n_bulks")


def slopes_and_contrasts(df, agg, per_rock, val_unit):

    def seed_mean_per_rock(sub):
        return np.stack([np.mean([per_rock[t] for t in df[(df.series == s) & (df.n_bulks == n)
                                                          & (df.depth.astype(str) == d)].tag], axis=0)
                         for s, n, d in zip(sub.series, sub.n_bulks, sub.depth)], axis=1)

    slopes = {}
    for series, depth in [("grid", d) for d in DEPTHS] + [(s, s) for s in ("stop nll", "stop mae", "corpus val")]:
        sub = line(agg, series, depth)
        sub = sub[sub.n_bulks.isin(SIZES)] if series != "corpus val" else sub
        if len(sub) < 3:
            continue
        x = np.log2(sub.n_bulks.values.astype(float))
        b_zero = float(np.polyfit(x, sub.mae_mean_mean.values, 1)[0])
        xc = x - x.mean()
        s = seed_mean_per_rock(sub) @ xc / (xc ** 2).sum()
        boot = paired_bootstrap(s, np.zeros_like(s), val_unit)
        key = f"{series} {depth}" if series == "grid" else series
        slopes[key] = {"zero_shot_mae_per_doubling": b_zero, "recal_val_joint_per_doubling": boot,
                       "steps": dict(zip(sub.n_bulks.astype(int).tolist(), sub.steps.round().astype(int).tolist()))}
        print(f"{key:14s} zero-shot mae {b_zero:+.4f}/doubling   recal val {boot['mean']:+.3f} "
              f"[{boot['lo']:+.3f}, {boot['hi']:+.3f}]/doubling   steps {slopes[key]['steps']}")
    contrasts = {t: paired_bootstrap(per_rock[PRODUCTION], per_rock[t], val_unit)
                 for t in per_rock if t != PRODUCTION}
    return slopes, contrasts


def constant_forecast(benchmark, train_mask):
    """Absolute error per category of the train-split mean measured share, on
    every quantified evaluated rock: the zero-shot panels' reference."""
    scored = benchmark.clean & benchmark.has_magnitude
    out = {}
    for j, group in enumerate(GROUPS):
        const = benchmark.magnitude[train_mask & benchmark.numeric[:, j], j].mean()
        quantified = scored & benchmark.numeric[:, j]
        out[group] = float(np.abs(benchmark.magnitude[quantified, j] - const).mean())
    return out


def references(masks, benchmark, latents, device, prod):
    from cnmineral.training.measured_mode_head import ablate_latent, fit_measured_head, score_measured_head
    gp = build_group_predictions(prod.build_mode_head().to(device), benchmark, latents)
    nochem = fit_measured_head(ablate_latent(gp.subset(masks["train"])), weight_decay=10.0)
    nochem_val = score_measured_head(nochem, ablate_latent(gp.subset(masks["val"])))
    prod_metrics = json.loads((REPO / "runs" / "real_rock" / PRODUCTION / "metrics.json").read_text())
    constant = constant_forecast(benchmark, masks["train"])
    reference = {"mae_mean": (float(np.mean([constant[g] for g in GROUPS])), "constant forecast"),
                 "headline_nll": (prod_metrics["climatology"]["joint_nll"], "constant forecast"),
                 "recal_val_joint": (float(nochem_val["joint"].mean()), "head given no chemistry"),
                 "recal_share_mae": (nochem_val["share_mae_mean"], "head given no chemistry")}
    reference.update({f"mae_{g}": (constant[g], "constant forecast") for g in GROUPS})
    return reference


def draw(agg, reference):
    """The scaling figure (four rulers) and its per-category companion."""
    cmap = plt.get_cmap("viridis")
    colors = {d: cmap(i / (len(SHOW_DEPTHS) - 1) * 0.85) for i, d in enumerate(SHOW_DEPTHS)}
    prod_row = agg[agg.tag == PRODUCTION].iloc[0]

    def panel(ax, col):
        for d in SHOW_DEPTHS:
            sub = line(agg, "grid", d)
            capped = (sub.series == "grid") & (sub.steps < 0.8 * d)
            ax.plot(sub.n_bulks, sub[f"{col}_mean"], "-", color=colors[d], lw=1)
            for mask, mfc in ((~capped, colors[d]), (capped, "none")):
                s2 = sub[mask]
                ax.errorbar(s2.n_bulks, s2[f"{col}_mean"],
                            yerr=[s2[f"{col}_mean"] - s2[f"{col}_min"], s2[f"{col}_max"] - s2[f"{col}_mean"]],
                            fmt="o", color=colors[d], mfc=mfc, capsize=2, lw=0.8,
                            label=f"{d} steps" if mfc != "none" else None)
        name, label = SHOW_RULE
        sub = line(agg, name, name)
        if not sub.empty:
            ax.errorbar(sub.n_bulks, sub[f"{col}_mean"],
                        yerr=[sub[f"{col}_mean"] - sub[f"{col}_min"], sub[f"{col}_max"] - sub[f"{col}_mean"]],
                        color="black", ls="--", marker="^", capsize=2, lw=1.2, label=label)
        ax.plot([10000], [prod_row[f"{col}_mean"]], "*", color="red", ms=11, label="production")
        value, ref_label = reference[col]
        ax.axhline(value, ls="--", color="#d4542a", lw=1, label=ref_label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(SIZES, [f"{n:,}" for n in SIZES], fontsize=8)
        ax.minorticks_off()
        ax.set_xlim(55, 14000)  # errorbar on a log axis wrecks the autoscale window
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(alpha=0.3)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
    for ax, (col, label) in zip(axes, [("mae_mean", "zero-shot absolute error, mean of seven"),
                                       ("headline_nll", "zero-shot headline NLL (640)"),
                                       ("recal_val_joint", "matrix-recalibrated val joint NLL"),
                                       ("recal_share_mae", "recalibrated val share error")]):
        panel(ax, col)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("simulated compositions", fontsize=10)
        ax.legend(fontsize=6)
    fig.suptitle("corpus size at fixed fitting depth (colour) and stopped on the real-rock train split "
                 "(black); bars = seed range; dashed = the no-information reference for that ruler", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "depth_ladder.png", dpi=150)

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    for ax, g in zip(axes.ravel(), (*GROUPS, "mean")):
        panel(ax, f"mae_{g}" if g != "mean" else "mae_mean")
        ax.set_title(g if g != "mean" else "mean of seven", fontsize=9)
    axes[1, 3].legend(fontsize=6, loc="upper right")
    fig.supxlabel("simulated compositions in the training corpus", fontsize=10)
    fig.supylabel("zero-shot absolute error (proportion)", fontsize=10, x=0.01)
    fig.suptitle("per mineral category, corpus size at fixed fitting depth (colour) and stopped on the "
                 "real-rock train split (black); bars = seed range; dashed = constant forecast", fontsize=9)
    fig.tight_layout(rect=(0.02, 0, 1, 1))
    fig.savefig(RESULTS / "depth_ladder_groups.png", dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-only", action="store_true",
                        help="redraw from results/depth_ladder.{csv,json} without recomputing")
    args = parser.parse_args()
    if args.figures_only:
        df = pd.read_csv(RESULTS / "depth_ladder.csv")
        saved = json.loads((RESULTS / "depth_ladder.json").read_text())
        reference = {k: tuple(v) for k, v in saved["references"].items()}
    else:
        df, per_rock, masks, val_unit, benchmark, latents, device, prod = score_all()
        slopes, contrasts = slopes_and_contrasts(df, aggregate(df), per_rock, val_unit)
        reference = references(masks, benchmark, latents, device, prod)
        print("references:", {k: round(v[0], 4) for k, v in reference.items()})
        (RESULTS / "depth_ladder.json").write_text(json.dumps(
            {"slopes": slopes, "production_minus_tag_recal_val_joint": contrasts,
             "references": {k: [float(v[0]), v[1]] for k, v in reference.items()}}, indent=1))
    draw(aggregate(df), reference)
    print(f"wrote {RESULTS / 'depth_ladder.csv'}, .json, .png and depth_ladder_groups.png")


if __name__ == "__main__":
    main()
