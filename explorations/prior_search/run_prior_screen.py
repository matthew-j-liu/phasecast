"""Stage 0 of the generator-prior search (#79): rank candidate (P, H2O)
priors by importance-reweighting the #78 broad-box draws on the benchmark
rocks. No MAGEMin, no model.

  python explorations/prior_search/run_prior_screen.py
  python explorations/prior_search/run_prior_screen.py --figures-only

Reads the #78 run under `--work` (128 broad-box draws x 31 steps per
evaluated rock, `q` on every draw), coarsens every point onto the seven
categories as `generator_reach.run_read` does, and for each scheme weights
each draw by its density over the broad box's. The reweighted mean share
over the pooled rung-50 and rung-30 rows is the marginal a corpus drawn
under that scheme would present for that rock. Per subset, the per-category
absolute error from the measured share averages over the rocks quantifying
the category; the four-silicate mean on the train split ranks the schemes,
the val split reads, test is untouched. Writes `schemes.csv`, `screen.json`
and two figures under `--results`, and the five nested rosters of the
scaling ladder under `rosters/` when missing.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cnmineral.data.condition_proposal import bulk_proposal
from cnmineral.data.phase_vocab import GROUPS
from cnmineral.eval.real_rock import BENCHMARK_DIR, load_benchmark
from explorations.corpus_scaling.run_ladder import hash_order
from explorations.generator_reach import run_read
from explorations.generator_reach.compositions import load_compositions
from explorations.prior_search.schemes import (
    CONTROLS, GRID, P_CENTERS, W_CENTERS, WIDTHS, area_fraction, density)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ROSTERS = HERE / "rosters"
REACH = REPO / "explorations" / "generator_reach"
WORK = REACH / "work" / "broad128"
TRAIN_BULKS = REPO / "data" / "corpus" / "train_bulks.csv"

LADDER = (128, 256, 512, 1024, 2048)
SUBSETS = ("train", "val")
SILICATES = ("fsp", "px", "amp", "bi")
OTHERS = ("ol", "opq", "q")
TRACES = ("th_ppm", "nb_ppm", "yb_ppm", "v_ppm", "y_ppm", "tio2")
THIN = 10
NAVY, ORANGE = "#1f4e79", "#d4542a"


def write_rosters(out=ROSTERS, sample_ids=None):
    """The ladder's nested rosters as prefixes of the md5 hash order the
    corpus_scaling ladder uses, written sorted by id as its rosters are. An
    existing file must match: a moved roster invalidates every cell trained
    on it."""
    if sample_ids is None:
        sample_ids = pd.read_csv(TRAIN_BULKS)["sample_id"].tolist()
    ordered = hash_order(sample_ids)
    out.mkdir(parents=True, exist_ok=True)
    for n in LADDER:
        ids = sorted(ordered[:n])
        path = out / f"n{n}.csv"
        if not path.exists():
            pd.DataFrame({"sample_id": ids}).to_csv(path, index=False)
        elif pd.read_csv(path)["sample_id"].tolist() != ids:
            raise SystemExit(f"{path} is not the hash-order prefix; a moved roster "
                             "invalidates every cell trained on it")


def load_samples(benchmark_dir=BENCHMARK_DIR):
    samples = pd.read_csv(Path(benchmark_dir) / "samples.csv", dtype={"sample_id": str})
    samples["key"] = samples["source"] + "|" + samples["sample_id"]
    return samples


def scheme_weights(draws, comps, samples, schemes=GRID):
    """One column per scheme: each draw's scheme density over the broad-box
    density it was drawn from. `broad` is 1 everywhere; `traced` evaluates
    the production trace-conditional proposal on the rock's TAS class and
    its reported traces, the class tier where they are missing."""
    p, w, q = (draws[c].to_numpy(dtype=float) for c in ("P_kbar", "H2O", "q"))
    bulk = draws["bulk"].to_numpy()
    out = draws[["bulk", "draw"]].copy()
    out["broad"] = 1.0
    traced = np.full(len(draws), np.nan)
    rows = samples.set_index("key").reindex(comps["key"])
    for b, cls, (_, rec) in zip(comps["bulk"], comps["rock_class"], rows.iterrows()):
        m = bulk == b
        if m.any():
            traces = {k: rec.get(k) for k in TRACES}
            traced[m] = bulk_proposal(cls, traces).q(p[m], w[m]) / q[m]
    out["traced"] = traced
    for s in schemes:
        out[s.name] = density(s, p, w) / q
    return out


def reweighted_marginals(rungs, weights):
    """Per rock: the weighted mean share per category over the pooled rung
    rows (both rung rows of a draw carry its weight), the weighted
    within-rock spread of the share, and the effective draw count
    `(sum w)^2 / sum w^2` over the draws that reached a rung."""
    rw = rungs.merge(weights[["bulk", "draw", "w"]], on=["bulk", "draw"])
    cols = [f"s_{g}" for g in GROUPS]
    by = rw["bulk"]
    denom = rw["w"].groupby(by).sum()
    mean = rw[cols].mul(rw["w"], axis=0).groupby(by).sum().div(denom, axis=0).where(denom > 0)
    dev = (rw[cols].to_numpy() - mean.reindex(by).to_numpy()) ** 2
    spread = (pd.DataFrame(dev, columns=cols).mul(rw["w"].to_numpy(), axis=0)
              .groupby(by.to_numpy()).sum().div(denom, axis=0).where(denom > 0) ** 0.5)
    reached = weights.merge(rungs[["bulk", "draw"]].drop_duplicates(), on=["bulk", "draw"])
    sums = reached.groupby("bulk")["w"].agg(total="sum", square=lambda v: (v ** 2).sum())
    n_eff = (sums["total"] ** 2 / sums["square"].where(sums["square"] > 0)).fillna(0.0)
    out = pd.concat([mean.rename(columns=lambda c: "mean_" + c[2:]),
                     spread.rename(columns=lambda c: "spread_" + c[2:])], axis=1)
    out["n_eff"] = n_eff.reindex(out.index).fillna(0.0)
    return out


def _mean(values):
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def score_scheme(marg, comps, magnitude, numeric):
    """Per subset: the per-category absolute error averaged over the rocks
    quantifying it (the head's share-error definition), its four-silicate
    and other-category means, the silicate spread, and the rocks no draw
    reached; then the effective-draw facts over all rocks."""
    bulks = marg.index.to_numpy()
    num = numeric[bulks]
    mean = marg[[f"mean_{g}" for g in GROUPS]].to_numpy()
    err = np.where(num, np.abs(mean - magnitude[bulks]), np.nan)
    spread = np.where(num, marg[[f"spread_{g}" for g in GROUPS]].to_numpy(), np.nan)
    subset = comps.set_index("bulk").loc[bulks, "subset"].to_numpy()
    sil = [GROUPS.index(g) for g in SILICATES]
    out = {}
    for s in SUBSETS:
        m = subset == s
        per = {g: _mean(err[m, i]) for i, g in enumerate(GROUPS)}
        out.update({f"{g}_{s}": per[g] for g in GROUPS})
        out[f"silicate_{s}"] = _mean([per[g] for g in SILICATES])
        out[f"other_{s}"] = _mean([per[g] for g in OTHERS])
        out[f"mean7_{s}"] = _mean(list(per.values()))
        out[f"spread_silicate_{s}"] = _mean([_mean(row[sil]) for row in spread[m]])
        out[f"n_rocks_{s}"] = int(m.sum())
        out[f"n_empty_{s}"] = int(np.isnan(mean[m]).all(axis=1).sum())
    n_eff = marg["n_eff"].to_numpy()
    out["n_eff_median"] = float(np.median(n_eff))
    out["n_eff_q10"] = float(np.quantile(n_eff, 0.1))
    out["frac_thin"] = float(np.mean(n_eff < THIN))
    return out


def screen(points, draws, comps, magnitude, numeric, samples, schemes=GRID):
    """One row per control and scheme: its rectangle, area fraction and
    scores. Rows are in ranking order, controls first."""
    rungs = run_read.rung_rows(points, draws)
    weights = scheme_weights(draws, comps, samples, schemes)
    rows = []
    for name in (*CONTROLS, *(s.name for s in schemes)):
        marg = reweighted_marginals(rungs, weights[["bulk", "draw", name]].rename(columns={name: "w"}))
        scheme = next((s for s in schemes if s.name == name), None)
        row = dict(name=name, kind="control" if scheme is None else "grid")
        if scheme is not None:
            row.update(p_center=scheme.p_center, w_center=scheme.w_center,
                       p_half=scheme.p_half, w_half=scheme.w_half,
                       p_lo=scheme.p_lo, p_hi=scheme.p_hi, w_lo=scheme.w_lo, w_hi=scheme.w_hi,
                       area_frac=area_fraction(scheme))
        row.update(score_scheme(marg, comps, magnitude, numeric))
        rows.append(row)
    table = pd.DataFrame(rows)
    grid = table[table["kind"] == "grid"].sort_values("silicate_train", kind="stable")
    return pd.concat([table[table["kind"] == "control"], grid], ignore_index=True)


def reproduction_check(table, distances):
    """The broad control must reproduce #78's `d_marginal` means on the same
    rocks and subsets; returns the largest absolute difference."""
    broad = table.set_index("name").loc["broad"]
    ref = distances.groupby(["subset", "category"])["d_marginal"].mean()
    diffs = [abs(broad[f"{g}_{s}"] - ref.get((s, g), np.nan))
             for s in SUBSETS for g in GROUPS if (s, g) in ref.index]
    return float(np.nanmax(diffs)) if diffs else np.nan


def summarize(table, facts, reproduction=None, top=3):
    grid = table[table["kind"] == "grid"]
    keep = ["silicate_train", "silicate_val", "other_val", "mean7_val", "n_eff_median", "frac_thin"]
    return {
        "run": facts,
        "settings": {"ladder": list(LADDER), "widths": WIDTHS, "p_centers": list(P_CENTERS),
                     "w_centers": list(W_CENTERS), "score": "mean absolute share error over "
                     "fsp, px, amp, bi; per category averaged over the rocks quantifying it",
                     "selects_on": "train", "reads_on": "val", "thin_floor": THIN},
        "reproduction_max_abs_diff": reproduction,
        "controls": {n: {k: table.set_index("name").loc[n, k] for k in keep}
                     for n in CONTROLS if n in set(table["name"])},
        "ranking": [{"name": r["name"], **{k: r[k] for k in keep}} for _, r in grid.iterrows()],
        "top": grid["name"].head(top).tolist(),
    }


def print_ranking(table):
    ctrl = table[table["kind"] == "control"].set_index("name")
    print("scheme                silicate train / val   other val   n_eff med   thin")
    for name in ctrl.index:
        r = ctrl.loc[name]
        print(f"{name:20s}  {r['silicate_train']:.4f} / {r['silicate_val']:.4f}   "
              f"{r['other_val']:.4f}   {r['n_eff_median']:7.1f}   {r['frac_thin']:.2f}")
    for _, r in table[table["kind"] == "grid"].iterrows():
        print(f"{r['name']:20s}  {r['silicate_train']:.4f} / {r['silicate_val']:.4f}   "
              f"{r['other_val']:.4f}   {r['n_eff_median']:7.1f}   {r['frac_thin']:.2f}")


def draw_figures(table, out):
    out.mkdir(parents=True, exist_ok=True)
    grid = table[table["kind"] == "grid"].sort_values("silicate_train", kind="stable")
    ctrl = table[table["kind"] == "control"].set_index("name")
    x = np.arange(len(grid))

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, key, title in zip(axes, ("silicate", "other"),
                              ("four silicates (fsp, px, amp, bi)", "cost on the others (ol, opq, q)")):
        ax.plot(x, grid[f"{key}_val"], "o", color=NAVY, label="val (reads)")
        ax.plot(x, grid[f"{key}_train"], "o", mfc="none", color=NAVY, label="train (selects)")
        for name, ls in (("broad", "--"), ("traced", ":")):
            if name in ctrl.index:
                ax.axhline(ctrl.loc[name, f"{key}_val"], color=ORANGE, ls=ls, lw=1,
                           label=f"{name} prior, val")
        ax.set_ylabel("absolute share error", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    lo, hi = axes[0].get_ylim()
    for xi, n in zip(x, grid["n_eff_median"]):
        axes[0].text(xi, hi, f"{n:.0f}", ha="center", va="bottom", fontsize=7, color="#555555")
    axes[0].set_ylim(lo, hi + 0.06 * (hi - lo))
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].set_xticks(x, grid["name"], rotation=60, ha="right", fontsize=8)
    fig.suptitle("candidate priors ranked by the reweighted marginal's share error on the train "
                 "split; grey numbers = median effective draws per rock", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "prior_ranking.png", dpi=150)
    plt.close(fig)

    by_name = table.set_index("name")
    fig, axes = plt.subplots(len(WIDTHS), 2, figsize=(8, 3.3 * len(WIDTHS)))
    for r, width in enumerate(WIDTHS):
        for c, (key, title) in enumerate((("silicate_val", "four silicates"),
                                          ("other_val", "cost on the others"))):
            ax = axes[r, c]
            m = np.full((len(P_CENTERS), len(W_CENTERS)), np.nan)
            for i, p in enumerate(P_CENTERS):
                for j, w in enumerate(W_CENTERS):
                    name = f"p{p:g}_w{w:g}_{width}"
                    if name in by_name.index:
                        m[i, j] = by_name.loc[name, key]
            ax.imshow(m, origin="lower", cmap="viridis_r", aspect="auto")
            for i in range(len(P_CENTERS)):
                for j in range(len(W_CENTERS)):
                    if np.isfinite(m[i, j]):
                        ax.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center", fontsize=8,
                                color="white")
            ax.set_xticks(range(len(W_CENTERS)), [f"{w:g}" for w in W_CENTERS])
            ax.set_yticks(range(len(P_CENTERS)), [f"{p:g}" for p in P_CENTERS])
            ax.set_xlabel("H2O center (wt%)", fontsize=9)
            ax.set_ylabel("P center (kbar)", fontsize=9)
            ax.set_title(f"{title}, {width} (+-{WIDTHS[width][0]:g} kbar, +-{WIDTHS[width][1]:g} wt%)",
                         fontsize=9)
    refs = ", ".join(f"{n} {ctrl.loc[n, 'silicate_val']:.3f} / {ctrl.loc[n, 'other_val']:.3f}"
                     for n in ctrl.index)
    fig.suptitle(f"val share error by rectangle center; controls (silicates / others): {refs}",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "prior_grid.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=WORK)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    write_rosters()

    if args.figures_only:
        table = pd.read_csv(args.results / "schemes.csv")
    else:
        benchmark = load_benchmark()
        mask = benchmark.clean & benchmark.has_magnitude
        comps = load_compositions(REACH / "results" / "compositions.csv")
        if list(comps["key"]) != list(benchmark.bulk_ids[mask]):
            raise SystemExit("generator_reach/results/compositions.csv does not match the "
                             "benchmark's evaluated rocks")
        magnitude, numeric = benchmark.magnitude[mask], benchmark.numeric[mask]
        points = run_read.shares_and_distances(run_read.build_points(args.work), magnitude, numeric)
        draws = run_read.load_draws(args.work)
        table = screen(points, draws, comps, magnitude, numeric, samples=load_samples())
        distances = REACH / "results" / "distances.csv"
        repro = reproduction_check(table, pd.read_csv(distances)) if distances.exists() else None
        summary = summarize(table, run_read.run_facts(args.work), repro)
        table.to_csv(args.results / "schemes.csv", index=False)
        (args.results / "screen.json").write_text(json.dumps(summary, indent=1))
        print_ranking(table)
        print(f"broad control vs #78 d_marginal, max abs diff: {repro}")
        print(f"top {len(summary['top'])} by train silicate error: {summary['top']}")

    draw_figures(table, args.results)
    print(f"wrote {args.results / 'schemes.csv'}, screen.json, prior_ranking.png, prior_grid.png")


if __name__ == "__main__":
    main()
