"""Read the generator run on the benchmark rocks: per rock and mineral
category, how close the draw marginal, the best draw and the best temperature
step come to the measured share, and the per-category bins that follow.

  python explorations/generator_reach/run_read.py
  python explorations/generator_reach/run_read.py --work explorations/generator_reach/work/smoke \\
      --results explorations/generator_reach/work/smoke/read
  python explorations/generator_reach/run_read.py --figures-only

Reads the manifested chunks under `--work` (a partial run gives a partial
read; the rock count is printed) and coarsens every converged ladder point
onto the seven benchmark categories: melt and fluid dropped, the crystals
renormalized over the categories the rock quantifies, as the head's share is
scored. Per (rock, category) three absolute errors against the measured
share: the mean share over the pooled rung-50 and rung-30 rows of all draws
(the marginal the corpus presents), the best single (draw, rung) row, and the
best step anywhere along that draw's ladder, with no melt-fraction rule.
`--tau` (default 0.05) sorts each (rock, category) into reproduced / buried /
wrong_stage / wrong_recipe and `--tau-grid` repeats the fractions at other
tolerances. Writes `distances.csv`, `summary.json` and three figures under
`--results`; `--figures-only` redraws from them.
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
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cnmineral.data.phase_vocab import GROUPS, MODEL_TO_GROUP, NON_SOLID_PHASES, PHASE_MERGES
from cnmineral.eval.real_rock import load_benchmark
from explorations.generator_reach.compositions import load_compositions

RESULTS = Path(__file__).resolve().parent / "results"
WORK = Path(__file__).resolve().parent / "work"
REFERENCE = REPO / "explorations" / "corpus_scaling" / "results" / "depth_ladder.csv"
PRODUCTION = "joint_heads_allrock_tuned_seed0"

RUNGS = (50, 30)
BINS = ("reproduced", "buried", "wrong_stage", "wrong_recipe")
DISTANCES = ("d_marginal", "d_best_draw", "d_best_step")
# MAGEMin's raw phase names onto the seven benchmark categories: the lumped
# fsp needs no plag/afs split at this level, the silica polymorphs fold into q
# as in the label vocabulary, and every other solid is `other`
RAW_TO_GROUP = {"fsp": "fsp",
                **{p: g for p, g in MODEL_TO_GROUP.items() if p not in ("plag", "afs")},
                **{m: MODEL_TO_GROUP[t] for m, t in PHASE_MERGES.items()}}
MASS_COLS = [f"m_{g}" for g in GROUPS] + ["m_other"]
KEYS = ["bulk", "draw", "T_celsius"]

LABELS = {"d_marginal": "draw mean at rungs 50 + 30",
          "d_best_draw": "best draw at the rungs",
          "d_best_step": "best step of that draw"}
COLORS = {"d_marginal": "#1f4e79", "d_best_draw": "#d4542a", "d_best_step": "#3a8f3a"}
BIN_COLORS = {"reproduced": "#3a8f3a", "buried": "#1f4e79",
              "wrong_stage": "#e0a030", "wrong_recipe": "#8a8a8a"}


def read_manifest(work):
    path = work / "manifest.jsonl"
    if not path.exists():
        return {}
    return {rec["chunk"]: rec for rec in map(json.loads, open(path))}


def chunk_files(work, table):
    done = read_manifest(work)
    return [p for p in sorted(work.glob(f"chunk_*_{table}.parquet"))
            if int(p.name.split("_")[1]) in done]


def point_masses(ladder):
    """One row per converged (bulk, draw, T_celsius): melt, fluid and the
    crystal mass in each category; `other` is every solid outside the seven."""
    solids = ladder[~ladder["phase"].isin(NON_SOLID_PHASES)]
    col = solids["phase"].map(lambda p: f"m_{RAW_TO_GROUP.get(p, 'other')}")
    masses = (solids.assign(col=col)
              .pivot_table(index=KEYS, columns="col", values="mode_wt", aggfunc="sum", fill_value=0.0)
              .reindex(columns=MASS_COLS, fill_value=0.0))
    out = ladder.groupby(KEYS).agg(liq_frac=("liq_frac", "first"))
    out["fl_frac"] = ladder[ladder["phase"] == "fl"].groupby(KEYS)["mode_wt"].sum()
    out = out.join(masses).fillna(0.0).reset_index()
    return out


def build_points(work, rebuild=False):
    """Every converged point of every manifested chunk, cached as
    `points.parquet` beside the chunks."""
    cache = work / "points.parquet"
    ladders = chunk_files(work, "ladder")
    if not ladders:
        raise SystemExit(f"no manifested chunks under {work}")
    if cache.exists() and not rebuild and \
            cache.stat().st_mtime >= max(p.stat().st_mtime for p in ladders):
        return pd.read_parquet(cache)
    parts, seen = [], set()
    for p in ladders:
        ladder = pd.read_parquet(p)
        seen |= set(ladder["phase"].unique())
        parts.append(point_masses(ladder))
    other = sorted(s for s in seen if s not in RAW_TO_GROUP and s not in NON_SOLID_PHASES)
    print(f"{len(ladders)} chunks; solids read as other: {other}")
    points = pd.concat(parts, ignore_index=True)
    points.to_parquet(cache, index=False)
    return points


def load_draws(work):
    return pd.concat([pd.read_parquet(p) for p in chunk_files(work, "draws")], ignore_index=True)


def shares_and_distances(points, magnitude, numeric):
    """Per point, the crystal share renormalized over the categories the rock
    quantifies and its absolute error from the measured share; a point with
    no crystal mass in those categories has share 0 everywhere, as the head's
    conditional mean does on an empty mask."""
    b = points["bulk"].to_numpy()
    mass = points[[f"m_{g}" for g in GROUPS]].to_numpy()
    num = numeric[b].astype(float)
    masked = mass * num
    denom = masked.sum(1, keepdims=True)
    share = np.divide(masked, denom, out=np.zeros_like(masked), where=denom > 0)
    dist = np.where(num > 0, np.abs(share - magnitude[b]), np.nan)
    out = points.copy()
    for i, g in enumerate(GROUPS):
        out[f"s_{g}"] = share[:, i]
        out[f"d_{g}"] = dist[:, i]
    out["zero_mass"] = denom[:, 0] <= 0
    return out


def rung_rows(points, draws):
    """The points read at rung 50 and rung 30 of each draw; a draw that never
    reaches a rung has no row there."""
    parts = []
    for r in RUNGS:
        ends = (draws[["bulk", "draw", f"T_end_{r}"]].dropna()
                .rename(columns={f"T_end_{r}": "T_celsius"}))
        parts.append(points.merge(ends, on=KEYS).assign(rung=r))
    return pd.concat(parts, ignore_index=True)


def reach_table(points, draws, comps, magnitude, numeric):
    """One row per (rock, quantified category): the measured share and the
    three distances, with the draw, rung and step behind each minimum."""
    rungs = rung_rows(points, draws)
    per_rock = comps.set_index("bulk")[["key", "subset", "total_crystals_pct"]]
    zero_mass = points.groupby("bulk")["zero_mass"].sum()
    frames = []
    for i, g in enumerate(GROUPS):
        s, d = f"s_{g}", f"d_{g}"
        bulks = np.intersect1d(np.flatnonzero(numeric[:, i]), points["bulk"].unique())
        frame = pd.DataFrame({"bulk": bulks, "measured": magnitude[bulks, i]})
        frame.insert(1, "category", g)

        rg = rungs[rungs["bulk"].isin(bulks)]
        by = rg.groupby("bulk")
        marg = pd.DataFrame({"mean_rungs": by[s].mean(), "d_q10_draw": by[d].quantile(0.1)})
        for r in RUNGS:
            sub = rg[rg["rung"] == r].groupby("bulk")[s]
            marg[f"mean_{r}"] = sub.mean()
            marg[f"n_draws_{r}"] = sub.size()
        best = (rg.loc[by[d].idxmin(), ["bulk", "draw", "rung", d]]
                .rename(columns={"draw": "best_draw", "rung": "best_rung", d: "d_best_draw"}))

        chosen = points.merge(best[["bulk", "best_draw"]],
                              left_on=["bulk", "draw"], right_on=["bulk", "best_draw"])
        at = chosen.loc[chosen.groupby("bulk")[d].idxmin()]
        solids = at[MASS_COLS].sum(1).to_numpy()
        step = pd.DataFrame({
            "bulk": at["bulk"].to_numpy(), "best_T": at["T_celsius"].to_numpy(),
            "best_liq": at["liq_frac"].to_numpy(),
            "best_other": np.divide(at["m_other"].to_numpy(), solids,
                                    out=np.zeros(len(at)), where=solids > 0),
            "d_best_step": at[d].to_numpy()})
        any_step = points[points["bulk"].isin(bulks)].groupby("bulk")[d].min().rename("d_best_any_step")

        frame = (frame.merge(marg, left_on="bulk", right_index=True, how="left")
                 .merge(best, on="bulk", how="left")
                 .merge(step, on="bulk", how="left")
                 .merge(any_step, left_on="bulk", right_index=True, how="left"))
        frames.append(frame)

    table = pd.concat(frames, ignore_index=True)
    table["d_marginal"] = (table["mean_rungs"] - table["measured"]).abs()
    for r in RUNGS:
        table[f"d_marginal_{r}"] = (table[f"mean_{r}"] - table["measured"]).abs()
        table[f"n_draws_{r}"] = table[f"n_draws_{r}"].fillna(0).astype(int)
    table["n_numeric"] = numeric.sum(1)[table["bulk"]]
    table["n_zero_mass"] = table["bulk"].map(zero_mass).fillna(0).astype(int)
    table = table.merge(per_rock, left_on="bulk", right_index=True, how="left")
    cols = ["key", "bulk", "subset", "category", "measured", "n_numeric", "total_crystals_pct",
            "n_draws_50", "n_draws_30", "n_zero_mass", "mean_50", "mean_30", "mean_rungs",
            "d_marginal", "d_marginal_50", "d_marginal_30", "d_q10_draw", "best_draw",
            "best_rung", "d_best_draw", "best_T", "best_liq", "best_other", "d_best_step",
            "d_best_any_step"]
    return table[cols].sort_values(["bulk", "category"]).reset_index(drop=True)


def bin_outcomes(table, tau):
    """reproduced if the marginal reaches, else buried if a draw at the rungs
    reaches, else wrong_stage if a step of that draw reaches, else
    wrong_recipe; a missing distance never reaches."""
    reached = [table[k].to_numpy() <= tau for k in DISTANCES]
    return np.select(reached, BINS[:3], default=BINS[3]).astype(object)


def bin_fractions(table, tau):
    if len(table) == 0:
        return {b: None for b in BINS}
    out = bin_outcomes(table, tau)
    return {b: round(float(np.mean(out == b)), 4) for b in BINS}


def _stat(values, fn):
    values = np.asarray(values, dtype=float)
    return None if np.all(np.isnan(values)) else round(float(fn(values)), 4)


def reference_mae(path=REFERENCE, tag=PRODUCTION):
    if not path.exists():
        return {}
    row = pd.read_csv(path).query("tag == @tag")
    return {} if row.empty else {g: round(float(row[f"mae_{g}"].iloc[0]), 4) for g in GROUPS}


def run_facts(work):
    recs = read_manifest(work)
    draws = load_draws(work)
    return dict(work=str(work), n_chunks=len(recs), n_rocks=int(draws["bulk"].nunique()),
                n_draws_per_rock=int(recs[min(recs)]["n_draws_per_bulk"]),
                n_points_run=int(sum(r["n_points_run"] for r in recs.values())),
                n_ladder_rows=int(sum(r["n_ladder_rows"] for r in recs.values())),
                mean_failed_per_draw=round(float(draws["n_failed"].mean()), 3),
                attrition_50=round(float(draws["attrition_50"].mean()), 4),
                attrition_30=round(float(draws["attrition_30"].mean()), 4),
                oversaturated=round(float(draws["oversaturated"].mean()), 4),
                solver_crashes=int(sum(len(r["solver_crashes"]) for r in recs.values())),
                wall_h=round(sum(r["wall_s"] for r in recs.values()) / 3600, 2))


def summarize(table, tau, taus, reference, run=None):
    per = {}
    for g in GROUPS:
        sub = table[table["category"] == g]
        multi = sub[sub["n_numeric"] >= 2]
        ok = sub.dropna(subset=["best_liq", "total_crystals_pct"])
        rho = (round(float(spearmanr(ok["best_liq"], 1 - ok["total_crystals_pct"] / 100).correlation), 4)
               if len(ok) > 2 else None)
        per[g] = {
            "n_rocks": int(len(sub)), "n_rocks_multi": int(len(multi)),
            "bins": {str(t): bin_fractions(sub, t) for t in taus},
            "bins_multi": {str(t): bin_fractions(multi, t) for t in taus},
            "quantiles": {k: [_stat(sub[k], lambda v, q=q: np.nanquantile(v, q))
                              for q in (0.1, 0.25, 0.5, 0.75, 0.9)] for k in DISTANCES},
            "mean": {k: _stat(sub[k], np.nanmean) for k in (*DISTANCES, "d_best_any_step")},
            "reference_mae": reference.get(g),
            "melt_spearman": rho,
            "n_no_rung": int(sub["d_marginal"].isna().sum()),
        }
    mean_bins = {str(t): {b: _stat([per[g]["bins"][str(t)][b] for g in GROUPS], np.nanmean)
                          for b in BINS} for t in taus}
    return {"run": run,
            "settings": {"tau": tau, "tau_grid": list(taus), "best_draw": "per category"},
            "reference": {"path": str(REFERENCE), "tag": PRODUCTION, "mae": reference},
            "per_category": per,
            "mean_of_categories": {
                "bins": mean_bins,
                "mean": {k: _stat([per[g]["mean"][k] for g in GROUPS], np.nanmean)
                         for k in DISTANCES}}}


def print_summary(summary, tau):
    run = summary["run"] or {}
    print(f"{run.get('n_rocks', '?')} rocks read; bins at tau {tau} as fractions of rocks, "
          "then the median of each distance")
    for g in GROUPS:
        p = summary["per_category"][g]
        b, q = p["bins"][str(tau)], p["quantiles"]
        if p["n_rocks"] == 0:
            print(f"{g}: no rocks")
            continue
        med = " / ".join("na" if q[k][2] is None else f"{q[k][2]:.3f}" for k in DISTANCES)
        print(f"{g}: n {p['n_rocks']}, reproduced {b['reproduced']:.2f}, buried {b['buried']:.2f}, "
              f"wrong stage {b['wrong_stage']:.2f}, wrong recipe {b['wrong_recipe']:.2f}; "
              f"median marginal / best draw / best step {med}")


def _panels(table):
    per_rock = table.groupby("bulk")[[*DISTANCES, "best_liq", "total_crystals_pct"]].mean()
    return [(g, table[table["category"] == g]) for g in GROUPS] + [("mean over categories", per_rock)]


def draw_figures(table, summary, out):
    out.mkdir(parents=True, exist_ok=True)
    tau = summary["settings"]["tau"]
    ref = summary["reference"]["mae"]
    taus = summary["settings"]["tau_grid"]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4), sharex=True, sharey=True)
    for ax, (name, sub) in zip(axes.flat, _panels(table)):
        n = len(sub)
        for k in DISTANCES:
            x = np.sort(sub[k].dropna().to_numpy())
            ax.step(np.r_[0, x], np.r_[0, np.arange(1, len(x) + 1)] / max(n, 1), where="post",
                    color=COLORS[k], lw=1.2, label=LABELS[k])
        ax.axvline(tau, color="black", ls=":", lw=0.9, label=f"tau {tau:g}")
        if name in ref:
            ax.axvline(ref[name], color="#8a8a8a", ls="--", lw=0.9, label="production zero-shot error")
        ax.set_title(f"{name} (n={n})", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("absolute share error", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of rocks", fontsize=10)
    axes.flat[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("how close equilibrium on the rock's own chemistry gets to its measured share",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "reach_cdf.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    width = 0.8 / len(taus)
    for ax, key, title in zip(axes, ("bins", "bins_multi"),
                              ("all evaluated rocks", "rocks quantifying at least two categories")):
        for j, t in enumerate(taus):
            x = np.arange(len(GROUPS)) + (j - (len(taus) - 1) / 2) * width
            bottom = np.zeros(len(GROUPS))
            for b in BINS:
                h = np.array([summary["per_category"][g][key][str(t)][b] or 0.0 for g in GROUPS],
                             dtype=float)
                ax.bar(x, h, width, bottom=bottom, color=BIN_COLORS[b], edgecolor="white",
                       lw=0.5, label=b if j == 0 else None)
                bottom += h
            for xi in x:
                ax.text(xi, 1.01, f"{t:g}", ha="center", va="bottom", fontsize=6.5, color="#555555")
        ax.set_xticks(np.arange(len(GROUPS)))
        ax.set_xticklabels(GROUPS)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("fraction of rocks", fontsize=10)
        ax.set_title(f"{title} (bars: tau {', '.join(f'{t:g}' for t in taus)})", fontsize=10)
    axes[0].legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    fig.tight_layout()
    fig.savefig(out / "reach_bins.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4), sharex=True, sharey=True)
    for ax, (name, sub) in zip(axes.flat, _panels(table)):
        x = 1 - sub["total_crystals_pct"].to_numpy(dtype=float) / 100
        y = sub["best_liq"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=8, alpha=0.5, color="#1f4e79", lw=0)
        ax.plot([0, 1], [0, 1], color="#8a8a8a", ls="--", lw=0.8)
        rho = spearmanr(x[ok], y[ok]).correlation if ok.sum() > 2 else np.nan
        ax.set_title(f"{name} (spearman {rho:.2f}, n={int(ok.sum())})", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("1 - crystallinity (measured, volume)", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("melt fraction at the best step (mass)", fontsize=10)
    fig.suptitle("diagnostic: where along the path the best step sits, against the rock's own "
                 "crystallinity", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "reach_melt.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=WORK / "broad128")
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--tau-grid", default="0.02,0.05,0.10")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--rebuild-points", action="store_true")
    args = ap.parse_args()
    taus = sorted({float(t) for t in args.tau_grid.split(",")} | {args.tau})
    args.results.mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        table = pd.read_csv(args.results / "distances.csv")
        summary = json.loads((args.results / "summary.json").read_text())
    else:
        benchmark = load_benchmark()
        mask = benchmark.clean & benchmark.has_magnitude
        comps = load_compositions(RESULTS / "compositions.csv")
        if list(comps["key"]) != list(benchmark.bulk_ids[mask]):
            raise SystemExit("results/compositions.csv does not match the benchmark's evaluated rocks")
        magnitude, numeric = benchmark.magnitude[mask], benchmark.numeric[mask]
        points = shares_and_distances(build_points(args.work, args.rebuild_points), magnitude, numeric)
        table = reach_table(points, load_draws(args.work), comps, magnitude, numeric)
        table["bin"] = bin_outcomes(table, args.tau)
        summary = summarize(table, args.tau, taus, reference_mae(), run_facts(args.work))
        table.to_csv(args.results / "distances.csv", index=False)
        (args.results / "summary.json").write_text(json.dumps(summary, indent=1))
        print_summary(summary, args.tau)

    draw_figures(table, summary, args.results)
    print(f"wrote {args.results / 'distances.csv'}, summary.json, reach_cdf.png, "
          "reach_bins.png and reach_melt.png")


if __name__ == "__main__":
    main()
