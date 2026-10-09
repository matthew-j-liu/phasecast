"""The seven-mineral ladder read on the kept equilibrium rows of both
benchmarks (#83, journal 2026-10-06): per draw, where each mineral's mass
peaks down the cooling ladder, how much of the peak remains at the
melt-fraction rungs, what grows as it declines (the gain budget: per unit of
the mineral's mass lost between its peak and the 10% step, the mass every
other category gained over the same steps), split wet against dry and
shallow against deep; and the prediction for the fractional read, the
no-resorption pile scored against the measured share beside the equilibrium
rung. Model-free, minutes.

  python explorations/generator_reach/run_ladder.py
  python explorations/generator_reach/run_ladder.py --benchmark plutonic

Rows: `run_read.build_points` on `work/plutonic128` and `work/broad128`, kept
to the evaluated train and val rocks of each composition table (test sealed); the
`m_<g>` columns are each category's crystal mass on the whole-system basis
at every converged step. A draw carries a category when its seven-category
crystal share at the peak step clears `PRESENCE_CUTOFF`; a rung's end is the
highest-T step with melt below it, as `run_signal.generator_reads` reads it.
The no-resorption pile at a step is the sum of the category's positive mass
increments down the path to that step: what the equilibrium path would have
accumulated had nothing it grew been consumed, which keeps each step's growth
but not fractionation's changed liquid line of descent, so it is a proxy for
the fractional run, not its result. Written under `results/`: `ladder.csv`
(one row per benchmark, category and split), `ladder.json` (settings, the
split medians, the kept counts and the proxy prediction) and `ladder.png`.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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

from cnmineral.data.phase_vocab import GROUP_INDEX, GROUPS, PRESENCE_CUTOFF
from cnmineral.eval.real_rock import BENCHMARK_DIR, load_benchmark
from explorations.generator_reach import run_read
from explorations.generator_reach.compositions import load_compositions
from explorations.prior_search.run_prior_screen import NAVY, ORANGE
from explorations.real_mineralogy.plutonic import plutonic_benchmark

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
WORK = HERE / "work"
BENCHMARKS = {"plutonic": ("plutonic128", "plutonic_compositions.csv"),
              "eruption": ("broad128", "compositions.csv")}
KEPT = ("train", "val")
RUNGS = (50, 30, 10, 5)
# a carried mineral has declined when less than this fraction of its peak is left
DECLINE = 0.9
SILICATES = ("fsp", "px", "amp", "bi")
PARTNERS = (*GROUPS, "other")
CURVES = ("wet", "dry", "deep", "shallow")


def kept_rows(benchmark):
    """The run's points and draws on the kept rocks, with P and H2O on every
    point; and the benchmark's measured share and quantified mask by bulk."""
    work_name, comps_name = BENCHMARKS[benchmark]
    comps = load_compositions(RESULTS / comps_name)
    kept = comps["subset"].isin(KEPT)
    if "evaluated" in comps:
        kept &= comps["evaluated"]
    kept = comps.loc[kept, "bulk"].to_numpy()
    draws = run_read.load_draws(WORK / work_name)
    draws = draws[draws["bulk"].isin(kept)]
    points = run_read.build_points(WORK / work_name)
    points = points[points["bulk"].isin(kept)].merge(draws[["bulk", "draw", "P_kbar", "H2O"]],
                                                      on=["bulk", "draw"])
    if benchmark == "plutonic":
        with plutonic_benchmark() as evaluated:
            truth = evaluated.subset(evaluated.has_magnitude)
        keys = truth.bulk_ids
    else:
        truth = load_benchmark(BENCHMARK_DIR)
        truth = truth.subset(truth.clean & truth.has_magnitude)
        keys = truth.bulk_ids
    bulk = comps.set_index("key").loc[keys, "bulk"].to_numpy()
    magnitude = np.full((len(comps), len(GROUPS)), np.nan)
    numeric = np.zeros((len(comps), len(GROUPS)), dtype=bool)
    magnitude[bulk], numeric[bulk] = truth.magnitude, truth.numeric
    return points.sort_values(["bulk", "draw", "T_celsius"], ascending=[True, True, False]), \
        draws, magnitude, numeric


def rung_ends(points, rungs=RUNGS):
    """{rung: (bulk, draw) -> T of the highest-T step with melt below it}."""
    return {r: points[points["liq_frac"] < r / 100].groupby(["bulk", "draw"])["T_celsius"].max()
            for r in rungs}


def no_resorption_pile(points):
    """Per category, the running sum of positive mass increments down each
    draw's path (the first step's mass counts whole): the crystals the
    equilibrium path grew, had nothing been consumed."""
    by = points.groupby(["bulk", "draw"], sort=False)
    out = points[["bulk", "draw", "T_celsius", "liq_frac"]].copy()
    for g in GROUPS:
        inc = by[f"m_{g}"].diff()
        inc = inc.fillna(points[f"m_{g}"]).clip(lower=0.0)
        out[f"m_{g}"] = inc.groupby([points["bulk"], points["draw"]]).cumsum()
    return out


def per_draw_table(points, ends):
    """One row per (bulk, draw, category): the peak mass and its T, whether
    the draw carries the category, the mass and the remaining fraction at
    each rung, and, when the category declined, its loss between the peak
    and the 10% step with every other category's gain over the same steps."""
    key = ["bulk", "draw"]
    mass = points[[f"m_{g}" for g in GROUPS]].to_numpy()
    seven = mass.sum(1, keepdims=True)
    share = np.divide(mass, seven, out=np.zeros_like(mass), where=seven > 0)
    rows = []
    at_rung = {r: points.merge(ends[r].rename("T_celsius").reset_index(), on=key + ["T_celsius"])
               .set_index(key) for r in ends}
    gains = {g: points.set_index(key + ["T_celsius"])[f"m_{g}"] for g in PARTNERS}
    for j, g in enumerate(GROUPS):
        col = f"m_{g}"
        p = points.assign(share=share[:, j])
        idx = p.groupby(key, sort=False)[col].idxmax()
        peak = p.loc[idx, key + ["T_celsius", col, "share", "P_kbar", "H2O"]].rename(
            columns={"T_celsius": "T_peak", col: "peak", "share": "peak_share"})
        peak["category"] = g
        peak["carries"] = peak["peak_share"] > PRESENCE_CUTOFF
        peak = peak.set_index(key)
        for r in ends:
            m = at_rung[r][col].reindex(peak.index)
            peak[f"m_{r}"] = m
            peak[f"remaining_{r}"] = np.where(peak["peak"] > 0, m / peak["peak"], np.nan)
            peak[f"T_{r}"] = at_rung[r]["T_celsius"].reindex(peak.index)
        declined = peak["carries"] & (peak["remaining_10"] < DECLINE)
        peak["declined"] = declined
        peak["loss"] = np.nan
        for h in PARTNERS:
            peak[f"gain_{h}"] = np.nan
        if declined.any():
            d = peak[declined]
            at = lambda T: pd.MultiIndex.from_arrays(
                [d.index.get_level_values(0), d.index.get_level_values(1), T])
            for h in PARTNERS:
                change = gains[h].reindex(at(d["T_10"])).to_numpy() - gains[h].reindex(at(d["T_peak"])).to_numpy()
                if h == g:
                    peak.loc[declined, "loss"] = -change
                else:
                    peak.loc[declined, f"gain_{h}"] = change
        rows.append(peak.reset_index())
    return pd.concat(rows, ignore_index=True)


def split_masks(table, medians):
    return {"all": np.ones(len(table), dtype=bool),
            "wet": table["H2O"].to_numpy() >= medians["H2O"],
            "dry": table["H2O"].to_numpy() < medians["H2O"],
            "deep": table["P_kbar"].to_numpy() >= medians["P_kbar"],
            "shallow": table["P_kbar"].to_numpy() < medians["P_kbar"]}


def summarize(table, medians, benchmark):
    """One row per category and split over the carrying draws; the gain
    budget is each other category's mean gain over the declining draws per
    unit of the category's mean loss."""
    masks = split_masks(table, medians)
    rows = []
    for g in GROUPS:
        for split, m in masks.items():
            t = table[(table["category"] == g) & table["carries"] & m]
            d = t[t["declined"]]
            loss = float(d["loss"].mean()) if len(d) else np.nan
            row = dict(benchmark=benchmark, category=g, split=split, n_draws=int(m.sum() // len(GROUPS)),
                       n_carrying=len(t), frac_declining=float(d.shape[0] / len(t)) if len(t) else np.nan,
                       median_T_peak=float(t["T_peak"].median()) if len(t) else np.nan,
                       median_peak_mass=float(t["peak"].median()) if len(t) else np.nan)
            for r in RUNGS:
                row[f"median_remaining_{r}"] = float(t[f"remaining_{r}"].median()) if len(t) else np.nan
                row[f"median_T_{r}"] = float(t[f"T_{r}"].median()) if len(t) else np.nan
            row["mean_loss"] = loss
            for h in PARTNERS:
                row[f"gain_{h}"] = float(d[f"gain_{h}"].mean() / loss) if len(d) and loss > 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def rock_shares(points, ends, magnitude, numeric):
    """Per rung, the draw-mean share over each rock's measured mineral list;
    `points` may carry equilibrium or proxy masses."""
    out = {}
    for r, end in ends.items():
        rows = points.merge(end.rename("T_celsius").reset_index(), on=["bulk", "draw", "T_celsius"])
        shared = run_read.shares_and_distances(rows, np.nan_to_num(magnitude), numeric)
        out[r] = shared.groupby("bulk")[[f"s_{g}" for g in GROUPS]].mean()
    return out


def prediction(points, ends, magnitude, numeric):
    """Per rung and category over the kept rocks quantifying it: the
    absolute error of the equilibrium rung share and of the no-resorption
    proxy against the measured share, and the proxy minus equilibrium."""
    eq = rock_shares(points, ends, magnitude, numeric)
    proxy = rock_shares(no_resorption_pile(points).merge(
        points[["bulk", "draw", "T_celsius", "P_kbar", "H2O"]], on=["bulk", "draw", "T_celsius"]),
        ends, magnitude, numeric)
    out = {}
    for r in ends:
        bulks = eq[r].index.to_numpy()
        res = {}
        for name, s in (("equilibrium", eq[r]), ("proxy", proxy[r])):
            err = np.abs(s.to_numpy() - magnitude[bulks])
            err[~numeric[bulks]] = np.nan
            per = {g: float(np.nanmean(err[:, j])) for j, g in enumerate(GROUPS) if numeric[bulks][:, j].any()}
            sil = [GROUP_INDEX[g] for g in SILICATES]
            has = numeric[bulks][:, sil].any(1)
            res[name] = {**per, "silicate": float(np.nanmean(err[has][:, sil], axis=1).mean())}
        res["proxy_minus_equilibrium"] = {k: res["proxy"][k] - res["equilibrium"][k] for k in res["proxy"]}
        res["n_rocks"] = {g: int(numeric[bulks][:, j].sum()) for j, g in enumerate(GROUPS)}
        res["n_rocks_read"] = int(len(bulks))
        out[str(r)] = res
    return out


def draw_figure(curves, medians_T10, out):
    fig, axes = plt.subplots(len(curves), len(GROUPS), figsize=(2.6 * len(GROUPS), 2.4 * len(curves)),
                             sharex=True, squeeze=False)
    styles = {"wet": (NAVY, "-"), "dry": (ORANGE, "-"), "deep": (NAVY, "--"), "shallow": (ORANGE, "--")}
    for i, (benchmark, by_split) in enumerate(curves.items()):
        for j, g in enumerate(GROUPS):
            ax = axes[i, j]
            for split, (color, ls) in styles.items():
                c = by_split[split]
                ax.plot(c.index, c[f"m_{g}"], color=color, ls=ls, lw=1.2, label=split)
            ax.axvline(medians_T10[benchmark], color="0.5", ls=":", lw=0.8)
            ax.set_title(f"{benchmark}: {g}", fontsize=9)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel("mean crystal mass, whole system", fontsize=8)
            if i == len(curves) - 1:
                ax.set_xlabel("T (C)", fontsize=8)
            ax.invert_xaxis()
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("kept draws, mean category mass down the equilibrium ladder by split; "
                 "dotted: median T of the 10% melt rung", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "ladder.png", dpi=150)
    plt.close(fig)


def read_benchmark(benchmark):
    points, draws, magnitude, numeric = kept_rows(benchmark)
    medians = {"H2O": float(draws["H2O"].median()), "P_kbar": float(draws["P_kbar"].median())}
    ends = rung_ends(points)
    table = per_draw_table(points, ends)
    summary = summarize(table, medians, benchmark)
    pred = prediction(points, ends, magnitude, numeric)
    masks = split_masks(points, medians)
    curves = {s: points[masks[s]].groupby("T_celsius")[[f"m_{g}" for g in GROUPS]].mean()
              for s in CURVES}
    facts = {"kept_rocks": int(points["bulk"].nunique()), "kept_draws": int(len(draws)),
             "split_medians": medians, "median_T_10": float(ends[10].median()),
             "draws_reaching": {str(r): int(len(ends[r])) for r in RUNGS}}
    return summary, pred, curves, facts


def print_summary(summary, pred):
    for benchmark, s in summary.groupby("benchmark", sort=False):
        print(f"{benchmark}, carrying draws, all splits: category, n carrying, fraction declining by 10% melt, "
              "median T at peak, median remaining at 50/30/10/5%, gain per unit lost")
        for row in s[s["split"] == "all"].itertuples():
            rem = " / ".join(f"{getattr(row, f'median_remaining_{r}'):.2f}" for r in RUNGS)
            gains = " ".join(f"{h} {getattr(row, f'gain_{h}'):+.2f}" for h in PARTNERS
                             if h != row.category and np.isfinite(getattr(row, f"gain_{h}")))
            print(f"  {row.category:4s} {row.n_carrying:6d}  {row.frac_declining:.2f}  "
                  f"{row.median_T_peak:5.0f}  {rem}  {gains}")
        print("  wet / dry / deep / shallow: fraction declining and median remaining at 10% melt")
        for g in GROUPS:
            parts = []
            for split in ("wet", "dry", "deep", "shallow"):
                r = s[(s["category"] == g) & (s["split"] == split)].iloc[0]
                parts.append(f"{r.frac_declining:.2f} {r.median_remaining_10:.2f}")
            print(f"  {g:4s} " + " / ".join(parts))
        p = pred[benchmark]
        print("  absolute error against the measured share, equilibrium / no-resorption proxy, per rung:")
        for r in RUNGS:
            e, x = p[str(r)]["equilibrium"], p[str(r)]["proxy"]
            print(f"    rung {r:2d}: " + ", ".join(
                f"{k} {e[k]:.3f}/{x[k]:.3f}" for k in [*GROUPS, "silicate"] if k in e))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", choices=tuple(BENCHMARKS), help="default: both")
    ap.add_argument("--results", type=Path, default=RESULTS)
    args = ap.parse_args()
    benchmarks = [args.benchmark] if args.benchmark else list(BENCHMARKS)

    summaries, preds, curves, facts, med = [], {}, {}, {}, {}
    for b in benchmarks:
        s, p, c, f = read_benchmark(b)
        summaries.append(s)
        preds[b], curves[b], facts[b], med[b] = p, c, f, f["median_T_10"]
    summary = pd.concat(summaries, ignore_index=True)
    print_summary(summary, preds)

    args.results.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.results / "ladder.csv", index=False)
    payload = {"settings": {"kept_subsets": list(KEPT), "rungs": list(RUNGS), "decline": DECLINE,
                            "presence_cutoff": PRESENCE_CUTOFF, "splits": "at the kept draws' medians",
                            "proxy": "sum of positive mass increments down the path to the rung step"},
               "benchmarks": {b: {**facts[b], "prediction": preds[b]} for b in benchmarks}}
    (args.results / "ladder.json").write_text(json.dumps(payload, indent=1, default=float))
    draw_figure(curves, med, args.results)
    print(f"wrote ladder.csv, ladder.json and ladder.png under {args.results}")


if __name__ == "__main__":
    main()
