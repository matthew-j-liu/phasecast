"""Can a rock's chemistry predict where its silicate hits sit? The read
that decides whether any chemistry-conditioned prior is worth writing
(#79, second family). No MAGEMin, no head.

  python explorations/prior_search/run_predictability.py

On the #78 draws, a rock's hits are the (draw, rung) rows whose mean
absolute share error over its quantified silicates is within `--tau`; a
rock with none keeps its `--fallback` best rows. Two ceilings come first:
the marginal under the hit rows themselves, and under a rectangle of the
stage-0 widths centered on each rock's own mean hit conditions. Then the
same rectangle is centered where a predictor fit on the train rocks puts
it (the train mean, the TAS-class mean, ridge and a random forest on the
ten anhydrous majors) and read on val, beside the predictor's R2 on the
hit centers. Everything scores as stage 0 does. Writes `predictability.csv`,
`predictability.json` and `predictability.png` under `--results`.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cnmineral.data.magemin_oxides import FE2O3_TO_FEO_MASS
from cnmineral.eval.real_rock import load_benchmark
from explorations.generator_reach import run_read
from explorations.generator_reach.compositions import load_compositions
from explorations.prior_search.run_prior_screen import (
    NAVY, ORANGE, REACH, RESULTS, SILICATES, SUBSETS, WORK, load_samples, reweighted_marginals,
    score_scheme)
from explorations.prior_search.schemes import WIDTHS, Scheme, density

MAJORS = ("sio2", "tio2", "al2o3", "feo", "mno", "mgo", "cao", "na2o", "k2o", "p2o5")
CENTER = ("hit_p", "hit_w")


def silicate_error(rungs, numeric):
    """Per rung row, the mean absolute share error over the rock's
    quantified silicate categories; NaN for a rock quantifying none."""
    cols = [f"d_{g}" for g in SILICATES]
    return rungs[cols].mean(axis=1, skipna=True)


def hit_rows(rungs, draws, tau, fallback):
    """The hit rows per rock with their conditions: rows within `tau`, else
    the `fallback` best rows. Returns the row table and a per-rock summary
    (hit count, whether it fell back, best error, mean and spread of the
    hit conditions)."""
    rows = rungs[["bulk", "draw", "rung", "e_sil"]].merge(
        draws[["bulk", "draw", "P_kbar", "H2O"]], on=["bulk", "draw"])
    rows = rows[rows["e_sil"].notna()]
    parts = []
    for bulk, sub in rows.groupby("bulk", sort=True):
        hit = sub[sub["e_sil"] <= tau]
        fell = hit.empty
        if fell:
            hit = sub.nsmallest(fallback, "e_sil")
        parts.append(hit.assign(fallback=fell))
    hits = pd.concat(parts, ignore_index=True)
    by = hits.groupby("bulk")
    summary = pd.DataFrame({
        "n_hits": by.size(), "fallback": by["fallback"].first(),
        "best_e": rows.groupby("bulk")["e_sil"].min(),
        "hit_p": by["P_kbar"].mean(), "hit_w": by["H2O"].mean(),
        "hit_p_std": by["P_kbar"].std().fillna(0.0), "hit_w_std": by["H2O"].std().fillna(0.0)})
    return hits, summary


def hit_weights(draws, hits):
    """Each draw weighted by the fraction of its rung rows that are hits."""
    frac = hits.groupby(["bulk", "draw"]).size() / 2.0
    out = draws[["bulk", "draw"]].copy()
    out["w"] = frac.reindex(pd.MultiIndex.from_frame(out)).fillna(0.0).to_numpy()
    return out


def window_weights(draws, centers, p_half, w_half):
    """Each draw weighted by a rectangle of the given half-widths centered
    on its rock's row of `centers` (columns hit_p, hit_w), over the broad-box
    density; a rock without a center gets zero weight."""
    out = draws[["bulk", "draw"]].copy()
    w = np.zeros(len(draws))
    bulk = draws["bulk"].to_numpy()
    p, h, q = (draws[c].to_numpy(dtype=float) for c in ("P_kbar", "H2O", "q"))
    for b, row in centers.iterrows():
        m = bulk == b
        if m.any() and np.isfinite(row["hit_p"]) and np.isfinite(row["hit_w"]):
            s = Scheme(str(b), float(row["hit_p"]), float(row["hit_w"]), p_half, w_half)
            w[m] = density(s, p[m], h[m]) / q[m]
    out["w"] = w
    return out


def features(comps):
    """The ten anhydrous majors renormalized to 100, total iron as FeO."""
    feo = comps["feo_tot"].fillna(comps["fe2o3_tot"] * FE2O3_TO_FEO_MASS).fillna(0.0)
    x = comps[[c for c in MAJORS if c != "feo"]].fillna(0.0).copy()
    x.insert(3, "feo", feo)
    return x.div(x.sum(axis=1), axis=0).mul(100.0)


def predictors(x, comps, summary, train):
    """Hit-center predictions for every rock from predictors fit on the
    train rocks: the train mean, the TAS-class mean, ridge and a random
    forest on the majors."""
    y = summary.loc[comps["bulk"], list(CENTER)].to_numpy()
    fit = train & np.isfinite(y).all(axis=1)
    xt, yt = x[fit].to_numpy(), y[fit]
    out = {"train_mean": np.tile(yt.mean(axis=0), (len(x), 1))}
    cls = comps["rock_class"].to_numpy()
    means = pd.DataFrame(yt, columns=CENTER).groupby(cls[fit]).mean()
    out["class_mean"] = np.array([means.loc[c].to_numpy() if c in means.index else yt.mean(axis=0)
                                  for c in cls])
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(xt, yt)
    out["ridge"] = ridge.predict(x.to_numpy())
    forest = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0).fit(xt, yt)
    out["forest"] = forest.predict(x.to_numpy())
    return {k: pd.DataFrame(v, columns=CENTER, index=comps["bulk"].to_numpy()) for k, v in out.items()}


def read(points, draws, comps, magnitude, numeric, tau=0.05, fallback=8):
    rungs = run_read.rung_rows(points, draws)
    rungs["e_sil"] = silicate_error(rungs, numeric)
    hits, summary = hit_rows(rungs, draws, tau, fallback)
    summary = summary.reindex(comps["bulk"])
    subset = comps["subset"].to_numpy()
    masks = {s: subset == s for s in SUBSETS}
    x = features(comps)
    preds = predictors(x, comps, summary, masks["train"])

    rows = []

    def add(name, kind, weights, r2=None):
        marg = reweighted_marginals(rungs, weights)
        row = dict(name=name, kind=kind, **score_scheme(marg, comps, magnitude, numeric))
        if r2 is not None:
            row.update(r2)
        rows.append(row)

    add("hit rows", "ceiling", hit_weights(draws, hits))
    truth = summary[list(CENTER)]
    for width, (ph, wh) in WIDTHS.items():
        add(f"own hit center, {width}", "ceiling", window_weights(draws, truth, ph, wh))
    known = np.isfinite(truth.to_numpy()).all(axis=1)
    for pname, centers in preds.items():
        r2 = {f"r2_{c}_{s}": float(r2_score(truth.loc[m & known, c], centers.loc[m & known, c]))
              for c in CENTER for s, m in masks.items()}
        for width, (ph, wh) in WIDTHS.items():
            add(f"{pname}, {width}", "predicted", window_weights(draws, centers, ph, wh), r2)
    table = pd.DataFrame(rows)
    facts = {
        "tau": tau, "fallback_rows": fallback,
        "n_rocks": int(summary["n_hits"].notna().sum()),
        "n_fallback": int(summary["fallback"].fillna(False).sum()),
        "hit_fraction_median": float((summary["n_hits"] / (2 * 128)).median()),
        "best_e_median": float(summary["best_e"].median()),
        "hit_p_std_median": float(summary["hit_p_std"].median()),
        "hit_w_std_median": float(summary["hit_w_std"].median()),
        "hit_p_iqr": [float(v) for v in summary["hit_p"].quantile([0.25, 0.75])],
        "hit_w_iqr": [float(v) for v in summary["hit_w"].quantile([0.25, 0.75])],
    }
    return table, summary, preds, facts


def references(path=RESULTS / "schemes.csv"):
    if not path.exists():
        return {}
    t = pd.read_csv(path).set_index("name")
    best = t[t["kind"] == "grid"].index[0]
    return {"broad": float(t.loc["broad", "silicate_val"]),
            f"best rectangle ({best})": float(t.loc[best, "silicate_val"])}


def print_table(table, facts, refs):
    print(f"{facts['n_rocks']} rocks; hits within {facts['tau']} on the silicates, {facts['n_fallback']} "
          f"rocks fell back to their {facts['fallback_rows']} best rows; median hit fraction "
          f"{facts['hit_fraction_median']:.3f}; median spread of a rock's hits {facts['hit_p_std_median']:.1f} "
          f"kbar / {facts['hit_w_std_median']:.1f} wt%")
    for k, v in refs.items():
        print(f"reference {k}: silicate val {v:.4f}")
    print("prior                        silicate train / val   other val   n_eff   R2 val P / H2O")
    for _, r in table.iterrows():
        r2 = (f"{r['r2_hit_p_val']:+.2f} / {r['r2_hit_w_val']:+.2f}"
              if "r2_hit_p_val" in r and np.isfinite(r.get("r2_hit_p_val", np.nan)) else "")
        print(f"{r['name']:28s} {r['silicate_train']:.4f} / {r['silicate_val']:.4f}   "
              f"{r['other_val']:.4f}   {r['n_eff_median']:5.1f}   {r2}")


def draw_figure(table, summary, preds, comps, refs, out):
    truth = summary.loc[comps["bulk"]]
    val = (comps["subset"].to_numpy() == "val") & np.isfinite(truth[list(CENTER)].to_numpy()).all(axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [2, 1, 1]})
    ax = axes[0]
    y = np.arange(len(table))
    for width, mfc in (("wide", NAVY), ("narrow", "none")):
        m = table["name"].str.endswith(width) | (table["name"] == "hit rows")
        ax.plot(table.loc[m, "silicate_val"], y[m], "o", color=NAVY, mfc=mfc,
                label=width if width == "wide" else "narrow (hollow)")
    for (k, v), ls in zip(refs.items(), ("--", ":")):
        ax.axvline(v, color=ORANGE, ls=ls, lw=1, label=k)
    ax.set_yticks(y, table["name"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("four-silicate share error, val", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("ceilings (top) and chemistry-predicted windows", fontsize=9)
    for ax, c, label, lim in zip(axes[1:], CENTER, ("P (kbar)", "H2O (wt%)"), ((0, 12.5), (0, 8.5))):
        for pname, color in (("forest", NAVY), ("ridge", ORANGE)):
            pred = preds[pname].loc[comps["bulk"], c].to_numpy()
            r2 = r2_score(truth.loc[val, c], pred[val])
            ax.scatter(truth.loc[val, c], pred[val], s=10, alpha=0.6, color=color, lw=0,
                       label=f"{pname}, R2 {r2:+.2f}")
        ax.plot(lim, lim, color="#8a8a8a", ls="--", lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"mean hit {label}, val rocks", fontsize=9)
        ax.set_ylabel(f"predicted from majors", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("does chemistry predict where a rock's silicate hits sit?", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "predictability.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=WORK)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--fallback", type=int, default=8)
    args = ap.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)

    benchmark = load_benchmark()
    mask = benchmark.clean & benchmark.has_magnitude
    comps = load_compositions(REACH / "results" / "compositions.csv")
    if list(comps["key"]) != list(benchmark.bulk_ids[mask]):
        raise SystemExit("generator_reach/results/compositions.csv does not match the "
                         "benchmark's evaluated rocks")
    magnitude, numeric = benchmark.magnitude[mask], benchmark.numeric[mask]
    points = run_read.shares_and_distances(run_read.build_points(args.work), magnitude, numeric)
    draws = run_read.load_draws(args.work)
    table, summary, preds, facts = read(points, draws, comps, magnitude, numeric,
                                        args.tau, args.fallback)
    refs = references()
    print_table(table, facts, refs)
    table.to_csv(args.results / "predictability.csv", index=False)
    (args.results / "predictability.json").write_text(json.dumps(
        {"facts": facts, "references": refs,
         "rows": table[["name", "kind", "silicate_train", "silicate_val", "other_val",
                        "n_eff_median"]].to_dict("records")}, indent=1))
    draw_figure(table, summary, preds, comps, refs, args.results)
    print(f"wrote {args.results / 'predictability.csv'}, predictability.json, predictability.png")


if __name__ == "__main__":
    main()
