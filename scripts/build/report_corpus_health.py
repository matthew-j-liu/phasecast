"""Corpus health report: the pilot gate for production launch.

Reads a generation output dir (chunk parquets + manifest) and reports what the
spec's pilot validates: attrition per rung, convergence failures, mass-balance
closure (mode sum and bulk reconstruction), per-phase coverage, and the label
mode distributions. Read-only; prints tables and writes health_*.csv next to
the chunks.

    conda run -n alignn_fix python scripts/build/report_corpus_health.py --corpus data/corpus/train
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cnmineral.data.magemin_corpus import RUNGS

CUTOFFS = (0.001, 0.01)   # 0.1 and 1 wt%, on weight-fraction modes


def load(corpus, name):
    files = sorted(corpus.glob(f"chunk_*_{name}.parquet"))
    if not files:
        raise SystemExit(f"no {name} chunks in {corpus}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    args = ap.parse_args()
    corpus = Path(args.corpus)

    draws = load(corpus, "draws")
    rungs = load(corpus, "rungs")

    print(f"{draws.bulk.nunique():,} bulks, {len(draws):,} draws, "
          f"{len(rungs):,} rung phase rows")
    print(f"failed points/draw: mean {draws.n_failed.mean():.2f}, "
          f"draws fully failed {(draws.n_points == 0).mean():.3%}")
    print(f"oversaturated draws (fl at hottest point): {draws.oversaturated.mean():.3%}")
    print("\nattrition per rung (draw never reached the melt target):")
    for m in RUNGS:
        col = f"attrition_{int(m * 100)}"
        print(f"  {int(m * 100)}% melt: {draws[col].mean():.3%}")

    print("\n=== mass-balance closure per (draw, rung) ===")
    key = ["bulk", "draw", "rung"]
    tot = rungs.groupby(key).mode_wt.sum()
    print(f"|sum(modes) - 1|: p50 {np.abs(tot - 1).median():.2e}, "
          f"p95 {np.abs(tot - 1).quantile(0.95):.2e}, max {np.abs(tot - 1).max():.2e}")

    recon = rungs[key + ["mode_wt"]].copy()
    wt_map = {"sio2": "wt_SiO2", "al2o3": "wt_Al2O3", "cao": "wt_CaO", "mgo": "wt_MgO",
              "feo": "wt_FeO", "k2o": "wt_K2O", "na2o": "wt_Na2O",
              "tio2": "wt_TiO2", "o": "wt_O", "cr2o3": "wt_Cr2O3", "h2o": "wt_H2O"}
    for ox, col in wt_map.items():
        recon[ox] = rungs.mode_wt * rungs[col]
    agg = recon.groupby(key)[list(wt_map)].sum().reset_index()
    in_cols = [f"in_{ox}" for ox in wt_map if ox not in ("o", "h2o", "feo")]
    agg = agg.merge(draws[["bulk", "draw", "H2O", "in_feo", "in_fe2o3"] + in_cols],
                    on=["bulk", "draw"])
    # input Fe (FeO + Fe2O3 wt%) -> the output's total-FeO + excess-O convention
    agg["in_feo_tot"] = agg.in_feo + agg.in_fe2o3 * (2 * 71.844 / 159.69)
    agg["in_o"] = agg.in_fe2o3 * (15.999 / 159.69)
    pairs = ([(ox, f"in_{ox}") for ox in wt_map if ox not in ("o", "h2o", "feo")]
             + [("feo", "in_feo_tot"), ("o", "in_o")])
    resid = pd.concat([(agg[ox] - agg[inc] / 100.0).abs() for ox, inc in pairs]
                      + [(agg.h2o - agg.H2O / 100.0).abs()])
    print(f"|reconstructed - input| per oxide (wt frac): p50 {resid.median():.2e}, "
          f"p95 {resid.quantile(0.95):.2e}, max {resid.max():.2e}")

    print("\n=== per-phase coverage and modes per rung ===")
    solids = rungs[~rungs.phase.isin(["liq", "fl"])]
    n_draws_rung = rungs.groupby("rung")[["bulk", "draw"]].apply(
        lambda g: len(g.drop_duplicates()))
    rows = []
    for (rung, phase), g in solids.groupby(["rung", "phase"]):
        pres = g.drop_duplicates(["bulk", "draw"])
        rows.append(dict(
            rung=rung, phase=phase,
            **{f"presence_{c}": round(len(pres[pres.mode_wt > c]) / n_draws_rung[rung], 4)
               for c in CUTOFFS},
            mode_med=round(float(g.mode_wt.median()), 4),
            mode_p90=round(float(g.mode_wt.quantile(0.9)), 4),
            has_em=round(float(g.em_json.notna().mean()), 3)))
    cov = pd.DataFrame(rows).sort_values(["rung", "presence_0.001"],
                                         ascending=[True, False])
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(cov.to_string(index=False))

    cov.to_csv(corpus / "health_phase_coverage.csv", index=False)
    print(f"\nwrote {corpus / 'health_phase_coverage.csv'}")


if __name__ == "__main__":
    main()
