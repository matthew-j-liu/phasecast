"""Corpus health report: the pilot gate for production launch.

Reads a generation output dir (chunk parquets + manifest) and reports what the
spec's pilot validates: attrition per rung, convergence failures, mass-balance
closure (mode sum and bulk reconstruction), per-phase coverage, and the label
mode distributions. Read-only; prints tables and writes health_*.csv next to
the chunks. `--figures` additionally writes the corpus-statistics EDA plots to
`<corpus>/figures/` (regenerable, gitignored; the sim-vs-real comparison plots
live with the rung pre-screen in explorations/archive/volcanic_proxy_definition/).

    conda run -n alignn_fix python scripts/build/report_corpus_health.py --corpus data/corpus/train --figures
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
    ap.add_argument("--figures", action="store_true",
                    help="write EDA plots to <corpus>/figures/")
    ap.add_argument("--bulks", nargs="+", help="full bulk table(s) for the "
                    "representativeness plot; a tier split across several "
                    "tables (train + expansion) takes them all "
                    "(default: <corpus>_bulks.csv next to the corpus dir)")
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

    if args.figures:
        bulks = ([Path(p) for p in args.bulks] if args.bulks
                 else [corpus.parent / f"{corpus.name}_bulks.csv"])
        write_figures(draws, rungs, cov, corpus, bulks)


def write_figures(draws, rungs, cov, corpus, bulks_paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cnmineral.data.condition_proposal import w_max

    out = corpus / "figures"
    out.mkdir(exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=150)
        plt.close(fig)
        print(f"wrote {out / name}")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    over = draws.oversaturated.astype(bool)
    ax.scatter(draws.P_kbar[~over], draws.H2O[~over], s=3, alpha=0.15,
               color="tab:blue", label="draws")
    ax.scatter(draws.P_kbar[over], draws.H2O[over], s=3, alpha=0.4,
               color="crimson", label=f"oversaturated ({over.mean():.1%})")
    p = np.linspace(draws.P_kbar.min(), draws.P_kbar.max(), 200)
    ax.plot(p, [w_max(float(x)) for x in p], color="k", lw=1.2,
            label="w_max(P) bound")
    ax.set_xlabel("P (kbar)")
    ax.set_ylabel("H2O (wt%)")
    ax.set_title("sampled (P, H2O) coverage")
    ax.legend(loc="lower right", markerscale=4)
    ax2.hist(draws.T_stop.dropna(), bins=60, color="tab:gray")
    ax2.set_xlabel("T_stop (C)")
    ax2.set_ylabel("draws")
    ax2.set_title("derived endpoint temperature")
    save(fig, "conditions.png")

    pres = cov.pivot(index="phase", columns="rung", values="presence_0.001")
    pres = pres.loc[pres.max(axis=1).sort_values().index]
    rung_order = sorted(pres.columns, reverse=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(pres) + 1.5))
    y = np.arange(len(pres))
    h = 0.8 / len(rung_order)
    for i, rung in enumerate(rung_order):
        ax.barh(y - 0.4 + (i + 0.5) * h, pres[rung], height=h, label=f"rung {rung}")
    ax.set_yticks(y, pres.index)
    ax.set_xlabel("presence frequency (mode > 0.1 wt%)")
    ax.legend(loc="lower right")
    ax.set_title("per-phase coverage by rung")
    save(fig, "phase_coverage.png")

    solids = rungs[~rungs.phase.isin(["liq", "fl"])]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, rung in zip(axes.flat, rung_order):
        g = solids[solids.rung == rung]
        top = g.phase.value_counts().head(12).index.tolist()
        ax.boxplot([g.mode_wt[g.phase == ph].values for ph in top], showfliers=False)
        ax.set_xticks(range(1, len(top) + 1), top, rotation=60)
        ax.set_ylabel("mode (wt frac)")
        ax.set_title(f"rung {rung}: mode distributions (12 commonest phases, when present)")
    save(fig, "mode_distributions.png")

    present = solids[solids.mode_wt > CUTOFFS[0]]
    key = ["bulk", "draw", "rung"]
    nph = present.groupby(key).size()
    asm = present.groupby(key).phase.agg(lambda s: "|".join(sorted(s)))
    nasm = asm.groupby(["bulk", "rung"]).nunique()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for rung in rung_order:
        v = nph.xs(rung, level="rung")
        ax.hist(v, bins=np.arange(0.5, v.max() + 1.5), histtype="step",
                label=f"rung {rung}")
        w = nasm.xs(rung, level="rung")
        ax2.hist(w, bins=np.arange(0.5, w.max() + 1.5), histtype="step",
                 label=f"rung {rung}")
    n_draws = draws.groupby("bulk").draw.nunique().median()
    ax.set_xlabel("solid phases per draw (mode > 0.1 wt%)")
    ax.set_ylabel("draws")
    ax.legend()
    ax2.set_xlabel(f"distinct assemblages per bulk (~{n_draws:.0f} draws each)")
    ax2.set_ylabel("bulks")
    ax2.set_title("within-bulk spread: the multimodality the heads learn")
    ax2.legend()
    save(fig, "assemblage_spread.png")

    missing = [p for p in bulks_paths if not p.exists()]
    if missing:
        print(f"no bulk table at {', '.join(map(str, missing))}; "
              f"skipping representativeness plot")
        return
    full = pd.concat([pd.read_csv(p) for p in bulks_paths], ignore_index=True)
    got = draws.drop_duplicates("bulk")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    freq = pd.DataFrame({
        "table": full.qap_name.value_counts(normalize=True),
        "generated": got.qap_name.value_counts(normalize=True),
    }).fillna(0.0).sort_values("table")
    y = np.arange(len(freq))
    ax.barh(y + 0.2, freq["table"], height=0.4,
            label=f"full table (n={len(full):,})")
    ax.barh(y - 0.2, freq["generated"], height=0.4,
            label=f"generated (n={len(got):,})")
    ax.set_yticks(y, freq.index, fontsize=7)
    ax.set_xlabel("class share")
    ax.set_title("qap class mix: generated bulks vs full table")
    ax.legend(loc="lower right")
    ax2.hist([got.sio2, full.sio2], bins=40, density=True, histtype="step",
             label=[f"generated (n={len(got):,})", f"full table (n={len(full):,})"])
    ax2.set_xlabel("SiO2 (wt%)")
    ax2.set_ylabel("density")
    ax2.set_title("SiO2 mix")
    ax2.legend()
    save(fig, "representativeness.png")


if __name__ == "__main__":
    main()
