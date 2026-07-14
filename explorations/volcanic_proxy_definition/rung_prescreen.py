"""Label-level pre-screen: pick the training rung against the real benchmark.

Runs on pilot (or production) corpus chunks. Per candidate rung (60/50/40/30%
residual melt) and presence cutoff (0.1 / 1 wt%), scores the synthetic labels
against real phenocryst observations on a shared 8-category vocabulary
(plag, kfsp, q, bi, amp, px, ol, opq -- pyroxenes and opaques lumped because
du Bray reports them unsplit):

  1. presence match: per-SiO2-band presence frequency vs du Bray (subalkaline
     sim bulks only, trace occurrences count as present), scored as MAE;
  2. rank match: Spearman rank correlation of mean crystals-renormalized
     proportions per band;
  3. magnitude context on the 13 point-counted samples (MSH/Pinatubo dacites)
     -- reported with the vol%-vs-wt% caveat, no density conversion here (the
     real-benchmark label adapter is a spec follow-up).

The owner picks the rung from the printed table; nothing is decided here.

    conda run -n alignn_fix python explorations/volcanic_proxy_definition/rung_prescreen.py \\
        --corpus data/corpus/train
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REAL = HERE.parent.parent / "data" / "real_mineralogy"
RESULTS = HERE / "results"

CATEGORIES = ["plag", "kfsp", "q", "bi", "amp", "px", "ol", "opq"]
CUTOFFS = (0.001, 0.01)   # weight-fraction modes: 0.1 and 1 wt%
BANDS = [0, 52, 63, 100]
BAND_LABELS = ["mafic", "intermediate", "felsic"]

SIM_MAP = {"q": "q", "bi": "bi", "amp": "amp", "ol": "ol",
           "opx": "px", "cpx": "px", "spl": "opq", "ilm": "opq",
           "mt": "opq", "hem": "opq"}
REAL_MAP = {"amp": "amp", "bi": "bi", "q": "q", "ol": "ol",
            "opx": "px", "cpx": "px", "px_unsplit": "px", "opq_unsplit": "opq"}


def load_sim(corpus):
    rungs = pd.concat([pd.read_parquet(f)
                       for f in sorted(corpus.glob("chunk_*_rungs.parquet"))],
                      ignore_index=True)
    draws = pd.concat([pd.read_parquet(f)
                       for f in sorted(corpus.glob("chunk_*_draws.parquet"))],
                      ignore_index=True)
    solids = rungs[~rungs.phase.isin(["liq", "fl"])].copy()
    is_fsp = solids.phase == "fsp"
    solids["cat"] = solids.phase.map(SIM_MAP)
    solids.loc[is_fsp, "cat"] = np.where(
        solids.loc[is_fsp, "wt_K2O"] > solids.loc[is_fsp, "wt_Na2O"], "kfsp", "plag")

    key = ["bulk", "draw", "rung"]
    total = solids.groupby(key).mode_wt.sum().rename("solid_tot")
    cat = (solids.dropna(subset=["cat"])
           .groupby(key + ["cat"]).mode_wt.sum().unstack(fill_value=0.0)
           .reindex(columns=CATEGORIES, fill_value=0.0))
    other = (solids[solids.cat.isna()]
             .groupby(key + ["phase"]).mode_wt.sum().unstack(fill_value=0.0))
    per_draw = cat.join(total).reset_index().merge(
        draws[["bulk", "draw", "sio2", "series", "qap_name"]], on=["bulk", "draw"])
    per_draw["band"] = pd.cut(per_draw.sio2, BANDS, labels=BAND_LABELS)
    return per_draw, other.reset_index().merge(
        draws[["bulk", "draw", "sio2"]], on=["bulk", "draw"])


def load_real():
    s = pd.read_csv(REAL / "samples.csv")
    m = pd.read_csv(REAL / "modes.csv")
    m["cat"] = m.magemin_phase.map(REAL_MAP)
    fsp = m.magemin_phase == "fsp"
    m.loc[fsp, "cat"] = np.where(m.loc[fsp, "phase"] == "plagioclase", "plag", "kfsp")

    dub_s = s[s.source.str.startswith("dubray") & s.sio2.notna()].set_index("sample_id")
    dub_m = m[m.source.str.startswith("dubray")]
    present = (dub_m.dropna(subset=["cat"])
               .groupby(["sample_id", "cat"]).size().unstack(fill_value=0) > 0)
    present = present.reindex(index=dub_s.index, columns=CATEGORIES, fill_value=False)
    present["band"] = pd.cut(dub_s.sio2, BANDS, labels=BAND_LABELS)
    presence = present.groupby("band", observed=True)[CATEGORIES].mean()

    ab = dub_m.dropna(subset=["cat"]).copy()
    ab["abundance_pct"] = pd.to_numeric(ab.abundance_pct, errors="coerce")
    ab = ab.dropna(subset=["abundance_pct"])
    share = (ab.groupby(["sample_id", "cat"]).abundance_pct.sum().unstack(fill_value=0.0)
             .reindex(columns=CATEGORIES, fill_value=0.0))
    share = share.div(share.sum(axis=1), axis=0).dropna()
    share["band"] = pd.cut(dub_s.sio2.reindex(share.index), BANDS, labels=BAND_LABELS)
    shares = share.groupby("band", observed=True)[CATEGORIES].mean()

    pc_s = s[s.mode_basis == "optical_vol_pointcount"].set_index("sample_id")
    pc_m = m[m.sample_id.isin(pc_s.index)].dropna(subset=["cat"]).copy()
    pc_m["abundance_pct"] = pd.to_numeric(pc_m.abundance_pct, errors="coerce")
    pc = (pc_m.dropna(subset=["abundance_pct"])
          .groupby(["sample_id", "cat"]).abundance_pct.sum().unstack(fill_value=0.0)
          .reindex(columns=CATEGORIES, fill_value=0.0))
    pc = pc.div(pc.sum(axis=1), axis=0)
    return presence, shares, pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    args = ap.parse_args()

    per_draw, other = load_sim(Path(args.corpus))
    real_presence, real_shares, pc = load_real()
    sub = per_draw[per_draw.series == "subalkaline"]
    print(f"{len(per_draw):,} sim (draw, rung) labels over "
          f"{per_draw.bulk.nunique():,} bulks; {len(sub):,} subalkaline for scoring")

    pick_rows = []
    for rung in sorted(sub.rung.unique(), reverse=True):
        g = sub[sub.rung == rung]
        row = dict(rung=rung, n=len(g))
        for cut in CUTOFFS:
            sim_p = g.groupby("band", observed=True)[CATEGORIES].agg(
                lambda x: (x > cut).mean())
            diff = (sim_p - real_presence).abs()
            row[f"presence_mae_{cut}"] = round(float(diff.mean().mean()), 4)
        shares = g[CATEGORIES].div(g.solid_tot, axis=0)
        shares["band"] = g.band.values
        sim_s = shares.groupby("band", observed=True)[CATEGORIES].mean()
        rho = [spearmanr(sim_s.loc[b], real_shares.loc[b]).statistic
               for b in BAND_LABELS if b in sim_s.index]
        row["rank_spearman_mean"] = round(float(np.mean(rho)), 3)
        pick_rows.append(row)
    pick = pd.DataFrame(pick_rows)

    print("\n=== rung pick table (owner decides) ===")
    print("presence MAE: |sim - real du Bray| presence frequency, "
          "8 phases x 3 SiO2 bands; rank: Spearman of mean crystals-"
          "renormalized shares per band")
    print(pick.to_string(index=False))

    print("\n=== per-band presence detail (cutoff 0.1 wt%) ===")
    detail = []
    for rung in sorted(sub.rung.unique(), reverse=True):
        g = sub[sub.rung == rung]
        sim_p = g.groupby("band", observed=True)[CATEGORIES].agg(
            lambda x: (x > CUTOFFS[0]).mean())
        for b in sim_p.index:
            detail.append(dict(rung=rung, band=b,
                               **{c: round(float(sim_p.loc[b, c]), 2) for c in CATEGORIES}))
    detail = pd.DataFrame(detail)
    for b in BAND_LABELS:
        print(f"\n{b} (real first):")
        real_row = real_presence.loc[b].round(2).to_frame().T.assign(rung="real")
        print(pd.concat([real_row, detail[detail.band == b].drop(columns="band")])
              .set_index("rung").to_string())

    print("\n=== sim-only phases (not in the real vocabulary), presence by rung ===")
    other_cols = [c for c in other.columns if c not in ("bulk", "draw", "rung", "sio2")]
    if other_cols:
        freq = other.groupby("rung")[other_cols].agg(lambda x: (x > CUTOFFS[0]).mean())
        print(freq.round(3).to_string())

    print("\n=== magnitude context: 13 point-counted dacites (vol%, unconverted) "
          "vs sim felsic mean wt shares ===")
    fel = per_draw[per_draw.band == "felsic"]
    mag = pd.DataFrame({"real_pointcount_mean": pc.mean()})
    for rung in sorted(fel.rung.unique(), reverse=True):
        g = fel[fel.rung == rung]
        mag[f"sim_{rung}"] = g[CATEGORIES].div(g.solid_tot, axis=0).mean()
    print(mag.round(3).to_string())
    print("(vol vs wt not converted -- ranks meaningful, magnitudes indicative)")

    RESULTS.mkdir(exist_ok=True)
    pick.to_csv(RESULTS / "rung_prescreen_pick.csv", index=False)
    detail.to_csv(RESULTS / "rung_prescreen_presence.csv", index=False)
    print(f"\nwrote results/rung_prescreen_pick.csv + rung_prescreen_presence.csv")


if __name__ == "__main__":
    main()
