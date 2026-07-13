"""Does a phenocryst-regime endpoint saturate the felsic diagnostic phases?

The granularity EDA (explorations/real_mineralogy/eda) shows quartz, alkali
feldspar, and biotite are felsic-restricted phenocrysts in real rocks. The
residual question for the endpoint decision: at an eruptible-crystallinity
endpoint (50-30% residual melt), do felsic bulks already saturate those phases
in the equilibrium cooling ladders, or do they only appear near-solidus --
which would argue for a composition-conditional (deeper felsic) stop?

Reads the stored ladders in ../equilibrium_multimodality/work (gitignored,
regenerable via that exploration's driver.py) plus the real benchmark in
data/real_mineralogy/. Writes results/. No new MAGEMin run.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
WORK = HERE.parent / "equilibrium_multimodality" / "work"
REAL = HERE.parent.parent / "data" / "real_mineralogy"
RESULTS = HERE / "results"

PRESENCE_EPS = 1e-3          # matches endpoint_repick.py / analyze.py
MELT_LADDER = [0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.01]
BANDS = [0, 52, 63, 100]
BAND_LABELS = ["mafic", "intermediate", "felsic"]

# kfsp = any feldspar instance that is K-dominant by weight (sanidine proxy;
# duplicates in the raw phase list are the two coexisting feldspars)
TRACKED = ["q", "bi", "kfsp"]


def load_points():
    meta = pd.read_csv(WORK / "meta.csv")
    rows = []
    for f in sorted((WORK / "raw").glob("bulk_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            liq = q = bi = 0.0
            kfsp = False
            for p in r["phases"]:
                mode = float(p["mode_wt"])
                if p["name"] == "liq":
                    liq += mode
                elif p["name"] == "q":
                    q += mode
                elif p["name"] == "bi":
                    bi += mode
                elif p["name"] == "fsp" and mode > PRESENCE_EPS:
                    c = p["comp_wt"]
                    kfsp = kfsp or c.get("K2O", 0) > c.get("Na2O", 0)
            # q_mode/bi_mode: meta.csv already owns `q` (the proposal density)
            rows.append(dict(sid=r["sid"], status=r["status"], G=r["G"],
                             liq=liq, q_mode=q, bi_mode=bi, kfsp=kfsp))
    pts = pd.DataFrame(rows).merge(meta, on="sid", how="inner")
    pts["ok"] = (pts.status == 0) & (pts.G < 0)
    return pts


def per_draw(pts):
    """One row per (bulk, draw): presence at each ladder endpoint + the melt
    fraction at which each tracked phase first saturates while cooling."""
    out = []
    for (bulk, draw), g in pts[pts.ok].groupby(["bulk", "draw"]):
        g = g.sort_values("T_celsius", ascending=False)
        present = dict(q=g.q_mode > PRESENCE_EPS, bi=g.bi_mode > PRESENCE_EPS,
                       kfsp=g.kfsp)
        row = dict(bulk=bulk, draw=int(draw), sio2_bulk=float(g.sio2_bulk.iloc[0]),
                   series=g.series.iloc[0])
        for ph in TRACKED:
            first = g.liq[present[ph]]
            row[f"{ph}_in_liq"] = float(first.iloc[0]) if len(first) else np.nan
        for melt in MELT_LADDER:
            sub = g[g.liq < melt]
            ep = sub.iloc[0] if len(sub) else g.iloc[-1]
            for ph in TRACKED:
                row[f"{ph}_at_{melt}"] = bool(present[ph].loc[ep.name])
        out.append(row)
    return pd.DataFrame(out)


def real_presence():
    """du Bray phenocryst presence frequency per SiO2 band (incl. trace rows)."""
    s = pd.read_csv(REAL / "samples.csv")
    m = pd.read_csv(REAL / "modes.csv")
    s = s[s.source.str.startswith("dubray") & s.sio2.notna()]
    m = m[m.source.str.startswith("dubray")]
    name_map = {"quartz": "q", "silica phases": "q", "biotite": "bi",
                "alkali feldspar": "kfsp"}
    m = m[m.phase.isin(name_map)].assign(ph=lambda d: d.phase.map(name_map))
    has = m.groupby(["sample_id", "ph"]).size().unstack(fill_value=0) > 0
    s = s.set_index("sample_id").join(has)
    for ph in TRACKED:
        s[ph] = s[ph].eq(True)
    s["band"] = pd.cut(s.sio2, BANDS, labels=BAND_LABELS)
    return s.groupby("band", observed=True)[TRACKED].mean(), s.band.value_counts()


def main():
    pts = load_points()
    conv = pts.groupby("bulk").ok.mean()
    dead = conv[conv == 0].index.tolist()
    draws = per_draw(pts)
    draws = draws[~draws.bulk.isin(dead)]
    draws["band"] = pd.cut(draws.sio2_bulk, BANDS, labels=BAND_LABELS)
    print(f"{len(draws):,} draws over {draws.bulk.nunique()} bulks "
          f"({len(dead)} non-converged excluded)")

    # du Bray is a subalkaline arc suite; the stratified sim bulks are half
    # alkaline by design, so the apples-to-apples row is subalkaline-only
    freq = draws.groupby(["series", "band"], observed=True)[
        [f"{ph}_at_{m}" for m in MELT_LADDER for ph in TRACKED]].mean()
    real, real_n = real_presence()

    print("\n=== sim presence frequency at each endpoint vs real du Bray phenocrysts"
          " (subalkaline sim bulks) ===")
    for ph in TRACKED:
        tab = freq.loc["subalkaline", [f"{ph}_at_{m}" for m in MELT_LADDER]].copy()
        tab.columns = [f"melt={m}" for m in MELT_LADDER]
        tab["real"] = real[ph]
        print(f"\n{ph}:\n{tab.round(2).to_string()}")

    print("\n=== melt fraction at first saturation (felsic bulks, per-draw) ===")
    fel = draws[draws.band == "felsic"]
    for ph in TRACKED:
        x = fel[f"{ph}_in_liq"]
        print(f"{ph}: never-in {x.isna().mean():.0%}, saturation melt frac "
              f"p90={np.nanpercentile(x, 90):.2f} median={np.nanmedian(x):.2f} "
              f"p10={np.nanpercentile(x, 10):.2f}")

    freq.to_csv(RESULTS / "felsic_saturation_by_band.csv")
    sub = draws[draws.series == "subalkaline"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, ph in zip(axes, TRACKED):
        for band in BAND_LABELS:
            y = [sub[sub.band == band][f"{ph}_at_{m}"].mean() for m in MELT_LADDER]
            ax.plot([m * 100 for m in MELT_LADDER], y, "o-", label=band)
            ax.axhline(real.loc[band, ph], ls=":", alpha=0.5,
                       color=ax.lines[-1].get_color())
        ax.set_title(f"{ph} presence vs endpoint, subalkaline sim\n"
                     "(dotted = real du Bray phenocryst freq)")
        ax.set_xlabel("residual melt at endpoint (%)")
        ax.invert_xaxis()
        ax.axvspan(40, 55, color="green", alpha=0.08)
    axes[0].set_ylabel("fraction of draws / samples with phase present")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "felsic_saturation.png", dpi=120)
    print("\nwrote results/felsic_saturation_by_band.csv + felsic_saturation.png")


if __name__ == "__main__":
    main()
