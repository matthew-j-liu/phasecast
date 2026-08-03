"""Water A/B: flat-box vs solubility-bounded (P, H2O) proposal, same 36 bulks.

The controlled before/after isolating the water redesign (corpus_regen_spec
"Water A/B"). A = the committed flat-box run (work/), B = the bounded-proposal
rerun (work/bounded/, via driver.py). Two questions:
  1. Does the amp/opx switch magnitude hold at the phenocryst-regime rungs?
  2. Does biotite presence move toward real arc frequencies? (Biotite is the
     sentinel -- it under-shoots at every rung under the flat box because that
     box samples dry high-P draws real wet-arc magmas don't.)

Reads both work dirs, reuses the endpoint-repick and felsic-saturation logic,
writes results/water_ab_{switch,presence}.csv.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from explorations.archive.volcanic_proxy_definition import endpoint_repick as er
from explorations.archive.volcanic_proxy_definition import felsic_saturation as fs
RESULTS = HERE / "results"
RUNS = {"flat": HERE / "work", "bounded": HERE / "work" / "bounded"}

RUNGS = [0.60, 0.50, 0.40, 0.30]


def switch_table(run, work):
    pts = er.load_points(work)
    conv = pts.groupby("bulk").ok.mean()
    dead = conv[conv == 0].index.tolist()
    rows = []
    for melt in RUNGS:
        ep = er.pick_at_melt(pts, melt)
        ep = ep[~ep.bulk.isin(dead)]
        recs = []
        for _, g in ep.groupby("bulk"):
            r = er.classify(g)
            r["sio2_bulk"] = float(g.sio2_bulk.iloc[0])
            recs.append(r)
        v = pd.DataFrame(recs)
        inter = v[(v.sio2_bulk > 52) & (v.sio2_bulk <= 63)]
        rows.append(dict(run=run, melt_target=melt, n_bulks=len(v),
                         frac_amp_opx_switch=round(float(v.amp_opx_switch.mean()), 3),
                         frac_amp_opx_switch_intermediate=round(
                             float(inter.amp_opx_switch.mean()), 3),
                         frac_identity_multimodal=round(
                             float(v.phase_identity_multimodal.mean()), 3),
                         mean_attrition=round(float(v.attrition.mean()), 3)))
    return pd.DataFrame(rows), dead


def presence_table(run, work):
    pts = fs.load_points(work)
    conv = pts.groupby("bulk").ok.mean()
    dead = conv[conv == 0].index.tolist()
    draws = fs.per_draw(pts)
    draws = draws[~draws.bulk.isin(dead)]
    draws["band"] = pd.cut(draws.sio2_bulk, fs.BANDS, labels=fs.BAND_LABELS)
    sub = draws[draws.series == "subalkaline"]
    rows = []
    for melt in RUNGS:
        for band in fs.BAND_LABELS:
            g = sub[sub.band == band]
            rows.append(dict(run=run, melt_target=melt, band=band, n_draws=len(g),
                             **{ph: round(float(g[f"{ph}_at_{melt}"].mean()), 3)
                                for ph in fs.TRACKED}))
    return pd.DataFrame(rows)


def main():
    for run, work in RUNS.items():
        if not (work / "meta.csv").exists():
            raise SystemExit(f"missing {work}/meta.csv -- "
                             f"{'rerun driver.py' if run == 'bounded' else 'flat-box baseline gone'}")

    switch = []
    for run, work in RUNS.items():
        tab, dead = switch_table(run, work)
        switch.append(tab)
        print(f"{run}: {len(dead)} non-converged bulks excluded {dead}")
    switch = pd.concat(switch, ignore_index=True)

    print("\n=== amp/opx switch magnitude, flat vs bounded (phenocryst-regime rungs) ===")
    piv = switch.pivot_table(
        index="melt_target",
        columns="run",
        values=["frac_amp_opx_switch", "frac_amp_opx_switch_intermediate"])
    print(piv.round(3).to_string())

    presence = pd.concat([presence_table(run, work) for run, work in RUNS.items()],
                         ignore_index=True)
    real, _ = fs.real_presence()

    print("\n=== presence frequency vs real du Bray phenocrysts (subalkaline bulks) ===")
    for ph in fs.TRACKED:
        tab = presence.pivot_table(index=["band"], columns=["melt_target", "run"],
                                   values=ph, observed=True)
        tab["real"] = real[ph]
        label = "biotite (SENTINEL)" if ph == "bi" else ph
        print(f"\n{label}:\n{tab.round(2).to_string()}")

    switch.to_csv(RESULTS / "water_ab_switch.csv", index=False)
    presence.to_csv(RESULTS / "water_ab_presence.csv", index=False)
    print("\nwrote results/water_ab_switch.csv + water_ab_presence.csv")


if __name__ == "__main__":
    main()
