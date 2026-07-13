"""Re-pick every draw's endpoint on a melt-fraction ladder to settle Task A.

The open question (proxy README, Task A): read the assemblage at near-solidus
(1-5% residual melt) or at an eruptible-crystallinity endpoint (~50% melt)?
Gemini argues near-solidus over-crystallizes and generates plutonic assemblages;
ChatGPT argues it is a fine identifiable reference state. Both agree the physics
is testable on data we already have.

The entire cooling ladder for every (P, H2O) draw is already stored in
equilibrium_multimodality/work/ (raw MAGEMin JSONL, gitignored, regenerable via
that exploration's driver.py). So we re-pick each draw's endpoint at a ladder of
residual-melt targets and ask, at each:
  (a) does the amp<->opx identity switch (the heads.md headline) survive?
  (b) how do the modal proportions and the phase list change?
  (c) where does residual-liq-as-glass become unphysical (silicic, low melt)?

No new MAGEMin run. Reads ../equilibrium_multimodality/work, writes results/.
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
RESULTS = HERE / "results"

PRESENCE_EPS = 1e-3   # mode wt frac above which a phase counts as present (matches
                      # equilibrium_multimodality/analyze.py so numbers are comparable)

# Residual-melt targets from eruptible lava (Gemini) down to near-solidus (ChatGPT).
# Rheological lock-up erupts at ~45-60 vol% crystals (Marsh 1981, Scaillet 1998),
# i.e. ~40-55% melt; 1% is the multimodality exploration's near-solidus rule.
# 0.70/0.60 added later: real phenocryst totals (du Bray ~10-30 vol%) put the
# median storage state shallower than 50% crystals, so probe that side too.
MELT_LADDER = [0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.01]

SILICATE_IDENTITY = ["amp", "opx", "bi", "g", "ol", "ne", "mu"]  # petrogenetic in/out switches
INTERMEDIATE = (0.10, 0.90)  # presence-freq band that signals a within-bulk flip


def load_points():
    """One row per MAGEMin point: (bulk, draw, T), liq frac, phase modes, liq SiO2."""
    meta = pd.read_csv(WORK / "meta.csv")
    rows = []
    for f in sorted((WORK / "raw").glob("bulk_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            ph, liq_sio2 = {}, np.nan
            for p in r["phases"]:
                ph[p["name"]] = ph.get(p["name"], 0.0) + float(p["mode_wt"])  # merge fsp/cpx dups
                if p["name"] == "liq":
                    liq_sio2 = float(p["comp_wt"].get("SiO2", np.nan))
            rows.append(dict(sid=r["sid"], status=r["status"], G=r["G"],
                             liq=ph.get("liq", 0.0), liq_sio2=liq_sio2, phases=ph))
    pts = pd.DataFrame(rows).merge(meta, on="sid", how="inner")
    pts["ok"] = (pts.status == 0) & (pts.G < 0)
    return pts


def pick_at_melt(pts, melt):
    """One row per (bulk, draw): assemblage at the highest-T converged point whose
    residual melt has dropped below `melt`. If a draw never crystallizes that far
    (stays melt-rich at the cooling floor), take the coldest converged point and
    flag it -- at high melt targets this is ~never; near-solidus it is the wet tail.
    """
    out = []
    for (bulk, draw), g in pts[pts.ok].groupby(["bulk", "draw"]):
        g = g.sort_values("T_celsius", ascending=False)
        sub = g[g.liq < melt]
        row, attr = (sub.iloc[0], False) if len(sub) else (g.iloc[-1], True)
        ph = dict(row["phases"])
        tot = sum(ph.values())
        cryst = {k: v / tot for k, v in ph.items()
                 if k not in ("liq", "fl") and tot > 0 and v / tot > PRESENCE_EPS}
        out.append(dict(bulk=bulk, draw=int(draw), qap_name=row.qap_name, series=row.series,
                        sio2_bulk=row.sio2_bulk, P=row.P_kbar, H2O=row.H2O, T_end=row.T_celsius,
                        attrition=attr, liq_end=row.liq, liq_sio2=row.liq_sio2,
                        n_cryst_phases=len(cryst), asm=cryst))
    return pd.DataFrame(out)


def amp_opx_status(ep_bulk):
    amp = np.array([1 if a.get("amp", 0) > PRESENCE_EPS else 0 for a in ep_bulk.asm])
    opx = np.array([1 if a.get("opx", 0) > PRESENCE_EPS else 0 for a in ep_bulk.asm])
    switch = (amp.mean() not in (0.0, 1.0)) and (opx.mean() not in (0.0, 1.0))
    return dict(amp_freq=float(amp.mean()), opx_freq=float(opx.mean()),
                amp_only=int(((amp == 1) & (opx == 0)).sum()),
                opx_only=int(((amp == 0) & (opx == 1)).sum()),
                both=int(((amp == 1) & (opx == 1)).sum()),
                neither=int(((amp == 0) & (opx == 0)).sum()),
                amp_opx_switch=bool(switch))


def classify(ep_bulk):
    """Per-bulk verdict at one endpoint: identity flips + amp/opx switch + modes."""
    phases = set().union(*ep_bulk.asm) if len(ep_bulk) else set()
    lo, hi = INTERMEDIATE
    freq = {p: float(np.mean([1.0 if a.get(p, 0) > PRESENCE_EPS else 0.0 for a in ep_bulk.asm]))
            for p in phases}
    id_flips = {p: freq[p] for p in SILICATE_IDENTITY if lo <= freq.get(p, 0.0) <= hi}
    v = dict(n_draws=len(ep_bulk),
             attrition=float(ep_bulk.attrition.mean()),
             liq_end_med=float(ep_bulk.liq_end.median()),
             T_end_med=float(ep_bulk.T_end.median()),
             n_cryst_med=float(ep_bulk.n_cryst_phases.median()),
             liq_sio2_med=float(ep_bulk.liq_sio2.median()),
             n_identity_flips=len(id_flips),
             identity_flips=";".join(f"{p}:{freq[p]:.2f}" for p in id_flips) or "-",
             phase_identity_multimodal=len(id_flips) > 0)
    v.update(amp_opx_status(ep_bulk))
    return v


def run_ladder(pts, dead):
    per_bulk, summary = [], []
    for melt in MELT_LADDER:
        ep = pick_at_melt(pts, melt)
        rows = []
        for bulk, g in ep.groupby("bulk"):
            r = dict(melt_target=melt, bulk=bulk, qap_name=g.qap_name.iloc[0],
                     series=g.series.iloc[0], sio2_bulk=round(float(g.sio2_bulk.iloc[0]), 1))
            r.update(classify(g))
            rows.append(r)
        df = pd.DataFrame(rows)
        live = df[~df.bulk.isin(dead)]
        per_bulk.append(live)
        band = pd.cut(live.sio2_bulk, [0, 52, 63, 100], labels=["mafic", "intermediate", "felsic"])
        summary.append(dict(
            melt_target=melt,
            n_bulks=len(live),
            frac_amp_opx_switch=round(float(live.amp_opx_switch.mean()), 3),
            frac_amp_opx_switch_intermediate=round(float(live[band == "intermediate"].amp_opx_switch.mean()), 3),
            frac_phase_identity_multimodal=round(float(live.phase_identity_multimodal.mean()), 3),
            mean_n_cryst_phases=round(float(live.n_cryst_med.mean()), 2),
            mean_T_end=round(float(live.T_end_med.mean()), 0),
            mean_attrition=round(float(live.attrition.mean()), 3),
            felsic_liq_sio2_med=round(float(live[band == "felsic"].liq_sio2_med.median()), 1),
        ))
    return pd.concat(per_bulk, ignore_index=True), pd.DataFrame(summary)


def make_figs(per_bulk, summary):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    x = summary.melt_target * 100

    ax[0].plot(x, summary.frac_amp_opx_switch, "o-", label="all bulks")
    ax[0].plot(x, summary.frac_amp_opx_switch_intermediate, "s--", label="intermediate 52-63% SiO2")
    ax[0].plot(x, summary.frac_phase_identity_multimodal, "^:", color="grey", label="any silicate flip")
    ax[0].set_xlabel("residual melt at endpoint (%)"); ax[0].set_ylabel("fraction of bulks")
    ax[0].set_title("does the amp<->opx switch survive the endpoint?")
    ax[0].invert_xaxis(); ax[0].legend(fontsize=7); ax[0].set_ylim(-0.03, 1.03)
    ax[0].axvspan(40, 55, color="green", alpha=0.08)  # eruptible window

    ax[1].plot(x, summary.mean_n_cryst_phases, "o-", color="C2")
    ax[1].set_xlabel("residual melt at endpoint (%)"); ax[1].set_ylabel("mean # crystalline phases")
    ax[1].set_title("assemblage richness vs endpoint\n(near-solidus grows the phase list)")
    ax[1].invert_xaxis(); ax[1].axvspan(40, 55, color="green", alpha=0.08)

    ax[2].plot(x, summary.felsic_liq_sio2_med, "o-", color="C3")
    ax[2].axhline(78, ls=":", color="grey"); ax[2].text(48, 78.4, "~rhyolite glass ceiling", fontsize=7)
    ax[2].set_xlabel("residual melt at endpoint (%)"); ax[2].set_ylabel("median residual-liq SiO2 (felsic bulks)")
    ax[2].set_title("(c) liq-as-glass sanity in silicic bulks\n(runaway SiO2 = unphysical near-solidus melt)")
    ax[2].invert_xaxis(); ax[2].axvspan(40, 55, color="green", alpha=0.08)

    fig.tight_layout(); fig.savefig(RESULTS / "endpoint_ladder.png", dpi=120)


def main():
    pts = load_points()
    conv = pts.groupby("bulk").ok.mean()
    dead = conv[conv == 0].index.tolist()
    print(f"loaded {len(pts):,} points, {pts.bulk.nunique()} bulks; "
          f"{len(dead)} non-converged (excluded): {dead}")

    per_bulk, summary = run_ladder(pts, dead)
    per_bulk.drop(columns=[], errors="ignore").to_csv(RESULTS / "endpoint_ladder_by_bulk.csv", index=False)
    summary.to_csv(RESULTS / "endpoint_ladder_summary.csv", index=False)

    print("\n=== amp<->opx switch and assemblage vs residual-melt endpoint ===")
    with pd.option_context("display.width", 200):
        print(summary.to_string(index=False))

    # amp/opx switch per bulk across the ladder (does the headline move with the endpoint?)
    print("\n=== per-bulk amp<->opx switch across the ladder (intermediate band) ===")
    inter = per_bulk[(per_bulk.sio2_bulk > 52) & (per_bulk.sio2_bulk <= 63)]
    piv = inter.pivot_table(index=["qap_name", "sio2_bulk", "bulk"],
                            columns="melt_target", values="amp_opx_switch")
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(piv.to_string())

    make_figs(per_bulk, summary)
    print("\nwrote results/endpoint_ladder_{summary,by_bulk}.csv + endpoint_ladder.png")


if __name__ == "__main__":
    main()
