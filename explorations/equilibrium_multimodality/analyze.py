"""Near-solidus endpoints + the multimodality verdict (corpus_regen_spec item 7).

For each test bulk, over its N (P, H2O) draws cooled to near-solidus, ask the
pivotal question: is the equilibrium assemblage posterior multimodal in PHASE
IDENTITY (a silicate switches in/out across paths -- the heads.md amp-in/opx-in
scenario claim), or does the identity set stay fixed and only proportions /
melt-fraction / accessory phases vary?

Runs the endpoint pick at tau = 1% and 5% liq (item 2 sensitivity) and writes
results/{endpoints,bulk_verdicts,summary}.csv + figures.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
RESULTS = HERE / "results"

PRESENCE_EPS = 1e-3          # mode wt frac above which a phase counts as present
INTERMEDIATE = (0.10, 0.90)  # presence freq band that signals a within-bulk flip

# Phases whose in/out is a petrogenetic IDENTITY switch (not framework, not
# melt/fluid, not an oxide accessory). amp (hydrous) vs opx (anhydrous) is the
# exact heads.md View-2 claim.
SILICATE_IDENTITY = ["amp", "opx", "bi", "g", "ol", "ne", "mu"]
FRAMEWORK = ["fsp", "cpx"]
ACCESSORY = ["ilm", "spl", "mag", "ru", "ap"]
MELT_FLUID = ["liq", "fl"]


def load_points():
    meta = pd.read_csv(WORK / "meta.csv")
    rows = []
    for f in sorted((WORK / "raw").glob("bulk_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            ph = {}
            for p in r["phases"]:
                ph[p["name"]] = ph.get(p["name"], 0.0) + float(p["mode_wt"])  # merge fsp/cpx dups
            rows.append(dict(sid=r["sid"], status=r["status"], G=r["G"],
                             liq=ph.get("liq", 0.0), phases=ph))
    pts = pd.DataFrame(rows).merge(meta, on="sid", how="inner")
    pts["ok"] = (pts.status == 0) & (pts.G < 0)
    return pts


def pick_endpoints(pts, tau):
    """One row per (bulk, draw): assemblage at the near-solidus endpoint.

    Highest-T CONVERGED point with liq mass frac < tau; if none, the coldest
    converged point, flagged attrition. Draws with no converged point are dropped.
    """
    out = []
    for (bulk, draw), g in pts[pts.ok].groupby(["bulk", "draw"]):
        g = g.sort_values("T_celsius", ascending=False)
        sub = g[g.liq < tau]
        if len(sub):
            row, attr = sub.iloc[0], False
        else:
            row, attr = g.iloc[-1], True
        ph = dict(row["phases"])
        tot = sum(ph.values())
        ph = {k: v / tot for k, v in ph.items() if tot > 0 and v / tot > PRESENCE_EPS}
        out.append(dict(bulk=bulk, draw=int(draw), qap_name=row.qap_name, series=row.series,
                        sio2_bulk=row.sio2_bulk, P=row.P_kbar, H2O=row.H2O,
                        T_end=row.T_celsius, attrition=attr, liq_end=row.liq, asm=ph))
    return pd.DataFrame(out)


def presence_freq(ep_bulk):
    """Per-phase presence frequency over one bulk's endpoint draws."""
    phases = set()
    for a in ep_bulk.asm:
        phases.update(a)
    n = len(ep_bulk)
    return {p: float(np.mean([1.0 if a.get(p, 0) > PRESENCE_EPS else 0.0 for a in ep_bulk.asm]))
            for p in phases}, n


def classify_bulk(ep_bulk):
    """Verdict for one bulk: which phases flip, and is any a silicate identity."""
    freq, n = presence_freq(ep_bulk)
    lo, hi = INTERMEDIATE
    intermediate = {p: f for p, f in freq.items() if lo <= f <= hi}
    id_flips = {p: freq[p] for p in SILICATE_IDENTITY if p in intermediate}

    # amp<->opx anti-correlation across draws (the exact heads.md switch)
    amp = np.array([1 if a.get("amp", 0) > PRESENCE_EPS else 0 for a in ep_bulk.asm])
    opx = np.array([1 if a.get("opx", 0) > PRESENCE_EPS else 0 for a in ep_bulk.asm])
    both = int(((amp == 1) & (opx == 1)).sum())
    amp_only = int(((amp == 1) & (opx == 0)).sum())
    opx_only = int(((amp == 0) & (opx == 1)).sum())
    neither = int(((amp == 0) & (opx == 0)).sum())
    amp_opx_switch = (amp.mean() not in (0.0, 1.0)) and (opx.mean() not in (0.0, 1.0))

    return dict(
        n_draws=n,
        attrition=float(ep_bulk.attrition.mean()),
        liq_end_med=float(ep_bulk.liq_end.median()),
        T_end_spread=float(ep_bulk.T_end.max() - ep_bulk.T_end.min()),
        amp_freq=freq.get("amp", 0.0), opx_freq=freq.get("opx", 0.0),
        n_intermediate=len(intermediate),
        n_identity_flips=len(id_flips),
        identity_flips=";".join(f"{p}:{freq[p]:.2f}" for p in id_flips) or "-",
        phase_identity_multimodal=len(id_flips) > 0,
        amp_opx_switch=bool(amp_opx_switch),
        amp_only=amp_only, opx_only=opx_only, both_amp_opx=both, neither=neither,
    )


def run_tau(pts, tau):
    ep = pick_endpoints(pts, tau)
    verdicts = []
    for bulk, g in ep.groupby("bulk"):
        v = dict(bulk=bulk, qap_name=g.qap_name.iloc[0], series=g.series.iloc[0],
                 sio2_bulk=round(float(g.sio2_bulk.iloc[0]), 1), tau=tau)
        v.update(classify_bulk(g))
        verdicts.append(v)
    return ep, pd.DataFrame(verdicts).sort_values(["series", "sio2_bulk"]).reset_index(drop=True)


def main():
    pts = load_points()
    n_bulks = pts.bulk.nunique()
    conv = pts.groupby("bulk").ok.mean()
    dead = conv[conv == 0].index.tolist()
    print(f"loaded {len(pts):,} points, {n_bulks} bulks; "
          f"{len(dead)} fully non-converged (excluded): {dead}")

    summary_rows = []
    for tau in (0.01, 0.05):
        ep, ver = run_tau(pts, tau)
        ep.drop(columns="asm").to_csv(RESULTS / f"endpoints_tau{int(tau*100):02d}.csv", index=False)
        ver.to_csv(RESULTS / f"bulk_verdicts_tau{int(tau*100):02d}.csv", index=False)
        live = ver[~ver.bulk.isin(dead)]
        frac_mm = live.phase_identity_multimodal.mean()
        frac_switch = live.amp_opx_switch.mean()
        print(f"\n=== tau = {tau:.0%} liq  ({len(live)} converged bulks) ===")
        print(f"phase-identity-multimodal: {live.phase_identity_multimodal.sum()}/{len(live)} "
              f"= {frac_mm:.0%}    amp<->opx switch: {live.amp_opx_switch.sum()}/{len(live)} = {frac_switch:.0%}")
        print(f"mean attrition (draws not reaching liq<tau): {live.attrition.mean():.1%}")
        by_series = live.groupby("series").phase_identity_multimodal.mean().round(2)
        print("multimodal fraction by series:", by_series.to_dict())
        cols = ["qap_name", "sio2_bulk", "series", "amp_freq", "opx_freq",
                "n_identity_flips", "identity_flips", "amp_opx_switch", "attrition"]
        with pd.option_context("display.width", 200, "display.max_rows", 100):
            print(live[cols].round(2).to_string(index=False))
        summary_rows.append(dict(tau=tau, n_bulks=len(live),
                                 frac_phase_identity_multimodal=round(frac_mm, 3),
                                 frac_amp_opx_switch=round(frac_switch, 3),
                                 mean_attrition=round(float(live.attrition.mean()), 3)))
        if tau == 0.01:
            make_figs(ep, live)

    pd.DataFrame(summary_rows).to_csv(RESULTS / "summary.csv", index=False)
    print("\nwrote results/{endpoints,bulk_verdicts,summary}*.csv + figures")


def make_figs(ep, ver):
    # 1. amp & opx presence frequency vs bulk SiO2, colored by series
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for series, mk in [("subalkaline", "o"), ("alkaline", "^")]:
        s = ver[ver.series == series]
        ax[0].scatter(s.sio2_bulk, s.amp_freq, marker=mk, s=55, label=f"amp ({series})", alpha=0.8)
        ax[0].scatter(s.sio2_bulk, s.opx_freq, marker=mk, s=55, facecolors="none",
                      edgecolors="C1", label=f"opx ({series})", alpha=0.8)
    ax[0].axhspan(*INTERMEDIATE, color="grey", alpha=0.12)
    ax[0].set_xlabel("bulk SiO2 wt%"); ax[0].set_ylabel("presence frequency over draws")
    ax[0].set_title("amp (filled) vs opx (open) presence\ngrey band = within-bulk flip")
    ax[0].legend(fontsize=6, ncol=2); ax[0].set_ylim(-0.05, 1.05)

    # 2. number of silicate-identity flips per bulk
    v = ver.sort_values("sio2_bulk")
    colors = ["#d62728" if m else "#1f77b4" for m in v.phase_identity_multimodal]
    ax[1].barh(range(len(v)), v.n_identity_flips, color=colors)
    ax[1].set_yticks(range(len(v)))
    ax[1].set_yticklabels([f"{q[:12]} {s:.0f}" for q, s in zip(v.qap_name, v.sio2_bulk)], fontsize=6)
    ax[1].set_xlabel("# silicate-identity phases with intermediate presence")
    ax[1].set_title("phase-identity multimodality per bulk\n(red = multimodal)")
    fig.tight_layout(); fig.savefig(RESULTS / "multimodality.png", dpi=120)

    # 3. degree-of-melting axis: T_end and liq_end spread per bulk (the alternative
    # structure that DOES survive equilibrium)
    fig2, ax2 = plt.subplots(figsize=(8, 4.4))
    for bulk, g in ep.groupby("bulk"):
        ax2.scatter([g.sio2_bulk.iloc[0]] * len(g), g.T_end, s=6, alpha=0.25, color="grey")
    med = ep.groupby("sio2_bulk").T_end.median()
    ax2.plot(med.index, med.values, "r.-", label="median near-solidus T_end")
    ax2.set_xlabel("bulk SiO2 wt%"); ax2.set_ylabel("near-solidus T_end (C)")
    ax2.set_title("degree-of-melting axis: T_end spread across (P,H2O) draws")
    ax2.legend(); fig2.tight_layout(); fig2.savefig(RESULTS / "melting_axis.png", dpi=120)


if __name__ == "__main__":
    main()
