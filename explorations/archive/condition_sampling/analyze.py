"""Analyze the (P,H2O) sampling illustration for gard 6875.

Reads mc_out.jsonl (per-point MAGEMin) + mc_meta.csv, finds the near-solidus
endpoint per (P,H2O) path, then:
  - builds the per-bulk assemblage cloud for the MC design
  - clusters it into scenarios and reports marginals
  - importance-reweights the uniform-q cloud to example priors (w = p/q), with ESS
  - compares the marginal across MC vs three product grids (the density artifact)
  - shows error-bar vs N (bootstrap) and ESS vs prior concentration
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LIQ_THRESH = 0.02   # near-solidus endpoint: highest T with liq mass frac < 2%

# proposal box (must match gen_inputs.py)
P_LO, P_HI = 1.0, 13.0
W_LO, W_HI = 0.5, 8.0


def load():
    meta = pd.read_csv("mc_meta.csv").set_index("sample_id")
    recs = {}
    with open("mc_out.jsonl") as f:
        for line in f:
            r = json.loads(line)
            ph = {}
            for nm, d in r["phases"].items():
                ph[nm] = ph.get(nm, 0.0) + float(d["mode"])   # merge fsp/cpx duplicates
            recs[int(r["sid"])] = ph
    meta["phases"] = meta.index.map(recs)
    meta["liq"] = meta["phases"].map(lambda p: p.get("liq", 0.0))
    return meta


def endpoints(meta):
    """One row per (design,draw): assemblage at the near-solidus endpoint."""
    out = []
    for (design, draw), g in meta.groupby(["design", "draw"]):
        g = g.sort_values("T_celsius", ascending=False)
        sub = g[g.liq < LIQ_THRESH]
        if len(sub):
            row = sub.iloc[0]; attr = False           # highest T below liq threshold
        else:
            row = g.iloc[-1]; attr = True             # never crossed -> coldest, flag
        ph = dict(row["phases"])
        tot = sum(ph.values())
        ph = {k: v / tot for k, v in ph.items() if v / tot > 1e-4}
        out.append(dict(design=design, draw=int(draw), P=row.P_kbar, H2O=row.H2O,
                        T_end=row.T_celsius, attrition=attr, asm=ph))
    return pd.DataFrame(out)


PHASES = ["liq", "fl", "fsp", "cpx", "opx", "amp", "ol", "spl", "ilm", "bi", "mu", "g"]


def modal_matrix(df):
    M = np.zeros((len(df), len(PHASES)))
    for i, asm in enumerate(df.asm.values):
        for j, p in enumerate(PHASES):
            M[i, j] = asm.get(p, 0.0)
    return M


def presence(df, w=None):
    M = modal_matrix(df)
    pres = (M > 1e-3).astype(float)
    if w is None:
        w = np.ones(len(df))
    w = w / w.sum()
    return dict(zip(PHASES, pres.T @ w))


def ess(w):
    w = np.asarray(w, float)
    return w.sum() ** 2 / np.sum(w ** 2)


def prior_weights(df, kind):
    """w_i = p(C_i)/q(C_i); q is uniform on the box so q is constant and w ~ p(C_i)."""
    P, W = df.P.values, df.H2O.values
    if kind == "uniform":              # p == q
        return np.ones(len(df))
    if kind == "wet_arc":              # deep & wet emphasis
        return np.exp(-0.5 * (((W - 5.0) / 1.2) ** 2 + ((P - 8.0) / 3.0) ** 2))
    if kind == "lit_andesite":         # ADR p25-p75 box, hard support
        return ((P >= 2) & (P <= 10) & (W >= 2.8) & (W <= 4.8)).astype(float)
    if kind == "dry_shallow":
        return np.exp(-0.5 * (((W - 1.5) / 0.8) ** 2 + ((P - 3.0) / 2.0) ** 2))
    raise ValueError(kind)


def main():
    meta = load()
    ep = endpoints(meta)
    ep.to_csv("endpoints.csv", index=False)
    mc = ep[ep.design == "mc256"].reset_index(drop=True)

    n_attr = int(ep.attrition.sum())
    print(f"endpoints: {len(ep)} paths, {n_attr} hit ladder floor (attrition)")
    print(f"mc256: {len(mc)} paths, T_end {mc.T_end.min():.0f}-{mc.T_end.max():.0f} C\n")

    print("=== marginal presence freq: MC(uniform-q) vs product grids ===")
    cols = {"mc256(q)": presence(mc),
            "gridU 8x8": presence(ep[ep.design == "gridU"]),
            "gridP 16x4": presence(ep[ep.design == "gridP"]),
            "gridW 4x16": presence(ep[ep.design == "gridW"])}
    tab = pd.DataFrame(cols).round(3)
    tab = tab[(tab.T > 0.01).any()]
    print(tab.to_string())
    print()

    print("=== same MC cloud, reweighted to priors (defer prior to inference) ===")
    pri = {}
    for k in ["uniform", "lit_andesite", "wet_arc", "dry_shallow"]:
        w = prior_weights(mc, k)
        pri[f"{k} (ESS={ess(w):.0f})"] = presence(mc, w)
    ptab = pd.DataFrame(pri).round(3)
    ptab = ptab[(ptab.T > 0.01).any()]
    print(ptab.to_string())
    print()

    # scenario clustering on the modal vector. NB: at near-solidus EQUILIBRIUM
    # (the single-point proxy used here) this wet andesite is amphibole-saturated
    # across the whole envelope, so the amph-vs-opx switch that production FC would
    # expose is muted; the structure that survives is melt/fluid-bearing vs dry.
    from sklearn.cluster import KMeans
    M = modal_matrix(mc)
    Z = (M - M.mean(0)) / (M.std(0) + 1e-9)
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Z)
    mc = mc.copy(); mc["k"] = km.labels_
    # name clusters by their dominant discriminating phases
    names = {}
    for k, g in mc.groupby("k"):
        a = modal_matrix(g).mean(0)
        top = ", ".join(f"{PHASES[j]} {a[j]:.2f}" for j in np.argsort(a)[::-1][:3])
        names[k] = top
    print("=== scenarios from clustering the MC cloud (uniform-q weights) ===")
    for k, g in mc.groupby("k"):
        print(f"cluster {k}  weight {len(g)/len(mc):.2f}  "
              f"P {g.P.mean():4.1f}+/-{g.P.std():.1f} kbar  "
              f"H2O {g.H2O.mean():.1f}+/-{g.H2O.std():.1f} wt%  "
              f"T_end {g.T_end.mean():.0f} C  liqfrac {modal_matrix(g)[:,0].mean():.2f}  | {names[k]}")
    print()

    print("=== scenario (cluster) weights under different priors (reweight, no new MAGEMin) ===")
    for kind in ["uniform", "lit_andesite", "wet_arc", "dry_shallow"]:
        w = prior_weights(mc, kind); w = w / w.sum()
        wsum = mc.assign(w=w).groupby("k").w.sum()
        print(f"{kind:14s} " + "  ".join(f"c{k}={wsum.get(k,0):.2f}" for k in sorted(mc.k.unique())))
    print()

    # error-bar vs N: ilmenite presence (intermediate freq -> real variance) and
    # the continuous residual-melt fraction; uniform-q vs a concentrated prior.
    print("=== error bar vs N (bootstrap) ===")
    rng = np.random.default_rng(7)
    Milm = (modal_matrix(mc)[:, PHASES.index("ilm")] > 1e-3).astype(float)
    liqf = modal_matrix(mc)[:, PHASES.index("liq")]
    wlit = prior_weights(mc, "lit_andesite")
    print(f"{'N':>4} {'ilm SE (q)':>11} {'ilm SE (lit)':>13} {'liqfrac SE (q)':>15} "
          f"{'median ESS(lit)':>16} {'frac empty':>11}")
    for N in [8, 16, 32, 64, 128, 256]:
        su, sp, sl, esl, empty = [], [], [], [], 0
        for _ in range(600):
            idx = rng.integers(0, len(mc), N)
            su.append(Milm[idx].mean()); sl.append(liqf[idx].mean())
            wl = wlit[idx]; s = wl.sum()
            if s > 0:
                sp.append((Milm[idx] * wl).sum() / s); esl.append(ess(wl))
            else:
                empty += 1
        print(f"{N:>4} {np.std(su):>11.3f} {np.nanstd(sp):>13.3f} {np.std(sl):>15.3f} "
              f"{(np.median(esl) if esl else float('nan')):>16.1f} {empty/600:>11.2f}")

    make_plots(mc, ep)
    print("\nwrote cloud.png, marginals.png, nscaling.png")


def make_plots(mc, ep):
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for k, g in mc.groupby("k"):
        ax[0].scatter(g.H2O, g.P, s=22, c=palette[k % len(palette)], label=f"cluster {k}",
                      alpha=0.8, edgecolor="none")
    ax[0].set_xlabel("H2O wt% (sampled)"); ax[0].set_ylabel("P kbar (sampled)")
    ax[0].set_title("MC cloud over q, colored by cluster\n(gard 6875 andesite, N=256)")
    ax[0].legend(fontsize=7, loc="upper left"); ax[0].invert_yaxis()
    sc = ax[1].scatter(mc.H2O, mc.P, s=22, c=mc.T_end, cmap="viridis")
    ax[1].set_xlabel("H2O wt%"); ax[1].set_ylabel("P kbar")
    ax[1].set_title("near-solidus endpoint T_end (C)"); ax[1].invert_yaxis()
    plt.colorbar(sc, ax=ax[1]); fig.tight_layout(); fig.savefig("cloud.png", dpi=110)

    cols = {"MC(q)": presence(mc), "gridU 8x8": presence(ep[ep.design == "gridU"]),
            "gridP 16x4": presence(ep[ep.design == "gridP"]), "gridW 4x16": presence(ep[ep.design == "gridW"])}
    tab = pd.DataFrame(cols)
    tab = tab[(tab.T > 0.02).any()]
    fig, axb = plt.subplots(figsize=(8, 4))
    tab.plot.bar(ax=axb); axb.set_ylabel("presence frequency")
    axb.set_title("marginal presence: MC vs product grids (same support, same N=64)")
    axb.tick_params(axis="x", rotation=0); fig.tight_layout(); fig.savefig("marginals.png", dpi=110)

    rng = np.random.default_rng(7)
    Milm = (modal_matrix(mc)[:, PHASES.index("ilm")] > 1e-3).astype(float)
    p = Milm.mean()
    Ns = [8, 16, 32, 64, 128, 256]
    se = [np.std([Milm[rng.integers(0, len(mc), N)].mean() for _ in range(600)]) for N in Ns]
    fig, axn = plt.subplots(figsize=(5.5, 4))
    axn.plot(Ns, se, "o-", label="bootstrap SE (ilm presence)")
    axn.plot(Ns, np.sqrt(p * (1 - p) / np.array(Ns)), "k--", label="sqrt(p(1-p)/N)")
    axn.set_xscale("log"); axn.set_xlabel("N draws per bulk"); axn.set_ylabel("SE of presence freq")
    axn.set_title("error bar vs N (uniform-q)"); axn.legend(); axn.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("nscaling.png", dpi=110)


if __name__ == "__main__":
    main()
