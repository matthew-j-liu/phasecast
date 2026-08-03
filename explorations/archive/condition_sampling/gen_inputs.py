"""Build the MAGEMin T-ladder input for the (P,H2O) sampling illustration on gard 6875.

Designs emitted into one input CSV (joined back via meta):
  mc256  -- 256 Latin-Hypercube draws from q = uniform(P,H2O) on the arc envelope
  gridU  -- 8x8 = 64 uniform product grid (same envelope)
  gridP  -- 16x4 grid, dense in P  (64 pts, same support, different density)
  gridW  -- 4x16 grid, dense in H2O (64 pts, same support, different density)

Each (P,H2O) sample is expanded over a cooling T ladder; the near-solidus
endpoint (highest T with liq < 2 wt%) is picked downstream.
"""
import numpy as np
import pandas as pd

# gard sample_id=6875 (qap_name andesite). Total Fe (FeO_tot 7.86) split at
# Fe3+/SumFe ~ 0.15 by mole (andesite Middlemost-class default), matching the
# existing hand-built example; tiny Cr2O3 added for the ig database.
OX = dict(sio2=57.76, al2o3=15.36, cao=6.76, mgo=6.17,
          feo=6.681000113487244, fe2o3=1.310373839073895,
          k2o=1.08, na2o=4.74, tio2=1.23, cr2o3=0.02)

# Proposal envelope q: broad uniform over the plausible arc (P, H2O).
P_LO, P_HI = 1.0, 13.0      # kbar  (~0.1 - 1.3 GPa: shallow crust -> lower-crustal hot zone)
W_LO, W_HI = 0.5, 8.0       # wt% H2O (nominally dry -> very wet arc)

# Cooling ladder: liquidus-ish down to sub-solidus, find liq<2% endpoint per path.
T_LADDER = np.arange(1180.0, 639.0, -20.0)   # 1180..660, 27 steps

SEED = 20260630


def lhs(n, rng):
    """2D Latin Hypercube on the unit square (stratified uniform, unbiased marginals)."""
    u = np.empty((n, 2))
    for d in range(2):
        perm = rng.permutation(n)
        u[:, d] = (perm + rng.random(n)) / n
    return u


def scale(u):
    P = P_LO + u[:, 0] * (P_HI - P_LO)
    W = W_LO + u[:, 1] * (W_HI - W_LO)
    return P, W


def grid(nP, nW):
    # cell-centered so points sit inside the envelope, not on its edges
    Pc = P_LO + (np.arange(nP) + 0.5) / nP * (P_HI - P_LO)
    Wc = W_LO + (np.arange(nW) + 0.5) / nW * (W_HI - W_LO)
    PP, WW = np.meshgrid(Pc, Wc)
    return PP.ravel(), WW.ravel()


def main():
    rng = np.random.default_rng(SEED)
    designs = {}
    Pmc, Wmc = scale(lhs(256, rng))
    designs["mc256"] = (Pmc, Wmc)
    designs["gridU"] = grid(8, 8)
    designs["gridP"] = grid(16, 4)
    designs["gridW"] = grid(4, 16)

    rows, meta = [], []
    sid = 0
    for design, (Pv, Wv) in designs.items():
        for draw, (P, W) in enumerate(zip(Pv, Wv)):
            for T in T_LADDER:
                sid += 1
                row = dict(sample_id=sid, T_celsius=float(T), P_kbar=float(P),
                           h2o_tot=float(W), **OX)
                rows.append(row)
                meta.append(dict(sample_id=sid, design=design, draw=draw,
                                 P_kbar=float(P), H2O=float(W), T_celsius=float(T)))

    cols = ["sample_id", "T_celsius", "P_kbar", "sio2", "al2o3", "cao", "mgo",
            "feo", "fe2o3", "k2o", "na2o", "tio2", "cr2o3", "h2o_tot"]
    pd.DataFrame(rows)[cols].to_csv("mc_in.csv", index=False)
    pd.DataFrame(meta).to_csv("mc_meta.csv", index=False)
    print(f"wrote mc_in.csv  {len(rows)} points  ({sid} sids)")
    print(f"  designs: " + ", ".join(f"{k}={len(v[0])}" for k, v in designs.items()))
    print(f"  T ladder: {T_LADDER[0]:.0f}..{T_LADDER[-1]:.0f} C, {len(T_LADDER)} steps")


if __name__ == "__main__":
    main()
