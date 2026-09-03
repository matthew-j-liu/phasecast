"""Input building and output parsing for the Track B MAGEMin label corpus.

Sits between the trace-conditional (P, H2O) proposal (`condition_proposal`)
and the chunked orchestrator (`scripts/build/generate_mineralogy_corpus.py`).
Per bulk: Fe-split the majors, draw N conditions, expand each draw over the
cooling ladder into `run_magemin.jl` input rows; then parse the returned JSONL
into the three corpus tables (ADR 004 "Storage shape"):

    draws  -- one row per (bulk, draw): passthrough bulk columns + traces, the
              renormalized MAGEMin input oxides, (P, H2O, q), the proposal's
              tier/rung routing, per-rung T_end / G_sys / attrition, the
              oversaturation tag, failure counts.
    ladder -- one row per converged ladder point x phase instance, modes only,
              truncated at the first point below the 30% residual-melt stop.
    rungs  -- one row per (draw, rung, phase instance) at the four re-picked
              endpoints (60/50/40/30% melt): raw modes, per-phase oxide wt%,
              endmember fractions (diagnostic, JSON), liq/fl rows included for
              mass-balance pinning.

Raw modes only are stored; crystals-renormalized labels and presence targets
are training-time rebuilds, never a regen.
"""

import json

import numpy as np
import pandas as pd

from .condition_proposal import bulk_proposal
from .fe_redox import ClassRatioStrategy
from .magemin_oxides import bulk_to_magemin

T_LADDER = np.arange(1200.0, 599.0, -20.0)
RUNGS = (0.60, 0.50, 0.40, 0.30)
STOP_MELT = 0.30

OXIDE_KEYS = ["sio2", "al2o3", "cao", "mgo", "feo", "fe2o3",
              "k2o", "na2o", "tio2", "cr2o3"]
INPUT_COLS = ["sample_id", "T_celsius", "P_kbar", *OXIDE_KEYS, "h2o_tot"]
TRACE_COLS = ["ba_ppm", "ce_ppm", "la_ppm", "nd_ppm", "ni_ppm", "nb_ppm",
              "rb_ppm", "sr_ppm", "th_ppm", "v_ppm", "y_ppm", "yb_ppm", "zr_ppm"]
# MAGEMin's OUTPUT vocabulary: total Fe as FeO plus excess O for the ferric
# component (Fe2O3 = 2 FeO + O by mass), unlike the FeO+Fe2O3 INPUT schema.
COMP_OXIDES = ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O",
               "Na2O", "TiO2", "O", "Cr2O3", "H2O"]


def build_inputs(bulks, majors, n_draws, seed):
    """Expand a bulk table into `run_magemin.jl` input rows + per-point meta.

    `bulks` needs `sample_id` (+ any passthrough columns, kept on the draws
    table); `majors` is gard2019 indexed by sample_id. The per-bulk rng is
    seeded `seed + sample_id`, so draws are independent of chunk boundaries
    and resume-stable. Returns (input_df, meta_df, draws_df); `sid` links
    input rows to meta rows.
    """
    fe3 = ClassRatioStrategy()
    passthrough = [c for c in bulks.columns if c != "sample_id"]
    inputs, meta, draws = [], [], []
    sid = 0
    for _, b in bulks.iterrows():
        bid = int(b.sample_id)
        row = majors.loc[bid]
        base = bulk_to_magemin(row, b.get("qap_name"), h2o_wt=0.0, fe3_strategy=fe3)
        anh = {k: base[k] for k in OXIDE_KEYS}
        anh_sum = sum(anh.values())
        traces = {c: float(row[c]) if pd.notna(row.get(c)) else np.nan
                  for c in TRACE_COLS if c in majors.columns}
        bp = bulk_proposal(b.get("qap_name"),
                           {**traces, "tio2": float(row["tio2"])
                            if pd.notna(row.get("tio2")) else np.nan})
        rng = np.random.default_rng(seed + bid)
        P, W, Q = bp.draw(n_draws, rng)
        # draw() returns draws grouped by mixture box; permute so a draw-index
        # prefix stays an unbiased subsample (train_draw_limit takes prefixes)
        order = rng.permutation(n_draws)
        P, W, Q = P[order], W[order], Q[order]
        for draw, (p, w, q) in enumerate(zip(P, W, Q)):
            scale = (100.0 - w) / anh_sum
            ox = {k: v * scale for k, v in anh.items()}
            for T in T_LADDER:
                sid += 1
                inputs.append(dict(sample_id=sid, T_celsius=float(T),
                                   P_kbar=float(p), h2o_tot=float(w), **ox))
                meta.append(dict(sid=sid, bulk=bid, draw=draw, T_celsius=float(T)))
            draws.append(dict(bulk=bid, draw=draw, P_kbar=float(p), H2O=float(w),
                              q=float(q), proposal_tier=bp.tier,
                              proposal_rung=bp.rung,
                              fe_provenance=base["fe_provenance"],
                              **{f"in_{k}": ox[k] for k in OXIDE_KEYS},
                              **traces,
                              **{c: b[c] for c in passthrough}))
    return (pd.DataFrame(inputs)[INPUT_COLS], pd.DataFrame(meta),
            pd.DataFrame(draws))


def _phase_rows(rec, keep_comp):
    ss = list(rec.get("ss", []))
    rows = []
    for inst, p in enumerate(rec["phases"]):
        r = dict(phase=p["name"], instance=inst, mode_wt=float(p["mode_wt"]))
        if keep_comp:
            comp = p.get("comp_wt", {})
            r.update({f"wt_{ox}": float(comp.get(ox, 0.0)) for ox in COMP_OXIDES})
            # ss carries SS phases in out.ph order; match sequentially by name
            # (duplicate fsp/cpx instances stay separate)
            if ss and ss[0]["name"] == p["name"]:
                e = ss.pop(0)
                r["em_json"] = json.dumps(
                    {"names": list(e["emNames"]), "frac": list(e["emFrac_wt"])})
            else:
                r["em_json"] = None
        rows.append(r)
    return rows


def parse_chunk(jsonl_path, meta):
    """Parse one chunk's JSONL into (ladder_df, rungs_df, draw_stats_df).

    Per (bulk, draw), on the T-descending converged points: the ladder is kept
    down to the first point below `STOP_MELT` (inclusive); each rung label is
    the highest-T converged point below that rung's melt fraction. A rung a
    draw never reaches is attrition (flagged on draw_stats, no rung rows).
    """
    recs = {}
    with open(jsonl_path) as fh:
        for line in fh:
            r = json.loads(line)
            recs[r["sid"]] = r

    ladder, rungs, stats = [], [], []
    for (bulk, draw), g in meta.groupby(["bulk", "draw"], sort=False):
        g = g.sort_values("T_celsius", ascending=False)
        pts = []
        n_failed = 0
        for sid, T in zip(g.sid, g.T_celsius):
            r = recs.get(sid)
            if r is None or r["status"] != 0 or not np.isfinite(r["G"]) or r["G"] >= 0:
                n_failed += 1
                continue
            liq = sum(p["mode_wt"] for p in r["phases"] if p["name"] == "liq")
            pts.append((T, liq, r))

        st = dict(bulk=bulk, draw=draw, n_failed=n_failed, n_points=len(pts))
        if not pts:
            st.update(oversaturated=False, T_stop=np.nan,
                      **{f"attrition_{int(m*100)}": True for m in RUNGS})
            stats.append(st)
            continue

        st["oversaturated"] = any(p["name"] == "fl" for p in pts[0][2]["phases"])

        stop = next((i for i, (_, liq, _) in enumerate(pts) if liq < STOP_MELT),
                    len(pts) - 1)
        for T, liq, r in pts[:stop + 1]:
            for row in _phase_rows(r, keep_comp=False):
                ladder.append(dict(bulk=bulk, draw=draw, T_celsius=T, **row))
        st["T_stop"] = pts[stop][0]

        for m in RUNGS:
            hit = next(((T, liq, r) for T, liq, r in pts if liq < m), None)
            st[f"attrition_{int(m*100)}"] = hit is None
            if hit is None:
                continue
            T, liq, r = hit
            for row in _phase_rows(r, keep_comp=True):
                rungs.append(dict(bulk=bulk, draw=draw, rung=int(m * 100),
                                  T_end=T, liq_frac=liq, G_sys=float(r["G"]),
                                  **row))
        stats.append(st)

    return pd.DataFrame(ladder), pd.DataFrame(rungs), pd.DataFrame(stats)
