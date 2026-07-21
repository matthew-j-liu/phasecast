"""Build crystals-renormalized labels from the mineralogy-corpus tables.

Turns the long (bulk, draw, rung, phase, instance) rungs rows into
per-(bulk, draw, rung) targets over `phase_vocab.SOLID_PHASES`: feldspar
instances are routed to plag/afs by `feldspar_rule`, the merge/drop rules
applied, instance modes summed, liq/fl dropped and the solids renormalized to
sum to 1. `load_joint_labels` emits the modal vector plus the CompHead oxide
targets, the CondHead (T, P, H2O) targets, and the teacher-forced
mass-balance fixings (melt/fluid fractions and compositions, input bulk in
the 11-oxide basis). The presence cutoff and the present-only Dirichlet
target are derived downstream in `mineralogy_dataset` -- label semantics are
training-time rebuilds from raw modes, never materialized to disk.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .feldspar_rule import classify_feldspar
from .magemin_corpus import COMP_OXIDES
from .magemin_oxides import FE2O3_TO_FEO_MASS
from .phase_vocab import (
    DROPPED_PHASES,
    NON_SOLID_PHASES,
    PHASE_MERGES,
    SOLID_PHASES,
)

_WT_COLUMNS = [f"wt_{o}" for o in COMP_OXIDES]
_IN_COLUMNS = [
    "in_sio2", "in_al2o3", "in_cao", "in_mgo", "in_feo", "in_fe2o3",
    "in_k2o", "in_na2o", "in_tio2", "in_cr2o3",
]


def _route_tokens(solids: pd.DataFrame) -> np.ndarray:
    """Map raw phase names to vocabulary tokens, feldspar split per instance."""
    token = solids["phase"].map(lambda p: PHASE_MERGES.get(p, p)).to_numpy(dtype=object)
    is_fsp = solids["phase"].to_numpy() == "fsp"
    if is_fsp.any():
        token[is_fsp] = classify_feldspar(
            solids["wt_CaO"].to_numpy()[is_fsp],
            solids["wt_Na2O"].to_numpy()[is_fsp],
            solids["wt_K2O"].to_numpy()[is_fsp],
        )
    unknown = set(token) - set(SOLID_PHASES)
    if unknown:
        raise ValueError(
            f"phases outside the vocabulary: {sorted(unknown)}; extend "
            "phase_vocab merge/drop rules before training on this corpus"
        )
    return token


def load_joint_labels(
    corpus_dir: str | Path,
    rungs: tuple[int, ...] = (50, 30),
    comp_eps: float = 1e-4,
) -> dict[str, np.ndarray | pd.DataFrame]:
    """Per-(bulk, draw, rung) targets for the joint mode/comp/cond trainer.

    Returns a dict of row-aligned arrays over the groups with nonzero solids:
    `index` (DataFrame bulk/draw/rung), `modes` [N, P] crystals-renormalized,
    `comp` [N, P, O] per-token mass-weighted oxide fractions (eps-clipped,
    renormalized; uniform at tokens absent from the group -- masked out of the
    loss), `cond` [N, 3] as (T_end, P_kbar, H2O), `f_liq`/`f_fl` [N],
    `c_liq`/`c_fl` [N, O] (zeros when the melt/fluid row is absent), and
    `bulk_11` [N, O] -- the MAGEMin input bulk converted to the output basis
    (Fe2O3 -> FeO-equivalent + excess O) and normalized to sum 1.
    """
    corpus_dir = Path(corpus_dir)
    rung_files = sorted(corpus_dir.glob("chunk_*_rungs.parquet"))
    if not rung_files:
        raise FileNotFoundError(f"no chunk_*_rungs.parquet under {corpus_dir}")
    cols = ["bulk", "draw", "rung", "T_end", "phase", "mode_wt", *_WT_COLUMNS]
    df = pd.concat(
        [
            part[part["rung"].isin(rungs)]
            for f in rung_files
            for part in [pd.read_parquet(f, columns=cols)]
        ],
        ignore_index=True,
    )

    solids = df[
        ~df["phase"].isin(NON_SOLID_PHASES) & ~df["phase"].isin(DROPPED_PHASES)
    ].copy()
    solids["token"] = _route_tokens(solids)
    for c in _WT_COLUMNS:
        solids[c] = solids[c] * solids["mode_wt"]
    grouped = solids.groupby(["bulk", "draw", "rung", "token"], sort=False)[
        ["mode_wt", *_WT_COLUMNS]
    ].sum()

    modes_wide = (
        grouped["mode_wt"]
        .unstack("token", fill_value=0.0)
        .reindex(columns=list(SOLID_PHASES), fill_value=0.0)
    )
    total = modes_wide.sum(axis=1)
    modes_wide = modes_wide[total > 0]
    index = modes_wide.index
    modes = modes_wide.div(total[total > 0], axis=0).to_numpy()

    comp_wide = (
        grouped[_WT_COLUMNS]
        .div(grouped["mode_wt"], axis=0)
        .unstack("token")
        .reindex(
            index=index,
            columns=pd.MultiIndex.from_product([_WT_COLUMNS, list(SOLID_PHASES)]),
        )
        .fillna(0.0)
    )
    comp = (
        comp_wide.to_numpy()
        .reshape(len(index), len(COMP_OXIDES), len(SOLID_PHASES))
        .transpose(0, 2, 1)
    )
    comp = np.clip(comp, comp_eps, None)
    comp = comp / comp.sum(axis=-1, keepdims=True)

    melt = df[df["phase"].isin(NON_SOLID_PHASES)].copy()
    for c in _WT_COLUMNS:
        melt[c] = melt[c] * melt["mode_wt"]
    melt_grouped = melt.groupby(["bulk", "draw", "rung", "phase"])[
        ["mode_wt", *_WT_COLUMNS]
    ].sum()
    fixings = {}
    for ph in ("liq", "fl"):
        if ph in melt_grouped.index.get_level_values("phase"):
            sub = melt_grouped.xs(ph, level="phase")
            frac = sub["mode_wt"].reindex(index).fillna(0.0).to_numpy()
            c_ph = (
                sub[_WT_COLUMNS]
                .div(sub["mode_wt"], axis=0)
                .reindex(index)
                .fillna(0.0)
                .to_numpy()
            )
        else:
            frac = np.zeros(len(index))
            c_ph = np.zeros((len(index), len(COMP_OXIDES)))
        fixings[ph] = (frac, c_ph)

    t_end = df.groupby(["bulk", "draw", "rung"])["T_end"].first().reindex(index)

    draw_files = sorted(corpus_dir.glob("chunk_*_draws.parquet"))
    if not draw_files:
        raise FileNotFoundError(f"no chunk_*_draws.parquet under {corpus_dir}")
    draws = pd.concat(
        [
            pd.read_parquet(f, columns=["bulk", "draw", "P_kbar", "H2O", *_IN_COLUMNS])
            for f in draw_files
        ],
        ignore_index=True,
    ).set_index(["bulk", "draw"])
    pairs = pd.MultiIndex.from_arrays(
        [index.get_level_values("bulk"), index.get_level_values("draw")]
    )
    d = draws.reindex(pairs)
    if d["P_kbar"].isna().any():
        raise ValueError("rungs rows without a matching draws row; corpus is inconsistent")

    cond = np.stack(
        [t_end.to_numpy(), d["P_kbar"].to_numpy(), d["H2O"].to_numpy()], axis=1
    )

    feo_total = d["in_feo"] + d["in_fe2o3"] * FE2O3_TO_FEO_MASS
    excess_o = d["in_fe2o3"] * (1.0 - FE2O3_TO_FEO_MASS)
    bulk_11 = np.stack(
        [
            d["in_sio2"], d["in_al2o3"], d["in_cao"], d["in_mgo"], feo_total,
            d["in_k2o"], d["in_na2o"], d["in_tio2"], excess_o, d["in_cr2o3"],
            d["H2O"],
        ],
        axis=1,
    )
    bulk_11 = bulk_11 / bulk_11.sum(axis=1, keepdims=True)

    return {
        "index": modes_wide.reset_index()[["bulk", "draw", "rung"]],
        "modes": modes,
        "comp": comp,
        "cond": cond,
        "f_liq": fixings["liq"][0],
        "c_liq": fixings["liq"][1],
        "f_fl": fixings["fl"][0],
        "c_fl": fixings["fl"][1],
        "bulk_11": bulk_11,
    }


def hurdle_targets(
    modes: np.ndarray, presence_cutoff: float
) -> tuple[np.ndarray, np.ndarray]:
    """Presence mask and present-only renormalized magnitude target.

    `modes` is [N, n_phases], rows summing to 1. Presence is `mode > cutoff`;
    the magnitude target renormalizes the present phases to sum to 1 and
    zeroes the rest (absent phases are owned by the Bernoullis, masked out of
    the Dirichlet loss). Every row has at least one present phase because the
    row max is >= 1/n_phases > cutoff.
    """
    presence = modes > presence_cutoff
    magnitude = np.where(presence, modes, 0.0)
    magnitude = magnitude / magnitude.sum(axis=1, keepdims=True)
    return presence, magnitude
