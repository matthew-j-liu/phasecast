"""Build crystals-renormalized modal labels from the mineralogy-corpus rungs tables.

Turns the long (bulk, draw, rung, phase, instance) rungs rows into one modal
vector per (bulk, draw, rung) over `phase_vocab.SOLID_PHASES`: feldspar
instances are routed to plag/afs by `feldspar_rule`, the merge/drop rules
applied, instance modes summed, liq/fl dropped and the solids renormalized to
sum to 1. The presence cutoff and the present-only Dirichlet target are
derived downstream in `mineralogy_dataset` -- label semantics are training-time
rebuilds from raw modes, never materialized to disk.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .feldspar_rule import classify_feldspar
from .phase_vocab import (
    DROPPED_PHASES,
    NON_SOLID_PHASES,
    PHASE_MERGES,
    SOLID_PHASES,
)

_RUNG_COLUMNS = [
    "bulk", "draw", "rung", "phase", "mode_wt", "wt_CaO", "wt_Na2O", "wt_K2O",
]


def load_mode_labels(
    corpus_dir: str | Path, rungs: tuple[int, ...] = (50, 30)
) -> pd.DataFrame:
    """One row per (bulk, draw, rung) with crystals-renormalized modes.

    Returns columns `bulk`, `draw`, `rung` plus one column per phase in
    `SOLID_PHASES`, each row's phase modes summing to 1. Draws that never
    reached a requested rung have no rungs rows and are naturally absent.
    """
    files = sorted(Path(corpus_dir).glob("chunk_*_rungs.parquet"))
    if not files:
        raise FileNotFoundError(f"no chunk_*_rungs.parquet under {corpus_dir}")

    parts = []
    for f in files:
        df = pd.read_parquet(f, columns=_RUNG_COLUMNS)
        parts.append(df[df["rung"].isin(rungs) & ~df["phase"].isin(NON_SOLID_PHASES)])
    df = pd.concat(parts, ignore_index=True)
    df = df[~df["phase"].isin(DROPPED_PHASES)]

    token = df["phase"].map(lambda p: PHASE_MERGES.get(p, p)).to_numpy(dtype=object)
    is_fsp = df["phase"].to_numpy() == "fsp"
    if is_fsp.any():
        token[is_fsp] = classify_feldspar(
            df["wt_CaO"].to_numpy()[is_fsp],
            df["wt_Na2O"].to_numpy()[is_fsp],
            df["wt_K2O"].to_numpy()[is_fsp],
        )
    unknown = set(token) - set(SOLID_PHASES)
    if unknown:
        raise ValueError(
            f"phases outside the vocabulary: {sorted(unknown)}; extend "
            "phase_vocab merge/drop rules before training on this corpus"
        )
    df = df.assign(token=token)

    wide = (
        df.groupby(["bulk", "draw", "rung", "token"], sort=False)["mode_wt"]
        .sum()
        .unstack("token", fill_value=0.0)
        .reindex(columns=list(SOLID_PHASES), fill_value=0.0)
    )
    total = wide.sum(axis=1)
    wide = wide[total > 0].div(total[total > 0], axis=0)
    return wide.reset_index()


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
