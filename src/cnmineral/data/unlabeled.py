"""V2 input vocabulary: 9 majors + 13 traces + unlabeled-corpus dataset for SSL.

The v2 model expands the encoder's token vocabulary beyond v1's 9 cations to
include 13 high-coverage trace elements from complete.csv. The same 22-token
schema is used both for SSL pretraining on the ~1M unlabeled rows here and
for supervised fine-tuning (`TrainFCv2Dataset` in `models/dataset.py`).

Conventions match v1: src=0 is padding, frac in [0, 1] (majors via wt% / 100,
traces via ppm / 1e6). The masked-token convention used during SSL training
(src=atomic_num, frac=0) is applied at batch time by the training loop, not
here -- the dataset emits raw populated/padded sequences.

Per the plan's "Lessons lifted from mineralML" section, the dataset:
  - applies the same quality filter as `scripts/02_filter_complete.py`
    (igneous only, sum 80-120 wt%, at least the 7 required majors)
  - excludes the supervised corpus bulks by default so SSL pretraining stays
    genuinely "unsupervised" (no leakage of test labels' bulks)
  - exposes a `supergroup` per row for weighted sampling (8 coarse
    petrogenetic supergroups vs the 60+ raw `qap_name` classes)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cnmineral.models.dataset import OXIDE_TO_CATION


# Majors: same 9 cations as v1 (Fe collapsed from FeO + Fe2O3 at conversion time).
V2_MAJORS_ORDER: list[tuple[str, int]] = [
    ("Si", 14), ("Al", 13), ("Ca", 20), ("Mg", 12), ("Fe", 26),
    ("K",  19), ("Na", 11), ("Ti", 22), ("Cr", 24),
]

# Traces, ordered by combined coverage + petrogenetic discriminant value.
# Coverage on the v1 supervised subset: Sr 76%, Zr 73%, Ba 72%, Y 70%, Rb 68%,
# Ni 65%, Nb 63%, V 58%, La 59%, Ce 58%, Nd 49%, Yb 45%, Th 53%.
V2_TRACES_ORDER: list[tuple[str, int]] = [
    ("Sr", 38), ("Zr", 40), ("Ba", 56), ("Y",  39), ("Rb", 37),
    ("Ni", 28), ("Nb", 41), ("V",  23), ("La", 57), ("Ce", 58),
    ("Nd", 60), ("Yb", 70), ("Th", 90),
]

V2_CATIONS_ORDER: list[tuple[str, int]] = V2_MAJORS_ORDER + V2_TRACES_ORDER
V2_N_TOKENS: int = len(V2_CATIONS_ORDER)  # 22

# Column names in complete.csv for the trace ppm values.
V2_TRACE_PPM_COLS: list[str] = [f"{sym.lower()}_ppm" for sym, _z in V2_TRACES_ORDER]

# Required majors and Fe columns mirror scripts/02_filter_complete.py so the
# SSL corpus uses the same quality bar as the supervised one.
_REQUIRED_MAJORS = ["sio2", "al2o3", "cao", "mgo", "k2o", "na2o", "tio2"]
_FE_COLS = ["feo", "fe2o3", "feo_tot", "fe2o3_tot"]
_OXIDE_SUM_COLS = _REQUIRED_MAJORS + ["mno", "p2o5", "cr2o3"]
_FE2O3_TO_FEO_MASS = (2 * 71.844) / 159.69


def _supergroup_of(qap_name: str | None) -> str:
    """Map a `qap_name` (IUGS Streckeisen class) to one of 8 petrogenetic supergroups.

    Used for class-balanced sampling during SSL pretrain and for coloring the
    latent-space UMAP. Returns "other" for unrecognized strings so the function
    is total.
    """
    if qap_name is None or not isinstance(qap_name, str):
        return "other"
    s = qap_name.strip().lower()
    if not s:
        return "other"
    # foid-bearing (alkaline silica-undersaturated): check first since several
    # qap_name strings are e.g. "foid-bearing diorite" or "phonolitic tephrite".
    if ("foid" in s or "nephelin" in s or "phonolit" in s
            or "tephrit" in s or "basanit" in s):
        return "foid-bearing"
    if ("dunit" in s or "harzburg" in s or "lherzolit" in s or "wehrlit" in s
            or "websterit" in s or "pyroxenit" in s or "peridotit" in s
            or "picrit" in s or "komatiit" in s):
        return "ultramafic"
    if "gabbr" in s or "norit" in s or "anorthosit" in s:
        return "mafic-plutonic"
    if "basalt" in s:
        return "mafic-volcanic"
    if "diorit" in s or "tonalit" in s or "monzonit" in s or "granodior" in s:
        return "intermediate-plutonic"
    if "andesit" in s or "latit" in s or "trachyt" in s:
        return "intermediate-volcanic"
    if "rhyolit" in s or "dacit" in s:
        return "felsic-volcanic"
    if "granit" in s or "syenit" in s:
        return "felsic-plutonic"
    return "other"


SUPERGROUPS: tuple[str, ...] = (
    "ultramafic", "mafic-volcanic", "mafic-plutonic",
    "intermediate-volcanic", "intermediate-plutonic",
    "felsic-volcanic", "felsic-plutonic", "foid-bearing", "other",
)


def _combine_fe(row: dict | pd.Series) -> float:
    """Total iron as FeO-equivalent, accepting any of (feo, fe2o3, feo_tot, fe2o3_tot)."""
    def _g(k: str) -> float:
        v = row.get(k)
        return 0.0 if (v is None or pd.isna(v)) else float(v)
    feo = _g("feo")
    fe2o3 = _g("fe2o3")
    if feo > 0 and fe2o3 > 0:
        return feo + fe2o3 * _FE2O3_TO_FEO_MASS
    if _g("feo_tot") > 0:
        return _g("feo_tot")
    if _g("fe2o3_tot") > 0:
        return _g("fe2o3_tot") * _FE2O3_TO_FEO_MASS
    if feo > 0:
        return feo
    if fe2o3 > 0:
        return fe2o3 * _FE2O3_TO_FEO_MASS
    return 0.0


def oxides_traces_to_tokens(
    majors: dict[str, float],
    traces_ppm: dict[str, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one rock's chemistry to (src [22], frac [22]) for the v2 encoder.

    `majors` keys are lowercased oxide names matching `OXIDE_TO_CATION` (sio2,
    al2o3, cao, mgo, feo, fe2o3, k2o, na2o, tio2, cr2o3). Missing oxides are
    treated as 0. `traces_ppm` keys are lowercased element symbols (sr, zr,
    ba, y, rb, ni, nb, v, la, ce, nd, yb, th); missing or None traces become
    padding tokens. `traces_ppm=None` produces a majors-only sample (all 13
    trace slots padded).

    Returns
    -------
    src : int64 [22]
        Atomic numbers in V2_CATIONS_ORDER. 0 indicates padding (element
        absent in this rock).
    frac : float32 [22]
        Mass fractions in [0, 1]: majors via cation wt% / 100, traces via
        ppm / 1e6. Padded positions are 0.
    """
    src = np.zeros(V2_N_TOKENS, dtype=np.int64)
    frac = np.zeros(V2_N_TOKENS, dtype=np.float32)

    # Majors: oxide wt% → cation wt% (Fe collapsed) → / 100.
    cation_wt = {sym: 0.0 for sym, _z in V2_MAJORS_ORDER}
    for oxide, val in majors.items():
        if val is None or pd.isna(val) or oxide not in OXIDE_TO_CATION:
            continue
        sym, _z, factor = OXIDE_TO_CATION[oxide]
        if sym in cation_wt:
            cation_wt[sym] += float(val) * factor

    for i, (sym, z) in enumerate(V2_MAJORS_ORDER):
        v = cation_wt[sym]
        if v > 0:
            src[i] = z
            frac[i] = v / 100.0

    if traces_ppm is None:
        return src, frac

    offset = len(V2_MAJORS_ORDER)
    for i, (sym, z) in enumerate(V2_TRACES_ORDER):
        v = traces_ppm.get(sym.lower())
        if v is None or pd.isna(v) or float(v) <= 0:
            continue
        src[offset + i] = z
        frac[offset + i] = float(v) / 1e6

    return src, frac


def igneous_quality_filter(
    df: pd.DataFrame,
    min_sum: float = 80.0,
    max_sum: float = 120.0,
) -> pd.DataFrame:
    """Lift of `scripts/02_filter_complete.py:filter_quality` for the SSL corpus.

    Keeps rows that (a) have all 7 required majors, (b) have at least one Fe
    column, and (c) sum (majors + Fe-as-FeO + Mn + P + Cr) within [min_sum,
    max_sum] wt%. Conservative bound consistent with the supervised filter.
    """
    req_mask = df[_REQUIRED_MAJORS].notna().all(axis=1)
    fe_mask = df[_FE_COLS].notna().any(axis=1)

    # Iron contribution for sum: prefer measured (feo + fe2o3), fall back to
    # whichever _tot is present.
    fe_for_sum = df[["feo", "fe2o3"]].fillna(0).sum(axis=1).where(
        df["feo"].notna() & df["fe2o3"].notna(),
        df[["feo_tot"]].fillna(0).max(axis=1).where(
            df["feo_tot"].notna(),
            df[["fe2o3_tot"]].fillna(0).max(axis=1) * _FE2O3_TO_FEO_MASS,
        ),
    )
    oxide_sum = (df[_REQUIRED_MAJORS].fillna(0).sum(axis=1)
                 + fe_for_sum
                 + df[["mno", "p2o5", "cr2o3"]].fillna(0).sum(axis=1))
    sum_mask = (oxide_sum >= min_sum) & (oxide_sum <= max_sum)
    return df.loc[req_mask & fe_mask & sum_mask].copy()


class UnlabeledV2Dataset(Dataset):
    """Per-row (src, frac, qap_name, supergroup) over igneous-filtered complete.csv.

    No mineralogy labels; consumed by the SSL pretraining loop. Trace columns
    are read from complete.csv where present; absent traces produce padding
    tokens that kingcrab's `Encoder` already handles via the `src == 0`
    convention.

    Parameters
    ----------
    complete_csv_path : str | Path
        Path to data/complete.csv.
    exclude_sample_ids : iterable[int] | None
        Bulks to drop from the unlabeled pool to avoid label-snooping. Pass
        the union of supervised train/val/test sample_ids (or just the test
        sample_ids if you want a lenient "no test leak only" policy). Default
        None keeps everything.
    min_sum, max_sum : float
        Quality-filter bounds on oxide sum in wt%.
    cache_arrays : bool
        If True (default), pre-compute the [N, 22] src and frac arrays at
        construction time. Trades memory for fast __getitem__ — same pattern
        as v1's eager loading.
    """

    def __init__(
        self,
        complete_csv_path: str | Path,
        exclude_sample_ids: np.ndarray | list[int] | None = None,
        min_sum: float = 80.0,
        max_sum: float = 120.0,
        cache_arrays: bool = True,
    ) -> None:
        complete_csv_path = Path(complete_csv_path)
        use_cols = (["sample_id", "rock_group", "qap_name"]
                    + _REQUIRED_MAJORS + _FE_COLS
                    + ["mno", "p2o5", "cr2o3", "h2o_tot"]
                    + V2_TRACE_PPM_COLS)
        use_set = set(use_cols)
        df = pd.read_csv(
            complete_csv_path,
            encoding="latin-1",
            low_memory=False,
            usecols=lambda c: c.lower() in use_set,
        )
        df.columns = [c.lower() for c in df.columns]
        df = df[df["rock_group"] == "igneous"].copy()
        df = igneous_quality_filter(df, min_sum=min_sum, max_sum=max_sum)

        if exclude_sample_ids is not None:
            excl = pd.Series(exclude_sample_ids).astype(np.int64).unique()
            df = df[~df["sample_id"].isin(excl)].reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f"No rows survived filtering in {complete_csv_path}")

        # Collapse Fe to a single column (mirror the supervised dataset's
        # convention: FeO + Fe2O3 → one Fe cation). We synthesize FeO and Fe2O3
        # from whichever cols are present so oxides_traces_to_tokens can
        # consume a uniform schema.
        majors_for_tokens = df[_REQUIRED_MAJORS + ["cr2o3"]].copy()
        # Total Fe as FeO-equivalent → put it all in the `feo` column so the
        # mass-balance factor (0.7773) applies cleanly.
        fe_feo_eq = df.apply(_combine_fe, axis=1)
        majors_for_tokens["feo"] = fe_feo_eq
        majors_for_tokens["fe2o3"] = 0.0

        traces_df = df[V2_TRACE_PPM_COLS]

        self._cache_arrays = cache_arrays
        if cache_arrays:
            srcs = np.zeros((len(df), V2_N_TOKENS), dtype=np.int64)
            fracs = np.zeros((len(df), V2_N_TOKENS), dtype=np.float32)
            for i in range(len(df)):
                m_row = majors_for_tokens.iloc[i].to_dict()
                t_row = {sym.lower(): traces_df.iloc[i][f"{sym.lower()}_ppm"]
                         for sym, _z in V2_TRACES_ORDER}
                src, frac = oxides_traces_to_tokens(m_row, t_row)
                srcs[i] = src
                fracs[i] = frac
            self._src = srcs
            self._frac = fracs

        self._majors = majors_for_tokens
        self._traces = traces_df
        self._sample_ids = df["sample_id"].to_numpy(dtype=np.int64)
        self._qap_names = df["qap_name"].fillna("").to_numpy(dtype=object)
        self._supergroups = np.array(
            [_supergroup_of(q) for q in self._qap_names], dtype=object)

        self.sample_ids: np.ndarray = self._sample_ids
        self.qap_names: np.ndarray = self._qap_names
        self.supergroups: np.ndarray = self._supergroups
        self.n_tokens: int = V2_N_TOKENS

    def __len__(self) -> int:
        return len(self._sample_ids)

    def __getitem__(self, idx: int) -> dict:
        if self._cache_arrays:
            src = self._src[idx]
            frac = self._frac[idx]
        else:
            m_row = self._majors.iloc[idx].to_dict()
            t_row = {sym.lower(): self._traces.iloc[idx][f"{sym.lower()}_ppm"]
                     for sym, _z in V2_TRACES_ORDER}
            src, frac = oxides_traces_to_tokens(m_row, t_row)
        return {
            "src": torch.from_numpy(np.asarray(src, dtype=np.int64)),
            "frac": torch.from_numpy(np.asarray(frac, dtype=np.float32)),
            "qap_name": str(self._qap_names[idx]),
            "supergroup": str(self._supergroups[idx]),
            "sample_id": int(self._sample_ids[idx]),
        }

    def supergroup_counts(self) -> dict[str, int]:
        """For class-balanced sampling: count rows per supergroup."""
        unique, counts = np.unique(self._supergroups, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
