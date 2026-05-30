"""PyTorch datasets.

- `TrainMageminDataset`: v0 H5 corpus (`data/train_magemin.h5`).
- `TrainFCv1Dataset`: v1 fractional-crystallization parquet
  (`data/train_fc_v1.parquet`). 11 oxides as inputs; modal wt% + (T, P, H2O)
  as supervision; conditions are z-scored at load.

Both share the same oxide -> cation conversion (FeO + Fe2O3 collapsed to a
single Fe column) and the same `src`/`frac` output convention CrabNet's
encoder expects.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Map each MAGEMin oxide column (lowercase) to (cation_symbol, atomic_number,
# wt%-conversion-factor). Factor = (n_cations * MW_cation) / MW_oxide; multiply
# by oxide wt% to get cation wt%.
#
# MW reference values:
#   SiO2  = 60.08    Si  = 28.086   factor = 28.086 / 60.08  = 0.4675
#   Al2O3 = 101.96   Al  = 26.982   factor = 2*26.982/101.96 = 0.5293
#   CaO   = 56.077   Ca  = 40.078   factor = 40.078 / 56.077 = 0.7147
#   MgO   = 40.304   Mg  = 24.305   factor = 24.305 / 40.304 = 0.6030
#   FeO   = 71.844   Fe  = 55.845   factor = 55.845 / 71.844 = 0.7773
#   Fe2O3 = 159.69   Fe  = 55.845   factor = 2*55.845/159.69 = 0.6994
#   K2O   = 94.196   K   = 39.098   factor = 2*39.098/94.196 = 0.8302
#   Na2O  = 61.979   Na  = 22.990   factor = 2*22.990/61.979 = 0.7419
#   TiO2  = 79.866   Ti  = 47.867   factor = 47.867 / 79.866 = 0.5995
#   Cr2O3 = 151.99   Cr  = 51.996   factor = 2*51.996/151.99 = 0.6842
#   H2O   = 18.015   H   = 1.008    factor = 2*1.008/18.015  = 0.1119
OXIDE_TO_CATION: dict[str, tuple[str, int, float]] = {
    "sio2":    ("Si", 14, 0.4675),
    "al2o3":   ("Al", 13, 0.5293),
    "cao":     ("Ca", 20, 0.7147),
    "mgo":     ("Mg", 12, 0.6030),
    "feo":     ("Fe", 26, 0.7773),
    "fe2o3":   ("Fe", 26, 0.6994),    # Same cation as feo; summed during conversion
    "k2o":     ("K",  19, 0.8302),
    "na2o":    ("Na", 11, 0.7419),
    "tio2":    ("Ti", 22, 0.5995),
    "cr2o3":   ("Cr", 24, 0.6842),
    "h2o_tot": ("H",   1, 0.1119),
}

# Final per-sample cation order (after summing FeO + Fe2O3 → Fe).
# Atomic numbers in this same order will be the `src` tensor.
CATIONS_ORDER: list[tuple[str, int]] = [
    ("Si", 14), ("Al", 13), ("Ca", 20), ("Mg", 12), ("Fe", 26),
    ("K",  19), ("Na", 11), ("Ti", 22), ("Cr", 24), ("H",   1),
]


def oxides_to_cations(
    X_oxide: np.ndarray,
    x_cols: list[str],
    cations_order: list[tuple[str, int]] = CATIONS_ORDER,
) -> np.ndarray:
    """Convert oxide wt% → cation wt% in `cations_order`.

    FeO and Fe2O3 both contribute to the Fe column. Any column in `x_cols` that
    isn't in OXIDE_TO_CATION (e.g., T_celsius, P_kbar), or whose cation isn't
    in `cations_order` (e.g., H once h2o_tot is dropped), is silently ignored.
    """
    n_samples = X_oxide.shape[0]
    out = np.zeros((n_samples, len(cations_order)), dtype=np.float32)
    cation_to_idx = {sym: i for i, (sym, _z) in enumerate(cations_order)}

    for col_idx, col in enumerate(x_cols):
        if col not in OXIDE_TO_CATION:
            continue
        cation_sym, _atomic_num, factor = OXIDE_TO_CATION[col]
        if cation_sym not in cation_to_idx:
            continue
        out[:, cation_to_idx[cation_sym]] += X_oxide[:, col_idx] * factor

    return out


class TrainMageminDataset(Dataset):
    """Per-sample (src, frac, y_wt, qap_name) for MAGEMin-derived training H5.

    H5 schema produced by `scripts/parse_magemin.py`:
      /X         [N, 13]  bulk oxides (11) + T_celsius + P_kbar
      /X_cols    str array, 13 names
      /Y_wt      [N, P]   mineral wt fractions
      /Y_cols    str array, phase names
      /qap_name  [N]      bytestrings; per-sample IUGS rock class
      (/Y_ss, /Y_energy, /fe_prov, /h2o_prov also present but not used in v0)

    Output per sample:
      {
        "src":  int64 tensor [10] of atomic numbers (Si, Al, Ca, Mg, Fe, K, Na, Ti, Cr, H)
        "frac": float32 tensor [10] of cation wt%
        "y_wt": float32 tensor [P] of mineral wt fractions
        "qap_name": str
      }

    Attributes populated at construction:
      n_minerals (int): number of phases discovered in the H5
      phase_names (list[str]): decoded /Y_cols
      qap_names (np.ndarray[str]): per-sample IUGS class labels, used by
        downstream stratified splitting
    """

    def __init__(self, h5_path: str | Path) -> None:
        h5_path = Path(h5_path)
        with h5py.File(h5_path, "r") as f:
            X_full = f["X"][:]
            x_cols = [c.decode() if isinstance(c, bytes) else c for c in f["X_cols"][:]]
            Y_wt = f["Y_wt"][:].astype(np.float32)
            phase_names = [c.decode() if isinstance(c, bytes) else c for c in f["Y_cols"][:]]
            qap = f["qap_name"][:]
            qap_names = np.array([
                q.decode() if isinstance(q, (bytes, np.bytes_)) else str(q)
                for q in qap
            ], dtype=object)

        # Pre-compute cation conversion once; cheap and avoids per-batch overhead.
        cation_wt_pct = oxides_to_cations(X_full, x_cols)
        # CrabNet's FractionalEncoder expects fractions in [0, 1], not wt% in
        # [0, 100]. Our LOI-normalized inputs sum to exactly 100 wt%, so dividing
        # by 100 gives mass fractions per cation that sum to ~1.
        cation_fraction = (cation_wt_pct / 100.0).astype(np.float32)

        self._frac = cation_fraction                            # [N, 10] float32
        self._y_wt = Y_wt                                       # [N, P] float32
        self._qap_names = qap_names                             # [N] object
        self._src = np.array(
            [z for _sym, z in CATIONS_ORDER],
            dtype=np.int64,
        )                                                       # [10] int64

        self.n_minerals: int = Y_wt.shape[1]
        self.phase_names: list[str] = phase_names
        self.qap_names: np.ndarray = qap_names

    def __len__(self) -> int:
        return len(self._frac)

    def __getitem__(self, idx: int) -> dict:
        return {
            "src":      torch.from_numpy(self._src),
            "frac":     torch.from_numpy(self._frac[idx]),
            "y_wt":     torch.from_numpy(self._y_wt[idx]),
            "qap_name": str(self._qap_names[idx]),
        }


# h2o_tot is excluded from model inputs: in the FC corpus it equals the
# CondHead H2O target h2o_path_wt (corr 1.0), so feeding it in target-leaks
# the H2O axis. The model takes the 10 anhydrous oxides; CondHead infers H2O.
V1_OXIDE_COLS = [
    "sio2", "al2o3", "cao", "mgo", "feo", "fe2o3",
    "k2o", "na2o", "tio2", "cr2o3",
]
V1_DROPPED_INPUT_COLS = ["h2o_tot"]
V1_COND_COLS = ("T_FC_end_used", "P_kbar", "h2o_path_wt")
V1_META_COLS = {
    "sample_id", "path_idx", "qap_name", "fe_provenance", "h2o_provenance",
    "fc_status", "T_FC_end_attrition",
    *V1_COND_COLS, *V1_OXIDE_COLS, *V1_DROPPED_INPUT_COLS,
}

# 9 cations after dropping h2o_tot (no H). v0's CATIONS_ORDER keeps H.
V1_CATIONS_ORDER: list[tuple[str, int]] = [
    ("Si", 14), ("Al", 13), ("Ca", 20), ("Mg", 12), ("Fe", 26),
    ("K",  19), ("Na", 11), ("Ti", 22), ("Cr", 24),
]


class TrainFCv1Dataset(Dataset):
    """v1 corpus dataset over `data/train_fc_v1.parquet`.

    Parquet on disk is untouched. Load-time filters:
      - keep `fc_status == "ok"` only
      - drop rows with `liq > liq_threshold` (default 0.10): physically real
        FC endpoints but reintroduce the v0 melt-inflation problem.

    Per-sample outputs:
      src         LongTensor  [9]    atomic numbers in V1_CATIONS_ORDER
      frac        FloatTensor [9]    cation mass fractions (wt% / 100)
      y_modal     FloatTensor [P]    mineral wt fractions (P = 18 in v1)
      y_cond_z    FloatTensor [3]    z-scored (T, P, H2O); inverse via
                                     `cond_mean`/`cond_std` attributes
      qap_name    str

    Class attributes after construction:
      n_minerals (int), phase_names (list[str]), qap_names (np.ndarray[str])
      cond_mean, cond_std (np.ndarray[3] float32)  -- saved for de-norm at
        inference; must be checkpointed alongside model weights.

    Parameters
    ----------
    parquet_path : str | Path
    liq_threshold : float
        Inclusive upper bound for liq mass fraction. ADR 006 / 2026-05-19
        journal pick 0.10.
    sample_id_filter : array-like[int] | None
        If provided, restrict to these sample_ids before stratification.
        Useful for `--limit N` smoke runs that subset bulks rather than rows
        (preserves the 8-paths-per-bulk multimodality signal).
    """

    def __init__(
        self,
        parquet_path: str | Path,
        liq_threshold: float = 0.10,
        sample_id_filter: np.ndarray | list[int] | None = None,
        cond_norm: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        parquet_path = Path(parquet_path)
        df = pd.read_parquet(parquet_path)

        df = df[df["fc_status"] == "ok"]
        df = df[df["liq"] <= liq_threshold].reset_index(drop=True)
        if sample_id_filter is not None:
            keep = pd.Series(sample_id_filter).unique()
            df = df[df["sample_id"].isin(keep)].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No rows survived filtering in {parquet_path}")

        phase_names = [c for c in df.columns if c not in V1_META_COLS]
        if len(phase_names) == 0:
            raise ValueError("No phase columns discovered; check column lists")

        X_oxide = df[V1_OXIDE_COLS].to_numpy(dtype=np.float32)
        cation_wt_pct = oxides_to_cations(X_oxide, V1_OXIDE_COLS,
                                          cations_order=V1_CATIONS_ORDER)
        cation_fraction = (cation_wt_pct / 100.0).astype(np.float32)

        y_modal = df[phase_names].to_numpy(dtype=np.float32)
        y_cond_raw = df[list(V1_COND_COLS)].to_numpy(dtype=np.float32)

        if cond_norm is None:
            cond_mean = y_cond_raw.mean(axis=0).astype(np.float32)
            cond_std = y_cond_raw.std(axis=0).astype(np.float32)
            cond_std = np.maximum(cond_std, 1e-6)
        else:
            cond_mean, cond_std = cond_norm
            cond_mean = np.asarray(cond_mean, dtype=np.float32)
            cond_std = np.asarray(cond_std, dtype=np.float32)
        y_cond_z = ((y_cond_raw - cond_mean) / cond_std).astype(np.float32)

        self._frac = cation_fraction
        self._y_modal = y_modal
        self._y_cond_z = y_cond_z
        self._y_cond_raw = y_cond_raw
        self._qap_names = df["qap_name"].to_numpy(dtype=object)
        self._sample_ids = df["sample_id"].to_numpy(dtype=np.int64)
        self._path_idx = df["path_idx"].to_numpy(dtype=np.int64)
        self._src = np.array([z for _sym, z in V1_CATIONS_ORDER], dtype=np.int64)

        self.n_minerals: int = y_modal.shape[1]
        self.phase_names: list[str] = phase_names
        self.qap_names: np.ndarray = self._qap_names
        self.sample_ids: np.ndarray = self._sample_ids
        self.path_idx: np.ndarray = self._path_idx
        self.cond_mean: np.ndarray = cond_mean
        self.cond_std: np.ndarray = cond_std
        self.cond_axes: tuple[str, ...] = V1_COND_COLS
        self.liq_threshold: float = liq_threshold
        self.parquet_path: Path = parquet_path

    def __len__(self) -> int:
        return len(self._frac)

    def __getitem__(self, idx: int) -> dict:
        return {
            "src":      torch.from_numpy(self._src),
            "frac":     torch.from_numpy(self._frac[idx]),
            "y_modal":  torch.from_numpy(self._y_modal[idx]),
            "y_cond_z": torch.from_numpy(self._y_cond_z[idx]),
            "qap_name": str(self._qap_names[idx]),
        }


class TrainFCv2Dataset(Dataset):
    """v2 supervised dataset: v1 corpus joined to complete.csv for trace inputs.

    Same modal + condition supervision as `TrainFCv1Dataset` (18 phases,
    3-axis z-scored conditions). Different INPUTS only: emits the 22-token
    v2 vocabulary (9 majors + 13 traces) instead of v1's 9-token majors-only.

    Trace coverage on the corpus subset is partial (Sr 76% / Zr 73% / Ba 72%
    / etc., 31% have all 8 high-coverage traces). Absent traces emit padding
    tokens (`src=0, frac=0`); the kingcrab Encoder's masked-mean pool already
    handles this via the `src == 0` convention.

    Parameters
    ----------
    parquet_path : str | Path
        v1 FC corpus parquet (data/train_fc_v1.parquet).
    complete_csv_path : str | Path
        complete.csv for the trace join.
    liq_threshold : float
        Same as v1.
    sample_id_filter : array-like[int] | None
        Same as v1.
    cond_norm : (mean, std) | None
        Same as v1.

    Attributes mirror `TrainFCv1Dataset` for compatibility with the existing
    training/eval loops; the only behavioral difference is per-sample shape:
    `src` and `frac` are `[22]` instead of `[9]`.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        complete_csv_path: str | Path,
        liq_threshold: float = 0.10,
        sample_id_filter: np.ndarray | list[int] | None = None,
        cond_norm: tuple[np.ndarray, np.ndarray] | None = None,
        majors_only: bool = False,
    ) -> None:
        # Lazy import: avoid a circular import (unlabeled imports OXIDE_TO_CATION
        # from this module at top-level).
        from cnmineral.data.unlabeled import (
            V2_N_TOKENS, V2_TRACES_ORDER, V2_TRACE_PPM_COLS,
            _combine_fe, oxides_traces_to_tokens,
        )

        parquet_path = Path(parquet_path)
        df = pd.read_parquet(parquet_path)
        df = df[df["fc_status"] == "ok"]
        df = df[df["liq"] <= liq_threshold].reset_index(drop=True)
        if sample_id_filter is not None:
            keep = pd.Series(sample_id_filter).unique()
            df = df[df["sample_id"].isin(keep)].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No rows survived filtering in {parquet_path}")

        phase_names = [c for c in df.columns if c not in V1_META_COLS]
        if len(phase_names) == 0:
            raise ValueError("No phase columns discovered; check column lists")

        # Trace join: one fetch from complete.csv keyed on sample_id. complete.csv
        # is ~790 MB but we only read sample_id + 13 ppm columns, ~tens of MB.
        complete_csv_path = Path(complete_csv_path)
        trace_use = ["sample_id"] + V2_TRACE_PPM_COLS
        trace_use_set = set(trace_use)
        traces_full = pd.read_csv(
            complete_csv_path,
            encoding="latin-1",
            low_memory=False,
            usecols=lambda c: c.lower() in trace_use_set,
        )
        traces_full.columns = [c.lower() for c in traces_full.columns]
        # complete.csv has duplicates on sample_id (multiple analyses of same
        # rock); take the first row per sample_id deterministically.
        traces_full = traces_full.drop_duplicates(subset=["sample_id"], keep="first")
        traces_lookup = traces_full.set_index("sample_id")

        # Build per-row (src [22], frac [22]). Majors come from the parquet's
        # own oxide columns (fe_provenance-adjusted FeO + Fe2O3 already split);
        # combine to total-Fe-as-FeO for the v2 helper's single-Fe convention.
        n = len(df)
        srcs = np.zeros((n, V2_N_TOKENS), dtype=np.int64)
        fracs = np.zeros((n, V2_N_TOKENS), dtype=np.float32)
        trace_present_count = 0
        for i in range(n):
            row = df.iloc[i]
            fe_eq = _combine_fe(row.to_dict())
            majors = {
                "sio2": float(row["sio2"]), "al2o3": float(row["al2o3"]),
                "cao": float(row["cao"]), "mgo": float(row["mgo"]),
                "feo": fe_eq, "fe2o3": 0.0,
                "k2o": float(row["k2o"]), "na2o": float(row["na2o"]),
                "tio2": float(row["tio2"]), "cr2o3": float(row["cr2o3"]),
            }
            sid = int(row["sample_id"])
            if sid in traces_lookup.index:
                trace_row = traces_lookup.loc[sid]
                traces_ppm = {sym.lower(): trace_row.get(f"{sym.lower()}_ppm")
                              for sym, _z in V2_TRACES_ORDER}
                if any(v is not None and not pd.isna(v) and float(v) > 0
                       for v in traces_ppm.values()):
                    trace_present_count += 1
            else:
                traces_ppm = None
            src, frac = oxides_traces_to_tokens(majors, traces_ppm)
            srcs[i] = src
            fracs[i] = frac

        y_modal = df[phase_names].to_numpy(dtype=np.float32)
        y_cond_raw = df[list(V1_COND_COLS)].to_numpy(dtype=np.float32)
        if cond_norm is None:
            cond_mean = y_cond_raw.mean(axis=0).astype(np.float32)
            cond_std = y_cond_raw.std(axis=0).astype(np.float32)
            cond_std = np.maximum(cond_std, 1e-6)
        else:
            cond_mean, cond_std = cond_norm
            cond_mean = np.asarray(cond_mean, dtype=np.float32)
            cond_std = np.asarray(cond_std, dtype=np.float32)
        y_cond_z = ((y_cond_raw - cond_mean) / cond_std).astype(np.float32)

        if majors_only:
            # Ablation cell (c): pretrained init evaluated on majors-only
            # input. Zero out the trace token slots so the encoder sees only
            # the 9 major tokens (kingcrab padding via src=0). The pretrained
            # encoder still benefits from having learned major<->trace
            # correlations during pretrain, even though it sees only majors at
            # supervised time.
            from cnmineral.data.unlabeled import V2_MAJORS_ORDER
            n_maj = len(V2_MAJORS_ORDER)
            srcs[:, n_maj:] = 0
            fracs[:, n_maj:] = 0.0

        self._src = srcs
        self._frac = fracs
        self._y_modal = y_modal
        self._y_cond_z = y_cond_z
        self._y_cond_raw = y_cond_raw
        self._qap_names = df["qap_name"].to_numpy(dtype=object)
        self._sample_ids = df["sample_id"].to_numpy(dtype=np.int64)
        self._path_idx = df["path_idx"].to_numpy(dtype=np.int64)

        self.n_minerals: int = y_modal.shape[1]
        self.phase_names: list[str] = phase_names
        self.qap_names: np.ndarray = self._qap_names
        self.sample_ids: np.ndarray = self._sample_ids
        self.path_idx: np.ndarray = self._path_idx
        self.cond_mean: np.ndarray = cond_mean
        self.cond_std: np.ndarray = cond_std
        self.cond_axes: tuple[str, ...] = V1_COND_COLS
        self.liq_threshold: float = liq_threshold
        self.parquet_path: Path = parquet_path
        self.n_tokens: int = V2_N_TOKENS
        self.trace_coverage: float = trace_present_count / max(n, 1)

    def __len__(self) -> int:
        return len(self._sample_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "src":      torch.from_numpy(self._src[idx]),
            "frac":     torch.from_numpy(self._frac[idx]),
            "y_modal":  torch.from_numpy(self._y_modal[idx]),
            "y_cond_z": torch.from_numpy(self._y_cond_z[idx]),
            "qap_name": str(self._qap_names[idx]),
        }
