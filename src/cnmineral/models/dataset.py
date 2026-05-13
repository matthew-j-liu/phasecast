"""PyTorch dataset for the MAGEMin-derived training H5.

Loads `data/train_magemin.h5` (or any H5 matching the schema produced by
`scripts/parse_magemin.py`) and converts the 11-oxide MAGEMin schema into the
elemental wt% format CrabNet's encoder expects. Specifically:

- Drop nothing — the H5 already has the MAGEMin 11-oxide system after Position C
  conversion (SiO2, Al2O3, CaO, MgO, FeO, Fe2O3, K2O, Na2O, TiO2, Cr2O3, H2O).
- Convert each oxide column to a cation wt% via molar-mass factor.
- Sum FeO + Fe2O3 into Fe total (single cation column).
- Build a shared `src` tensor of atomic numbers, same for every sample
  (matches upstream CrabNet's MineralDataSet / SyntheticRockDataset
  convention).

Result: 10 cation columns (Si, Al, Ca, Mg, Fe, K, Na, Ti, Cr, H), one fraction
value per sample per cation. T and P live in /X but are not (yet) part of the
encoder input — they're stored for downstream conditional analysis.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
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


def oxides_to_cations(X_oxide: np.ndarray, x_cols: list[str]) -> np.ndarray:
    """Convert [N, 11] oxide wt% → [N, 10] cation wt% in CATIONS_ORDER.

    FeO and Fe2O3 both contribute to the Fe column. Any column in `x_cols` that
    isn't in OXIDE_TO_CATION (e.g., T_celsius, P_kbar) is silently ignored.
    """
    n_samples = X_oxide.shape[0]
    out = np.zeros((n_samples, len(CATIONS_ORDER)), dtype=np.float32)
    cation_to_idx = {sym: i for i, (sym, _z) in enumerate(CATIONS_ORDER)}

    for col_idx, col in enumerate(x_cols):
        if col not in OXIDE_TO_CATION:
            continue
        cation_sym, _atomic_num, factor = OXIDE_TO_CATION[col]
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
