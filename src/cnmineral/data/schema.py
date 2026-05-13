"""Oxide schema conversion: mineralML 11-oxide ↔ MAGEMin 11-oxide.

Two schemas in play:

mineralML / ICP / our user input:
    SiO2, TiO2, Al2O3, FeOt, MnO, MgO, CaO, Na2O, K2O, P2O5, Cr2O3

MAGEMin Xoxides (igneous database):
    SiO2, Al2O3, CaO, MgO, FeO, Fe2O3, K2O, Na2O, TiO2, Cr2O3, H2O

Differences handled here:
    - FeOt ↔ FeO + Fe2O3   (split via Fe3RatioStrategy; sum on the way back)
    - MnO, P2O5            (dropped going INTO MAGEMin; restored as 0 coming back)
    - H2O                  (defaulted to 0 going IN if missing; ~98% of complete.csv has no H2O)

Tier-based Fe provenance tracked per row:
    T1 ("measured")           : both `feo` and `fe2o3` directly reported
    T2 ("assumed_class")      : only a "_tot" column; ratio applied
    T2c ("derived_partial")   : one of feo/fe2o3 + a "_tot" column; remainder derived
    T3 ("no_iron")            : no Fe info at all; row drops
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .fe_redox import ClassRatioStrategy, Fe3RatioStrategy
from .h2o import ClassRangeH2OStrategy, H2OStrategy


MINERALML_OXIDES: list[str] = [
    "SiO2", "TiO2", "Al2O3", "FeOt", "MnO", "MgO",
    "CaO", "Na2O", "K2O", "P2O5", "Cr2O3",
]

MAGEMIN_OXIDES: list[str] = [
    "SiO2", "Al2O3", "CaO", "MgO", "FeO", "Fe2O3",
    "K2O", "Na2O", "TiO2", "Cr2O3", "H2O",
]

# Lowercase column names as they appear in complete.csv input rows
MAGEMIN_OXIDES_LOWER: list[str] = [c.lower() if c != "H2O" else "h2o_tot" for c in MAGEMIN_OXIDES]

# Mass-conversion factor: 1 g Fe2O3 → mass-equivalent g FeO at same Fe count
# = (2 × MW_FeO) / MW_Fe2O3 = (2 × 71.844) / 159.69 = 0.8998
FE2O3_TO_FEO_MASS = (2 * 71.844) / 159.69

FeProvenance = Literal["measured", "assumed_class", "derived_partial", "no_iron"]
H2OProvenance = Literal["measured", "assumed_class"]


@dataclass
class FeSplit:
    """Result of partitioning total iron into FeO and Fe2O3 components."""
    feo: float
    fe2o3: float
    provenance: FeProvenance


def detect_fe_columns(row: pd.Series) -> dict[str, float]:
    """Return whichever Fe columns are present (non-NaN, non-zero)."""
    out = {}
    for col in ("feo", "fe2o3", "feo_tot", "fe2o3_tot"):
        val = row.get(col)
        if val is None or pd.isna(val):
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if v > 0:
            out[col] = v
    return out


def split_fe(
    row: pd.Series,
    rock_class: str,
    fe3_strategy: Fe3RatioStrategy,
) -> FeSplit:
    """Partition total iron into FeO and Fe2O3 in wt%.

    Priority:
        1. T1: `feo` and `fe2o3` both present  →  use directly, no assumption
        2. T2: one of `_tot` columns present   →  split using ratio from fe3_strategy
        3. T2c: partial (one of feo/fe2o3 + a _tot)  →  derive remainder
        4. T3: nothing                         →  return all zeros + "no_iron"

    Returns FeSplit with .feo, .fe2o3 (in wt%) and .provenance label.
    """
    fe = detect_fe_columns(row)

    if "feo" in fe and "fe2o3" in fe:
        return FeSplit(feo=fe["feo"], fe2o3=fe["fe2o3"], provenance="measured")

    # Helper: from a total-Fe wt% (already in FeO-equivalent), split using ratio
    def _split_from_feo_total(feo_total_wt: float, r: float) -> tuple[float, float]:
        # r = Fe³⁺ / ΣFe (mole basis). For our purposes (small r), treat as mass basis.
        # FeO mass = (1 - r) * feo_total
        # Fe2O3 mass = r * feo_total / 0.8998 (converting back from FeO-mass-equivalent)
        feo = (1.0 - r) * feo_total_wt
        fe2o3 = (r * feo_total_wt) / FE2O3_TO_FEO_MASS
        return feo, fe2o3

    r = fe3_strategy(rock_class)

    # T2a: only fe2o3_tot reported (total Fe as Fe2O3)
    if "fe2o3_tot" in fe and "feo" not in fe and "fe2o3" not in fe:
        feo_total_wt = fe["fe2o3_tot"] * FE2O3_TO_FEO_MASS
        feo, fe2o3 = _split_from_feo_total(feo_total_wt, r)
        return FeSplit(feo=feo, fe2o3=fe2o3, provenance="assumed_class")

    # T2b: only feo_tot reported (total Fe as FeO)
    if "feo_tot" in fe and "feo" not in fe and "fe2o3" not in fe:
        feo, fe2o3 = _split_from_feo_total(fe["feo_tot"], r)
        return FeSplit(feo=feo, fe2o3=fe2o3, provenance="assumed_class")

    # T2c: partial — one of feo/fe2o3 + a _tot. Use the explicit one, derive the other.
    if "feo" in fe and "fe2o3_tot" in fe:
        # Fe2O3 = total Fe (as Fe2O3) − (FeO_measured · 1/0.8998)
        fe2o3 = fe["fe2o3_tot"] - (fe["feo"] / FE2O3_TO_FEO_MASS)
        fe2o3 = max(fe2o3, 0.0)
        return FeSplit(feo=fe["feo"], fe2o3=fe2o3, provenance="derived_partial")

    if "fe2o3" in fe and "feo_tot" in fe:
        feo = fe["feo_tot"] - (fe["fe2o3"] * FE2O3_TO_FEO_MASS)
        feo = max(feo, 0.0)
        return FeSplit(feo=feo, fe2o3=fe["fe2o3"], provenance="derived_partial")

    # Single non-tot value with nothing else: treat as the only Fe present, apply ratio
    # This is a softer case than T3 — we have some Fe info.
    if "feo" in fe:
        feo_total_wt = fe["feo"]  # treat as total Fe
        feo, fe2o3 = _split_from_feo_total(feo_total_wt, r)
        return FeSplit(feo=feo, fe2o3=fe2o3, provenance="derived_partial")

    if "fe2o3" in fe:
        feo_total_wt = fe["fe2o3"] * FE2O3_TO_FEO_MASS
        feo, fe2o3 = _split_from_feo_total(feo_total_wt, r)
        return FeSplit(feo=feo, fe2o3=fe2o3, provenance="derived_partial")

    # T3: nothing
    return FeSplit(feo=0.0, fe2o3=0.0, provenance="no_iron")


def mineralml_to_magemin(
    row: pd.Series,
    rock_class: str,
    fe3_strategy: Fe3RatioStrategy | None = None,
    h2o_strategy: H2OStrategy | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, float | str]:
    """Convert one row of complete.csv → MAGEMin 11-oxide schema + provenance.

    Returns dict with keys matching MAGEMIN_OXIDES_LOWER plus `fe_provenance`
    and `h2o_provenance`. Caller is responsible for LOI-normalizing if needed.

    Missing MnO/P2O5: silently dropped (not in MAGEMin schema).
    Missing Cr2O3: set to 0 (MAGEMin handles cleanly per docs).

    H2O handling (tiered, parallel to Fe redox):
        T1 ("measured"):       h2o_tot reported in the row → use directly
        T2 ("assumed_class"):  h2o_tot missing → sample from h2o_strategy
                               (default: Wallace 2005 + Plank 2013 per-class range)
    """
    if fe3_strategy is None:
        fe3_strategy = ClassRatioStrategy()
    if h2o_strategy is None:
        h2o_strategy = ClassRangeH2OStrategy()
    if rng is None:
        rng = np.random.default_rng()

    fe_split = split_fe(row, rock_class, fe3_strategy)

    # Read other oxides defensively (NaN → 0)
    def _get(key: str) -> float:
        v = row.get(key)
        if v is None or pd.isna(v):
            return 0.0
        return float(v)

    # H2O: T1 if reported (and > 0), else T2 sample from strategy
    h2o_raw = row.get("h2o_tot")
    if h2o_raw is not None and not pd.isna(h2o_raw) and float(h2o_raw) > 0.0:
        h2o_value = float(h2o_raw)
        h2o_prov: H2OProvenance = "measured"
    else:
        h2o_value = float(h2o_strategy(rock_class, rng))
        h2o_prov = "assumed_class"

    out: dict[str, float | str] = {
        "sio2":     _get("sio2"),
        "al2o3":    _get("al2o3"),
        "cao":      _get("cao"),
        "mgo":      _get("mgo"),
        "feo":      fe_split.feo,
        "fe2o3":    fe_split.fe2o3,
        "k2o":      _get("k2o"),
        "na2o":     _get("na2o"),
        "tio2":     _get("tio2"),
        "cr2o3":    _get("cr2o3"),
        "h2o_tot":  h2o_value,
        "fe_provenance": fe_split.provenance,
        "h2o_provenance": h2o_prov,
    }
    return out


def magemin_to_mineralml(comp_wt: dict[str, float]) -> dict[str, float]:
    """Convert a MAGEMin phase composition (11 oxides) → mineralML 11-oxide schema.

    Sums FeO + Fe2O3 → FeOt; restores MnO/P2O5 as 0 (not in MAGEMin output).
    Drops H2O.
    """
    feo = float(comp_wt.get("FeO", comp_wt.get("feo", 0.0)) or 0.0)
    fe2o3 = float(comp_wt.get("Fe2O3", comp_wt.get("fe2o3", 0.0)) or 0.0)
    feot = feo + fe2o3 * FE2O3_TO_FEO_MASS

    def _g(key: str) -> float:
        # MAGEMin output uses mixed case; accept either case
        return float(comp_wt.get(key, comp_wt.get(key.lower(), 0.0)) or 0.0)

    return {
        "SiO2":   _g("SiO2"),
        "TiO2":   _g("TiO2"),
        "Al2O3":  _g("Al2O3"),
        "FeOt":   feot,
        "MnO":    0.0,    # not in MAGEMin output
        "MgO":    _g("MgO"),
        "CaO":    _g("CaO"),
        "Na2O":   _g("Na2O"),
        "K2O":    _g("K2O"),
        "P2O5":   0.0,    # not in MAGEMin output
        "Cr2O3":  _g("Cr2O3"),
    }


def loi_normalize(oxides: dict[str, float], target_sum: float = 100.0) -> dict[str, float]:
    """Scale 11 oxide values so they sum exactly to target_sum (default 100 wt%).

    Operates on the numeric keys only; non-numeric keys (like fe_provenance) are
    passed through unchanged.
    """
    numeric = {k: v for k, v in oxides.items() if isinstance(v, (int, float))}
    total = sum(numeric.values())
    if total <= 0:
        return dict(oxides)
    scale = target_sum / total
    out = dict(oxides)
    for k, v in numeric.items():
        out[k] = v * scale
    return out
