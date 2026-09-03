"""Bulk -> MAGEMin oxide conversion with tiered Fe provenance.

Two schemas in play:

    input bulk (gard / mineralML / ICP):
        SiO2, TiO2, Al2O3, FeOt (or FeO+Fe2O3), MnO, MgO, CaO, Na2O, K2O, P2O5, Cr2O3
    MAGEMin Xoxides (igneous database):
        SiO2, Al2O3, CaO, MgO, FeO, Fe2O3, K2O, Na2O, TiO2, Cr2O3, H2O

Handled here:
    - FeOt <-> FeO + Fe2O3   (split via `Fe3RatioStrategy`)
    - MnO, P2O5              (dropped going INTO MAGEMin)
    - H2O                    (injected by the caller -- sampled per (P, H2O) draw,
                              not read from the bulk; see ADR 004 "Conditions")

Fe provenance tier per bulk (also the item-10 provenance seed):
    measured         : both `feo` and `fe2o3` reported          -> no assumption
    assumed_class    : only a `_tot` column; class ratio applied
    derived_partial  : one of feo/fe2o3 + a `_tot`; remainder derived (a real
                       per-rock ratio).  Also a lone feo/fe2o3 with no total,
                       which is treated as total iron and takes the class
                       ratio; that path dominates the tier, so the name
                       oversells how much of it carries redox information.
    no_iron          : no Fe info at all
"""

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

from .fe_redox import ClassRatioStrategy, Fe3RatioStrategy


MAGEMIN_OXIDES = [
    "SiO2", "Al2O3", "CaO", "MgO", "FeO", "Fe2O3",
    "K2O", "Na2O", "TiO2", "Cr2O3", "H2O",
]
MAGEMIN_OXIDES_LOWER = [c.lower() if c != "H2O" else "h2o_tot" for c in MAGEMIN_OXIDES]

# 1 g Fe2O3 -> mass-equivalent g FeO at the same Fe count = 2*MW_FeO / MW_Fe2O3.
FE2O3_TO_FEO_MASS = (2 * 71.844) / 159.69

FeProvenance = Literal["measured", "assumed_class", "derived_partial", "no_iron"]


@dataclass(frozen=True)
class FeSplit:
    """Result of partitioning total iron into FeO and Fe2O3 (wt%)."""
    feo: float
    fe2o3: float
    provenance: FeProvenance


def detect_fe_columns(row) -> dict:
    """Return whichever Fe columns are present (non-NaN, positive)."""
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


def _split_from_feo_total(feo_total_wt, r):
    # r = Fe3+/SigmaFe. FeO mass = (1-r)*total; Fe2O3 = r*total / mass-factor
    # (converting back out of FeO-mass-equivalent). Treated on a mass basis for
    # the small r regime we operate in.
    feo = (1.0 - r) * feo_total_wt
    fe2o3 = (r * feo_total_wt) / FE2O3_TO_FEO_MASS
    return feo, fe2o3


def split_fe(row, rock_class, fe3_strategy) -> FeSplit:
    """Partition total iron into FeO and Fe2O3 (wt%) with a provenance tier.

    Priority: both measured -> single `_tot` + class ratio -> partial derive ->
    a lone feo/fe2o3 treated as total -> no iron.
    """
    fe = detect_fe_columns(row)

    if "feo" in fe and "fe2o3" in fe:
        return FeSplit(feo=fe["feo"], fe2o3=fe["fe2o3"], provenance="measured")

    r = fe3_strategy(rock_class)

    if "fe2o3_tot" in fe and "feo" not in fe and "fe2o3" not in fe:
        feo_total_wt = fe["fe2o3_tot"] * FE2O3_TO_FEO_MASS
        feo, fe2o3 = _split_from_feo_total(feo_total_wt, r)
        return FeSplit(feo=feo, fe2o3=fe2o3, provenance="assumed_class")

    if "feo_tot" in fe and "feo" not in fe and "fe2o3" not in fe:
        feo, fe2o3 = _split_from_feo_total(fe["feo_tot"], r)
        return FeSplit(feo=feo, fe2o3=fe2o3, provenance="assumed_class")

    if "feo" in fe and "fe2o3_tot" in fe:
        fe2o3 = max(fe["fe2o3_tot"] - (fe["feo"] / FE2O3_TO_FEO_MASS), 0.0)
        return FeSplit(feo=fe["feo"], fe2o3=fe2o3, provenance="derived_partial")

    if "fe2o3" in fe and "feo_tot" in fe:
        feo = max(fe["feo_tot"] - (fe["fe2o3"] * FE2O3_TO_FEO_MASS), 0.0)
        return FeSplit(feo=feo, fe2o3=fe["fe2o3"], provenance="derived_partial")

    if "feo" in fe:
        feo, fe2o3 = _split_from_feo_total(fe["feo"], r)
        return FeSplit(feo=feo, fe2o3=fe2o3, provenance="derived_partial")

    if "fe2o3" in fe:
        feo, fe2o3 = _split_from_feo_total(fe["fe2o3"] * FE2O3_TO_FEO_MASS, r)
        return FeSplit(feo=feo, fe2o3=fe2o3, provenance="derived_partial")

    return FeSplit(feo=0.0, fe2o3=0.0, provenance="no_iron")


def bulk_to_magemin(row, rock_class, h2o_wt, fe3_strategy: "Optional[Fe3RatioStrategy]" = None) -> dict:
    """Convert one bulk row -> MAGEMin 11-oxide input (lowercase keys) + provenance.

    `h2o_wt` is injected by the caller (the sampled H2O for this draw), not read
    from the bulk. MnO/P2O5 are dropped (absent from the MAGEMin schema); missing
    Cr2O3 becomes 0.
    """
    if fe3_strategy is None:
        fe3_strategy = ClassRatioStrategy()

    fe = split_fe(row, rock_class, fe3_strategy)

    def _get(key):
        v = row.get(key)
        return 0.0 if v is None or pd.isna(v) else float(v)

    return {
        "sio2": _get("sio2"),
        "al2o3": _get("al2o3"),
        "cao": _get("cao"),
        "mgo": _get("mgo"),
        "feo": fe.feo,
        "fe2o3": fe.fe2o3,
        "k2o": _get("k2o"),
        "na2o": _get("na2o"),
        "tio2": _get("tio2"),
        "cr2o3": _get("cr2o3"),
        "h2o_tot": float(h2o_wt),
        "fe_provenance": fe.provenance,
    }


def loi_normalize(oxides, target_sum=100.0) -> dict:
    """Scale the numeric oxide values so they sum to `target_sum` (default 100 wt%).

    Non-numeric keys (e.g. `fe_provenance`) pass through unchanged.
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
