"""Route MAGEMin feldspar instances to the `plag` / `afs` vocabulary tokens.

MAGEMin's igneous database has one feldspar solution model, so plagioclase and
alkali feldspar both arrive as `fsp` rows (two instances when the solvus is
open). This module recovers the identity from the composition, vendored from
mineralML's stoichiometric `FeldsparClassifier` (Shi et al.): molar An/Ab/Or
ternary coordinates classified against the Deer-Howie-Zussman two-feldspar
boundary curves as adapted by Thermobar/mineralML (Or tolerance widened to
0.075 for temperature dependence). Compositions inside the miscibility gap --
ternary feldspars the curves refuse to name -- are routed by the dominant
molar component (An >= Or -> plag, else afs). Called by `mineralogy_labels`
at label-build time; no mineralML runtime dependency.
"""

import numpy as np

_M_CAO = 56.0774
_M_NA2O = 61.9789
_M_K2O = 94.1960

# DHZ/Thermobar boundary curves (mineralML values), ascending-x for np.interp.
# Plagioclase field: Or <= f(An); alkali field: An <= f(Or); between is gap.
_AN_PLAG = np.array([0.00, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90, 1.00])
_OR_PLAG = np.array([0.10, 0.10, 0.15, 0.10, 0.075, 0.075, 0.075, 0.075, 0.075])
_OR_KFELD = np.array([0.10, 0.15, 0.37, 0.95])
_AN_KFELD = np.array([0.10, 0.15, 0.075, 0.075])


def anorthite_orthoclase(wt_cao, wt_na2o, wt_k2o):
    """Molar (An, Or) from feldspar CaO / Na2O / K2O mass fractions.

    An = Ca/(Ca+Na+K), Or = K/(Ca+Na+K) in cation moles (Na2O and K2O carry
    two cations per formula unit). Scale-invariant, so any consistent mass
    basis works. Inputs may be scalars or arrays; NaN is treated as 0.
    """
    wt_cao, wt_na2o, wt_k2o = np.broadcast_arrays(
        np.nan_to_num(np.asarray(wt_cao, dtype=float)),
        np.nan_to_num(np.asarray(wt_na2o, dtype=float)),
        np.nan_to_num(np.asarray(wt_k2o, dtype=float)),
    )
    ca = wt_cao / _M_CAO
    na = 2.0 * wt_na2o / _M_NA2O
    k = 2.0 * wt_k2o / _M_K2O
    total = ca + na + k
    with np.errstate(invalid="ignore", divide="ignore"):
        an = np.where(total > 0, ca / total, 0.0)
        orr = np.where(total > 0, k / total, 0.0)
    return an, orr


def classify_feldspar(wt_cao, wt_na2o, wt_k2o) -> np.ndarray:
    """Classify feldspar compositions -> array of "plag" / "afs" tokens."""
    an, orr = anorthite_orthoclase(wt_cao, wt_na2o, wt_k2o)
    an = np.atleast_1d(an)
    orr = np.atleast_1d(orr)
    in_plag_field = orr <= np.interp(an, _AN_PLAG, _OR_PLAG)
    in_afs_field = ~in_plag_field & (an <= np.interp(orr, _OR_KFELD, _AN_KFELD))
    gap_to_plag = ~in_plag_field & ~in_afs_field & (an >= orr)
    return np.where(in_plag_field | gap_to_plag, "plag", "afs")
