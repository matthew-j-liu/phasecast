"""Candidate (P, H2O) priors for the generator-prior search (#79): a uniform
rectangle clipped to the broad box and the solubility bound, the coarse grid
of centers and widths the screen ranks, and the names of the two production
controls. `density` and `draw` copy the box construction of
`cnmineral.data.condition_proposal` (uniform in P, uniform in H2O given P) so
a corpus drawn from a scheme carries exactly the density the screen weighted
the #78 draws by.
"""

from dataclasses import dataclass

import numpy as np

from cnmineral.data.condition_proposal import P_HI, P_LO, W_CAP, lhs, w_max


@dataclass(frozen=True)
class Scheme:
    name: str
    p_center: float
    w_center: float
    p_half: float
    w_half: float

    @property
    def p_lo(self):
        return max(P_LO, self.p_center - self.p_half)

    @property
    def p_hi(self):
        return min(P_HI, self.p_center + self.p_half)

    @property
    def w_lo(self):
        return max(0.0, self.w_center - self.w_half)

    @property
    def w_hi(self):
        return min(W_CAP, self.w_center + self.w_half)


def density(scheme, p_kbar, h2o_wt):
    """Scheme density at (P, H2O); 0 outside the clipped rectangle."""
    p = np.asarray(p_kbar, dtype=float)
    w = np.asarray(h2o_wt, dtype=float)
    top = np.minimum(scheme.w_hi, w_max(p))
    span = top - scheme.w_lo
    inside = ((p >= scheme.p_lo) & (p <= scheme.p_hi)
              & (w >= scheme.w_lo) & (w <= top) & (span > 0))
    q = 1.0 / ((scheme.p_hi - scheme.p_lo) * np.where(span > 0, span, 1.0))
    return np.where(inside, q, 0.0)


def draw(scheme, n, rng):
    """Latin-hypercube draws from the scheme; returns (P, H2O, density)."""
    u = lhs(n, rng)
    p = scheme.p_lo + u[:, 0] * (scheme.p_hi - scheme.p_lo)
    top = np.minimum(scheme.w_hi, w_max(p))
    if np.any(top <= scheme.w_lo):
        raise ValueError(f"{scheme.name}: the solubility bound closes the H2O "
                         "interval inside the scheme's pressure range")
    w = scheme.w_lo + u[:, 1] * (top - scheme.w_lo)
    return p, w, density(scheme, p, w)


def area_fraction(scheme, n=4001):
    """The scheme's (P, H2O) area as a fraction of the broad box's."""
    p = np.linspace(P_LO, P_HI, n)
    broad = np.trapz(w_max(p), p)
    ps = np.linspace(scheme.p_lo, scheme.p_hi, n)
    span = np.clip(np.minimum(scheme.w_hi, w_max(ps)) - scheme.w_lo, 0.0, None)
    return float(np.trapz(span, ps) / broad)


P_CENTERS = (6.0, 8.0, 10.0)
W_CENTERS = (3.0, 4.5, 6.0)
WIDTHS = {"narrow": (2.0, 1.5), "wide": (3.0, 2.5)}
GRID = tuple(Scheme(f"p{p:g}_w{w:g}_{width}", p, w, p_half, w_half)
             for width, (p_half, w_half) in WIDTHS.items()
             for p in P_CENTERS for w in W_CENTERS)
CONTROLS = ("broad", "traced")
