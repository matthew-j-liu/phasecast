"""The locked (P, H2O) proposal for Track B corpus generation.

Implements corpus_regen_spec "Bounded sampler": Latin-Hypercube draws from
`q = Uniform(P [0.5, 12] kbar x H2O [0, min(8, S(P))] wt%)`, where `S(P)` is
the Newman & Lowenstern 2002 (VolatileCalc) rhyolite-curve H2O solubility and
8 wt% is the melt-inclusion supply ceiling. `q` is uniform in P and uniform in
H2O conditional on P, so the closed-form density stored per row is
`1 / ((P_HI - P_LO) * w_max(P))`; the inference layer reweights by `p/q`.

The solubility solve is a direct port of the pure-H2O rhyolite branch of
VolatileCalc (Redlich-Kwong H2O fugacity + Silver/Stolper melt speciation),
evaluated at the model's 850 C rhyolite standard state.
"""

import numpy as np

P_LO, P_HI = 0.5, 12.0
W_CAP = 8.0
T_CURVE_C = 850.0


def _fna(tk):
    return (166800000 - 193080 * (tk - 273.15) + 186.4 * (tk - 273.15) ** 2
            - 0.071288 * (tk - 273.15) ** 3) * 1.01325


def _mrk_fh2o(p_bar, tk):
    """Redlich-Kwong fugacity of pure H2O (bar)."""
    r = 83.14321
    b = 14.6
    a = _fna(tk)

    def fnf(v):
        return r * tk / (v - b) - a / ((v * v + b * v) * tk ** 0.5) - p_bar

    v2 = b + 5.0
    q = 1.0
    v1 = 0.0
    while abs(v2 - v1) >= 0.00001:
        v1 = v2
        f1 = (fnf(v1 + 0.01) - fnf(v1)) / 0.01
        v2 = v1 - q * fnf(v1) / f1
        f2 = (fnf(v2 + 0.01) - fnf(v2)) / 0.01
        if f2 * f1 <= 0:
            q = q / 2.0
        if abs(v2 - v1) > 0.00001:
            f1 = f2
    v = v2
    g = (np.log(v / (v - b)) + b / (v - b)
         - 2 * a * np.log((v + b) / v) / (r * tk ** 1.5 * b)
         + (np.log((v + b) / v) - b / (v + b)) * a * b / (r * tk ** 1.5 * b ** 2)
         - np.log(p_bar * v / (r * tk)))
    return np.exp(g) * p_bar


def _xh2om(wt_h2o):
    """Mole fraction of molecular H2O in rhyolite melt (Silver/Stolper speciation)."""
    if wt_h2o < 0.6:
        return np.exp(-5.526) * wt_h2o ** 1.977
    snh2o = wt_h2o / 18.015
    sno = (100 - wt_h2o) / 32.5
    xh2o = snh2o / (snh2o + sno)
    xo = 1 - xh2o
    xohm = 0.01
    if xh2o - 0.6 * xohm <= 0:
        xohm = 0.999 * xh2o + 0.005
    derath = 1.0
    while abs(derath) >= 0.00001:
        fx = (9.345 - 4.304 * (xohm - 1) - 2 * 6.277 * (xo - xohm)
              - 2 * 2.328 * (xh2o - xohm)
              + np.log(xohm ** 2 / ((xh2o - 0.5 * xohm) * (xo - 0.5 * xohm))))
        fxp = (-4.304 + 2 * (6.277 + 2.328)
               + (2 * (xh2o - 0.5 * xohm) * (xo - 0.5 * xohm)
                  + 0.5 * xohm * ((xh2o - 0.5 * xohm) + (xo - 0.5 * xohm)))
               / (xohm * (xh2o - 0.5 * xohm) * (xo - 0.5 * xohm)))
        derath = fx / fxp
        xohm = xohm - derath
    return xh2o - 0.5 * xohm


def nl_rhyolite_solubility(p_kbar, t_c=T_CURVE_C):
    """Dissolved H2O (wt%) at pure-H2O saturation, N&L 2002 rhyolite curve.

    Scalar. Solves for the melt water content whose equilibrium vapor is pure
    H2O (GH2O = 1) at `p_kbar`, by bisection.
    """
    tk = t_c + 273.15
    p_bar = p_kbar * 1000.0
    fh2o = _mrk_fh2o(p_bar, tk)
    # GH2O = 1 condition, rhyolite standard state 3.45 wt% / 799 bar / 850 C
    # (xH2Om 0.0323, fH2O 712.2 bar), partial molar volume 5 cc/mol shift,
    # Delta H 4420 cal/mol (Silver et al. 1990).
    target = 0.0323 * (fh2o / 712.2) * np.exp(
        -5 * (p_bar - 799) / (41.84 * 1.9872 * tk)
        + 4420 / 1.9872 * (1 / tk - 1 / 1123.16))
    lo, hi = 0.01, 20.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if _xh2om(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


_GRID = None


def _grid():
    # Dense S(P) table from P_LO up to the cap crossover (S = W_CAP); above it
    # w_max is the constant cap, so N&L is never evaluated past ~3 kbar --
    # inside its calibration range, per the spec's argument.
    global _GRID
    if _GRID is None:
        lo, hi = P_LO, P_HI
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if nl_rhyolite_solubility(mid) < W_CAP:
                lo = mid
            else:
                hi = mid
        p_star = 0.5 * (lo + hi)
        p = np.arange(P_LO, p_star + 0.01, 0.01)
        s = np.array([nl_rhyolite_solubility(pi) for pi in p])
        _GRID = (p, np.minimum(s, W_CAP), p_star)
    return _GRID


def w_max(p_kbar):
    """Upper H2O bound of the proposal at pressure `p_kbar`: min(8, S(P)). Vectorized."""
    p, s, p_star = _grid()
    return np.where(np.asarray(p_kbar, dtype=float) >= p_star,
                    W_CAP, np.interp(p_kbar, p, s))


def q_density(p_kbar, h2o_wt):
    """Proposal density q(P, H2O); 0 outside the support. Vectorized."""
    p = np.asarray(p_kbar, dtype=float)
    w = np.asarray(h2o_wt, dtype=float)
    wm = w_max(p)
    q = 1.0 / ((P_HI - P_LO) * wm)
    inside = (p >= P_LO) & (p <= P_HI) & (w >= 0.0) & (w <= wm)
    return np.where(inside, q, 0.0)


def lhs(n, rng, d=2):
    """d-dim Latin Hypercube on the unit cube (stratified, unbiased marginals)."""
    u = np.empty((n, d))
    for k in range(d):
        perm = rng.permutation(n)
        u[:, k] = (perm + rng.random(n)) / n
    return u


def draw_conditions(n, rng):
    """LHS draws from the locked proposal. Returns (P kbar, H2O wt%, q) arrays.

    P is uniform on [P_LO, P_HI]; H2O is uniform on [0, w_max(P)] given P (the
    unit-square LHS is warped vertically by the solubility bound, preserving
    stratification in rank space).
    """
    u = lhs(n, rng)
    p = P_LO + u[:, 0] * (P_HI - P_LO)
    w = u[:, 1] * w_max(p)
    return p, w, q_density(p, w)
