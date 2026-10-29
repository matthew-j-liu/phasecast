"""The locked (P, H2O) proposal for Track B corpus generation.

Implements ADR 004 "Conditions (the bounded sampler)": Latin-Hypercube draws from
`q = Uniform(P [0.5, 12] kbar x H2O [0, min(8, S(P))] wt%)`, where `S(P)` is
the Newman & Lowenstern 2002 (VolatileCalc) rhyolite-curve H2O solubility and
8 wt% is the melt-inclusion supply ceiling. `q` is uniform in P and uniform in
H2O conditional on P, so the closed-form density stored per row is
`1 / ((P_HI - P_LO) * w_max(P))`; the inference layer reweights by `p/q`.

The solubility solve is a direct port of the pure-H2O rhyolite branch of
VolatileCalc (Redlich-Kwong H2O fugacity + Silver/Stolper melt speciation),
evaluated at the model's 850 C rhyolite standard state.

The second half of the module is the trace-conditional proposal (#45/#48),
adopted as the production sampler at the 2026-07-23 regen and wired into
`magemin_corpus.build_inputs`: an immobile-element discriminant cascade
assigns each bulk a soft tectonic-setting posterior, and draws come from a
posterior-weighted mixture of per-(maficity, setting) literature boxes --
falling back to per-class Wallace/Plank + Putirka ranges where traces are
missing -- plus a broad floor of the bounded proposal above so importance
weights stay finite on its full support. `draw_conditions` above is now that
floor component (and the sampler behind the pre-regen corpus); the stored `q`
carries the exact mixture density either way, so the inference `p/q`
reweighting is unchanged.
"""

import math
from dataclasses import dataclass

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


SETTINGS = ("arc", "ridge", "intraplate")

# Discriminant boundaries, all applied as logistic gates in log10-ratio space.
# Pearce 2008: the MORB-OIB mantle array's upper edge is the chondritic Th/Nb
# line (Sun & McDonough 1989 chondrite: Th 0.029 / Nb 0.246 ppm); arc rocks
# plot above it.  Within the array, ridge vs intraplate splits at the
# geometric mean of the N-MORB (2.33/3.05) and E-MORB (8.3/2.37) Nb/Yb of the
# same reference set -- the enriched-source transition.
TH_NB_CHONDRITE = 0.029 / 0.246
NBYB_RIDGE_INTRA = math.sqrt((2.33 / 3.05) * (8.3 / 2.37))
# Shervais 1982: Ti/V < 20 arc tholeiites, 20-50 MORB, > 50 OIB/alkaline.
TIV_ARC, TIV_OIB = 20.0, 50.0
# Winchester & Floyd 1977 subalkaline | alkaline boundary.
NB_Y_ALKALINE = 0.67
TI_FRAC_TIO2 = 47.867 / 79.866

_MAFICITY = {
    "ultramafic": "mafic",
    "mafic-volcanic": "mafic",
    "mafic-plutonic": "mafic",
    "intermediate-volcanic": "intermediate",
    "intermediate-plutonic": "intermediate",
    "felsic-volcanic": "felsic",
    "felsic-plutonic": "felsic",
    "foid-bearing": "intermediate",
    "other": "intermediate",
}

# (p_lo kbar, p_hi kbar, w_lo wt%, w_hi wt%) per (maficity, setting).
# H2O anchors: arc from Plank et al. 2013 (4 +/- 2 wt%, felsic melts at the
# wet end), ridge and OIB from Wallace 2005, alkaline intraplate and evolved
# ridge/intraplate widened toward the wet side of their mafic parents.
# P narrows less than H2O: storage depth is only weakly setting-determined.
TRACE_TIER_BOXES = {
    ("mafic", "arc"): (0.5, 6.0, 2.0, 6.0),
    ("intermediate", "arc"): (0.5, 6.0, 2.0, 6.0),
    ("felsic", "arc"): (0.5, 6.0, 3.0, 6.0),
    ("mafic", "ridge"): (0.5, 4.0, 0.05, 0.5),
    ("intermediate", "ridge"): (0.5, 4.0, 0.2, 1.5),
    ("felsic", "ridge"): (0.5, 4.0, 1.0, 3.0),
    ("mafic", "intraplate"): (1.0, 8.0, 0.3, 1.5),
    ("intermediate", "intraplate"): (1.0, 8.0, 0.5, 3.0),
    ("felsic", "intraplate"): (1.0, 8.0, 1.0, 4.0),
}

# Fallback tier: ADR 003 per-class H2O ranges (Wallace 2005 / Plank et al.
# 2013 / Burnham 1979 / Holtz et al. 2001 / Best 2003) and Putirka 2008
# per-class P ranges, both recovered from the pre-equilibrium-corpus sampler.
WALLACE_PLANK_2013 = {
    "basalt": (0.1, 1.0),
    "trachybasalt": (0.5, 3.0),
    "basanite": (0.5, 2.5),
    "tephrite": (1.0, 4.0),
    "picrite": (0.1, 1.0),
    "komatiite": (0.05, 0.5),
    "subalkalic gabbro": (0.1, 1.5),
    "alkalic gabbro": (0.5, 2.5),
    "gabbro": (0.5, 2.0),
    "norite": (0.5, 2.0),
    "anorthosite": (0.1, 1.0),
    "troctolite": (0.2, 1.5),
    "andesite": (2.0, 6.0),
    "basaltic andesite": (2.0, 5.0),
    "trachyandesite": (1.0, 4.0),
    "phonotephrite": (1.0, 4.0),
    "tephriphonolite": (1.0, 4.0),
    "latite": (1.0, 4.0),
    "diorite": (2.0, 5.0),
    "gabbroic diorite": (1.5, 4.5),
    "monzodiorite": (1.0, 4.0),
    "monzogabbro": (1.0, 3.5),
    "monzonite": (1.5, 4.0),
    "syenite": (1.0, 4.0),
    "alkali feldspar syenite": (1.0, 3.5),
    "quartz monzonite": (2.0, 5.0),
    "quartz monzodiorite": (1.5, 4.5),
    "quartz diorite": (2.0, 5.0),
    "quartz syenite": (1.5, 4.0),
    "dacite": (3.0, 7.0),
    "rhyolite": (2.0, 7.0),
    "trachyte": (1.0, 5.0),
    "trachydacite": (1.5, 5.0),
    "phonolite": (1.0, 5.0),
    "tonalite": (3.0, 6.0),
    "granodiorite": (3.0, 6.0),
    "monzogranite": (3.0, 6.0),
    "syenogranite": (3.0, 6.0),
    "granite": (3.0, 6.0),
    "alkali feldspar granite": (2.0, 5.0),
    "foidite": (1.0, 5.0),
    "nephelinite": (1.0, 5.0),
    "leucitite": (1.0, 4.0),
    "foid syenite": (1.0, 4.0),
    "foid gabbro": (1.0, 3.5),
    "foid monzosyenite": (1.0, 3.5),
    "foid monzodiorite": (1.0, 3.5),
    "foid-bearing diorite": (1.5, 4.0),
    "foid-bearing latite": (1.0, 4.0),
    "dunite": (0.0, 0.1),
    "harzburgite": (0.0, 0.1),
    "lherzolite": (0.0, 0.2),
    "wehrlite": (0.0, 0.2),
    "websterite": (0.0, 0.2),
    "orthopyroxenite": (0.0, 0.2),
    "clinopyroxenite": (0.0, 0.2),
    "peridotite": (0.0, 0.1),
    "mantle peridotite": (0.0, 0.1),
    "mantle pyroxenite": (0.0, 0.2),
    "default": (0.5, 3.0),
}

PUTIRKA_2008_P = {
    "basalt": (1.0, 5.0),
    "trachybasalt": (1.0, 5.0),
    "basanite": (2.0, 10.0),
    "tephrite": (2.0, 7.0),
    "picrite": (1.0, 10.0),
    "komatiite": (1.0, 10.0),
    "subalkalic gabbro": (2.0, 7.0),
    "alkalic gabbro": (2.0, 7.0),
    "gabbro": (2.0, 7.0),
    "norite": (2.0, 7.0),
    "anorthosite": (5.0, 10.0),
    "troctolite": (2.0, 7.0),
    "andesite": (1.0, 5.0),
    "basaltic andesite": (1.0, 5.0),
    "trachyandesite": (1.0, 5.0),
    "phonotephrite": (2.0, 7.0),
    "tephriphonolite": (2.0, 7.0),
    "latite": (1.0, 5.0),
    "diorite": (2.0, 7.0),
    "gabbroic diorite": (2.0, 7.0),
    "monzodiorite": (2.0, 7.0),
    "monzogabbro": (2.0, 7.0),
    "monzonite": (2.0, 7.0),
    "syenite": (1.0, 5.0),
    "alkali feldspar syenite": (1.0, 5.0),
    "quartz monzonite": (2.0, 5.0),
    "quartz monzodiorite": (2.0, 5.0),
    "quartz diorite": (2.0, 5.0),
    "quartz syenite": (2.0, 5.0),
    "dacite": (1.0, 3.0),
    "rhyolite": (1.0, 3.0),
    "trachyte": (1.0, 3.0),
    "trachydacite": (1.0, 3.0),
    "phonolite": (1.0, 3.0),
    "tonalite": (2.0, 7.0),
    "granodiorite": (2.0, 5.0),
    "monzogranite": (2.0, 5.0),
    "syenogranite": (2.0, 5.0),
    "granite": (2.0, 5.0),
    "alkali feldspar granite": (2.0, 5.0),
    "foidite": (2.0, 10.0),
    "nephelinite": (2.0, 10.0),
    "leucitite": (1.0, 7.0),
    "foid syenite": (1.0, 7.0),
    "foid gabbro": (2.0, 7.0),
    "foid monzosyenite": (2.0, 7.0),
    "foid monzodiorite": (2.0, 7.0),
    "foid-bearing diorite": (2.0, 7.0),
    "foid-bearing latite": (1.0, 5.0),
    "dunite": (5.0, 25.0),
    "harzburgite": (5.0, 25.0),
    "lherzolite": (5.0, 30.0),
    "wehrlite": (5.0, 20.0),
    "websterite": (5.0, 20.0),
    "orthopyroxenite": (5.0, 20.0),
    "clinopyroxenite": (5.0, 20.0),
    "peridotite": (5.0, 25.0),
    "mantle peridotite": (10.0, 30.0),
    "mantle pyroxenite": (10.0, 25.0),
    "default": (2.0, 10.0),
}


SETTING_SIGMA = 0.15
FELSIC_SIGMA_MULT = 2.0
FLOOR_EPS = 0.08


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, z))))


def _ratio(traces, num, den):
    a, b = traces.get(num), traces.get(den)
    if a is None or b is None:
        return None
    a, b = float(a), float(b)
    if not (math.isfinite(a) and math.isfinite(b)) or a <= 0 or b <= 0:
        return None
    return a / b


def setting_posterior(traces, sigma):
    """Soft tectonic-setting weights from immobile-element ratios.

    `traces` maps gard2019 column names (`th_ppm`, `nb_ppm`, `yb_ppm`,
    `v_ppm`, `y_ppm`, plus `tio2` in wt%) to values; missing/NaN/<=0 entries
    disqualify a ratio.  Returns `((w_arc, w_ridge, w_intraplate, w_fallback),
    rung)` summing to 1.  First applicable discriminant wins; the Nb/Y rung
    resolves only the alkaline (intraplate) share, so its subalkaline mass
    falls through to the class tables; no usable ratio puts all mass on
    fallback with `rung=None`.
    """
    th_nb = _ratio(traces, "th_ppm", "nb_ppm")
    nb_yb = _ratio(traces, "nb_ppm", "yb_ppm")
    if th_nb is not None and nb_yb is not None:
        p_arc = _sigmoid((math.log10(th_nb) - math.log10(TH_NB_CHONDRITE)) / sigma)
        p_en = _sigmoid((math.log10(nb_yb) - math.log10(NBYB_RIDGE_INTRA)) / sigma)
        return (p_arc, (1 - p_arc) * (1 - p_en), (1 - p_arc) * p_en, 0.0), "th_nb_yb"
    tio2, v = traces.get("tio2"), traces.get("v_ppm")
    if (tio2 is not None and v is not None
            and math.isfinite(float(tio2)) and math.isfinite(float(v))
            and float(tio2) > 0 and float(v) > 0):
        r = math.log10(float(tio2) * 1e4 * TI_FRAC_TIO2 / float(v))
        p_arc = _sigmoid((math.log10(TIV_ARC) - r) / sigma)
        p_oib = _sigmoid((r - math.log10(TIV_OIB)) / sigma)
        w_ridge = max(0.0, 1.0 - p_arc - p_oib)
        tot = p_arc + p_oib + w_ridge
        return (p_arc / tot, w_ridge / tot, p_oib / tot, 0.0), "ti_v"
    nb_y = _ratio(traces, "nb_ppm", "y_ppm")
    if nb_y is not None:
        p_alk = _sigmoid((math.log10(nb_y) - math.log10(NB_Y_ALKALINE)) / sigma)
        return (0.0, 0.0, p_alk, 1.0 - p_alk), "nb_y"
    return (0.0, 0.0, 0.0, 1.0), None


@dataclass(frozen=True)
class Box:
    """A (P, H2O) rectangle with P bounds already clipped to the proposal
    support and to pressures where the solubility bound leaves at least a
    0.05 wt% conditional H2O interval above `w_lo`."""

    p_lo: float
    p_hi: float
    w_lo: float
    w_hi: float


def _make_box(p_lo, p_hi, w_lo, w_hi):
    p, s, p_star = _grid()
    if w_lo + 0.05 > s[0]:
        p_floor = float(np.interp(w_lo + 0.05, s, p))
    else:
        p_floor = P_LO
    return Box(max(P_LO, p_lo, p_floor), min(P_HI, p_hi), w_lo, w_hi)


def _box_q(box, p, w):
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)
    wm = np.minimum(box.w_hi, w_max(p))
    span = wm - box.w_lo
    inside = ((p >= box.p_lo) & (p <= box.p_hi)
              & (w >= box.w_lo) & (w <= wm) & (span > 0))
    q = 1.0 / ((box.p_hi - box.p_lo) * np.where(span > 0, span, 1.0))
    return np.where(inside, q, 0.0)


def _box_draw(box, n, rng):
    u = lhs(n, rng)
    p = box.p_lo + u[:, 0] * (box.p_hi - box.p_lo)
    w = box.w_lo + u[:, 1] * (np.minimum(box.w_hi, w_max(p)) - box.w_lo)
    return p, w


def _largest_remainder(n, weights):
    ideal = np.asarray(weights, dtype=float) * n
    counts = np.floor(ideal).astype(int)
    order = np.argsort(-(ideal - counts), kind="stable")
    for i in order[: n - counts.sum()]:
        counts[i] += 1
    return counts


@dataclass(frozen=True)
class BulkProposal:
    """One bulk's trace-conditional proposal: a weighted mixture of boxes
    plus the broad floor.  `weights` aligns with `boxes` and sums with
    `floor` to 1; `q` is the exact mixture density (each box uses the same
    uniform-P x uniform-H2O|P construction as the broad sampler, so no area
    integral is needed)."""

    boxes: tuple
    weights: tuple
    floor: float
    tier: str
    rung: object
    setting_weights: tuple

    def q(self, p_kbar, h2o_wt):
        q = self.floor * q_density(p_kbar, h2o_wt)
        for box, wt in zip(self.boxes, self.weights):
            q = q + wt * _box_q(box, p_kbar, h2o_wt)
        return q

    def draw(self, n, rng):
        counts = _largest_remainder(n, self.weights + (self.floor,))
        ps, ws = [], []
        for box, k in zip(self.boxes, counts[:-1]):
            if k > 0:
                p, w = _box_draw(box, int(k), rng)
                ps.append(p)
                ws.append(w)
        if counts[-1] > 0:
            p, w, _ = draw_conditions(int(counts[-1]), rng)
            ps.append(p)
            ws.append(w)
        p = np.concatenate(ps)
        w = np.concatenate(ws)
        return p, w, self.q(p, w)


def bulk_proposal(qap_name, traces):
    """Build the trace-conditional proposal for one bulk.

    Cascade posterior -> posterior-weighted (maficity, setting) boxes, with
    the fallback share drawn from the per-class Wallace/Plank + Putirka box;
    every bulk keeps `FLOOR_EPS` mass on the broad floor.  `tier` is
    'trace' when the discriminant carries the majority of the non-floor
    mass, else 'fallback'.
    """
    from cnmineral.eval.supergroup import qap_to_supergroup

    bucket = _MAFICITY.get(qap_to_supergroup(qap_name), "intermediate")
    sigma = SETTING_SIGMA * (FELSIC_SIGMA_MULT if bucket == "felsic" else 1.0)
    (w_arc, w_ridge, w_intra, w_fb), rung = setting_posterior(traces, sigma)

    boxes, weights = [], []
    for setting, wt in zip(SETTINGS, (w_arc, w_ridge, w_intra)):
        if wt > 1e-6:
            boxes.append(_make_box(*TRACE_TIER_BOXES[(bucket, setting)]))
            weights.append(wt)
    if w_fb > 1e-6:
        key = qap_name if qap_name in WALLACE_PLANK_2013 else "default"
        w_lo, w_hi = WALLACE_PLANK_2013[key]
        p_lo, p_hi = PUTIRKA_2008_P[key if key in PUTIRKA_2008_P else "default"]
        boxes.append(_make_box(p_lo, p_hi, w_lo, w_hi))
        weights.append(w_fb)
    tot = sum(weights)
    weights = tuple(wt / tot * (1.0 - FLOOR_EPS) for wt in weights)
    tier = "trace" if rung is not None and (w_arc + w_ridge + w_intra) >= w_fb else "fallback"
    return BulkProposal(tuple(boxes), weights, FLOOR_EPS, tier, rung,
                        (w_arc, w_ridge, w_intra))
