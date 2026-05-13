"""Rock-class-conditional T/P sampling strategies for MAGEMin runs.

Mirrors the Fe³⁺/ΣFe strategy pattern in `fe_redox.py`. Each strategy takes
a rock class (qap_name from complete.csv) and returns K samples of (T, P)
from a distribution conditioned on that class.

Default strategy `TPRangeStrategy` draws uniformly from per-class T/P ranges
from Putirka (2008) thermobarometer literature and Best (2003) textbook.
This implements Position B's "marginalize over rock-class-conditional T/P
prior" framing.

Alternative `TPGridStrategy` uses the same brute 28-point grid as Stage 1
(T = 800-1400, P = 1-20 kbar) for ablation/reproduction.

References:
    Putirka K. (2008) "Thermometers and barometers for volcanic systems."
        Reviews in Mineralogy and Geochemistry 69:61-120.
    Best M.G. (2003) Igneous and Metamorphic Petrology, 2nd ed. Wiley-Blackwell.
    Stolper E. (1980) "A phase diagram for mid-ocean ridge basalts: preliminary
        results and implications for petrogenesis." CMP 74:13-27.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


# Per-qap_name T/P ranges. T in °C, P in kbar.
#
# Volcanic classes use shallow crustal pressures (1-5 kbar) reflecting eruption
# conditions. Plutonic equivalents use slightly higher pressures (2-7 kbar)
# reflecting mid-crustal emplacement. Ultramafic mantle-derived rocks use
# mantle pressures (5-25 kbar).
PUTIRKA_2008: dict[str, dict[str, tuple[float, float]]] = {
    # --- Mafic volcanic ---
    "basalt":              {"T_C": (1000, 1300), "P_kbar": (1, 5)},
    "trachybasalt":        {"T_C": (1050, 1250), "P_kbar": (1, 5)},
    "basanite":            {"T_C": (1100, 1300), "P_kbar": (2, 10)},
    "tephrite":            {"T_C": (1050, 1250), "P_kbar": (2, 7)},
    "picrite":             {"T_C": (1300, 1500), "P_kbar": (1, 10)},
    "komatiite":           {"T_C": (1400, 1600), "P_kbar": (1, 10)},

    # --- Mafic plutonic (slightly deeper, slower-cooled) ---
    "subalkalic gabbro":   {"T_C": (1050, 1250), "P_kbar": (2, 7)},
    "alkalic gabbro":      {"T_C": (1000, 1200), "P_kbar": (2, 7)},
    "gabbro":              {"T_C": (1050, 1250), "P_kbar": (2, 7)},
    "norite":              {"T_C": (1050, 1200), "P_kbar": (2, 7)},
    "anorthosite":         {"T_C": (1100, 1250), "P_kbar": (5, 10)},
    "troctolite":          {"T_C": (1100, 1250), "P_kbar": (2, 7)},

    # --- Intermediate volcanic ---
    "andesite":            {"T_C": (950, 1150),  "P_kbar": (1, 5)},
    "basaltic andesite":   {"T_C": (1000, 1200), "P_kbar": (1, 5)},
    "trachyandesite":      {"T_C": (900, 1100),  "P_kbar": (1, 5)},
    "phonotephrite":       {"T_C": (1000, 1200), "P_kbar": (2, 7)},
    "tephriphonolite":     {"T_C": (950, 1150),  "P_kbar": (2, 7)},
    "latite":              {"T_C": (900, 1100),  "P_kbar": (1, 5)},

    # --- Intermediate plutonic ---
    "diorite":             {"T_C": (900, 1050),  "P_kbar": (2, 7)},
    "gabbroic diorite":    {"T_C": (950, 1150),  "P_kbar": (2, 7)},
    "monzodiorite":        {"T_C": (900, 1050),  "P_kbar": (2, 7)},
    "monzogabbro":         {"T_C": (1000, 1200), "P_kbar": (2, 7)},
    "monzonite":           {"T_C": (850, 1000),  "P_kbar": (2, 7)},
    "syenite":             {"T_C": (800, 950),   "P_kbar": (1, 5)},
    "alkali feldspar syenite": {"T_C": (800, 950), "P_kbar": (1, 5)},
    "quartz monzonite":    {"T_C": (800, 950),   "P_kbar": (2, 5)},
    "quartz monzodiorite": {"T_C": (850, 1000),  "P_kbar": (2, 5)},
    "quartz diorite":      {"T_C": (850, 1000),  "P_kbar": (2, 5)},
    "quartz syenite":      {"T_C": (800, 950),   "P_kbar": (2, 5)},

    # --- Felsic volcanic ---
    "dacite":              {"T_C": (850, 1000),  "P_kbar": (1, 3)},
    "rhyolite":            {"T_C": (750, 900),   "P_kbar": (1, 3)},
    "trachyte":            {"T_C": (800, 950),   "P_kbar": (1, 3)},
    "trachydacite":        {"T_C": (800, 950),   "P_kbar": (1, 3)},
    "phonolite":           {"T_C": (750, 900),   "P_kbar": (1, 3)},

    # --- Felsic plutonic ---
    "tonalite":            {"T_C": (800, 950),   "P_kbar": (2, 7)},
    "granodiorite":        {"T_C": (750, 900),   "P_kbar": (2, 5)},
    "monzogranite":        {"T_C": (700, 850),   "P_kbar": (2, 5)},
    "syenogranite":        {"T_C": (700, 850),   "P_kbar": (2, 5)},
    "granite":             {"T_C": (700, 850),   "P_kbar": (2, 5)},
    "alkali feldspar granite": {"T_C": (700, 850), "P_kbar": (2, 5)},

    # --- Silica-undersaturated alkaline ---
    "foidite":             {"T_C": (1100, 1250), "P_kbar": (2, 10)},
    "nephelinite":         {"T_C": (1100, 1250), "P_kbar": (2, 10)},
    "leucitite":           {"T_C": (1100, 1250), "P_kbar": (1, 7)},
    "foid syenite":        {"T_C": (800, 1000),  "P_kbar": (1, 7)},
    "foid gabbro":         {"T_C": (1000, 1200), "P_kbar": (2, 7)},
    "foid monzosyenite":   {"T_C": (850, 1000),  "P_kbar": (2, 7)},
    "foid monzodiorite":   {"T_C": (900, 1050),  "P_kbar": (2, 7)},
    "foid-bearing diorite":      {"T_C": (900, 1050), "P_kbar": (2, 7)},
    "foid-bearing latite":       {"T_C": (900, 1100), "P_kbar": (1, 5)},

    # --- Ultramafic (mantle conditions) ---
    "dunite":              {"T_C": (1200, 1400), "P_kbar": (5, 25)},
    "harzburgite":         {"T_C": (1200, 1400), "P_kbar": (5, 25)},
    "lherzolite":          {"T_C": (1100, 1400), "P_kbar": (5, 30)},
    "wehrlite":            {"T_C": (1150, 1350), "P_kbar": (5, 20)},
    "websterite":          {"T_C": (1100, 1300), "P_kbar": (5, 20)},
    "orthopyroxenite":     {"T_C": (1100, 1300), "P_kbar": (5, 20)},
    "clinopyroxenite":     {"T_C": (1100, 1300), "P_kbar": (5, 20)},
    "peridotite":          {"T_C": (1200, 1400), "P_kbar": (5, 25)},
    "mantle peridotite":   {"T_C": (1200, 1400), "P_kbar": (10, 30)},
    "mantle pyroxenite":   {"T_C": (1100, 1300), "P_kbar": (10, 25)},

    # --- Fallback ---
    "default":             {"T_C": (900, 1200),  "P_kbar": (2, 10)},
}


# Stage 1 brute grid (for ablation / reproduction)
_T_GRID = (800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0)
_P_GRID = (1.0, 5.0, 10.0, 20.0)


class TPStrategy(Protocol):
    """A callable that returns K (T, P) samples for a given rock class."""

    def sample(self, rock_class: str, rng: np.random.Generator) -> list[tuple[float, float]]:
        ...


class TPRangeStrategy:
    """Uniform sample within rock-class T/P range. Default: Putirka 2008.

    K samples per bulk; same K across all classes. Use this as the Position C
    default — the per-class ranges encode the empirical (T, P) prior so
    MAGEMin labels reflect realistic crystallization conditions.

    Examples:
        >>> rng = np.random.default_rng(42)
        >>> strat = TPRangeStrategy(k_per_bulk=5)
        >>> strat.sample("basalt", rng)   # 5 (T, P) tuples within basalt range
    """

    def __init__(
        self,
        ranges: dict[str, dict[str, tuple[float, float]]] | None = None,
        k_per_bulk: int = 20,
    ) -> None:
        self.ranges = dict(ranges) if ranges is not None else dict(PUTIRKA_2008)
        if "default" not in self.ranges:
            raise ValueError("ranges dict must include a 'default' entry for unknown classes")
        if k_per_bulk < 1:
            raise ValueError(f"k_per_bulk must be >= 1, got {k_per_bulk}")
        self.k_per_bulk = int(k_per_bulk)

    def _lookup(self, rock_class: str) -> dict[str, tuple[float, float]]:
        key = (rock_class or "").strip().lower()
        return self.ranges.get(key, self.ranges["default"])

    def sample(self, rock_class: str, rng: np.random.Generator) -> list[tuple[float, float]]:
        cfg = self._lookup(rock_class)
        t_lo, t_hi = cfg["T_C"]
        p_lo, p_hi = cfg["P_kbar"]
        T = rng.uniform(t_lo, t_hi, size=self.k_per_bulk)
        P = rng.uniform(p_lo, p_hi, size=self.k_per_bulk)
        return list(zip(T.tolist(), P.tolist()))


class TPGridStrategy:
    """Brute T/P grid (Stage 1 reproduction). 28 points per bulk regardless of class.

    Use this for ablation studies — does class-conditional T/P sampling help
    vs. the rock-agnostic grid?
    """

    def __init__(
        self,
        t_grid: tuple[float, ...] = _T_GRID,
        p_grid: tuple[float, ...] = _P_GRID,
    ) -> None:
        self.t_grid = tuple(t_grid)
        self.p_grid = tuple(p_grid)

    @property
    def k_per_bulk(self) -> int:
        return len(self.t_grid) * len(self.p_grid)

    def sample(self, rock_class: str, rng: np.random.Generator) -> list[tuple[float, float]]:
        return [(t, p) for p in self.p_grid for t in self.t_grid]


class ConstantTPStrategy:
    """Single fixed (T, P) for every bulk. Useful as a sensitivity baseline."""

    def __init__(self, T_C: float, P_kbar: float, k_per_bulk: int = 1) -> None:
        self.T_C = float(T_C)
        self.P_kbar = float(P_kbar)
        self.k_per_bulk = int(k_per_bulk)

    def sample(self, rock_class: str, rng: np.random.Generator) -> list[tuple[float, float]]:
        return [(self.T_C, self.P_kbar)] * self.k_per_bulk
