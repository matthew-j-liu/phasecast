"""H2O sampling strategies for bulk → MAGEMin conversion.

H2O is rarely measured in whole-rock analyses — only ~1% of complete.csv has
an `h2o_tot` column populated. When defaulting to 0 wt% (the upstream MAGEMin
behavior), the equilibrium model returns "dry" assemblages that don't
crystallize hydrous phases (amphibole, biotite, muscovite) even at conditions
where real rocks routinely contain them. This degenerates training labels for
hydrous phases to ~0% presence.

We expose H2O as an injectable strategy paralleling `fe_redox.py` and
`tp_priors.py`:
- Tier 1 (measured) is handled at the caller: if the row reports `h2o_tot`,
  use it directly.
- Tier 2 (assumed_class) is what these strategies provide: sample H2O from a
  per-rock-class distribution.

Default `ClassRangeH2OStrategy` draws uniformly from per-class ranges
compiled from:
    Wallace P. (2005) "Volatiles in subduction zone magmas..." JVGR 140:217-240
    Plank T., Kelley K., Zimmer M., Hauri E., Wallace P. (2013) "Why do
        mafic arc magmas contain ~4 wt% water on average?" EPSL 364:168-179
    Burnham C.W. (1979) "The importance of volatile constituents." In:
        Yoder H.S. (ed), The Evolution of the Igneous Rocks, 439-482.
    Holtz F., Becker A., Freise M., Johannes W. (2001) "The water-undersaturated
        and dry Qz–Ab–Or system revisited." CMP 141:347-357
    Best M.G. (2003) Igneous and Metamorphic Petrology, 2nd ed.

Within-class H2O variance is genuinely large (arc rocks span 1-8 wt%); the
range-based design encodes that variance rather than committing to a single
literature point estimate.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


# Per-qap_name H2O ranges in wt%. Lower bound first.
WALLACE_PLANK_2013: dict[str, tuple[float, float]] = {
    # --- Mafic volcanic ---
    "basalt":              (0.1, 1.0),
    "trachybasalt":        (0.5, 3.0),
    "basanite":            (0.5, 2.5),
    "tephrite":            (1.0, 4.0),
    "picrite":             (0.1, 1.0),
    "komatiite":           (0.05, 0.5),

    # --- Mafic plutonic (inherit basalt but slightly higher; slower-cooled retains volatiles) ---
    "subalkalic gabbro":   (0.1, 1.5),
    "alkalic gabbro":      (0.5, 2.5),
    "gabbro":              (0.5, 2.0),
    "norite":              (0.5, 2.0),
    "anorthosite":         (0.1, 1.0),
    "troctolite":          (0.2, 1.5),

    # --- Intermediate volcanic (arc-dominated, wet) ---
    "andesite":            (2.0, 6.0),
    "basaltic andesite":   (2.0, 5.0),
    "trachyandesite":      (1.0, 4.0),
    "phonotephrite":       (1.0, 4.0),
    "tephriphonolite":     (1.0, 4.0),
    "latite":              (1.0, 4.0),

    # --- Intermediate plutonic ---
    "diorite":             (2.0, 5.0),
    "gabbroic diorite":    (1.5, 4.5),
    "monzodiorite":        (1.0, 4.0),
    "monzogabbro":         (1.0, 3.5),
    "monzonite":           (1.5, 4.0),
    "syenite":             (1.0, 4.0),
    "alkali feldspar syenite": (1.0, 3.5),
    "quartz monzonite":    (2.0, 5.0),
    "quartz monzodiorite": (1.5, 4.5),
    "quartz diorite":      (2.0, 5.0),
    "quartz syenite":      (1.5, 4.0),

    # --- Felsic volcanic ---
    "dacite":              (3.0, 7.0),
    "rhyolite":            (2.0, 7.0),
    "trachyte":            (1.0, 5.0),
    "trachydacite":        (1.5, 5.0),
    "phonolite":           (1.0, 5.0),

    # --- Felsic plutonic ---
    "tonalite":            (3.0, 6.0),
    "granodiorite":        (3.0, 6.0),
    "monzogranite":        (3.0, 6.0),
    "syenogranite":        (3.0, 6.0),
    "granite":             (3.0, 6.0),
    "alkali feldspar granite": (2.0, 5.0),     # often A-type; drier

    # --- Silica-undersaturated alkaline ---
    "foidite":             (1.0, 5.0),
    "nephelinite":         (1.0, 5.0),
    "leucitite":           (1.0, 4.0),
    "foid syenite":        (1.0, 4.0),
    "foid gabbro":         (1.0, 3.5),
    "foid monzosyenite":   (1.0, 3.5),
    "foid monzodiorite":   (1.0, 3.5),
    "foid-bearing diorite":      (1.5, 4.0),
    "foid-bearing latite":       (1.0, 4.0),

    # --- Ultramafic (dry mantle conditions) ---
    "dunite":              (0.0, 0.1),
    "harzburgite":         (0.0, 0.1),
    "lherzolite":          (0.0, 0.2),
    "wehrlite":            (0.0, 0.2),
    "websterite":          (0.0, 0.2),
    "orthopyroxenite":     (0.0, 0.2),
    "clinopyroxenite":     (0.0, 0.2),
    "peridotite":          (0.0, 0.1),
    "mantle peridotite":   (0.0, 0.1),
    "mantle pyroxenite":   (0.0, 0.2),

    # --- Fallback ---
    "default":             (0.5, 3.0),
}


class H2OStrategy(Protocol):
    """A callable that returns one H2O wt% sample given rock class + rng."""

    def __call__(self, rock_class: str, rng: np.random.Generator) -> float:
        ...


class ClassRangeH2OStrategy:
    """Uniform sample within a per-class H2O range. Default: Wallace 2005 + Plank 2013.

    Examples:
        >>> rng = np.random.default_rng(42)
        >>> strat = ClassRangeH2OStrategy()
        >>> strat("basalt", rng)        # uniform in [0.1, 1.0]
        >>> strat("andesite", rng)      # uniform in [2.0, 6.0]
        >>> strat("unknown_class", rng) # uniform in default range [0.5, 3.0]
    """

    def __init__(self, ranges: dict[str, tuple[float, float]] | None = None) -> None:
        self.ranges = dict(ranges) if ranges is not None else dict(WALLACE_PLANK_2013)
        if "default" not in self.ranges:
            raise ValueError("ranges dict must include a 'default' entry for unknown classes")

    def _lookup(self, rock_class: str) -> tuple[float, float]:
        key = (rock_class or "").strip().lower()
        return self.ranges.get(key, self.ranges["default"])

    def __call__(self, rock_class: str, rng: np.random.Generator) -> float:
        lo, hi = self._lookup(rock_class)
        return float(rng.uniform(lo, hi))


class ConstantH2OStrategy:
    """Same H2O wt% for all rocks. Useful as a sensitivity baseline.

    Examples:
        >>> dry = ConstantH2OStrategy(0.0)       # MAGEMin-original behavior
        >>> wet = ConstantH2OStrategy(3.0)       # uniformly wet, all rocks
    """

    def __init__(self, h2o_wt_pct: float) -> None:
        if h2o_wt_pct < 0:
            raise ValueError(f"h2o_wt_pct must be >= 0, got {h2o_wt_pct}")
        self.h2o_wt_pct = float(h2o_wt_pct)

    def __call__(self, rock_class: str, rng: np.random.Generator) -> float:
        return self.h2o_wt_pct
