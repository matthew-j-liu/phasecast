"""Fe³⁺/ΣFe ratio strategies for bulk → MAGEMin conversion.

When complete.csv reports only total iron (as `feo_tot` or `fe2o3_tot`), we
need to split it into FeO and Fe2O3 for MAGEMin's input. The ratio depends on
the magma's oxidation state, which varies by tectonic setting and rock class.

We expose this as a strategy pattern so sensitivity analysis is trivial:
swap the strategy at config time, rerun training, compare. The default is
Middlemost 1989 / Le Maitre 1976 per-class lookups, mapped to qap_name
categories observed in complete.csv.

References:
    Middlemost (1989) "Iron oxidation ratios, norms and the classification
    of volcanic rocks." Chemical Geology 77:19-26.
    Le Maitre (1976) "Some problems of the projection of chemical data into
    mineralogical classifications." Contrib. Mineral. Petrol. 56:181-189.
"""

from __future__ import annotations

from typing import Protocol


# Per-qap_name Fe³⁺/ΣFe defaults. Values derived from Middlemost 1989 / Le Maitre
# 1976 compilations, mapped to qap_name's modal-mineralogy categories. Plutonic
# classes (gabbro, granite, etc.) inherit values from their volcanic equivalents
# (basalt, rhyolite, etc.) since the oxidation state during crystallization is
# similar at the magmatic stage.
MIDDLEMOST_1989: dict[str, float] = {
    # Ultramafic (low oxidation, mantle-derived)
    "dunite":            0.05,
    "harzburgite":       0.05,
    "lherzolite":        0.05,
    "wehrlite":          0.05,
    "websterite":        0.05,
    "orthopyroxenite":   0.05,
    "clinopyroxenite":   0.05,
    "peridotite":        0.05,

    # Mafic volcanic
    "basalt":            0.15,
    "picrite":           0.10,
    "komatiite":         0.10,
    "trachybasalt":      0.20,
    "basanite":          0.20,
    "tephrite":          0.20,

    # Mafic plutonic (inherits from basalt)
    "subalkalic gabbro": 0.15,
    "alkalic gabbro":    0.20,
    "gabbro":            0.15,
    "norite":            0.15,
    "anorthosite":       0.15,
    "troctolite":        0.15,

    # Intermediate volcanic
    "andesite":              0.30,
    "basaltic andesite":     0.25,
    "trachyandesite":        0.35,
    "phonotephrite":         0.30,
    "tephriphonolite":       0.32,

    # Intermediate plutonic
    "diorite":               0.25,
    "gabbroic diorite":      0.20,
    "monzodiorite":          0.30,
    "monzogabbro":           0.20,
    "monzonite":             0.30,
    "syenite":               0.35,
    "alkali feldspar syenite": 0.35,

    # Felsic volcanic
    "dacite":            0.40,
    "rhyolite":          0.45,
    "trachyte":          0.40,
    "trachydacite":      0.40,
    "phonolite":         0.35,

    # Felsic plutonic (inherits from dacite/rhyolite)
    "tonalite":          0.35,
    "granodiorite":      0.40,
    "monzogranite":      0.45,
    "syenogranite":      0.45,
    "granite":           0.45,
    "alkali feldspar granite": 0.45,

    # Silica-undersaturated alkaline
    "foidite":           0.25,
    "nephelinite":       0.25,
    "leucitite":         0.25,
    "foid syenite":      0.30,
    "foid gabbro":       0.20,
    "foid monzosyenite": 0.30,
    "foid monzodiorite": 0.25,

    # Quartz-bearing intermediates (between dacite and granite)
    "quartz monzonite":   0.40,
    "quartz monzodiorite": 0.35,
    "quartz diorite":     0.30,
    "quartz syenite":     0.40,

    # Fallback for any class not in the table
    "default":           0.15,
}


class Fe3RatioStrategy(Protocol):
    """A callable that returns Fe³⁺/ΣFe given rock class + (optionally) oxides."""

    def __call__(self, rock_class: str, oxides: dict[str, float] | None = None) -> float:
        ...


class ClassRatioStrategy:
    """Look up Fe³⁺/ΣFe from a per-class table. Default is Middlemost 1989.

    Sensitivity analysis pattern: instantiate with a perturbed copy of the table
    to see how predictions shift when the assumed ratios change.

    Examples:
        >>> default = ClassRatioStrategy()
        >>> default("basalt")
        0.15
        >>> tweaked = ClassRatioStrategy({**MIDDLEMOST_1989, "basalt": 0.20})
        >>> tweaked("basalt")
        0.20
    """

    def __init__(self, ratios: dict[str, float] | None = None) -> None:
        self.ratios = dict(ratios) if ratios is not None else dict(MIDDLEMOST_1989)
        if "default" not in self.ratios:
            raise ValueError("ratios dict must include a 'default' key for unknown classes")

    def __call__(self, rock_class: str, oxides: dict[str, float] | None = None) -> float:
        key = (rock_class or "").strip().lower()
        return self.ratios.get(key, self.ratios["default"])


class ConstantRatioStrategy:
    """Same Fe³⁺/ΣFe ratio for all rocks. Useful as an ablation baseline.

    Examples:
        >>> reduced = ConstantRatioStrategy(0.10)    # all rocks reduced
        >>> oxidized = ConstantRatioStrategy(0.30)   # all rocks oxidized
    """

    def __init__(self, ratio: float) -> None:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"ratio must be in [0, 1], got {ratio}")
        self.ratio = float(ratio)

    def __call__(self, rock_class: str, oxides: dict[str, float] | None = None) -> float:
        return self.ratio
