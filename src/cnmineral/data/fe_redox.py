"""Fe3+/SigmaFe ratio strategies for bulk -> MAGEMin conversion.

When a bulk reports only total iron (as `feo_tot` or `fe2o3_tot`), we need to
split it into FeO and Fe2O3 for MAGEMin's input. The ratio depends on the
magma's oxidation state, which varies by tectonic setting and rock class.

We expose this as a strategy pattern so sensitivity analysis is trivial: swap
the strategy at config time, rerun, compare. The default is Middlemost 1989 /
Le Maitre 1976 per-class lookups, mapped to `qap_name` categories.

References:
    Middlemost (1989) "Iron oxidation ratios, norms and the classification of
    volcanic rocks." Chemical Geology 77:19-26.
    Le Maitre (1976) "Some problems of the projection of chemical data into
    mineralogical classifications." Contrib. Mineral. Petrol. 56:181-189.
"""


# Per-qap_name Fe3+/SigmaFe defaults. Values derived from Middlemost 1989 / Le
# Maitre 1976 compilations, mapped to qap_name's modal-mineralogy categories.
# Plutonic classes (gabbro, granite, etc.) inherit values from their volcanic
# equivalents (basalt, rhyolite, etc.) since the oxidation state during
# crystallization is similar at the magmatic stage.
MIDDLEMOST_1989 = {
    # Ultramafic (low oxidation, mantle-derived)
    "dunite":            0.05,
    "harzburgite":       0.05,
    "lherzolite":        0.05,
    "wehrlite":          0.05,
    "wherlite":          0.05,   # gard spelling variant
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
    "latite":                0.35,
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
    "quartz-rich rhyolite":    0.45,
    "alkali feldspar rhyolite": 0.45,
    "quartz latite":     0.40,
    "trachyte":          0.40,
    "quartz trachyte":   0.40,
    "alkali feldspar trachyte":        0.40,
    "alkali feldspar quartz trachyte": 0.40,
    "trachydacite":      0.40,
    "phonolite":         0.35,

    # Felsic plutonic (inherits from dacite/rhyolite)
    "tonalite":          0.35,
    "granodiorite":      0.40,
    "monzogranite":      0.45,
    "syenogranite":      0.45,
    "granite":           0.45,
    "alkali feldspar granite": 0.45,

    # Silica-undersaturated alkaline (volcanic)
    "foidite":           0.25,
    "phonolitic foidite": 0.25,
    "tephritic foidite":  0.25,
    "phonolitic tephrite": 0.30,
    "tephritic phonolite": 0.32,
    "phonolitic basanite": 0.22,
    "foid-bearing latite":  0.32,
    "foid-bearing trachyte": 0.35,
    "foid-bearing alkali feldspar trachyte": 0.35,
    "nephelinite":       0.25,
    "leucitite":         0.25,

    # Silica-undersaturated alkaline (plutonic)
    "foid syenite":      0.30,
    "foid gabbro":       0.20,
    "foid monzosyenite": 0.30,
    "foid monzodiorite": 0.25,

    # Quartz-bearing intermediates (between dacite and granite)
    "quartz monzonite":   0.40,
    "quartz monzodiorite": 0.35,
    "quartz diorite":     0.30,
    "quartz syenite":     0.40,

    # Silica rock (>90% SiO2, negligible Fe; QC-flagged upstream)
    "silexite":          0.45,

    # Fallback for any class not in the table
    "default":           0.15,
}


class ClassRatioStrategy:
    """Look up Fe3+/SigmaFe from a per-class table. Default is Middlemost 1989.

    Sensitivity analysis pattern: instantiate with a perturbed copy of the table
    to see how predictions shift when the assumed ratios change.

        >>> ClassRatioStrategy()("basalt")
        0.15
        >>> ClassRatioStrategy({**MIDDLEMOST_1989, "basalt": 0.20})("basalt")
        0.2
    """

    def __init__(self, ratios=None):
        self.ratios = dict(ratios) if ratios is not None else dict(MIDDLEMOST_1989)
        if "default" not in self.ratios:
            raise ValueError("ratios dict must include a 'default' key for unknown classes")

    def __call__(self, rock_class, oxides=None):
        key = (rock_class or "").strip().lower()
        return self.ratios.get(key, self.ratios["default"])
