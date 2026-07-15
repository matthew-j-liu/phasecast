"""Solid-phase vocabulary for the mineralogy heads.

Defines the label vocabulary that `mineralogy_labels` builds targets over and
the heads predict over. The vocabulary is derived from the MAGEMin corpus
phase names with three rules: MAGEMin's lumped `fsp` is split into `plag` /
`afs` by `feldspar_rule`; the SiO2 polymorphs `trd` / `crst` are folded into
`q` (identical chemistry, indistinguishable from bulk composition); and
phases below a learnability floor (`chl`, `fper`) are dropped, their mass
renormalized away. `liq` and `fl` are not vocabulary: the label is the
crystals-renormalized solid assemblage, and melt/fluid enter only through the
mass-balance closure term.
"""

SOLID_PHASES: tuple[str, ...] = (
    "plag", "afs", "cpx", "spl", "amp", "ilm", "opx", "q", "bi", "ol", "g",
    "ru", "sph", "sill", "cd", "mu", "ep", "ne", "ky", "and", "cor",
)
N_PHASES: int = len(SOLID_PHASES)
PHASE_INDEX: dict[str, int] = {p: i for i, p in enumerate(SOLID_PHASES)}

PHASE_MERGES: dict[str, str] = {"trd": "q", "crst": "q"}
DROPPED_PHASES: frozenset[str] = frozenset({"chl", "fper"})
NON_SOLID_PHASES: frozenset[str] = frozenset({"liq", "fl"})

# presence target: crystals-renormalized mode > 0.1 wt%
PRESENCE_CUTOFF: float = 1e-3
