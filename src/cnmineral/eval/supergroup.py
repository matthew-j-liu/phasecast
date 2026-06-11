"""qap_name -> 9-supergroup lookup.

Coarse petrogenetic supergroup labels used for SSL-encoder probe diagnostics
and UMAP coloring.  The qap_name column on `BulkChemistryDataset` has 70+
IUGS Streckeisen classes; the 9-supergroup collapse is a stable color palette
for UMAP and a cleaner secondary probe target.

Ported verbatim from the v2 string-rule lookup in
`v2-traces-ssl-final:src/cnmineral/data/unlabeled.py::_supergroup_of`, which
predates v3's CompHead work and has the historical name "other" for the
catch-all bucket.
"""

import numpy as np

SUPERGROUPS: tuple[str, ...] = (
    "ultramafic",
    "mafic-volcanic",
    "mafic-plutonic",
    "intermediate-volcanic",
    "intermediate-plutonic",
    "felsic-volcanic",
    "felsic-plutonic",
    "foid-bearing",
    "other",
)

SUPERGROUP_COLORS: dict[str, str] = {
    "ultramafic": "#7f3b08",
    "mafic-volcanic": "#b35806",
    "mafic-plutonic": "#e08214",
    "intermediate-volcanic": "#fdb863",
    "intermediate-plutonic": "#b2abd2",
    "felsic-volcanic": "#8073ac",
    "felsic-plutonic": "#542788",
    "foid-bearing": "#2d004b",
    "other": "#bbbbbb",
}


def qap_to_supergroup(qap_name: str | None) -> str:
    """Collapse an IUGS Streckeisen qap_name to one of 9 supergroups.

    Returns "other" for unrecognized strings so the function is total.
    The foid-bearing branch fires first because several qap strings
    combine a foid keyword with a non-foid root (e.g. "foid-bearing
    diorite", "phonolitic tephrite").
    """
    if qap_name is None or not isinstance(qap_name, str):
        return "other"
    s = qap_name.strip().lower()
    if not s:
        return "other"
    if (
        "foid" in s
        or "nephelin" in s
        or "phonolit" in s
        or "tephrit" in s
        or "basanit" in s
    ):
        return "foid-bearing"
    if (
        "dunit" in s
        or "harzburg" in s
        or "lherzolit" in s
        or "wehrlit" in s
        or "websterit" in s
        or "pyroxenit" in s
        or "peridotit" in s
        or "picrit" in s
        or "komatiit" in s
    ):
        return "ultramafic"
    if "gabbr" in s or "norit" in s or "anorthosit" in s:
        return "mafic-plutonic"
    if "basalt" in s:
        return "mafic-volcanic"
    if "diorit" in s or "tonalit" in s or "monzonit" in s or "granodior" in s:
        return "intermediate-plutonic"
    if "andesit" in s or "latit" in s or "trachyt" in s:
        return "intermediate-volcanic"
    if "rhyolit" in s or "dacit" in s:
        return "felsic-volcanic"
    if "granit" in s or "syenit" in s:
        return "felsic-plutonic"
    return "other"


def map_qap_array(qap_names) -> np.ndarray:
    """Apply `qap_to_supergroup` over an iterable; return an object ndarray."""
    return np.array(
        [qap_to_supergroup(q) for q in qap_names], dtype=object,
    )
