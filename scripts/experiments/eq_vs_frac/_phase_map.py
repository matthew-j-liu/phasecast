"""PetThermoTools (MAGEMinCalc) phase name -> MAGEMin short abbreviation.

ptt returns keys like 'clinopyroxene1', 'plagioclase1', 'alkali-feldspar1',
'rhm-oxide1', 'q1', 'sph1', 'bi1'. Strip trailing digits, then map prefix.
plagioclase1 + alkali-feldspar1 both -> 'fsp' (MAGEMin's fsp SS combines them).
"""
import re

_TRAIL_DIGITS = re.compile(r"\d+$")

PREFIX_TO_SHORT = {
    "clinopyroxene": "cpx",
    "orthopyroxene": "opx",
    "olivine": "ol",
    "plagioclase": "fsp",
    "alkali-feldspar": "fsp",
    "kfeldspar": "fsp",
    "spinel": "spl",
    "rhm-oxide": "ilm",
    "ilmenite": "ilm",
    "magnetite": "spl",
    "garnet": "g",
    "amphibole": "amp",
    "hornblende": "amp",
    "biotite": "bi",
    "muscovite": "mu",
    "cordierite": "cd",
    "epidote": "ep",
    "sillimanite": "sill",
    "andalusite": "and",
    "kyanite": "ky",
    "nepheline": "ne",
    "leucite": "lct",
    "feldspathoid": "ne",
    "quartz": "q",
    "tridymite": "trd",
    "sphene": "sph",
    "titanite": "sph",
    "rutile": "ru",
    "corundum": "cor",
    "ferropericlase": "fper",
    "fluid": "fl",
    "liquid": "liq",
    "bi": "bi",
    "mu": "mu",
    "q": "q",
    "sph": "sph",
    "ne": "ne",
    "ru": "ru",
    "cor": "cor",
    "ep": "ep",
    "g": "g",
    "cd": "cd",
    "amp": "amp",
}


def to_short(ptt_phase: str) -> str | None:
    """Map ptt phase name (with optional trailing digits) to MAGEMin short code.

    Returns None for unknown/system keys (e.g. 'All', 'PhaseList', 'sys',
    'Conditions') so the caller can skip them.
    """
    base = _TRAIL_DIGITS.sub("", ptt_phase)
    return PREFIX_TO_SHORT.get(base)
