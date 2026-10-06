"""The plutonic benchmark as the real-rock instrument reads it.

`harvest.py` writes the plutonic tables to their own directory; this is the
one place that loads them for evaluation.  `cnmineral.eval.real_rock` finds a
benchmark's split through its module global `SPLIT_PATH`, so
`plutonic_benchmark` points that global at the plutonic split for the length
of a `with` block and restores the eruption split afterwards.  Every plutonic
script evaluates inside that block rather than reassigning the global itself.
"""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from cnmineral.data.phase_vocab import GROUP_INDEX
from cnmineral.eval import real_rock
from cnmineral.eval.real_rock import MINERAL_DENSITY, RealRockBenchmark, load_benchmark
from explorations.real_mineralogy.harvest import PLUTONIC_DIR

PLUTONIC_SPLIT_PATH = Path(__file__).resolve().parent / "plutonic_split.csv"

# The split pyroxenes take the densities of their volcanic counterparts.  The
# rows outside the seven categories are never converted to mass, so they carry
# no density; a NaN makes an accidental conversion loud.
PLUTONIC_DENSITY = {
    **MINERAL_DENSITY,
    "orthopyroxene": MINERAL_DENSITY["hypersthene"],
    "clinopyroxene": MINERAL_DENSITY["augite"],
    **dict.fromkeys(
        ["muscovite", "accessory minerals", "accessory minerals and Fe-Ti oxides",
         "mafic minerals", "secondary minerals"],
        np.nan,
    ),
}


def _phases_by_rock(modes: pd.DataFrame) -> pd.Series:
    key = modes["source"] + "|" + modes["sample_id"].astype(str)
    return modes.groupby(key)["magemin_phase"].agg(set)


def oxides_lumped_keys(modes: pd.DataFrame) -> set[str]:
    """Rocks whose Fe-Ti oxides went uncounted: they carry the
    accessories-with-oxides lump and no oxide row of their own.  Every other
    category was counted, so they are scored with the oxides unmeasured."""
    phases = _phases_by_rock(modes)
    return set(phases.index[phases.map(lambda p: "accessory_opq" in p and "opq_unsplit" not in p)])


def mafic_total_keys(modes: pd.DataFrame) -> set[str]:
    """Rocks reporting only a mafic total, so no mafic category was counted
    on its own; these stay outside the evaluated set."""
    phases = _phases_by_rock(modes)
    return set(phases.index[phases.map(lambda p: "mafic_unsplit" in p)])


def load_plutonic() -> RealRockBenchmark:
    """Every plutonic rock with chemistry and a mode; `clean` marks the evaluated.

    Neither source grades alteration per rock, and each already removed the
    rocks its own chemical screens flag, so every rock counts as fresh.  The
    rocks `mafic_total_keys` names stay in the benchmark, and so in the split,
    but outside `clean`.  The rocks `oxides_lumped_keys` names are evaluated
    with the oxides marked unmeasured, so their presence 0 there is not read
    as an absence (#83, 2026-10-05; #82's committed results predate this and
    cover 507 rocks).
    """
    benchmark = load_benchmark(PLUTONIC_DIR, densities=PLUTONIC_DENSITY)
    modes = pd.read_csv(PLUTONIC_DIR / "modes.csv")
    measured = benchmark.measured.copy()
    measured[np.isin(benchmark.bulk_ids, sorted(oxides_lumped_keys(modes))), GROUP_INDEX["opq"]] = False
    return replace(benchmark, measured=measured,
                   clean=~np.isin(benchmark.bulk_ids, sorted(mafic_total_keys(modes))))


@contextmanager
def plutonic_benchmark():
    """The evaluated plutonic rocks, with the instrument reading their split.

    Every rock must appear in the split file: a rock the split does not know
    silently leaves the climatology null's fitting set, and with no plutonic
    rock known at all the null fits on nothing.
    """
    benchmark = load_plutonic()
    roster = pd.read_csv(PLUTONIC_SPLIT_PATH)
    known = set(roster["source"] + "|" + roster["sample_id"].astype(str))
    missing = set(benchmark.bulk_ids) - known
    if missing:
        raise ValueError(
            f"{len(missing)} plutonic rocks are not in {PLUTONIC_SPLIT_PATH.name}, "
            f"e.g. {sorted(missing)[:3]}; the harvest changed since the split "
            "was built"
        )
    eruption_split = real_rock.SPLIT_PATH
    real_rock.SPLIT_PATH = PLUTONIC_SPLIT_PATH
    try:
        yield benchmark.subset(benchmark.clean)
    finally:
        real_rock.SPLIT_PATH = eruption_split
