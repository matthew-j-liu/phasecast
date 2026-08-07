"""Andrews XRD sedimentary rocks shaped into the real-rock benchmark dataclass.

Mock-up only.  Provenance is unrecovered -- the samples are anonymised
`Rock_N` with no locality, formation or freshness grading -- so nothing built
here is a reportable benchmark number until the source publication is found.

The point of going through `RealRockBenchmark` rather than a bespoke container
is that every shipped eval and plot function then runs unmodified.
"""

import sys
import types
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from cnmineral.data.bulk_chemistry import OXIDE_TO_CATION_FACTOR, tokenize_bulk_frame
from cnmineral.data.phase_vocab import GROUPS
# `_truth_arrays` is private, and reached into deliberately: reusing the exact
# coarsening the eruption benchmark gets is the whole point of the exercise.
from cnmineral.eval.real_rock import RealRockBenchmark, _truth_arrays

PKL = (Path.home() / "git_clones/CrabNet_PyPi/CrabNet-Mineral/alignn"
       / "datasets/dp_crab_Andrews.pkl")
SOURCE = "andrews_xrd"

# Andrews mineral -> the `magemin_phase` key `TRUTH_TO_GROUP` coarsens.  The
# eleven minerals left out (calcite, dolomite, iron-dolomite, siderite, halite,
# pyrite, kaolinite, smectite, illite, iron-chlorite, muscovite) have no
# counterpart in the 21-phase vocabulary.  Omitting them is what renormalises
# truth over the scoreable groups -- the same move `groundmass` gets on the
# eruption benchmark, at a much worse ratio.
MINERAL_TO_PHASE: dict[str, str] = {
    "SiO2": "q",
    "K2O Al2O3 (SiO2)6": "fsp",
    "anorthoclase": "fsp",
    "albite": "fsp",
    "oligoclase": "fsp",
    "bytownite": "fsp",
    "anorthite": "fsp",
    "pyroxene": "px_unsplit",
    "ferrotschermakite": "amp",
    "magnetite": "opq_unsplit",
    "Fe2O3": "opq_unsplit",
}

# element wt fraction -> the oxide wt% column `tokenize_bulk_frame` reads.  Fe
# goes to `feo_tot` so it takes `_resolve_iron`'s documented fallback branch.
ELEMENT_TO_OXIDE: dict[str, str] = {
    "Si": "sio2", "Al": "al2o3", "Ca": "cao", "Mg": "mgo",
    "K": "k2o", "Na": "na2o", "Fe": "feo_tot",
}

# `_resolve_iron` scales `feo_tot` by the FeO factor, so that column's inverse
# has to use the same key.
_FACTOR_KEY: dict[str, str] = {"feo_tot": "feo"}

# present in the source table, unrepresentable in the 22-token vocabulary.  Cl
# and S are what predict halite and pyrite, both of which the truth reports.
UNUSED_ELEMENTS: tuple[str, ...] = ("Cl", "S")


def _read_package(pkl_path: Path):
    """Unpickle the Andrews package.

    Its globals reference a module named `DataFrame` that exists nowhere in the
    old repo, so lookups are proxied to pandas / numpy.
    """

    class Shim(types.ModuleType):
        def __getattr__(self, name):
            for mod in (pd, np, np.core.multiarray):
                if hasattr(mod, name):
                    return getattr(mod, name)
            raise AttributeError(name)

    sys.modules.setdefault("DataFrame", Shim("DataFrame"))
    return joblib.load(pkl_path)


def load_andrews(pkl_path: Path = PKL) -> tuple[RealRockBenchmark, dict]:
    """The benchmark plus the provenance facts the dataclass has no slot for."""
    package = _read_package(pkl_path)
    frame, minerals = package["df"], package["mineral_names"]
    modes_matrix = np.vstack(frame["Y"].values).astype(np.float64)
    sample_ids = frame["Sample_ID"].to_numpy()
    keys = np.array([f"{SOURCE}|{s}" for s in sample_ids])

    samples = pd.DataFrame({"sample_id": sample_ids})
    for element, column in ELEMENT_TO_OXIDE.items():
        factor = OXIDE_TO_CATION_FACTOR[_FACTOR_KEY.get(column, column)]
        samples[column] = frame[element].to_numpy(dtype=np.float64) * 100.0 / factor
    src, frac = tokenize_bulk_frame(samples)

    scored = [(j, name) for j, name in enumerate(minerals)
              if name in MINERAL_TO_PHASE]
    columns = [j for j, _ in scored]
    names = [name for _, name in scored]
    modes = pd.DataFrame({
        "key": np.repeat(keys, len(scored)),
        "phase": np.tile(names, len(keys)),
        "magemin_phase": np.tile([MINERAL_TO_PHASE[n] for n in names], len(keys)),
        "abundance_pct": modes_matrix[:, columns].ravel() * 100.0,
    })

    # XRD reports mass directly, so the volume-to-mass step the eruption
    # benchmark needs is the identity here and its mid-series solid-solution
    # density assumption is simply absent.  No censored rows either: every
    # mineral carries a number, and a reported zero means looked-for-not-found,
    # which `_truth_arrays` already skips.
    truth = _truth_arrays(modes, keys, {n: 1.0 for n in names}, 0.0)

    reported = modes_matrix.sum(1)
    scoreable = modes_matrix[:, columns].sum(1)
    sio2 = samples["sio2"].to_numpy(dtype=np.float64)

    benchmark = RealRockBenchmark(
        bulk_ids=keys,
        bulk_src=torch.from_numpy(src),
        bulk_frac=torch.from_numpy(frac),
        source=np.full(len(keys), SOURCE),
        precision_tier=np.zeros(len(keys), dtype=bool),
        grade=np.full(len(keys), -1, dtype=np.int64),
        clean=np.ones(len(keys), dtype=bool),
        sio2=sio2,
        # no locality of any kind, so every rock is its own cluster and the
        # unit bootstrap degenerates to an ordinary per-rock one.
        unit=keys.copy(),
        **truth,
    )
    info = {
        "n_rocks": len(keys),
        "off_vocab_share": 1.0 - scoreable / reported,
        "reported_modal_total": reported,
        "unused_elements": list(UNUSED_ELEMENTS),
        "dropped_minerals": sorted(set(minerals) - set(MINERAL_TO_PHASE)),
        "element_wt_frac": {
            element: frame[element].to_numpy(dtype=np.float64)
            for element in ELEMENT_TO_OXIDE
        },
        "groups_absent_from_truth": [
            group for i, group in enumerate(GROUPS)
            if not truth["presence"][:, i].any()
        ],
    }
    return benchmark, info


def write_split(
    benchmark: RealRockBenchmark, out_path: Path, seed: int = 0,
) -> Path:
    """A random 60/15/25 carve-out in the committed split file's schema.

    Random rather than unit-grouped because there are no units to group on.
    Only the climatology null reads it -- the model is scored zero-shot -- so
    this decides which rocks the do-nothing forecast is fit on, nothing else.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(benchmark))
    cuts = np.cumsum([0.60, 0.15]) * len(order)
    subset = np.empty(len(order), dtype=object)
    for name, rows in zip(
        ("train", "val", "test"), np.split(order, cuts.astype(int))
    ):
        subset[rows] = name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "source": benchmark.source,
        "sample_id": [k.split("|", 1)[1] for k in benchmark.bulk_ids],
        "unit": benchmark.unit,
        "subset": subset,
    }).to_csv(out_path, index=False)
    return out_path
