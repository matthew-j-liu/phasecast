"""The evaluated benchmark rocks as a MAGEMin bulk table: the grade 1 and 2
rocks with a quantified mode, in benchmark order, with a rock class for the
class-conditional iron split.

Benchmark ids are strings and many collide numerically with gard2019 ids, so
the table carries its own integer `bulk` index beside the benchmark key. The
iron ratio table (`MIDDLEMOST_1989`) is keyed by rock name and the benchmark
carries none, so each rock is classified on the total alkali versus silica
diagram (Le Bas et al. 1986), the volcanic chemical classification those
ratios are defined on. Basaltic trachyandesite is not a key and takes
basaltic andesite's ratio (the silica-matched neighbour between trachybasalt
and trachyandesite); picrobasalt takes picrite's. A composition outside every
field (anhydrous SiO2 above 87.5 wt%) takes the nearest field and is flagged.
"""

import numpy as np
import pandas as pd
from matplotlib.path import Path

from cnmineral.data.fe_redox import MIDDLEMOST_1989, ClassRatioStrategy
from cnmineral.data.magemin_oxides import FE2O3_TO_FEO_MASS, split_fe
from cnmineral.eval.real_rock import SPLIT_SUBSETS, RealRockBenchmark, split_mask

# Le Bas et al. 1986 fields as (SiO2, Na2O + K2O) wt% polygons on an anhydrous
# basis, transcribed from pyrolite's TAS classifier
TAS_FIELDS = {
    "picrobasalt": [(41, 0), (41, 3), (45, 3), (45, 0)],
    "basalt": [(45, 0), (45, 5), (52, 5), (52, 0)],
    "basaltic andesite": [(52, 0), (52, 5), (57, 5.9), (57, 0)],
    "andesite": [(57, 0), (57, 5.9), (63, 7), (63, 0)],
    "dacite": [(63, 0), (63, 7), (69, 8), (77.3, 0)],
    "rhyolite": [(69, 8), (71.8, 13.5), (85.9, 6.8), (87.5, 4.7), (77.3, 0)],
    "trachybasalt": [(45, 5), (49.4, 7.3), (52, 5)],
    "basaltic trachyandesite": [(49.4, 7.3), (53, 9.3), (57, 5.9), (52, 5)],
    "trachyandesite": [(53, 9.3), (57.6, 11.7), (61, 8.6), (63, 7), (57, 5.9)],
    "trachyte": [(57.6, 11.7), (61, 13.5), (63, 16.2), (71.8, 13.5), (61, 8.6)],
    "trachydacite": [(61, 8.6), (71.8, 13.5), (69, 8), (63, 7)],
    "tephrite": [(41, 3), (41, 7), (45, 9.4), (49.4, 7.3), (45, 5), (45, 3)],
    "phonotephrite": [(45, 9.4), (48.4, 11.5), (53, 9.3), (49.4, 7.3)],
    "tephriphonolite": [(48.4, 11.5), (52.5, 14), (57.6, 11.7), (53, 9.3)],
    "phonolite": [(52.5, 14), (52.5, 18), (57, 18), (63, 16.2), (61, 13.5), (57.6, 11.7)],
    "foidite": [(35, 9), (37, 14), (52.5, 18), (52.5, 14), (48.4, 11.5), (45, 9.4),
                (41, 7), (41, 3), (37, 3)],
}
TAS_TO_CLASS = {"picrobasalt": "picrite", "basaltic trachyandesite": "basaltic andesite"}

ANHYDROUS = ("sio2", "tio2", "al2o3", "mno", "mgo", "cao", "na2o", "k2o", "p2o5")
PASSTHROUGH = ("source", "sample_id", "unit", "total_crystals_pct", "alteration_extent",
               "mode_basis", "sio2", "tio2", "al2o3", "feo_tot", "fe2o3_tot", "mno", "mgo",
               "cao", "na2o", "k2o", "p2o5", "loi")

_PATHS = {name: Path(v) for name, v in TAS_FIELDS.items()}


def _distance_to_polygon(x, y, vertices):
    v = np.asarray(vertices, dtype=float)
    a, b = v, np.roll(v, -1, axis=0)
    ab = b - a
    t = ((x - a[:, 0]) * ab[:, 0] + (y - a[:, 1]) * ab[:, 1]) / (ab ** 2).sum(1)
    proj = a + np.clip(t, 0.0, 1.0)[:, None] * ab
    return float(np.sqrt(((proj - (x, y)) ** 2).sum(1)).min())


def tas_field(sio2_anh, alk_anh):
    """TAS field name per composition and whether it lay inside a field; a
    composition outside every field takes the nearest one."""
    names, inside = [], []
    for s, a in zip(np.asarray(sio2_anh, dtype=float), np.asarray(alk_anh, dtype=float)):
        hit = next((n for n, path in _PATHS.items()
                    if path.contains_point((s, a), radius=1e-9)
                    or path.contains_point((s, a), radius=-1e-9)), None)
        inside.append(hit is not None)
        if hit is None:
            hit = min(TAS_FIELDS, key=lambda n: _distance_to_polygon(s, a, TAS_FIELDS[n]))
        names.append(hit)
    return np.array(names, dtype=object), np.array(inside, dtype=bool)


def rock_class(field):
    cls = TAS_TO_CLASS.get(field, field)
    if cls not in MIDDLEMOST_1989:
        raise KeyError(f"TAS field {field!r} has no iron ratio; extend TAS_TO_CLASS")
    return cls


def anhydrous_tas_inputs(rows):
    """SiO2 and Na2O + K2O renormalized over the anhydrous majors, total iron
    counted as FeO."""
    feo = rows["feo_tot"].fillna(rows["fe2o3_tot"] * FE2O3_TO_FEO_MASS).fillna(0.0)
    total = rows[list(ANHYDROUS)].fillna(0.0).sum(axis=1) + feo
    scale = 100.0 / total
    return (rows["sio2"] * scale).to_numpy(), ((rows["na2o"] + rows["k2o"]) * scale).to_numpy()


def composition_table(benchmark: RealRockBenchmark, samples: pd.DataFrame) -> pd.DataFrame:
    """The evaluated rocks (clean, with a quantified mode) in benchmark order,
    their reported chemistry, TAS field, iron class and split provenance."""
    mask = benchmark.clean & benchmark.has_magnitude
    keys = benchmark.bulk_ids[mask]
    samples = samples.copy()
    samples["key"] = samples["source"] + "|" + samples["sample_id"].astype(str)
    for col in PASSTHROUGH:
        if col not in samples.columns:
            samples[col] = np.nan
    rows = samples.set_index("key").loc[keys].reset_index()

    subset = np.full(len(benchmark), "none", dtype=object)
    for s in SPLIT_SUBSETS:
        subset[split_mask(benchmark, s)] = s

    sio2_anh, alk_anh = anhydrous_tas_inputs(rows)
    field, inside = tas_field(sio2_anh, alk_anh)
    classes = [rock_class(f) for f in field]
    fe3 = ClassRatioStrategy()
    provenance = [split_fe(r, c, fe3).provenance for (_, r), c in zip(rows.iterrows(), classes)]

    out = pd.DataFrame({"bulk": np.arange(len(rows)), "key": rows["key"]})
    for col in PASSTHROUGH:
        out[col] = rows[col].to_numpy()
    out.insert(6, "subset", subset[mask])
    out["tas_sio2"] = sio2_anh
    out["tas_alk"] = alk_anh
    out["tas_field"] = field
    out["tas_inside"] = inside
    out["rock_class"] = classes
    out["fe3_ratio"] = [fe3(c) for c in classes]
    out["fe_provenance"] = provenance
    return out


def load_compositions(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"sample_id": str})
