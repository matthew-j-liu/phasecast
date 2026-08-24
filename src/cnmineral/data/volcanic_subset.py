"""Volcanic + has-iron subset of gard2019: the supervised-corpus core.

Builds the volcanic-core subset: 25 aphanitic QAPF names, has-iron, and
oxide-sum QC (flagged, not hard-dropped). The corpus bulk tables
(scripts/build/build_corpus_bulk_tables.py) draw from this subset.
"""
import numpy as np
import pandas as pd

from cnmineral.data.magemin_oxides import FE2O3_TO_FEO_MASS

PARQUET = "data/bulks/gard2019.parquet"

# Aphanitic (extrusive) QAPF names -- ~100% `volcanic` in the full-CSV rock_origin
# cross-check, and they carry the mass. The plutonic-texture-but-volcanic-tagged
# names (gabbro/norite/anorthosite) are an owner decision, left OUT of the core.
VOLCANIC_QAP = [
    "basalt", "andesite", "dacite", "rhyolite",
    "trachyte", "latite", "phonolite", "tephrite", "basanite", "foidite",
    "quartz latite", "quartz-rich rhyolite", "alkali feldspar rhyolite",
    "quartz trachyte", "alkali feldspar trachyte", "alkali feldspar quartz trachyte",
    "foid-bearing trachyte", "foid-bearing alkali feldspar trachyte",
    "foid-bearing latite", "phonolitic tephrite", "tephritic phonolite",
    "phonolitic basanite", "phonolitic foidite", "tephritic foidite", "silexite",
]

# Alkaline QAPF series (silica-undersaturated / high-alkali). Everything else in
# VOLCANIC_QAP is subalkaline. Used to guarantee the test spans both limbs.
ALKALINE_QAP = {
    "trachyte", "latite", "phonolite", "tephrite", "basanite", "foidite",
    "quartz trachyte", "alkali feldspar trachyte", "alkali feldspar quartz trachyte",
    "foid-bearing trachyte", "foid-bearing alkali feldspar trachyte",
    "foid-bearing latite", "phonolitic tephrite", "tephritic phonolite",
    "phonolitic basanite", "phonolitic foidite", "tephritic foidite",
}


def total_fe_as_feo(df):
    """Total iron expressed as FeO wt% (for the anhydrous oxide sum), NaN if none."""
    feo = df.feo.fillna(0) + df.fe2o3.fillna(0) * FE2O3_TO_FEO_MASS
    feo = feo.where(df.feo.notna() | df.fe2o3.notna())
    feo = feo.fillna(df.feo_tot)
    feo = feo.fillna(df.fe2o3_tot * FE2O3_TO_FEO_MASS)
    return feo


def expansion_quotas(pool, have, avail, budget, alpha=0.5):
    """Per-class quotas for adding `budget` bulks to an existing tier.

    Temperature-smoothed allocation (square-root sampling at the default
    alpha, per Mahajan et al. 2018): each class targets a `pool**alpha` share
    of the expanded tier, less what it already holds, capped by availability;
    mass freed by capped classes respills to the open ones by the same
    weights. `pool`/`have`/`avail` are aligned Series indexed by class;
    returns integer quotas summing to min(budget, avail.sum()).
    """
    w = pool.astype(float) ** alpha
    target = w / w.sum() * (float(have.sum()) + budget)
    want = (target - have).clip(lower=0.0)
    if want.sum() > budget:
        want = want * (budget / want.sum())
    grant = np.minimum(want, avail.astype(float))
    while grant.sum() < budget - 1e-9:
        headroom = avail - grant
        open_ = headroom > 1e-9
        if not open_.any():
            break
        add = w[open_] / w[open_].sum() * (budget - grant.sum())
        grant[open_] = np.minimum(grant[open_] + add, avail[open_].astype(float))
    total = int(round(grant.sum()))
    quotas = grant.astype(int)
    for qap in (grant - quotas).sort_values(ascending=False).index:
        if quotas.sum() >= total:
            break
        if quotas[qap] < avail[qap]:
            quotas[qap] += 1
    return quotas


def build_subset():
    df = pd.read_parquet(PARQUET)
    df = df[df.rock_group == "igneous"].copy()
    df = df[df.qap_name.astype(str).isin(VOLCANIC_QAP)].copy()

    fe = df[["feo", "fe2o3", "feo_tot", "fe2o3_tot"]]
    df["has_fe"] = (fe.fillna(0) > 0).any(axis=1)
    df["sio2"] = pd.to_numeric(df.sio2, errors="coerce")
    df = df[df.has_fe & (df.sio2 > 0)].copy()

    df["feo_star"] = total_fe_as_feo(df)
    majors = ["sio2", "tio2", "al2o3", "mgo", "cao", "k2o", "na2o"]
    df["oxide_sum"] = df[majors].fillna(0).sum(axis=1) + df.feo_star.fillna(0)
    df["total_alkali"] = df.na2o.fillna(0) + df.k2o.fillna(0)
    df["series"] = np.where(df.qap_name.astype(str).isin(ALKALINE_QAP), "alkaline", "subalkaline")

    # QC flags (informational; not hard filters at this stage)
    df["flag_oxidesum"] = (df.oxide_sum < 90) | (df.oxide_sum > 102)
    df["flag_supersilicic"] = df.sio2 > 80
    df["flag_ultramafic"] = df.mgo.fillna(0) > 18   # picrite/komatiite-ish
    return df
