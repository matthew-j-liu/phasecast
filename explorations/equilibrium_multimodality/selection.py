"""Volcanic + has-iron subset of gard2019 and the composition-stratified test bulks.

Builds the volcanic-core subset (item 3): 25 aphanitic QAPF names, has-iron, and
oxide-sum QC (flagged, not hard-dropped). Then selects a few dozen bulks
stratified across SiO2 (mafic -> felsic) x alkalinity (subalkaline vs alkaline)
for the multimodality test. Water is a sampled condition, not a selection axis.

Writes results/subset_class_counts.csv and results/selected_bulks.csv.
"""
import numpy as np
import pandas as pd

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

FE2O3_TO_FEO = (2 * 71.844) / 159.69


def total_fe_as_feo(df):
    """Total iron expressed as FeO wt% (for the anhydrous oxide sum), NaN if none."""
    feo = df.feo.fillna(0) + df.fe2o3.fillna(0) * FE2O3_TO_FEO
    feo = feo.where(df.feo.notna() | df.fe2o3.notna())
    feo = feo.fillna(df.feo_tot)
    feo = feo.fillna(df.fe2o3_tot * FE2O3_TO_FEO)
    return feo


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


def class_counts(df):
    d = df.assign(
        qap=df.qap_name.astype(str),
        any_flag=df[["flag_oxidesum", "flag_supersilicic", "flag_ultramafic"]].any(axis=1),
    )
    g = d.groupby("qap")
    tab = pd.DataFrame({
        "n_bulks": g.size(),
        "sio2_med": g.sio2.median().round(1),
        "n_flagged": g.any_flag.sum(),
    }).sort_values("n_bulks", ascending=False)
    return tab


# Anchor classes spanning SiO2 x alkalinity, with how many bulks to draw from each.
# Mafic-intermediate subalkaline (where the heads.md amp/opx competition lives) is
# oversampled for resolution. Bulks are drawn at evenly spaced SiO2 quantiles
# WITHIN each class, so they are class-representative rather than mislabeled tails.
ANCHORS = [
    ("basalt", 5), ("andesite", 6), ("dacite", 4), ("rhyolite", 4),
    ("quartz-rich rhyolite", 1),
    ("basanite", 2), ("tephrite", 2), ("phonolitic tephrite", 2),
    ("tephritic phonolite", 1), ("phonolite", 2), ("foidite", 1),
    ("trachyte", 1), ("quartz trachyte", 1), ("latite", 2),
    ("foid-bearing latite", 1), ("foid-bearing trachyte", 1),
]


def select_bulks(df, consistency_wt=8.0):
    """Draw class-representative bulks at evenly spaced within-class SiO2 quantiles.

    A per-class consistency guard (|SiO2 - class median| <= consistency_wt) drops
    gross mislabels (e.g. a `dacite` reported at 47% SiO2) before quantile picking.
    Deterministic.
    """
    clean = df[~(df.flag_oxidesum | df.flag_supersilicic | df.flag_ultramafic)].copy()
    picks = []
    for qap, n in ANCHORS:
        pool = clean[clean.qap_name.astype(str) == qap]
        if pool.empty:
            continue
        med = pool.sio2.median()
        pool = pool[(pool.sio2 - med).abs() <= consistency_wt]
        qs = np.linspace(0.15, 0.85, n)
        targets = pool.sio2.quantile(qs).values
        used = set()
        for t in targets:
            cand = pool[~pool.sample_id.isin(used)]
            if cand.empty:
                break
            r = cand.iloc[(cand.sio2 - t).abs().argmin()]
            used.add(int(r.sample_id))
            picks.append(dict(sample_id=int(r.sample_id), qap_name=qap,
                              series=str(r.series), sio2=round(float(r.sio2), 2),
                              total_alkali=round(float(r.total_alkali), 2)))
    sel = pd.DataFrame(picks).drop_duplicates("sample_id").reset_index(drop=True)
    sel = sel.sort_values(["series", "sio2"]).reset_index(drop=True)
    return sel


def main():
    df = build_subset()
    print(f"volcanic-core subset: {len(df)} bulks across {df.qap_name.nunique()} qap classes")
    counts = class_counts(df)
    counts.to_csv("explorations/equilibrium_multimodality/results/subset_class_counts.csv")
    print(counts.to_string())
    print(f"\nQC flags: oxidesum {int(df.flag_oxidesum.sum())}, "
          f"supersilicic {int(df.flag_supersilicic.sum())}, "
          f"ultramafic {int(df.flag_ultramafic.sum())}")

    sel = select_bulks(df)
    sel.to_csv("explorations/equilibrium_multimodality/results/selected_bulks.csv", index=False)
    print(f"\nselected {len(sel)} test bulks "
          f"({(sel.series=='subalkaline').sum()} subalk / {(sel.series=='alkaline').sum()} alk)")
    print(sel.to_string())


if __name__ == "__main__":
    main()
