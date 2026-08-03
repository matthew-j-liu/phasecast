"""Extract a crustal-volcanic (bulk chemistry -> mineral assemblage) validation
table from the LEPR experimental-petrology dump.

Filters to starting SiO2 in [45, 78] wt%, P <= 1 GPa, and assemblages with a
silicate melt plus at least one solid phase; maps LEPR phase codes to the
model's coarse mineral vocabulary.  Output lands in results/ next to this file.
"""

import os
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEPR_PATH = os.path.join(REPO_ROOT, "data", "experiments", "ExPetDB_download_LEPR-2007-08-17.xlsx")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

OXIDES = ["SiO2", "TiO2", "Al2O3", "FeO", "Fe2O3", "MgO", "CaO", "Na2O", "K2O"]
# LEPR's "H2O value" column is entirely empty; water is split across
# "H2OP value" (H2O+) and "H2OM value" (H2O-), summed here into one H2O
H2O_COLS = ["H2OP", "H2OM"]

COARSE_MAP = {
    "plag": "feldspar", "kfs": "feldspar",
    "cpx": "clinopyroxene", "pig": "clinopyroxene",
    "opx": "orthopyroxene",
    "amph": "amphibole",
    "ol": "olivine",
    "gt": "garnet",
    "ilm": "oxide", "ox": "oxide", "spn": "oxide", "mgt": "oxide",
    "tmg": "oxide", "hem": "oxide", "rut": "oxide",
    "qz": "quartz/silica", "trid": "quartz/silica", "crst": "quartz/silica",
    "bi": "mica", "phl": "mica", "ms": "mica",
    "neph": "feldspathoid", "leuc": "feldspathoid",
    "liq": "melt",
    "fl": "fluid",
    # accessory / minor crustal phases -> "other"
    "ap": "other", "tit": "other", "zo": "other", "ep": "other", "all": "other",
    "wh": "other", "carb": "other", "sul": "other", "anhy": "other",
    "ky": "other", "sil": "other", "cor": "other", "crd": "other", "st": "other",
    "wo": "other", "mel": "other", "tc": "other", "chevk": "other", "ha": "other",
    "montd": "other", "clay": "other",
}

# deep-mantle / metal / exotic high-P codes: dropped from assemblages entirely
DROP_CODES = {
    "capv", "mgpv", "mw", "per", "cas", "stis", "coe", "met", "fe", "nals",
    "hol", "pv", "phsE", "armc", "chm", "tisi", "fb",
    "liqc",  # carbonatite melt; not the silicate liq we validate against
}

MELT_FLUID = {"melt", "fluid"}

# `Experiment` IDs repeat across papers (each lab numbers its own runs), so the
# join key is (Citation, Experiment)
KEY = ["Citation", "Experiment"]


def load_sheets():
    xl = pd.ExcelFile(LEPR_PATH, engine="calamine")
    exp = xl.parse("Experiment")
    bulk = xl.parse("Bulk (starting material)")
    return exp, bulk


def dedup_experiments(exp):
    exp = exp.drop_duplicates(subset=KEY + ["T (C)", "P (GPa)", "Phases"])
    dup = exp.duplicated(KEY, keep=False)
    n_conflict = exp.loc[dup].groupby(KEY).ngroups
    return exp[~dup].copy(), n_conflict


def dedup_bulk(bulk):
    value_cols = [f"{ox} value" for ox in OXIDES + H2O_COLS if f"{ox} value" in bulk.columns]
    b = bulk[KEY + value_cols].copy()
    for c in value_cols:
        b[c] = pd.to_numeric(b[c], errors="coerce")

    conflict_keys = set()
    for key, grp in b.groupby(KEY):
        if len(grp) == 1:
            continue
        vals = grp[value_cols]
        # disagreement = any oxide whose duplicate values differ by > 0.01 wt%
        spread = vals.max() - vals.min()
        if (spread.fillna(0) > 0.01).any():
            conflict_keys.add(key)

    deduped = b.groupby(KEY, as_index=False).first()
    keys = list(zip(deduped["Citation"], deduped["Experiment"]))
    deduped["bulk_conflict"] = [k in conflict_keys for k in keys]
    return deduped, conflict_keys


def parse_phases(s):
    return [c.strip() for c in str(s).split(",") if c.strip()]


def main():
    exp, bulk = load_sheets()
    n_exp, n_bulk = len(exp), len(bulk)

    exp, n_exp_conflict = dedup_experiments(exp)
    deduped, conflict_keys = dedup_bulk(bulk)
    merged = exp.merge(deduped, on=KEY, how="inner", validate="one_to_one")
    print(f"{n_exp} experiments, {n_bulk} bulk rows -> {len(merged)} with a bulk "
          f"({n_exp_conflict} (citation, id) keys dropped for conflicting T/P/Phases; "
          f"{len(conflict_keys)} had conflicting duplicate bulks)")

    merged["P_kbar"] = pd.to_numeric(merged["P (GPa)"], errors="coerce") * 10.0
    merged["T_C"] = pd.to_numeric(merged["T (C)"], errors="coerce")

    h2op, h2om = merged["H2OP value"], merged["H2OM value"]
    merged["H2O"] = h2op.fillna(0) + h2om.fillna(0)
    merged.loc[h2op.isna() & h2om.isna(), "H2O"] = pd.NA

    feo = merged["FeO value"]
    fe2o3 = merged["Fe2O3 value"]
    merged["fe_total"] = feo.fillna(0) + 0.8998 * fe2o3.fillna(0)
    merged.loc[feo.isna() & fe2o3.isna(), "fe_total"] = pd.NA
    merged["fe_fields"] = (
        feo.notna().map({True: "FeO", False: ""})
        + (feo.notna() & fe2o3.notna()).map({True: "+", False: ""})
        + fe2o3.notna().map({True: "Fe2O3", False: ""})
    )

    codes = merged["Phases"].apply(parse_phases)
    merged["phases_raw"] = codes.apply(",".join)
    merged["unmapped_codes"] = codes.apply(
        lambda cs: ",".join(sorted({c for c in cs if c not in COARSE_MAP and c not in DROP_CODES})))
    merged["dropped_codes"] = codes.apply(
        lambda cs: ",".join(sorted({c for c in cs if c in DROP_CODES})))
    coarse = codes.apply(
        lambda cs: sorted({COARSE_MAP[c] for c in cs if c in COARSE_MAP}))
    merged["assemblage_coarse"] = coarse.apply(",".join)

    has_liq = codes.apply(lambda cs: "liq" in cs)
    has_solid = coarse.apply(lambda ph: any(p not in MELT_FLUID for p in ph))
    sio2 = merged["SiO2 value"]
    in_sio2 = sio2.between(45, 78)
    in_p = merged["P_kbar"] <= 10

    print(f"filters (of {len(merged)} merged): SiO2 in [45,78]: {in_sio2.sum()}, "
          f"P<=10 kbar: {in_p.sum()}, has liq: {has_liq.sum()}, "
          f"has >=1 solid: {has_solid.sum()}")

    keep = merged[in_sio2 & in_p & has_liq & has_solid].copy()
    print(f"surviving experiments: {len(keep)}")

    out = pd.DataFrame({
        "experiment_id": keep["Experiment"],
        "citation": keep["Citation"],
        "material": keep["Material"],
        **{ox: keep[f"{ox} value"] for ox in OXIDES},
        "H2O": keep["H2O"],
        "fe_total": keep["fe_total"],
        "fe_fields": keep["fe_fields"],
        "assemblage_coarse": keep["assemblage_coarse"],
        "phases_raw": keep["phases_raw"],
        "unmapped_codes": keep["unmapped_codes"],
        "dropped_codes": keep["dropped_codes"],
        "bulk_conflict": keep["bulk_conflict"],
        "P_kbar": keep["P_kbar"],
        "T_C": keep["T_C"],
    })

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "lepr_crustal_volcanic.csv")
    out.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    print("\n-- summary --")
    bands = pd.cut(out["SiO2"], [45, 52, 63, 68, 78],
                   labels=["basalt 45-52", "andesite 52-63", "dacite 63-68", "rhyolite 68-78"],
                   include_lowest=True)
    print("SiO2 bands:")
    print(bands.value_counts().sort_index().to_string())

    print("\ncoarse phase counts (experiments containing each):")
    from collections import Counter
    cnt = Counter()
    for a in out["assemblage_coarse"]:
        cnt.update(a.split(","))
    for k, v in cnt.most_common():
        print(f"  {k}: {v}")

    print("\ndata quality:")
    for ox in OXIDES + ["H2O"]:
        n_missing = out[ox].isna().sum()
        if n_missing:
            print(f"  missing {ox}: {n_missing}")
    totals = out[OXIDES].sum(axis=1, skipna=True)
    weird = ((totals < 95) | (totals > 102)).sum()
    print(f"  anhydrous oxide total outside [95, 102]: {weird} "
          f"(min {totals.min():.1f}, max {totals.max():.1f})")
    print(f"  bulk_conflict rows kept: {out['bulk_conflict'].sum()}")
    print(f"  rows with unmapped phase codes: {(out['unmapped_codes'] != '').sum()} "
          f"({sorted(set(','.join(out['unmapped_codes']).split(',')) - {''})})")
    print(f"  rows with dropped deep-mantle/metal codes: {(out['dropped_codes'] != '').sum()}")
    print(f"  material column non-null: {out['material'].notna().sum()} (LEPR left it blank)")
    print(f"  fe_fields breakdown: {out['fe_fields'].value_counts(dropna=False).to_dict()}")
    print(f"  amphibole-bearing: {cnt.get('amphibole', 0)}")


if __name__ == "__main__":
    main()
