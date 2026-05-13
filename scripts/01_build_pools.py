"""Build per-(mineral, context) composition pools from min_df_v1.csv.

Reads mineralML's curated grain corpus, groups by (Mineral, Tectonic Setting),
computes basic solid-solution endmembers (Fo, An/Ab/Or, En/Fs/Wo, Mg#/Cr#),
and writes one parquet file per pool to data/pools/.

These pools feed the Position C training-data generator: when MAGEMin predicts
a phase at, say, Fo80 in a MORB context, we sample a real grain from
data/pools/Olivine__MORB.parquet that matches the predicted endmember.

Run:
    python scripts/01_build_pools.py [--min-grains 30] [--input PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Canonical 11-oxide order; matches mineralML's OXIDES constant
OXIDES = ["SiO2", "TiO2", "Al2O3", "FeOt", "MnO", "MgO", "CaO",
          "Na2O", "K2O", "P2O5", "Cr2O3"]

# Molar masses, g/mol (from mineralML's constants.py OXIDE_MASSES)
OXIDE_MW = {
    "SiO2": 60.08, "TiO2": 79.866, "Al2O3": 101.96, "FeOt": 71.844,
    "MnO": 70.9374, "MgO": 40.3044, "CaO": 56.0774, "Na2O": 61.9789,
    "K2O": 94.196, "P2O5": 141.9445, "Cr2O3": 151.99,
}

# Cations per formula unit
OXIDE_CATIONS = {
    "SiO2": 1, "TiO2": 1, "Al2O3": 2, "FeOt": 1, "MnO": 1, "MgO": 1,
    "CaO": 1, "Na2O": 2, "K2O": 2, "P2O5": 2, "Cr2O3": 2,
}

OXIDE_TO_CATION = {
    "SiO2": "Si", "TiO2": "Ti", "Al2O3": "Al", "FeOt": "Fe", "MnO": "Mn",
    "MgO": "Mg", "CaO": "Ca", "Na2O": "Na", "K2O": "K", "P2O5": "P", "Cr2O3": "Cr",
}

# Skip these from pool generation: too sparse in min_df_v1 (Quartz=86, Zircon=657)
# AND they're better handled at inference by empirical short-circuit rules
SKIP_MINERALS = {"Quartz", "Zircon"}


def cation_moles(df: pd.DataFrame) -> pd.DataFrame:
    """Convert oxide wt% to cation moles per gram (relative units, suitable for ratios)."""
    out = pd.DataFrame(index=df.index)
    for ox in OXIDES:
        if ox not in df.columns:
            continue
        wt = pd.to_numeric(df[ox], errors="coerce").fillna(0)
        out[OXIDE_TO_CATION[ox]] = wt / OXIDE_MW[ox] * OXIDE_CATIONS[ox]
    return out


def add_endmembers(df: pd.DataFrame, mineral: str) -> pd.DataFrame:
    """Append simple solid-solution endmember columns based on mineral identity.

    Inlined formulas (not via mineralML's stoichiometry classifiers) for v0 simplicity.
    More rigorous site-occupancy calculations come when we vendor mineralML's calculators.
    """
    cat = cation_moles(df)
    df = df.copy()

    if mineral == "Olivine":
        denom = (cat["Mg"] + cat["Fe"]).replace(0, np.nan)
        df["Fo"] = cat["Mg"] / denom

    elif mineral in {"Plagioclase", "KFeldspar"}:
        denom = (cat["Ca"] + cat["Na"] + cat["K"]).replace(0, np.nan)
        df["An"] = cat["Ca"] / denom
        df["Ab"] = cat["Na"] / denom
        df["Or"] = cat["K"] / denom

    elif mineral in {"Clinopyroxene", "Orthopyroxene"}:
        denom = (cat["Mg"] + cat["Fe"] + cat["Ca"]).replace(0, np.nan)
        df["En"] = cat["Mg"] / denom
        df["Fs"] = cat["Fe"] / denom
        df["Wo"] = cat["Ca"] / denom

    elif mineral == "Spinel":
        denom_mg = (cat["Mg"] + cat["Fe"]).replace(0, np.nan)
        denom_cr = (cat["Cr"] + cat["Al"]).replace(0, np.nan)
        df["MgNum"] = cat["Mg"] / denom_mg
        df["CrNum"] = cat["Cr"] / denom_cr

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "min_df_v1.csv",
        help="Path to min_df_v1.csv (originally from Zenodo deposit sarahshi-mineralML-20f9557; expected in repo's data/ dir)",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "pools",
        help="Output directory for per-pool parquet files",
    )
    parser.add_argument(
        "--min-grains", type=int, default=30,
        help="Drop (mineral, context) cells with fewer than this many grains",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"reading {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"  loaded {len(df):,} rows")

    # basic cleanup
    df = df[~df["Mineral"].isin(SKIP_MINERALS)].copy()
    df["Tectonic Setting"] = df["Tectonic Setting"].fillna("Unknown")

    # fill oxide NaNs with 0 (mineralML convention from paper §2.4)
    for ox in OXIDES:
        if ox not in df.columns:
            df[ox] = 0.0
        df[ox] = pd.to_numeric(df[ox], errors="coerce").fillna(0.0)

    # drop rows with <3 nonzero oxides (mineralML's exclusion rule, paper §2.4)
    nonzero = (df[OXIDES] > 0).sum(axis=1)
    df = df[nonzero >= 3]
    print(f"  after cleanup: {len(df):,} rows, {df['Mineral'].nunique()} minerals")

    # group + write
    summary_rows = []
    n_written = 0
    n_skipped = 0
    for (mineral, ctx), grp in df.groupby(["Mineral", "Tectonic Setting"]):
        if len(grp) < args.min_grains:
            n_skipped += 1
            continue

        out_df = add_endmembers(grp, mineral)
        ctx_slug = str(ctx).replace("/", "_").replace(" ", "_")
        path = args.out / f"{mineral}__{ctx_slug}.parquet"
        out_df.to_parquet(path, index=False)
        n_written += 1

        row = {"mineral": mineral, "context": ctx, "n_grains": len(grp)}
        for em in ("Fo", "An", "Ab", "Or", "En", "Fs", "Wo", "MgNum", "CrNum"):
            if em in out_df.columns:
                row[f"{em}_mean"] = out_df[em].mean()
                row[f"{em}_std"] = out_df[em].std()
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(["mineral", "context"])
    summary_path = args.out / "summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\nwrote {n_written} pool files, skipped {n_skipped} cells (<{args.min_grains} grains)")
    print(f"summary: {summary_path}")
    print("\nper-mineral pool counts:")
    print(summary.groupby("mineral").size().to_string())


if __name__ == "__main__":
    main()
