"""Filter complete.csv → MAGEMin-ready inputs CSV, stratified by qap_name.

Replaces the existing CrabNet-Mineral filter pipeline with:
  - Permissive Fe handling (Tier 1 measured + Tier 2 class-ratio fallback)
  - Stage-2-permissive Cr2O3/H2O (fill missing with 0, don't require)
  - Quality bounds (sum 80-120 wt%, reject outliers)
  - Stratified sampling by qap_name with configurable per-class targets
  - Outputs MAGEMin's expected CSV schema + fe_provenance + qap_name metadata

See docs/magemin_handoff.md for the downstream Julia pipeline.

Run:
    # smoke test (~500 bulks total)
    python scripts/02_filter_complete.py --scale smoke

    # full v1 corpus (~22k bulks)
    python scripts/02_filter_complete.py --scale full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cnmineral.data.fe_redox import ClassRatioStrategy, ConstantRatioStrategy
from cnmineral.data.h2o import ClassRangeH2OStrategy, ConstantH2OStrategy
from cnmineral.data.schema import (
    MAGEMIN_OXIDES_LOWER,
    loi_normalize,
    mineralml_to_magemin,
)


COMPLETE_PATH_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "complete.csv"

REQUIRED_MAJORS = ["sio2", "al2o3", "cao", "mgo", "k2o", "na2o", "tio2"]
FE_COLS = ["feo", "fe2o3", "feo_tot", "fe2o3_tot"]
META_COLS = ["sample_id", "rock_group", "qap_name", "rock_type", "rock_name", "rock_origin"]


# Per-class targets for the "full" v1 corpus. Tunable; the smoke test scales these down.
FULL_TARGETS: dict[str, int] = {
    "basalt": 3000,
    "andesite": 2500,
    "dunite": 2000,
    "dacite": 2000,
    "monzogranite": 2000,
    "granodiorite": 2000,
    "rhyolite": 2000,
    "tonalite": 1500,
    "diorite": 1500,
    "subalkalic gabbro": 2000,
    "alkalic gabbro": 1500,
    "syenite": 1000,
    "monzonite": 1000,
    "phonolite": 800,
    "trachyte": 800,
    "harzburgite": 1000,
    "lherzolite": 1000,
    "foidite": 500,
    "tephrite": 500,
    "picrite": 500,
    "_other": 1000,    # everything not above; sampled flat from remaining classes
}

SCALES: dict[str, dict[str, int]] = {
    "smoke":  {k: max(10, v // 50) for k, v in FULL_TARGETS.items()},     # ~500 bulks
    "dev":    {k: max(50, v // 10) for k, v in FULL_TARGETS.items()},     # ~2.5k bulks
    "full":   FULL_TARGETS,
}


def load_igneous(path: Path) -> pd.DataFrame:
    """Load complete.csv, restrict to igneous, keep only the columns we need."""
    keep = META_COLS + REQUIRED_MAJORS + FE_COLS + ["mno", "p2o5", "cr2o3", "h2o_tot"]
    keep_set = set(keep)
    df = pd.read_csv(
        path,
        encoding="latin-1",
        low_memory=False,
        usecols=lambda c: c.lower() in keep_set,
    )
    df.columns = [c.lower() for c in df.columns]
    df = df[df["rock_group"] == "igneous"].copy()
    return df


def filter_quality(df: pd.DataFrame, min_sum: float = 80.0, max_sum: float = 120.0) -> pd.DataFrame:
    """Drop rows with missing required majors, bad oxide sums, or unusable iron."""
    # Required majors must all be present
    req_mask = df[REQUIRED_MAJORS].notna().all(axis=1)

    # Must have SOME iron column
    fe_mask = df[FE_COLS].notna().any(axis=1)

    # Compute total-Fe-as-FeO equivalent for sum check
    feo_equiv = (
        df["feo"].fillna(0)
        + df["fe2o3"].fillna(0) * (2 * 71.844 / 159.69)
        + df["feo_tot"].fillna(0)
        + df["fe2o3_tot"].fillna(0) * (2 * 71.844 / 159.69)
    )
    # Don't double-count if multiple cols present — pick the largest (most complete) source
    fe_one = df[FE_COLS].fillna(0).max(axis=1)
    # Conservative: estimate iron contribution as min of (sum of measured) and FeO-equiv of largest tot
    fe_for_sum = df[["feo", "fe2o3"]].fillna(0).sum(axis=1).where(
        df["feo"].notna() & df["fe2o3"].notna(),
        df[["feo_tot"]].fillna(0).max(axis=1).where(
            df["feo_tot"].notna(),
            df[["fe2o3_tot"]].fillna(0).max(axis=1) * (2 * 71.844 / 159.69),
        ),
    )

    oxide_sum = (
        df[REQUIRED_MAJORS].fillna(0).sum(axis=1)
        + fe_for_sum
        + df[["mno", "p2o5", "cr2o3"]].fillna(0).sum(axis=1)
    )
    sum_mask = (oxide_sum >= min_sum) & (oxide_sum <= max_sum)

    mask = req_mask & fe_mask & sum_mask
    return df.loc[mask].copy()


def stratified_sample(
    df: pd.DataFrame,
    targets: dict[str, int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample N rows per qap_name class, with `_other` absorbing unlisted classes."""
    named_classes = [k for k in targets if k != "_other"]
    parts = []

    for cls in named_classes:
        pool = df[df["qap_name"] == cls]
        n = targets[cls]
        if len(pool) == 0:
            continue
        if len(pool) <= n:
            parts.append(pool.copy())
        else:
            parts.append(pool.sample(n=n, random_state=rng.integers(2**31)))

    # Other classes: pool everything not in named_classes, sample flat
    other_pool = df[~df["qap_name"].isin(named_classes)]
    n_other = targets.get("_other", 0)
    if n_other > 0 and len(other_pool) > 0:
        n_take = min(n_other, len(other_pool))
        parts.append(other_pool.sample(n=n_take, random_state=rng.integers(2**31)))

    return pd.concat(parts, ignore_index=True) if parts else df.iloc[:0].copy()


def _make_fe_strategy(spec: str):
    if spec == "middlemost":
        return ClassRatioStrategy()
    if spec.startswith("const:"):
        return ConstantRatioStrategy(float(spec.split(":")[1]))
    raise ValueError(f"unknown fe strategy '{spec}'")


def _make_h2o_strategy(spec: str):
    if spec == "wallace_plank":
        return ClassRangeH2OStrategy()
    if spec.startswith("const:"):
        return ConstantH2OStrategy(float(spec.split(":")[1]))
    raise ValueError(f"unknown h2o strategy '{spec}'")


def convert_rows(
    df: pd.DataFrame,
    fe_strategy_name: str = "middlemost",
    h2o_strategy_name: str = "wallace_plank",
    seed: int = 42,
) -> pd.DataFrame:
    """Convert each row to MAGEMin's 11-oxide schema, LOI-normalize, tag provenance.

    Both Fe³⁺/ΣFe and H2O handled via injectable strategies; provenance for each
    is carried through to the output CSV for paper-side methodological transparency.
    """
    fe_strategy = _make_fe_strategy(fe_strategy_name)
    h2o_strategy = _make_h2o_strategy(h2o_strategy_name)
    rng = np.random.default_rng(seed)

    records = []
    for _, row in df.iterrows():
        qap = (row.get("qap_name") or "").strip().lower()
        magemin = mineralml_to_magemin(
            row, rock_class=qap,
            fe3_strategy=fe_strategy,
            h2o_strategy=h2o_strategy,
            rng=rng,
        )
        if magemin["fe_provenance"] == "no_iron":
            continue

        # LOI-normalize numeric oxides
        numeric_only = {k: v for k, v in magemin.items() if k in MAGEMIN_OXIDES_LOWER}
        normalized = loi_normalize(numeric_only, target_sum=100.0)

        record = {"sample_id": row.get("sample_id")}
        record.update(normalized)
        record["fe_provenance"] = magemin["fe_provenance"]
        record["h2o_provenance"] = magemin["h2o_provenance"]
        record["qap_name"] = qap
        records.append(record)

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=COMPLETE_PATH_DEFAULT,
                        help="Path to complete.csv (Gard et al. 2019)")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent.parent / "data" / "magemin_inputs.csv",
                        help="Output CSV in MAGEMin's expected schema")
    parser.add_argument("--scale", choices=list(SCALES.keys()), default="smoke",
                        help="Scale preset: smoke (~500), dev (~2.5k), full (~22k)")
    parser.add_argument("--fe-strategy", default="middlemost",
                        help="'middlemost' (default), or 'const:<ratio>' (e.g. const:0.15)")
    parser.add_argument("--h2o-strategy", default="wallace_plank",
                        help="'wallace_plank' (default, per-class range), or 'const:<wt%>' (e.g. const:0.0 for dry)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"loading {args.input}")
    df = load_igneous(args.input)
    print(f"  igneous rows: {len(df):,}")

    print("applying quality filter")
    df = filter_quality(df)
    print(f"  after quality filter: {len(df):,}")
    print(f"  qap_name non-null: {df['qap_name'].notna().sum():,}")

    print(f"stratified sampling with scale='{args.scale}'")
    targets = SCALES[args.scale]
    df_sampled = stratified_sample(df, targets, rng)
    print(f"  sampled: {len(df_sampled):,}")
    print("  per-class counts:")
    cnt = df_sampled["qap_name"].fillna("(unlisted)").value_counts()
    print(cnt.head(15).to_string())

    print(f"converting to MAGEMin schema (fe={args.fe_strategy}, h2o={args.h2o_strategy})")
    df_out = convert_rows(
        df_sampled,
        fe_strategy_name=args.fe_strategy,
        h2o_strategy_name=args.h2o_strategy,
        seed=args.seed,
    )
    print(f"  converted: {len(df_out):,}")
    print("  Fe provenance distribution:")
    print(df_out["fe_provenance"].value_counts().to_string())
    print("  H2O provenance distribution:")
    print(df_out["h2o_provenance"].value_counts().to_string())
    print(f"  H2O range in output: {df_out['h2o_tot'].min():.3f} – {df_out['h2o_tot'].max():.3f} wt%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
