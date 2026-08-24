"""Build the deterministic 2000-row test corpus from data/bulks/gard2019.csv.

Output: tests/fixtures/complete_subset.parquet (~113 KB; checked in).
Re-run if `_load_raw_dataframe`'s column list changes or the seed
needs to be redrawn.  Idempotent: same seed produces the same bytes.

    conda run -n mineral python scripts/build/build_test_fixture.py
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

from cnmineral.data.bulk_chemistry import _load_raw_dataframe

SOURCE_CSV = REPO_ROOT / "data" / "bulks" / "gard2019.csv"
OUTPUT_PARQUET = REPO_ROOT / "tests" / "fixtures" / "complete_subset.parquet"
SAMPLE_SIZE = 2000
SEED = 42


def main() -> None:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"source not found: {SOURCE_CSV}")

    df = _load_raw_dataframe(SOURCE_CSV)
    df = df[df["rock_group"] == "igneous"]

    rng = np.random.default_rng(seed=SEED)
    idx = rng.choice(len(df), size=SAMPLE_SIZE, replace=False)
    idx.sort()
    sample = df.iloc[idx].reset_index(drop=True)

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"wrote {len(sample)} rows -> {OUTPUT_PARQUET.relative_to(REPO_ROOT)}")
    print(f"file size: {OUTPUT_PARQUET.stat().st_size/1024:.1f} KB")
    print(f"unique qap_name classes in sample: {sample['qap_name'].nunique(dropna=True)}")


if __name__ == "__main__":
    main()
