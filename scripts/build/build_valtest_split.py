"""Generate the sealed valtest val/test roster, one time.

Reads data/corpus/valtest_bulks.csv, drops the excluded bulk, and divides
the remaining 499 bulks 16:20 into val/test stratified by `qap_name`,
writing src/cnmineral/data/valtest_split.csv (committed; read at runtime by
`cnmineral.data.mineralogy_dataset.load_valtest_split`). The assignment is
generated once and versioned rather than recomputed at load time -- the test
subset is sealed, so membership must not drift with sklearn versions.
Refuses to overwrite an existing roster unless --force is passed.

    conda run -n alignn_fix python scripts/build/build_valtest_split.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
BULKS_CSV = REPO_ROOT / "data" / "corpus" / "valtest_bulks.csv"
OUTPUT_CSV = REPO_ROOT / "src" / "cnmineral" / "data" / "valtest_split.csv"

# near-total solver failure across its draws; full record in journal 2026-07-17
DROPPED_BULKS = [411132]

TEST_FRACTION = 20 / 36
RANDOM_STATE = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing roster")
    args = parser.parse_args()

    if OUTPUT_CSV.exists() and not args.force:
        sys.exit(f"{OUTPUT_CSV} exists; the roster is sealed. "
                 f"Pass --force to regenerate.")

    bulks = pd.read_csv(BULKS_CSV)
    bulks = bulks[~bulks["sample_id"].isin(DROPPED_BULKS)]
    val_ids, test_ids = train_test_split(
        bulks["sample_id"].to_numpy(),
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
        stratify=bulks["qap_name"],
    )

    roster = bulks[["sample_id", "qap_name"]].copy()
    roster["subset"] = "test"
    roster.loc[roster["sample_id"].isin(val_ids), "subset"] = "val"
    roster = roster.sort_values("sample_id")
    roster.to_csv(OUTPUT_CSV, index=False)

    print(f"wrote {OUTPUT_CSV}")
    print(f"val={len(val_ids)} test={len(test_ids)} of {len(roster)} bulks")
    tally = roster.groupby(["qap_name", "subset"]).size().unstack(fill_value=0)
    print(tally.to_string())


if __name__ == "__main__":
    main()
