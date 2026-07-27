"""Generate the sealed valtest val/test roster, one time.

Reads data/corpus/valtest_bulks.csv, drops the excluded bulk, and divides
the remaining bulks 16:20 into val/test stratified by `qap_name`, writing
src/cnmineral/data/valtest_split.csv (committed; read at runtime by
`cnmineral.data.mineralogy_dataset.load_valtest_split`). The assignment is
generated once and versioned rather than recomputed at load time -- the test
subset is sealed, so membership must not drift with sklearn versions.
Refuses to overwrite an existing roster unless --force is passed.

    conda run -n alignn_fix python scripts/build/build_valtest_split.py

`--extend` assigns only the bulks a grown valtest_bulks.csv adds, at the same
ratio and stratification, and leaves every committed assignment exactly where
it was. That is what keeps the seal intact across a roster expansion: growing
the tier must not move a bulk from test to val.
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


def assign(bulks: pd.DataFrame) -> pd.DataFrame:
    """Stratified val/test assignment over `bulks`."""
    thin = bulks["qap_name"].value_counts()
    thin = thin[thin < 2]
    if len(thin):
        sys.exit(f"cannot stratify {thin.to_dict()}: each class needs 2+ bulks")
    val_ids, _ = train_test_split(
        bulks["sample_id"].to_numpy(),
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
        stratify=bulks["qap_name"],
    )
    roster = bulks[["sample_id", "qap_name"]].copy()
    roster["subset"] = "test"
    roster.loc[roster["sample_id"].isin(val_ids), "subset"] = "val"
    return roster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing roster")
    parser.add_argument("--extend", action="store_true",
                        help="assign only the bulks the table adds, freezing "
                             "every committed assignment")
    args = parser.parse_args()
    if args.force and args.extend:
        parser.error("--extend never overwrites an assignment; --force is moot")

    bulks = pd.read_csv(BULKS_CSV)
    bulks = bulks[~bulks["sample_id"].isin(DROPPED_BULKS)]

    if args.extend:
        if not OUTPUT_CSV.exists():
            sys.exit(f"{OUTPUT_CSV} does not exist; build the roster first")
        held = pd.read_csv(OUTPUT_CSV)
        new = bulks[~bulks["sample_id"].isin(held["sample_id"])]
        if new.empty:
            sys.exit(f"{OUTPUT_CSV} already covers every bulk in {BULKS_CSV.name}")
        roster = pd.concat([held, assign(new)], ignore_index=True)
        print(f"extending by {len(new)} bulks; "
              f"{len(held)} committed assignments frozen")
    else:
        if OUTPUT_CSV.exists() and not args.force:
            sys.exit(f"{OUTPUT_CSV} exists; the roster is sealed. "
                     f"Pass --force to regenerate, or --extend to grow it.")
        roster = assign(bulks)

    roster = roster.sort_values("sample_id")
    roster.to_csv(OUTPUT_CSV, index=False)

    counts = roster["subset"].value_counts()
    print(f"wrote {OUTPUT_CSV}")
    print(f"val={counts.get('val', 0)} test={counts.get('test', 0)} "
          f"of {len(roster)} bulks")
    tally = roster.groupby(["qap_name", "subset"]).size().unstack(fill_value=0)
    print(tally.to_string())


if __name__ == "__main__":
    main()
