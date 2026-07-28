"""Derive the SSL pretrain test roster from a checkpoint (+ optional merges).

Reads a checkpoint's stamped test rows (test_sample_ids where present,
else test_indices resolved through the checkpoint's own reconstructed
dataset) and writes their sample_ids as a one-column CSV.  A later
pretrain corpus pins the surviving rows into its own test split via the
`test_roster_path` config key (`cnmineral.training.splits.reserve_test`),
so successive encoders stay scoreable on shared held-out rows.

`--merge` unions in additional sample_id CSVs -- e.g. the supervised
valtest roster (`cnmineral/data/valtest_split.csv`), so no bulk any eval
is measured on ever enters SSL training.

    # inherited-only roster (the shared-row eval set)
    python scripts/build/build_pretrain_test_roster.py \
        --checkpoint checkpoints/pretrain_v3_logmse_std_seed0/epoch_best.pt

    # training-time holdout: inherited + supervised valtest bulks
    python scripts/build/build_pretrain_test_roster.py \
        --checkpoint checkpoints/pretrain_v3_logmse_std_seed0/epoch_best.pt \
        --merge src/cnmineral/data/valtest_split.csv \
        --out src/cnmineral/data/pretrain_test_roster.csv
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import pandas as pd

from cnmineral.training.build_from_config import create_dataset
from cnmineral.training.checkpoint import Checkpoint

DEFAULT_OUT = Path("src/cnmineral/data/pretrain_inherited_test_roster.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--merge", type=Path, action="append", default=[],
        help="additional sample_id CSV(s) to union into the roster",
    )
    args = parser.parse_args()

    ckpt = Checkpoint.load(args.checkpoint)
    if ckpt.test_sample_ids is not None:
        sample_ids = {int(s) for s in ckpt.test_sample_ids}
    else:
        if ckpt.test_indices is None:
            raise ValueError(
                f"{args.checkpoint} stamps neither test_sample_ids nor "
                f"test_indices; nothing to derive a roster from"
            )
        dataset = create_dataset(ckpt.config)
        sample_ids = {
            int(dataset.sample_ids[i]) for i in ckpt.test_indices
        }
    n_ckpt = len(sample_ids)

    for merge_path in args.merge:
        merged = pd.read_csv(merge_path)["sample_id"].astype(int)
        before = len(sample_ids)
        sample_ids.update(merged)
        print(
            f"merged {merge_path}: {len(merged)} ids, "
            f"{len(sample_ids) - before} new"
        )

    out_ids = sorted(sample_ids)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": out_ids}).to_csv(args.out, index=False)
    print(
        f"{len(out_ids)} sample_ids ({n_ckpt} from checkpoint) -> {args.out}"
    )


if __name__ == "__main__":
    main()
