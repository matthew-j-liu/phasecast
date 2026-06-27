"""Print a human-readable summary of one or more checkpoint files.

    python scripts/view/inspect_checkpoint.py checkpoints/pretrain_v3_seed0/epoch_best.pt
    python scripts/view/inspect_checkpoint.py checkpoints/pretrain_v3_seed0/*.pt

Multiple paths print one summary block each, separated by blank
lines -- so eyeballing two checkpoints side-by-side in a terminal
shows architecture / loss / val-curve differences at a glance.

All business logic lives in `cnmineral.training.Checkpoint.summary`;
this file is the argparse shell.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path
from cnmineral.training import Checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="+", type=Path,
        help="one or more .pt files to summarize",
    )
    args = parser.parse_args()

    for i, path in enumerate(args.paths):
        if i > 0:
            print()
        if not path.exists():
            print(f"=== {path} ===\n  (file not found)")
            continue
        ckpt = Checkpoint.load(path)
        print(ckpt.summary(name=str(path)))


if __name__ == "__main__":
    main()
