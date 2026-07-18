"""Interrogate a trained ModeHead checkpoint on the valtest val roster.

  python scripts/eval/mode_head.py --checkpoint checkpoints/mode_head_seed0/head_best.pt

`--checkpoint` is a `ModeHeadCheckpoint` (head weights + encoder pointer +
label semantics). The val dataset is rebuilt from the checkpoint's stamped
`label_config` plus the committed valtest roster (`valtest_split.csv`).
Figures + `metrics.json` land at `results/mode_head/<checkpoint dir name>/`.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
from pathlib import Path

from cnmineral.eval.mode_head import run_mode_head_eval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="ModeHead checkpoint produced by scripts/training/mode_head.py",
    )
    args = parser.parse_args()
    run_mode_head_eval(args.checkpoint.resolve())


if __name__ == "__main__":
    main()
