"""Interrogate a trained ModeHead checkpoint on its provisional val split.

  python scripts/eval/mode_head.py --checkpoint checkpoints/mode_head_seed0/head_best.pt

`--checkpoint` is a `ModeHeadCheckpoint` (head weights + encoder pointer +
label semantics). The dataset and split are rebuilt from the checkpoint's
stamped `label_config` plus `--seed` / `--val-fraction`, which must match the
training config (the split is hashed, not stored). Figures + `metrics.json`
land at `results/mode_head/<checkpoint dir name>/`.
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
    parser.add_argument(
        "--seed", type=int, default=0,
        help="provisional-split seed used at training time",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.20,
        help="provisional-split val fraction used at training time",
    )
    args = parser.parse_args()
    run_mode_head_eval(
        args.checkpoint.resolve(), seed=args.seed, val_fraction=args.val_fraction
    )


if __name__ == "__main__":
    main()
