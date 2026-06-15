"""Impute held-out elements from a frozen encoder + pretrain head.

  python scripts/eval/impute_elements.py --checkpoint CKPT.pt

The dataset and Gard test split are reconstructed from `ckpt.config`;
artifacts land at `results/imputation/<checkpoint_tag>/`.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
from pathlib import Path

from cnmineral.eval.imputation import run_imputation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="encoder+head checkpoint produced by scripts/training/pretrain.py",
    )
    args = parser.parse_args()
    run_imputation(checkpoint_path=args.checkpoint.resolve())


if __name__ == "__main__":
    main()
