"""Score a trained log-space imputation head over its encoder's test split.

  python scripts/eval/eval_imputation_head.py --head-checkpoint HEAD.pt

The encoder and Gard test split are reconstructed from the head checkpoint's
encoder reference; artifacts land at
`results/imputation/<encoder_tag>/readout/`.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
from pathlib import Path

from cnmineral.eval.imputation_logspace import run_imputation_logspace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    run_imputation_logspace(head_checkpoint_path=args.head_checkpoint.resolve())


if __name__ == "__main__":
    main()
