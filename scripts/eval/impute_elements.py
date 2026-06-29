"""Score a trained imputation head in log space over its encoder's test split.

  python scripts/eval/impute_elements.py --checkpoint HEAD_CKPT.pt

`--checkpoint` is an `ImputationHeadCheckpoint` (head weights + the encoder it
was trained against).  The encoder and Gard test split are reconstructed from
the encoder checkpoint's config; artifacts land at
`results/imputation/<encoder_tag>/readout/`.
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
        help="imputation head checkpoint produced by scripts/training/imputation_head.py",
    )
    args = parser.parse_args()
    run_imputation(args.checkpoint.resolve())


if __name__ == "__main__":
    main()
