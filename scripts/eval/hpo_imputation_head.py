"""Optuna HPO sweep for the log-space imputation head on a frozen encoder.

  python scripts/eval/hpo_imputation_head.py --config CONFIG.yaml

The encoder, dataset, and train/val split are reconstructed from the encoder
checkpoint named in the config; only the head is tuned.  The sweep is
resumable (`load_if_exists`); rerun the same command to add trials.  Emits
`best_params` and a summary JSON; the final head ships via
`train_imputation_head` on the updated config.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
from pathlib import Path

from cnmineral.training.imputation_head_hpo import run_hpo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_hpo(args.config.resolve())


if __name__ == "__main__":
    main()
