"""Hurdle ModeHead on a frozen SSL encoder.

  python scripts/training/mode_head.py --config configs/heads/mode_head.yaml

Labels build from the corpus rungs tables named in the config; only the head
trains. Saves the head to `<head_checkpoint_dir>/<head_checkpoint_filename>`
and prints the val joint hurdle NLL and per-phase calibration table.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
from pathlib import Path

from cnmineral.training import pick_device
from cnmineral.training.mode_head import run_single


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    result = run_single(args.config.resolve(), pick_device())
    print(f"train_loss={result['train_loss']:.4f} val_loss={result['val_loss']:.4f}")


if __name__ == "__main__":
    main()
