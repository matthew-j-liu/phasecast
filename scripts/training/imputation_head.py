"""Log-space imputation head on a frozen SSL encoder.

  Single run (train, then score the test split once):
      python scripts/training/imputation_head.py --config CFG.yaml

  Optuna HPO sweep (head-only, encoder frozen):
      python scripts/training/imputation_head.py --hpo --config CFG.yaml

The encoder, dataset, and splits are reconstructed from the encoder checkpoint
named in the config; only the head trains.  A single run saves the head to
`<head_checkpoint_dir>/<head_checkpoint_filename>` and writes its test scorecard
under `results/imputation/<encoder_tag>/readout/`.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
from pathlib import Path

from cnmineral.training import pick_device
from cnmineral.training.imputation_head import run_single
from cnmineral.training.imputation_head_hpo import run_hpo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--hpo", action="store_true",
        help="run an Optuna head-HPO sweep instead of a single train+score pass",
    )
    args = parser.parse_args()

    if args.hpo:
        run_hpo(args.config.resolve())
        return

    result = run_single(args.config.resolve(), pick_device())
    print(f"train_loss={result['train_loss']:.4f} val_loss={result['val_loss']:.4f}")
    summary = result.get("trace_summary")
    if summary:
        print(f"test trace log-R2: median {summary['median_head_log_r2']:.3f} "
              f"over {summary['n_traces']} traces")


if __name__ == "__main__":
    main()
