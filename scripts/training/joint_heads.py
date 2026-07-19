"""Joint mode/comp/cond heads on a frozen SSL encoder.

  python scripts/training/joint_heads.py --config configs/heads/joint_heads.yaml
  python scripts/training/joint_heads.py --config configs/heads/joint_heads_hpo.yaml --hpo

Labels build from the corpus rungs + draws tables named in the config; only
the heads train, under the joint loss with the mass-balance closure term.
Plain runs save all three heads (each at its own best epoch) to
`<head_checkpoint_dir>/<head_checkpoint_filename>` and print the val joint
hurdle NLL and per-phase calibration table; `--hpo` runs the resumable
Optuna sweep instead (`cnmineral.training.joint_heads_hpo`).
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
from pathlib import Path

from cnmineral.training import pick_device
from cnmineral.training.joint_heads import run_single
from cnmineral.training.joint_heads_hpo import run_hpo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--hpo", action="store_true",
        help="run the resumable Optuna sweep instead of a single train",
    )
    args = parser.parse_args()

    if args.hpo:
        run_hpo(args.config.resolve())
        return
    result = run_single(args.config.resolve(), pick_device())
    print(" ".join(f"{name}_best={value:.4f}" for name, value in result.items()))


if __name__ == "__main__":
    main()
