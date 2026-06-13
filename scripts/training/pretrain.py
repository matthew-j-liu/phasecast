"""SSL pretrain CLI.

  Single run:
      python scripts/training/pretrain.py --config CFG.yaml

  Optuna HPO sweep:
      python scripts/training/pretrain.py --hpo --config CFG.yaml
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path
from cnmineral.training import pick_device, run_hpo, run_single


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True,
        help="YAML config path; structure depends on --hpo",
    )
    parser.add_argument(
        "--hpo", action="store_true",
        help="treat --config as an HPO sweep config (configs/pretrain/pretrain_hpo.yaml)",
    )

    args = parser.parse_args()

    device = pick_device()
    print(f"device: {device}")

    if args.hpo:
        run_hpo(args.config, device)
    else:
        result = run_single(args.config, device)
        test_str = (
            f"{result['test_loss']:.4f}"
            if result["test_loss"] is not None else "n/a"
        )
        print(
            f"\nfinal: train={result['train_loss']:.4f} "
            f"val={result['val_loss']:.4f} test={test_str}"
        )


if __name__ == "__main__":
    main()
