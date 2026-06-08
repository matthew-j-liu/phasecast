"""Plot loss curves from a checkpoint or Optuna study.

Two modes, picked by which flag you pass:

  Single-run loss curve:
      python scripts/plot_training.py --checkpoint checkpoints/.../epoch_010.pt \\
          --out figures/run_curve.png

  HPO study (two-panel: optimization history + per-trial curves):
      python scripts/plot_training.py \\
          --study-storage data/optuna/v3_pretrain_hpo_smoke.journal \\
          --study-name v3_pretrain_hpo_smoke \\
          --out figures/study.png

Omitting `--out` opens an interactive window.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cnmineral.eval.training_curves import (
    plot_run_loss, plot_study_history, plot_trial_curves,
)


def plot_from_checkpoint(checkpoint_path: Path, out: Path | None) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    monitor = payload["monitor"]
    fig, ax = plt.subplots()
    plot_run_loss(monitor, ax=ax, title=f"{checkpoint_path.name}")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print(f"saved {out}")
    else:
        plt.show()


def plot_from_study(
    storage_path: Path, study_name: str, out: Path | None,
) -> None:
    storage = JournalStorage(JournalFileBackend(str(storage_path)))
    study = optuna.load_study(study_name=study_name, storage=storage)
    fig, (ax_history, ax_trials) = plt.subplots(
        nrows=2, ncols=1, figsize=(8, 8),
    )
    plot_study_history(study, ax=ax_history)
    plot_trial_curves(study, ax=ax_trials)
    fig.suptitle(study_name)
    fig.tight_layout()
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print(f"saved {out}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="path to a per-epoch checkpoint (.pt)",
    )
    parser.add_argument(
        "--study-storage", type=Path, default=None,
        help="path to an Optuna JournalStorage file",
    )
    parser.add_argument(
        "--study-name", type=str, default=None,
        help="study name inside the storage (required with --study-storage)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="if set, save figure here; otherwise show interactively",
    )
    args = parser.parse_args()

    if args.checkpoint is not None:
        plot_from_checkpoint(args.checkpoint, args.out)
    elif args.study_storage is not None:
        if args.study_name is None:
            parser.error("--study-storage requires --study-name")
        plot_from_study(args.study_storage, args.study_name, args.out)
    else:
        parser.error("must pass either --checkpoint or --study-storage")


if __name__ == "__main__":
    main()
