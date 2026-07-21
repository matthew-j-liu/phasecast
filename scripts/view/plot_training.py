"""
Plot loss curves from a checkpoint or Optuna study.

Two modes, picked by which flag you pass:

  Single-run loss curve:
      python scripts/view/plot_training.py --checkpoint checkpoints/.../epoch_best.pt \\
          --out runs/scratch/run_curve.png
      (joint-heads checkpoints are detected by their per-term monitor
      keys and render one panel per loss term)

  HPO study (optimization history + per-trial curves as a second
  `<out>_trials.png` file):
      python scripts/view/plot_training.py \\
          --study-storage data/optuna/journals/v3_pretrain_hpo_long.journal \\
          --study-name v3_pretrain_hpo_long \\
          --out runs/scratch/study.png

Omitting `--out` opens an interactive window.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import warnings
import optuna
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MaxNLocator
from optuna.exceptions import ExperimentalWarning
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.visualization.matplotlib import (
    plot_intermediate_values,
    plot_optimization_history,
)
from cnmineral.eval.training_curves import plot_joint_run_loss, plot_run_loss
from cnmineral.training.joint_heads import selection_series


def plot_from_checkpoint(checkpoint_path: Path, out: Path | None) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    monitor = payload["monitor"]
    if any(k.startswith("train_") for k in monitor):
        # joint-heads monitors carry per-head terms; the ModeHead trainer's
        # (train_bce / train_nll) also land here but have no selection curves
        selection = (
            selection_series(monitor) if "val_comp_nll" in monitor else None
        )
        fig = plot_joint_run_loss(
            monitor, selection=selection, title=checkpoint_path.name,
        )
    else:
        fig, ax = plt.subplots()
        plot_run_loss(monitor, ax=ax, title=f"{checkpoint_path.name}")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"saved {out}")
    else:
        plt.show()


def plot_from_study(
    storage_path: Path, study_name: str, out: Path | None,
) -> None:
    storage = JournalStorage(JournalFileBackend(str(storage_path)))
    study = optuna.load_study(study_name=study_name, storage=storage)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        history_ax = plot_optimization_history(study)
        trials_ax = plot_intermediate_values(study)
    trials_ax.set_xlabel("hpo_run_epoch")
    trials_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        trials_out = out.with_stem(out.stem + "_trials")
        history_ax.figure.savefig(out, dpi=300, bbox_inches="tight")
        trials_ax.figure.savefig(trials_out, dpi=300, bbox_inches="tight")
        print(f"saved {out}")
        print(f"saved {trials_out}")
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
