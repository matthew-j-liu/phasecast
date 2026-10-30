"""
Plot loss curves from a checkpoint.

    python scripts/view/plot_training.py --checkpoint checkpoints/.../epoch_best.pt \\
        --out runs/scratch/run_curve.png

Joint-heads checkpoints are detected by their per-term monitor keys and render
one panel per loss term. Omitting `--out` opens an interactive window.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="path to a per-epoch checkpoint (.pt)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="if set, save figure here; otherwise show interactively",
    )
    args = parser.parse_args()

    plot_from_checkpoint(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
