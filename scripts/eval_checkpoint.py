"""Evaluate a saved BaseModel checkpoint on its training H5 and dump per-phase
metrics to JSON.

Reloads model from `best.pt`, rebuilds the val split using the same seed +
val_frac stored in the checkpoint's config, runs `evaluate` once, and writes
a structured JSON summary. Useful for after-the-fact comparison between
training runs (smoke vs full, baseline vs ablation, etc.).

Run:
    python scripts/eval_checkpoint.py \
        --checkpoint checkpoints/base_v0_smoke/best.pt \
        --h5 data/train_magemin_smoke.h5 \
        --out checkpoints/base_v0_smoke/eval.json
"""

from __future__ import annotations

import os
# macOS-only: PyTorch + MAGEMin both link libomp; set before torch import
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cnmineral.models.dataset import TrainMageminDataset
from cnmineral.models.losses import mode_loss
from cnmineral.models.model import BaseModel
from cnmineral.models.training import evaluate, make_loaders


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--h5", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSON path (defaults to <checkpoint dir>/eval.json)")
    args = p.parse_args()

    print(f"loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    phase_names = ckpt["phase_names"]
    n_minerals = ckpt["n_minerals"]
    print(f"  n_minerals={n_minerals}, best_epoch={ckpt['best_epoch']}, "
          f"best_val_loss={ckpt['best_val_loss']:.5f}")

    print(f"\nloading dataset: {args.h5}")
    dataset = TrainMageminDataset(args.h5)
    assert dataset.n_minerals == n_minerals, (
        f"checkpoint n_minerals={n_minerals} != dataset n_minerals={dataset.n_minerals}"
    )

    _train_loader, val_loader = make_loaders(
        dataset,
        val_frac=cfg["data"]["val_frac"],
        batch_size=cfg["training"]["batch_size"],
        seed=cfg["training"]["seed"],
        stratify_by=cfg["data"]["stratify_by"],
        num_workers=0,
    )

    model = BaseModel(n_minerals=n_minerals, **cfg["model"])
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"\nrunning eval on {len(val_loader)} batches")
    loss_fn = lambda p, t: mode_loss(p, t, lambda_sum=cfg["loss"]["lambda_sum"])
    metrics = evaluate(model, val_loader, loss_fn, device, phase_names)

    # Compose summary
    r2 = metrics["r2_per_phase"]
    presence = metrics["phase_presence"]
    rows = [
        {"phase": ph, "presence": float(pr), "r2": float(r)}
        for ph, pr, r in zip(phase_names, presence, r2)
    ]
    rows.sort(key=lambda r: -r["presence"])

    summary = {
        "checkpoint": str(args.checkpoint),
        "h5_path": str(args.h5),
        "n_minerals": n_minerals,
        "best_epoch": ckpt["best_epoch"],
        "val_loss_at_best": ckpt["best_val_loss"],
        "val_loss_recomputed": metrics["total"],
        "val_mse": metrics["mse"],
        "abs_sum_violation": metrics["abs_sum_violation"],
        "sparsity": metrics["sparsity"],
        "per_phase": rows,
    }

    out_path = args.out or (args.checkpoint.parent / "eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_path}")

    # Pretty-print
    print("\nper-phase R² (sorted by val-set presence):")
    print(f"  {'phase':<8s} {'presence':>10s} {'R²':>8s}")
    for row in rows:
        if row["presence"] < 0.005:
            continue
        marker = "  ✓" if row["r2"] > 0.5 else ""
        print(f"  {row['phase']:<8s} {row['presence']:>10.3f} {row['r2']:>+8.3f}{marker}")


if __name__ == "__main__":
    main()
