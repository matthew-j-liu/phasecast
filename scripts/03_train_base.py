"""Train BaseModel (ModeHead-only) on the MAGEMin-derived training H5.

Single CLI argument: --config PATH. Everything else lives in the YAML.

Run:
    python scripts/03_train_base.py --config configs/base_v0.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from time import time

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cnmineral.models.dataset import TrainMageminDataset
from cnmineral.models.losses import mode_loss
from cnmineral.models.model import BaseModel
from cnmineral.models.training import (
    EarlyStopping,
    evaluate,
    make_loaders,
    train_one_epoch,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_paths(cfg: dict) -> dict:
    """Convert string paths in the config to repo-relative Path objects."""
    cfg["data"]["h5_path"] = REPO_ROOT / cfg["data"]["h5_path"]
    cfg["output"]["checkpoint_dir"] = REPO_ROOT / cfg["output"]["checkpoint_dir"]
    cfg["output"]["metrics_log"] = REPO_ROOT / cfg["output"]["metrics_log"]
    return cfg


def format_top_k_r2(phase_names: list[str], r2: list[float], presence: list[float], k: int = 5) -> str:
    """Sort phases by val-set presence and return 'name=r2' for the top k."""
    order = sorted(range(len(phase_names)), key=lambda i: -presence[i])[:k]
    return "  ".join(
        f"{phase_names[i]}={r2[i]:+.3f}" for i in order
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to training YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = resolve_paths(cfg)

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"config: {args.config}")

    # Data
    print(f"\nloading dataset: {cfg['data']['h5_path']}")
    dataset = TrainMageminDataset(cfg["data"]["h5_path"])
    print(f"  {len(dataset)} samples, {dataset.n_minerals} phases")
    print(f"  phases: {dataset.phase_names}")

    train_loader, val_loader = make_loaders(
        dataset,
        val_frac=cfg["data"]["val_frac"],
        batch_size=cfg["training"]["batch_size"],
        seed=cfg["training"]["seed"],
        stratify_by=cfg["data"]["stratify_by"],
        num_workers=cfg["data"]["num_workers"],
    )
    print(f"  train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    # Model
    model = BaseModel(
        n_minerals=dataset.n_minerals,
        **cfg["model"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nmodel: BaseModel(n_minerals={dataset.n_minerals}) — {n_params:,} params")

    # Optim + sched
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=cfg["training"]["cosine_eta_min"],
    )

    # Loss
    loss_fn = lambda p, t: mode_loss(p, t, lambda_sum=cfg["loss"]["lambda_sum"])

    # Early stopping + best-checkpoint restore
    stopper = EarlyStopping(
        patience=cfg["training"]["patience"],
        min_delta=cfg["training"]["min_delta"],
    )

    # Output dir + metrics log
    out_dir = cfg["output"]["checkpoint_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg["output"]["metrics_log"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_f = open(metrics_path, "w")

    # Training loop
    print("\nstarting training")
    t0 = time()
    epoch_lines = []
    for epoch in range(cfg["training"]["epochs"]):
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device, dataset.phase_names)
        scheduler.step()

        top_r2 = format_top_k_r2(
            dataset.phase_names,
            val_metrics["r2_per_phase"],
            val_metrics["phase_presence"],
            k=5,
        )
        line = (
            f"epoch {epoch:3d}  "
            f"train={train_metrics['total']:.5f}  "
            f"val={val_metrics['total']:.5f}  "
            f"mse={val_metrics['mse']:.5f}  "
            f"|Σ-1|={val_metrics['abs_sum_violation']:.4f}  "
            f"sparsity={val_metrics['sparsity']:.3f}\n"
            f"           top-5 R²: {top_r2}"
        )
        print(line)
        epoch_lines.append(line)

        # JSONL log
        log_row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": {
                k: v for k, v in val_metrics.items()
                if k not in ("r2_per_phase", "phase_presence")
            },
            "lr": scheduler.get_last_lr()[0],
        }
        metrics_f.write(json.dumps(log_row) + "\n")
        metrics_f.flush()

        stop = stopper(epoch, val_metrics["total"], model)
        if stop:
            print(f"\nearly stop at epoch {epoch} (best epoch {stopper.best_epoch}, "
                  f"best val loss {stopper.best_loss:.5f})")
            break

    metrics_f.close()
    elapsed = time() - t0

    # Restore best model + save final checkpoint
    stopper.restore_best(model)
    checkpoint_path = out_dir / "best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "phase_names": dataset.phase_names,
        "n_minerals": dataset.n_minerals,
        "best_epoch": stopper.best_epoch,
        "best_val_loss": stopper.best_loss,
    }, checkpoint_path)
    print(f"\nfinished in {elapsed:.1f}s. best checkpoint → {checkpoint_path}")

    # Final eval at best epoch
    final = evaluate(model, val_loader, loss_fn, device, dataset.phase_names)
    print(f"\nfinal val (best epoch {stopper.best_epoch}):")
    print(f"  total={final['total']:.5f}  mse={final['mse']:.5f}  "
          f"|Σ-1|={final['abs_sum_violation']:.4f}  sparsity={final['sparsity']:.3f}")
    print("  per-phase R² (sorted by val-set presence):")
    order = sorted(range(len(dataset.phase_names)), key=lambda i: -final["phase_presence"][i])
    for i in order:
        if final["phase_presence"][i] < 0.005:
            continue
        print(f"    {dataset.phase_names[i]:8s} "
              f"presence={final['phase_presence'][i]:.3f}  "
              f"R²={final['r2_per_phase'][i]:+.3f}")


if __name__ == "__main__":
    main()
