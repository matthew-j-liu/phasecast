"""Train BaseModel (v0) or BaseModelV1 (v1) on the corresponding corpus.

Dispatch is by the `model_version` key in the config:
  v0 -> BaseModel + MSE/sum-to-1 loss, reads data/train_magemin.h5
  v1 -> BaseModelV1 + joint Dirichlet+GMM NLL, reads data/train_fc_v1.parquet

Run:
    python scripts/03_train_base.py --config configs/base_v0.yaml
    python scripts/03_train_base.py --config configs/base_v1.yaml
    python scripts/03_train_base.py --config configs/base_v1.yaml --limit-bulks 1000 --epochs 10
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
import subprocess
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

from cnmineral.models.dataset import TrainMageminDataset, TrainFCv1Dataset
from cnmineral.models.losses import mode_loss, joint_loss
from cnmineral.models.model import BaseModel, BaseModelV1
from cnmineral.models.training import (
    EarlyStopping,
    evaluate,
    evaluate_v1,
    make_loaders,
    make_loaders_v1,
    make_loaders_v1_bulk_split,
    train_one_epoch,
    train_one_epoch_v1,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def pick_device() -> torch.device:
    """CUDA -> MPS (Apple Silicon) -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() \
            and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def format_top_k_r2(phase_names, r2, presence, k: int = 5) -> str:
    order = sorted(range(len(phase_names)), key=lambda i: -presence[i])[:k]
    return "  ".join(f"{phase_names[i]}={r2[i]:+.3f}" for i in order)


def run_v0(cfg, args, device):
    cfg["data"]["h5_path"] = REPO_ROOT / cfg["data"]["h5_path"]
    cfg["output"]["checkpoint_dir"] = REPO_ROOT / cfg["output"]["checkpoint_dir"]
    cfg["output"]["metrics_log"] = REPO_ROOT / cfg["output"]["metrics_log"]

    print(f"\nloading dataset: {cfg['data']['h5_path']}")
    dataset = TrainMageminDataset(cfg["data"]["h5_path"])
    print(f"  {len(dataset)} samples, {dataset.n_minerals} phases")

    train_loader, val_loader = make_loaders(
        dataset,
        val_frac=cfg["data"]["val_frac"],
        batch_size=cfg["training"]["batch_size"],
        seed=cfg["training"]["seed"],
        stratify_by=cfg["data"]["stratify_by"],
        num_workers=cfg["data"]["num_workers"],
    )

    model = BaseModel(n_minerals=dataset.n_minerals, **cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nmodel: BaseModel(n_minerals={dataset.n_minerals}) -- {n_params:,} params")

    optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"],
                      weight_decay=cfg["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"],
                                  eta_min=cfg["training"]["cosine_eta_min"])
    loss_fn = lambda p, t: mode_loss(p, t, lambda_sum=cfg["loss"]["lambda_sum"])
    stopper = EarlyStopping(patience=cfg["training"]["patience"],
                            min_delta=cfg["training"]["min_delta"])

    out_dir = cfg["output"]["checkpoint_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg["output"]["metrics_log"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_f = open(metrics_path, "w")
    meta = {"_meta": {
        "git_commit": get_git_commit(), "device": str(device),
        "seed": cfg["training"]["seed"], "n_minerals": dataset.n_minerals,
        "phase_names": dataset.phase_names, "n_samples": len(dataset),
        "n_params": n_params, "config": cfg,
    }}
    metrics_f.write(json.dumps(meta, default=str) + "\n"); metrics_f.flush()

    t0 = time()
    progress = bool(cfg["output"].get("progress_bar", True))
    for epoch in range(cfg["training"]["epochs"]):
        t_epoch = time()
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device,
                             progress=progress, desc=f"epoch {epoch:3d} train")
        va = evaluate(model, val_loader, loss_fn, device, dataset.phase_names,
                      progress=progress, desc=f"epoch {epoch:3d} val")
        scheduler.step()
        top_r2 = format_top_k_r2(dataset.phase_names, va["r2_per_phase"],
                                 va["phase_presence"], k=5)
        print(f"epoch {epoch:3d}  train={tr['total']:.5f}  val={va['total']:.5f}  "
              f"mse={va['mse']:.5f}  |sum-1|={va['abs_sum_violation']:.4f}\n"
              f"           top-5 R2: {top_r2}")
        metrics_f.write(json.dumps({
            "epoch": epoch, "epoch_wall_s": time() - t_epoch,
            "train": tr, "val": dict(va),
            "lr": scheduler.get_last_lr()[0],
        }) + "\n"); metrics_f.flush()
        if stopper(epoch, va["total"], model):
            print(f"\nearly stop at epoch {epoch} (best {stopper.best_epoch})")
            break

    metrics_f.close()
    stopper.restore_best(model)
    ckpt = out_dir / "best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg, "phase_names": dataset.phase_names,
        "n_minerals": dataset.n_minerals,
        "best_epoch": stopper.best_epoch, "best_val_loss": stopper.best_loss,
    }, ckpt)
    print(f"\nfinished in {time() - t0:.1f}s. best -> {ckpt}")


def run_v1(cfg, args, device):
    cfg["data"]["parquet_path"] = REPO_ROOT / cfg["data"]["parquet_path"]
    cfg["output"]["checkpoint_dir"] = REPO_ROOT / cfg["output"]["checkpoint_dir"]
    cfg["output"]["metrics_log"] = REPO_ROOT / cfg["output"]["metrics_log"]
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    print(f"\nloading v1 corpus: {cfg['data']['parquet_path']}")
    sample_id_filter = None
    if args.limit_bulks is not None:
        import pandas as pd
        all_ids = pd.read_parquet(cfg["data"]["parquet_path"], columns=["sample_id"])
        unique_ids = all_ids["sample_id"].unique()
        rng = np.random.default_rng(cfg["training"]["seed"])
        sample_id_filter = rng.choice(unique_ids, size=min(args.limit_bulks, len(unique_ids)),
                                       replace=False)
        print(f"  limiting to {len(sample_id_filter)} bulks (~{len(sample_id_filter)*8} rows)")

    dataset = TrainFCv1Dataset(
        cfg["data"]["parquet_path"],
        liq_threshold=cfg["data"]["liq_threshold"],
        sample_id_filter=sample_id_filter,
    )
    print(f"  {len(dataset)} path-rows, {dataset.n_minerals} phases")
    print(f"  cond_mean (T, P, H2O): {dataset.cond_mean}")
    print(f"  cond_std  (T, P, H2O): {dataset.cond_std}")

    splits_path = REPO_ROOT / cfg["data"].get("splits_path", "data/splits_v1.json")
    train_loader, val_loader, test_loader, split_info = make_loaders_v1_bulk_split(
        dataset,
        val_frac=cfg["data"].get("val_frac", 0.16),
        test_frac=cfg["data"].get("test_frac", 0.20),
        min_bulks_per_class=cfg["data"].get("min_bulks_per_class", 25),
        batch_size=cfg["training"]["batch_size"],
        seed=cfg["training"]["seed"],
        num_workers=cfg["data"]["num_workers"],
        splits_path=str(splits_path),
    )
    print(f"  split: train={split_info['n_train_rows']}/{split_info['n_train_bulks']}b  "
          f"val={split_info['n_val_rows']}/{split_info['n_val_bulks']}b  "
          f"test={split_info['n_test_rows']}/{split_info['n_test_bulks']}b  "
          f"(dropped {split_info['n_dropped_bulks']}b in {len(split_info['dropped_classes'])} rare classes)")

    model = BaseModelV1(n_minerals=dataset.n_minerals, **cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nmodel: BaseModelV1(n_minerals={dataset.n_minerals}, K_gmm={cfg['model']['K_gmm']})"
          f" -- {n_params:,} params")

    optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"],
                      weight_decay=cfg["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"],
                                  eta_min=cfg["training"]["cosine_eta_min"])

    lam = cfg["loss"]["lambda_cond"]
    eps = cfg["loss"]["eps_smooth"]
    loss_fn = lambda out, ymod, ycond: joint_loss(out, ymod, ycond,
                                                  lambda_cond=lam, eps_smooth=eps)

    stopper = EarlyStopping(patience=cfg["training"]["patience"],
                            min_delta=cfg["training"]["min_delta"])
    out_dir = cfg["output"]["checkpoint_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg["output"]["metrics_log"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_f = open(metrics_path, "w")
    meta = {"_meta": {
        "git_commit": get_git_commit(), "device": str(device),
        "seed": cfg["training"]["seed"], "n_minerals": dataset.n_minerals,
        "phase_names": dataset.phase_names, "n_samples": len(dataset),
        "n_params": n_params,
        "cond_mean": dataset.cond_mean.tolist(),
        "cond_std": dataset.cond_std.tolist(),
        "cond_axes": list(dataset.cond_axes),
        "liq_threshold": dataset.liq_threshold,
        "limit_bulks": args.limit_bulks,
        "config": cfg,
    }}
    metrics_f.write(json.dumps(meta, default=str) + "\n"); metrics_f.flush()

    t0 = time()
    progress = bool(cfg["output"].get("progress_bar", True))
    grad_clip = cfg["training"].get("grad_clip", 1.0)
    for epoch in range(cfg["training"]["epochs"]):
        t_epoch = time()
        tr = train_one_epoch_v1(model, train_loader, optimizer, loss_fn, device,
                                grad_clip=grad_clip, progress=progress,
                                desc=f"epoch {epoch:3d} train")
        va = evaluate_v1(model, val_loader, loss_fn, device,
                         dataset.cond_mean, dataset.cond_std,
                         progress=progress, desc=f"epoch {epoch:3d} val")
        scheduler.step()
        top_r2 = format_top_k_r2(dataset.phase_names, va["r2_per_phase"],
                                 va["phase_presence"], k=5)
        mae = va["cond_mae_raw"]
        print(f"epoch {epoch:3d}  train={tr['total']:.4f}  val={va['total']:.4f}  "
              f"dir={va['nll_dir']:+.3f}  gmm={va['nll_gmm']:+.3f}\n"
              f"           top-5 R2: {top_r2}\n"
              f"           cond MAE: T={mae[0]:.1f}C  P={mae[1]:.2f}kbar  H2O={mae[2]:.2f}wt%")
        metrics_f.write(json.dumps({
            "epoch": epoch, "epoch_wall_s": time() - t_epoch,
            "train": tr, "val": dict(va),
            "lr": scheduler.get_last_lr()[0],
        }) + "\n"); metrics_f.flush()
        if stopper(epoch, va["total"], model):
            print(f"\nearly stop at epoch {epoch} (best {stopper.best_epoch})")
            break

    metrics_f.close()
    stopper.restore_best(model)
    ckpt = out_dir / "best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg, "phase_names": dataset.phase_names,
        "n_minerals": dataset.n_minerals,
        "K_gmm": cfg["model"]["K_gmm"],
        "cond_mean": dataset.cond_mean.tolist(),
        "cond_std": dataset.cond_std.tolist(),
        "cond_axes": list(dataset.cond_axes),
        "liq_threshold": dataset.liq_threshold,
        "best_epoch": stopper.best_epoch, "best_val_loss": stopper.best_loss,
    }, ckpt)
    print(f"\nfinished in {time() - t0:.1f}s. best -> {ckpt}")

    final_val = evaluate_v1(model, val_loader, loss_fn, device,
                            dataset.cond_mean, dataset.cond_std, progress=False)
    final_test = evaluate_v1(model, test_loader, loss_fn, device,
                             dataset.cond_mean, dataset.cond_std, progress=False)

    def _rows(metrics):
        return sorted(
            [{"phase": ph, "presence": float(pr), "r2": float(r)}
             for ph, pr, r in zip(dataset.phase_names, metrics["phase_presence"],
                                  metrics["r2_per_phase"])],
            key=lambda r: -r["presence"],
        )

    eval_summary = {
        "checkpoint": str(ckpt),
        "parquet_path": str(cfg["data"]["parquet_path"]),
        "n_minerals": dataset.n_minerals,
        "best_epoch": stopper.best_epoch,
        "val_loss_at_best": stopper.best_loss,
        "split_info": {k: split_info[k] for k in split_info
                       if k not in ("train_sample_ids", "val_sample_ids",
                                    "test_sample_ids", "trainval_pool_sample_ids",
                                    "dropped_class_sample_ids")},
        "val": {
            "total": final_val["total"], "nll_dir": final_val["nll_dir"],
            "nll_gmm": final_val["nll_gmm"],
            "cond_mae_raw": final_val["cond_mae_raw"],
            "per_phase": _rows(final_val),
        },
        "test": {
            "total": final_test["total"], "nll_dir": final_test["nll_dir"],
            "nll_gmm": final_test["nll_gmm"],
            "cond_mae_raw": final_test["cond_mae_raw"],
            "per_phase": _rows(final_test),
        },
        "cond_axes": list(dataset.cond_axes),
    }
    eval_path = out_dir / "eval.json"
    with open(eval_path, "w") as f:
        json.dump(eval_summary, f, indent=2, default=str)
    print(f"\neval summary -> {eval_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit-bulks", type=int, default=None,
                        help="v1 only: subsample to N unique bulks "
                             "(keeps all 8 paths per bulk)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override config epochs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["training"]["seed"])
    device = pick_device()
    print(f"device: {device}")
    print(f"config: {args.config}")

    version = cfg.get("model_version", "v0")
    if version == "v0":
        run_v0(cfg, args, device)
    elif version == "v1":
        run_v1(cfg, args, device)
    else:
        raise ValueError(f"Unknown model_version: {version}")


if __name__ == "__main__":
    main()
