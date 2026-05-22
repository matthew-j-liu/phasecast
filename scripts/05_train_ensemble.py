"""Train a K-member deep ensemble at a fixed hyperparameter setting.

After HPO (scripts/04_hpo.py) selects best params, this script trains K
independent models with different seeds on the same train/val split, saving
each as `best_seed{N}.pt`. Aggregation (mean of Dirichlet alphas, mean of
GMM mixture-weight-renormalized posteriors) lives in eval code, not here.

Run:
    python scripts/05_train_ensemble.py --config configs/base_v1.yaml \\
        --hp-source data/optuna/v1_hpo_summary.json --K 5

If --hp-source is omitted, uses the configs/base_v1.yaml hyperparameters
verbatim (useful for ensembling at the v1 default config before HPO).
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
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

from cnmineral.models.dataset import TrainFCv1Dataset
from cnmineral.models.losses import joint_loss
from cnmineral.models.model import BaseModelV1
from cnmineral.models.training import (
    EarlyStopping,
    evaluate_v1,
    make_loaders_v1,
    make_loaders_v1_bulk_split,
    train_one_epoch_v1,
)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() \
            and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def merge_hp_into_cfg(cfg: dict, hp: dict) -> dict:
    """Overlay Optuna best_params onto the base_v1 config in-place.

    Hyperparameters that live in `model`: d_model, N, heads, dropout, K_gmm.
    In `training`: lr, weight_decay, batch_size.
    In `loss`: lambda_cond, eps_smooth.
    """
    for key in ("d_model", "N", "heads", "dropout", "K_gmm"):
        if key in hp:
            cfg["model"][key] = hp[key]
    for key in ("lr", "weight_decay", "batch_size"):
        if key in hp:
            cfg["training"][key] = hp[key]
    for key in ("lambda_cond", "eps_smooth"):
        if key in hp:
            cfg["loss"][key] = hp[key]
    return cfg


def train_one_member(
    seed: int,
    member_dir: Path,
    cfg: dict,
    dataset,
    device: torch.device,
    loaders: tuple,
) -> dict:
    """Train a single ensemble member; return its eval summary on val + test.

    `loaders` is a 3-tuple (train_loader, val_loader, test_loader) shared
    across ensemble members so they evaluate against the same held-out split.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, val_loader, test_loader = loaders

    model = BaseModelV1(n_minerals=dataset.n_minerals, **cfg["model"]).to(device)
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
    grad_clip = cfg["training"].get("grad_clip", 1.0)

    member_dir.mkdir(parents=True, exist_ok=True)
    metrics_f = open(member_dir / "metrics.jsonl", "w")
    metrics_f.write(json.dumps({
        "_meta": {"seed": seed, "config": cfg,
                  "cond_mean": dataset.cond_mean.tolist(),
                  "cond_std": dataset.cond_std.tolist()}
    }, default=str) + "\n")

    t0 = time()
    for epoch in range(cfg["training"]["epochs"]):
        tr = train_one_epoch_v1(model, train_loader, optimizer, loss_fn, device,
                                grad_clip=grad_clip, progress=False)
        va = evaluate_v1(model, val_loader, loss_fn, device,
                         dataset.cond_mean, dataset.cond_std, progress=False)
        scheduler.step()
        metrics_f.write(json.dumps({"epoch": epoch,
                                     "train": tr, "val": dict(va)}) + "\n")
        metrics_f.flush()
        if stopper(epoch, va["total"], model):
            break
    metrics_f.close()

    stopper.restore_best(model)
    ckpt = member_dir / "best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg, "phase_names": dataset.phase_names,
        "n_minerals": dataset.n_minerals,
        "K_gmm": cfg["model"]["K_gmm"], "seed": seed,
        "cond_mean": dataset.cond_mean.tolist(),
        "cond_std": dataset.cond_std.tolist(),
        "cond_axes": list(dataset.cond_axes),
        "liq_threshold": dataset.liq_threshold,
        "best_epoch": stopper.best_epoch, "best_val_loss": stopper.best_loss,
    }, ckpt)
    elapsed = time() - t0

    final_val = evaluate_v1(model, val_loader, loss_fn, device,
                            dataset.cond_mean, dataset.cond_std, progress=False)
    final_test = evaluate_v1(model, test_loader, loss_fn, device,
                             dataset.cond_mean, dataset.cond_std, progress=False)
    return {
        "seed": seed, "checkpoint": str(ckpt), "elapsed_s": elapsed,
        "best_epoch": stopper.best_epoch, "best_val_loss": stopper.best_loss,
        "val": {
            "total": final_val["total"], "nll_dir": final_val["nll_dir"],
            "nll_gmm": final_val["nll_gmm"],
            "cond_mae_raw": final_val["cond_mae_raw"],
            "r2_per_phase": final_val["r2_per_phase"],
            "phase_presence": final_val["phase_presence"],
        },
        "test": {
            "total": final_test["total"], "nll_dir": final_test["nll_dir"],
            "nll_gmm": final_test["nll_gmm"],
            "cond_mae_raw": final_test["cond_mae_raw"],
            "r2_per_phase": final_test["r2_per_phase"],
            "phase_presence": final_test["phase_presence"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hp-source", type=Path, default=None,
                        help="Optuna summary JSON to load best_params from")
    parser.add_argument("--K", type=int, default=5, help="ensemble size")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="override checkpoint_dir from config")
    parser.add_argument("--seed-base", type=int, default=1000,
                        help="member seed = seed_base + member_idx")
    parser.add_argument("--splits-path", type=Path,
                        default=REPO_ROOT / "data/splits_v1.json",
                        help="bulk-level train/val/test split manifest; "
                             "loaded if it exists, computed and written otherwise")
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--min-bulks-per-class", type=int, default=25)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.hp_source is not None:
        with open(args.hp_source) as f:
            summary = json.load(f)
        hp = summary["best_params"]
        cfg = merge_hp_into_cfg(cfg, hp)
        print(f"loaded best params from {args.hp_source}: {hp}")
    cfg["data"]["parquet_path"] = REPO_ROOT / cfg["data"]["parquet_path"]
    out_root = (Path(args.out_dir) if args.out_dir is not None
                else REPO_ROOT / cfg["output"]["checkpoint_dir"]).resolve()
    if str(out_root).startswith(str(REPO_ROOT.resolve())):
        pass
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"out_root: {out_root}")

    device = pick_device()
    print(f"device: {device}")

    dataset = TrainFCv1Dataset(
        cfg["data"]["parquet_path"],
        liq_threshold=cfg["data"]["liq_threshold"],
    )
    print(f"corpus: {len(dataset)} path-rows, {dataset.n_minerals} phases")

    train_loader, val_loader, test_loader, split_info = make_loaders_v1_bulk_split(
        dataset,
        val_frac=cfg["data"].get("val_frac", 0.16),
        test_frac=args.test_frac,
        min_bulks_per_class=args.min_bulks_per_class,
        batch_size=cfg["training"]["batch_size"],
        seed=cfg["training"]["seed"],
        num_workers=cfg["data"]["num_workers"],
        splits_path=str(args.splits_path),
    )
    print(f"split: train={split_info['n_train_rows']} rows / "
          f"{split_info['n_train_bulks']} bulks  "
          f"val={split_info['n_val_rows']}/{split_info['n_val_bulks']}  "
          f"test={split_info['n_test_rows']}/{split_info['n_test_bulks']}")

    loaders = (train_loader, val_loader, test_loader)
    members = []
    for k in range(args.K):
        seed = args.seed_base + k
        member_dir = out_root / f"seed_{seed}"
        print(f"\n=== member {k+1}/{args.K}  seed={seed} ===")
        info = train_one_member(seed, member_dir, cfg, dataset, device, loaders)
        print(f"  best epoch {info['best_epoch']:3d}  val={info['best_val_loss']:.4f}  "
              f"test={info['test']['total']:.4f}  "
              f"T_mae={info['test']['cond_mae_raw'][0]:.1f}  "
              f"P_mae={info['test']['cond_mae_raw'][1]:.2f}  "
              f"H2O_mae={info['test']['cond_mae_raw'][2]:.3f}  "
              f"({info['elapsed_s']:.0f}s)")
        members.append(info)

    summary = {
        "K": args.K, "device": str(device),
        "config": cfg, "members": members,
        "split_info": {k: split_info[k] for k in split_info
                       if k not in ("train_sample_ids", "val_sample_ids",
                                    "test_sample_ids", "trainval_pool_sample_ids",
                                    "dropped_class_sample_ids")},
        "splits_path": str(args.splits_path),
        "cond_mean": dataset.cond_mean.tolist(),
        "cond_std": dataset.cond_std.tolist(),
        "cond_axes": list(dataset.cond_axes),
        "phase_names": dataset.phase_names,
    }
    summary_path = out_root / "ensemble_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nensemble summary -> {summary_path}")


if __name__ == "__main__":
    main()
