"""v1 HPO sweep driver.

Optuna TPE sampler + JournalStorage (filesystem, resumable). Each trial does
k-fold stratified-on-qap_name CV; objective is the mean of fold best-val
joint NLL.

Run:
    # short verification sweep
    python scripts/04_hpo.py --config configs/hpo_v1.yaml --n-trials 30 \\
        --limit-bulks 4000 --cv-epochs 15

    # full sweep
    python scripts/04_hpo.py --config configs/hpo_v1.yaml --n-trials 150

To resume an interrupted study: re-run the same command; load_if_exists picks
up where it left off.
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
import optuna
import pandas as pd
import torch
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cnmineral.models.dataset import TrainFCv1Dataset
from cnmineral.models.objective import make_objective_v1
from cnmineral.models.training import make_loaders_v1_bulk_split


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() \
            and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n-trials", type=int, default=None,
                        help="override config n_trials")
    parser.add_argument("--limit-bulks", type=int, default=None,
                        help="override config data.limit_bulks (smoke runs)")
    parser.add_argument("--cv-epochs", type=int, default=None,
                        help="override config cv.cv_epochs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.n_trials is not None:
        cfg["n_trials"] = args.n_trials
    if args.limit_bulks is not None:
        cfg["data"]["limit_bulks"] = args.limit_bulks
    if args.cv_epochs is not None:
        cfg["cv"]["cv_epochs"] = args.cv_epochs

    device = pick_device()
    print(f"device: {device}")
    print(f"config: {args.config}")
    print(f"n_trials: {cfg['n_trials']}  cv_epochs: {cfg['cv']['cv_epochs']}  "
          f"k_folds: {cfg['cv']['k_folds']}")

    parquet_path = REPO_ROOT / cfg["data"]["parquet_path"]
    sample_id_filter = None
    limit = cfg["data"].get("limit_bulks")
    if limit is not None:
        all_ids = pd.read_parquet(parquet_path, columns=["sample_id"])
        unique_ids = all_ids["sample_id"].unique()
        rng = np.random.default_rng(cfg["sampler_seed"])
        sample_id_filter = rng.choice(unique_ids, size=min(limit, len(unique_ids)),
                                       replace=False)
        print(f"limit_bulks active: {len(sample_id_filter)} bulks (~{len(sample_id_filter)*8} rows)")

    t_load = time()
    dataset = TrainFCv1Dataset(
        parquet_path,
        liq_threshold=cfg["data"]["liq_threshold"],
        sample_id_filter=sample_id_filter,
    )
    print(f"loaded {len(dataset)} rows in {time() - t_load:.1f}s")

    splits_path = REPO_ROOT / cfg["data"].get("splits_path", "data/splits_v1.json")
    _, _, _, split_info = make_loaders_v1_bulk_split(
        dataset,
        val_frac=cfg["data"].get("val_frac", 0.16),
        test_frac=cfg["data"].get("test_frac", 0.20),
        min_bulks_per_class=cfg["data"].get("min_bulks_per_class", 25),
        batch_size=256,
        seed=cfg["cv"]["seed"],
        num_workers=0,
        splits_path=str(splits_path),
    )
    print(f"split: trainval_pool={split_info['n_trainval_pool_bulks']} bulks  "
          f"test={split_info['n_test_bulks']} bulks  "
          f"dropped={split_info['n_dropped_bulks']} bulks  "
          f"(threshold={split_info['min_bulks_per_class']})")

    storage_path = REPO_ROOT / cfg["storage_path"]
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(storage_path)))

    sampler = TPESampler(seed=cfg["sampler_seed"],
                         n_startup_trials=cfg["n_startup_trials"])
    pruner = MedianPruner(n_warmup_steps=cfg["pruner_warmup_folds"])

    study = optuna.create_study(
        study_name=cfg["study_name"],
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True,
    )
    n_done = sum(1 for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE)
    trials_left = max(0, cfg["n_trials"] - n_done)
    print(f"resuming study {cfg['study_name']!r}: {n_done} complete, "
          f"running {trials_left} more to reach n_trials={cfg['n_trials']}.")

    objective = make_objective_v1(
        dataset,
        trainval_pool_sample_ids=split_info["trainval_pool_sample_ids"],
        device=device,
        cv_epochs=cfg["cv"]["cv_epochs"],
        k_folds=cfg["cv"]["k_folds"],
        num_workers=cfg["data"]["num_workers"],
        seed=cfg["cv"]["seed"],
    )

    if trials_left > 0:
        study.optimize(objective, n_trials=trials_left, gc_after_trial=True)

    print("\nbest trial:")
    print(f"  value:  {study.best_value:.4f}")
    print(f"  params: {study.best_params}")

    out_dir = storage_path.parent
    summary = {
        "study_name": cfg["study_name"],
        "storage_path": str(storage_path),
        "n_trials_complete": sum(1 for t in study.trials
                                  if t.state == optuna.trial.TrialState.COMPLETE),
        "n_trials_pruned": sum(1 for t in study.trials
                                if t.state == optuna.trial.TrialState.PRUNED),
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "device": str(device),
        "config": cfg,
    }
    summary_path = out_dir / f"{cfg['study_name']}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nsummary -> {summary_path}")


if __name__ == "__main__":
    main()
