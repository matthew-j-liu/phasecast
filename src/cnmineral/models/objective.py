"""Optuna objective for v1 HPO.

Single-stage TPE search over (architecture, optimizer, loss, GMM-component
count, label-smoothing eps) dims listed in ADR 006. Each trial:

1. samples hyperparameters
2. prunes if d_model % heads != 0 (Optuna's `TrialPruned`)
3. runs `k_folds`-fold bulk-level stratified CV over the trainval pool;
   each fold trains a fresh BaseModelV1 for `cv_epochs`
4. returns the mean across folds of (0.5 * NLL_dir + 0.5 * NLL_gmm) at the
   best val epoch within the fold

The 20% held-out test set is NEVER seen by HPO; the caller (`04_hpo.py`)
constructs the split via `make_loaders_v1_bulk_split` first and passes only
the trainval_pool sample_ids in.
"""

import optuna
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from cnmineral.models.losses import joint_loss
from cnmineral.models.model import BaseModelV1
from cnmineral.models.training import (
    evaluate_v1,
    make_cv_folds_v1,
    train_one_epoch_v1,
)


def suggest_hyperparams(trial: optuna.trial.Trial) -> dict:
    """Sample the v1 HPO search dims.

    Expanded from the original ADR 006 space after the first HPO consistently
    pegged d_model, N, and heads at their upper bounds (search-space-too-narrow
    signal). Dropped the values that never won.
    """
    d_model = trial.suggest_categorical("d_model", [256, 384, 512])
    N = trial.suggest_int("N", 2, 4)
    heads = trial.suggest_categorical("heads", [8, 16])
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    lambda_cond = trial.suggest_float("lambda_cond", 0.25, 4.0, log=True)
    K_gmm = trial.suggest_int("K_gmm", 2, 4)
    eps_smooth = trial.suggest_float("eps_smooth", 1e-4, 1e-2, log=True)
    return {
        "d_model": d_model, "N": N, "heads": heads, "dropout": dropout,
        "lr": lr, "weight_decay": weight_decay, "batch_size": batch_size,
        "lambda_cond": lambda_cond, "K_gmm": K_gmm, "eps_smooth": eps_smooth,
    }


def _build_model(params: dict, n_minerals: int) -> BaseModelV1:
    return BaseModelV1(
        n_minerals=n_minerals,
        K_gmm=params["K_gmm"],
        d_model=params["d_model"],
        N=params["N"],
        heads=params["heads"],
        dropout=params["dropout"],
    )


def _train_one_fold(
    params: dict,
    dataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    device: torch.device,
    cv_epochs: int,
    num_workers: int,
) -> float:
    """Train one fold; return best (lowest) val total loss."""
    model = _build_model(params, dataset.n_minerals).to(device)
    optimizer = AdamW(model.parameters(), lr=params["lr"],
                      weight_decay=params["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cv_epochs, eta_min=1e-6)
    loss_fn = lambda out, ymod, ycond: joint_loss(
        out, ymod, ycond,
        lambda_cond=params["lambda_cond"],
        eps_smooth=params["eps_smooth"],
    )
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=params["batch_size"], shuffle=True,
        num_workers=num_workers, drop_last=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=params["batch_size"], shuffle=False,
        num_workers=num_workers, drop_last=False,
    )

    best_val = float("inf")
    for _ in range(cv_epochs):
        train_one_epoch_v1(model, train_loader, optimizer, loss_fn, device,
                           grad_clip=1.0, progress=False)
        scheduler.step()
        val = evaluate_v1(model, val_loader, loss_fn, device,
                          dataset.cond_mean, dataset.cond_std, progress=False)
        if val["total"] < best_val:
            best_val = float(val["total"])
    return best_val


def make_objective_v1(
    dataset,
    trainval_pool_sample_ids,
    device: torch.device,
    cv_epochs: int = 30,
    k_folds: int = 3,
    num_workers: int = 0,
    seed: int = 42,
):
    """Build a bulk-level CV Optuna objective over the trainval pool.

    Parameters
    ----------
    dataset : TrainFCv1Dataset
        Full filtered corpus (still includes dropped-class rows, but the CV
        folds are restricted to `trainval_pool_sample_ids` so those rows
        never appear in train or val).
    trainval_pool_sample_ids : array-like[int]
        Bulks that constitute the 80% pool (`info["trainval_pool_sample_ids"]`
        from `make_loaders_v1_bulk_split`). The 20% test bulks are excluded
        by virtue of not appearing here.
    cv_epochs, k_folds, num_workers, seed : standard.
    """
    fold_indices = make_cv_folds_v1(dataset, trainval_pool_sample_ids,
                                     k_folds=k_folds, seed=seed)

    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest_hyperparams(trial)
        if params["d_model"] % params["heads"] != 0:
            raise optuna.exceptions.TrialPruned()

        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
            torch.manual_seed(seed + 1000 * trial.number + fold_idx)
            best_val = _train_one_fold(
                params, dataset, train_idx, val_idx,
                device, cv_epochs, num_workers,
            )
            fold_scores.append(best_val)
            trial.report(best_val, step=fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return float(np.mean(fold_scores))

    return objective
