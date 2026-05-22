"""Training loop primitives + stratified loader split.

Not a script — `scripts/03_train_base.py` and `scripts/04_hpo.py` import
these. The v0 loop predates v1's dict-output model, so the v1 entry points
(`train_one_epoch_v1`, `evaluate_v1`) are kept separate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: torch.device,
    progress: bool = True,
    desc: str = "train",
) -> dict[str, float]:
    """Run one training epoch. Returns mean loss + components over all batches."""
    model.train()
    accum: dict[str, list[float]] = defaultdict(list)

    iterator = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True) if progress else loader
    for batch in iterator:
        src = batch["src"].to(device)
        frac = batch["frac"].to(device)
        target = batch["y_wt"].to(device)

        pred = model(src, frac)
        total, components = loss_fn(pred, target)

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        loss_val = float(total.item())
        accum["total"].append(loss_val)
        for k, v in components.items():
            accum[k].append(v)
        if progress:
            iterator.set_postfix(loss=f"{loss_val:.5f}")

    return {k: float(np.mean(v)) for k, v in accum.items()}


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    device: torch.device,
    phase_names: list[str],
    sparsity_threshold: float = 0.01,
    progress: bool = True,
    desc: str = "val",
) -> dict[str, float | list[float]]:
    """Run one eval pass. Returns total val loss + components + per-phase R²,
    mean prediction sparsity, mean sum-to-1 violation magnitude.

    Per-phase R² is computed only over the held-out val set. Returns the full
    vector; logging code can surface the top-k most-frequent phases.
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    accum: dict[str, list[float]] = defaultdict(list)

    iterator = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True) if progress else loader
    with torch.no_grad():
        for batch in iterator:
            src = batch["src"].to(device)
            frac = batch["frac"].to(device)
            target = batch["y_wt"].to(device)

            pred = model(src, frac)
            total, components = loss_fn(pred, target)

            accum["total"].append(float(total.item()))
            for k, v in components.items():
                accum[k].append(v)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    P = np.concatenate(all_preds, axis=0)        # [N, n_minerals]
    T = np.concatenate(all_targets, axis=0)

    # Per-phase R² = 1 - SSE / SST. SST is variance of target × N.
    ss_res = ((P - T) ** 2).sum(axis=0)
    target_means = T.mean(axis=0, keepdims=True)
    ss_tot = ((T - target_means) ** 2).sum(axis=0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    # Frequency of each phase in val targets (fraction of samples where wt > 0.01)
    presence = (T > sparsity_threshold).mean(axis=0)

    # Mean prediction sparsity: fraction of pred entries below threshold
    sparsity = float((P < sparsity_threshold).mean())

    # Sum-to-1 violation magnitude (not the squared loss term)
    abs_sum_violation = float(np.mean(np.abs(P.sum(axis=-1) - 1.0)))

    out: dict[str, float | list[float]] = {
        k: float(np.mean(v)) for k, v in accum.items()
    }
    out["r2_per_phase"] = r2.tolist()
    out["phase_presence"] = presence.tolist()
    out["sparsity"] = sparsity
    out["abs_sum_violation"] = abs_sum_violation
    return out


@dataclass
class EarlyStopping:
    """Patience-based early stopping. Mirror upstream alignn/scripts/train.py.

    Tracks the lowest val loss seen so far; signals stop when `patience`
    consecutive epochs have failed to improve by at least `min_delta`.
    """

    patience: int = 15
    min_delta: float = 5e-6
    best_loss: float = float("inf")
    counter: int = 0
    best_epoch: int = 0
    best_state: dict | None = field(default=None, repr=False)

    def __call__(
        self,
        epoch: int,
        val_loss: float,
        model: torch.nn.Module,
    ) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = float(val_loss)
            self.best_epoch = epoch
            self.counter = 0
            # Save best state for restore-at-end (don't reach for disk every epoch)
            self.best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            return False

        self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: torch.nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def make_loaders(
    dataset,
    val_frac: float = 0.15,
    batch_size: int = 256,
    seed: int = 42,
    stratify_by: str = "qap_name",
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Stratified train/val split using the dataset's `qap_names` array.

    StratifiedShuffleSplit ensures each rock class has proportional
    representation in val. Falls back to random split if a class has fewer
    than 2 samples (rare, but possible at smoke scale for very rare classes).
    """
    if stratify_by != "qap_name":
        raise NotImplementedError(f"stratify_by={stratify_by!r} not supported; only 'qap_name'")

    labels = np.asarray(dataset.qap_names)
    indices = np.arange(len(dataset))

    # StratifiedShuffleSplit needs every class to have >= 2 members.
    # If any singleton classes, demote them to a shared "_rare" bucket for split purposes.
    unique, counts = np.unique(labels, return_counts=True)
    singletons = set(unique[counts < 2])
    if singletons:
        labels_for_split = np.array([
            "_rare" if c in singletons else c for c in labels
        ], dtype=object)
    else:
        labels_for_split = labels

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_idx, val_idx = next(splitter.split(indices, labels_for_split))

    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader


def _coerce_labels_for_stratification(labels: np.ndarray) -> np.ndarray:
    """Bucket singleton classes into '_rare' so StratifiedShuffleSplit accepts them."""
    unique, counts = np.unique(labels, return_counts=True)
    singletons = set(unique[counts < 2])
    if not singletons:
        return labels
    return np.array(["_rare" if c in singletons else c for c in labels], dtype=object)


def make_loaders_v1(
    dataset,
    val_frac: float = 0.15,
    batch_size: int = 256,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """DEPRECATED. Row-level stratified split leaks bulk identity across
    train/val (all 8 path-rows of a bulk share identical encoder input). Use
    `make_loaders_v1_bulk_split` for any v1 work. Kept only so legacy code
    paths still import cleanly.
    """
    labels = np.asarray(dataset.qap_names)
    indices = np.arange(len(dataset))
    labels_for_split = _coerce_labels_for_stratification(labels)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_idx, val_idx = next(splitter.split(indices, labels_for_split))

    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )
    return train_loader, val_loader


def make_loaders_v1_bulk_split(
    dataset,
    val_frac: float = 0.16,
    test_frac: float = 0.20,
    min_bulks_per_class: int = 25,
    batch_size: int = 256,
    seed: int = 42,
    num_workers: int = 0,
    splits_path=None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Bulk-level 3-way stratified split on sample_id.

    Three-stage construction, all stratified by qap_name at the bulk level:

    1. Drop bulks from classes with fewer than `min_bulks_per_class` total
       bulks. These are removed from the corpus entirely so per-class test
       metrics stay interpretable.
    2. Stage A: outer 80/20 stratified train_test_split on the surviving
       bulks -> trainval_pool (80%) + test (20%).
    3. Stage B: inner 80/20 stratified split of trainval_pool -> ensemble
       train (64% of surviving) + ensemble val (16%).

    If `splits_path` is given and exists, the manifest is loaded; the saved
    seed / val_frac / test_frac / min_bulks_per_class must match the caller's
    args or a ValueError is raised (regenerate by deleting the file).

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    info : dict with keys
        train_sample_ids, val_sample_ids, test_sample_ids
        trainval_pool_sample_ids, dropped_class_sample_ids, dropped_classes
        n_*_rows, n_*_bulks, n_dropped_rows, n_dropped_bulks
        seed, val_frac, test_frac, min_bulks_per_class
    """
    import json
    sample_ids = dataset.sample_ids
    qap_names = dataset.qap_names

    if splits_path is not None and Path(splits_path).exists():
        with open(splits_path) as f:
            saved = json.load(f)
        for key, val in (("seed", seed), ("val_frac", val_frac),
                          ("test_frac", test_frac),
                          ("min_bulks_per_class", min_bulks_per_class)):
            if saved.get(key) != val:
                raise ValueError(
                    f"splits manifest {splits_path} has {key}={saved.get(key)!r} "
                    f"but caller passed {key}={val!r}. Delete the file to regenerate."
                )
        info = saved
    else:
        unique_bulks, first_idx = np.unique(sample_ids, return_index=True)
        bulk_qap = qap_names[first_idx]

        unique_classes, class_counts = np.unique(bulk_qap, return_counts=True)
        rare_classes = set(unique_classes[class_counts < min_bulks_per_class].tolist())
        is_rare = (np.isin(bulk_qap, list(rare_classes))
                   if rare_classes else np.zeros(len(unique_bulks), dtype=bool))
        dropped_bulks = unique_bulks[is_rare]
        surviving_bulks = unique_bulks[~is_rare]
        surviving_qap = bulk_qap[~is_rare]

        trainval_bulks, test_bulks, trainval_qap, _test_qap = train_test_split(
            surviving_bulks, surviving_qap,
            test_size=test_frac, stratify=surviving_qap,
            random_state=seed,
        )
        rel_val = val_frac / (1.0 - test_frac)
        ens_train_bulks, ens_val_bulks = train_test_split(
            trainval_bulks,
            test_size=rel_val, stratify=trainval_qap,
            random_state=seed,
        )

        info = {
            "train_sample_ids": sorted(int(x) for x in ens_train_bulks),
            "val_sample_ids": sorted(int(x) for x in ens_val_bulks),
            "test_sample_ids": sorted(int(x) for x in test_bulks),
            "trainval_pool_sample_ids": sorted(int(x) for x in trainval_bulks),
            "dropped_class_sample_ids": sorted(int(x) for x in dropped_bulks),
            "dropped_classes": sorted(rare_classes),
            "seed": seed,
            "val_frac": val_frac,
            "test_frac": test_frac,
            "min_bulks_per_class": min_bulks_per_class,
        }
        if splits_path is not None:
            Path(splits_path).parent.mkdir(parents=True, exist_ok=True)
            with open(splits_path, "w") as f:
                json.dump(info, f, indent=2)

    train_set_arr = np.asarray(info["train_sample_ids"], dtype=np.int64)
    val_set_arr = np.asarray(info["val_sample_ids"], dtype=np.int64)
    test_set_arr = np.asarray(info["test_sample_ids"], dtype=np.int64)
    dropped_arr = np.asarray(info.get("dropped_class_sample_ids", []), dtype=np.int64)

    train_rows = np.where(np.isin(sample_ids, train_set_arr))[0]
    val_rows = np.where(np.isin(sample_ids, val_set_arr))[0]
    test_rows = np.where(np.isin(sample_ids, test_set_arr))[0]
    dropped_rows = (np.where(np.isin(sample_ids, dropped_arr))[0]
                    if len(dropped_arr) else np.array([], dtype=np.int64))

    info.update({
        "n_train_rows": int(len(train_rows)),
        "n_val_rows": int(len(val_rows)),
        "n_test_rows": int(len(test_rows)),
        "n_train_bulks": int(len(train_set_arr)),
        "n_val_bulks": int(len(val_set_arr)),
        "n_test_bulks": int(len(test_set_arr)),
        "n_trainval_pool_bulks": int(len(train_set_arr) + len(val_set_arr)),
        "n_dropped_bulks": int(len(dropped_arr)),
        "n_dropped_rows": int(len(dropped_rows)),
    })

    train_loader = DataLoader(Subset(dataset, train_rows.tolist()),
                              batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(Subset(dataset, val_rows.tolist()),
                            batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(Subset(dataset, test_rows.tolist()),
                             batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False)
    return train_loader, val_loader, test_loader, info


def make_cv_folds_v1(
    dataset,
    trainval_pool_sample_ids,
    k_folds: int = 3,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Bulk-level stratified CV folds over the trainval pool.

    Uses sklearn.model_selection.StratifiedGroupKFold so each fold's val
    contains whole bulks (no row-level leak) AND class proportions are
    roughly even across folds. Caller is responsible for ensuring the pool
    only contains bulks from surviving classes (i.e. classes with at least
    `min_bulks_per_class` from `make_loaders_v1_bulk_split`).

    Returns
    -------
    fold_indices : list of (train_row_idx, val_row_idx) np.ndarray pairs,
        with row indices into `dataset` (not positions within the pool).
    """
    pool = np.asarray(trainval_pool_sample_ids, dtype=np.int64)
    sample_ids = dataset.sample_ids
    qap_names = dataset.qap_names

    pool_row_mask = np.isin(sample_ids, pool)
    pool_row_idx = np.where(pool_row_mask)[0]
    pool_groups = sample_ids[pool_row_mask]
    pool_y = qap_names[pool_row_mask]

    skf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    folds = []
    for tr_local, va_local in skf.split(np.zeros(len(pool_y)), pool_y, groups=pool_groups):
        folds.append((pool_row_idx[tr_local], pool_row_idx[va_local]))
    return folds


def train_one_epoch_v1(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: torch.device,
    grad_clip: float | None = 1.0,
    progress: bool = True,
    desc: str = "train",
) -> dict[str, float]:
    """One v1 training epoch. Loss function receives the model output dict
    and returns (total, components dict)."""
    model.train()
    accum: dict[str, list[float]] = defaultdict(list)

    iterator = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True) if progress else loader
    for batch in iterator:
        src = batch["src"].to(device)
        frac = batch["frac"].to(device)
        y_modal = batch["y_modal"].to(device)
        y_cond_z = batch["y_cond_z"].to(device)

        out = model(src, frac)
        total, components = loss_fn(out, y_modal, y_cond_z)

        optimizer.zero_grad()
        total.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        accum["total"].append(float(total.item()))
        for k, v in components.items():
            accum[k].append(v)
        if progress:
            iterator.set_postfix(loss=f"{total.item():.4f}")

    return {k: float(np.mean(v)) for k, v in accum.items()}


def _gmm_mean_1d(means: np.ndarray, mix_logits: np.ndarray) -> np.ndarray:
    """Per-axis GMM mean from raw component means and mixture logits.

    means, mix_logits : [B, A, K]. Returns [B, A].
    """
    w = np.exp(mix_logits - mix_logits.max(axis=-1, keepdims=True))
    w = w / w.sum(axis=-1, keepdims=True)
    return (w * means).sum(axis=-1)


def evaluate_v1(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    device: torch.device,
    cond_mean: np.ndarray,
    cond_std: np.ndarray,
    sparsity_threshold: float = 0.01,
    progress: bool = True,
    desc: str = "val",
) -> dict[str, float | list[float]]:
    """v1 eval pass.

    Reports total/component losses (mean over batches), per-phase R²
    computed against the Dirichlet mean = alpha / alpha.sum, per-axis
    condition MAE in the ORIGINAL (de-normalized) units, and presence
    counts.
    """
    model.eval()
    accum: dict[str, list[float]] = defaultdict(list)
    all_alpha: list[np.ndarray] = []
    all_target_modal: list[np.ndarray] = []
    all_cond_means: list[np.ndarray] = []
    all_cond_logits: list[np.ndarray] = []
    all_target_cond_z: list[np.ndarray] = []

    iterator = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True) if progress else loader
    with torch.no_grad():
        for batch in iterator:
            src = batch["src"].to(device)
            frac = batch["frac"].to(device)
            y_modal = batch["y_modal"].to(device)
            y_cond_z = batch["y_cond_z"].to(device)

            out = model(src, frac)
            total, components = loss_fn(out, y_modal, y_cond_z)

            accum["total"].append(float(total.item()))
            for k, v in components.items():
                accum[k].append(v)

            all_alpha.append(out["alpha"].cpu().numpy())
            all_target_modal.append(y_modal.cpu().numpy())
            all_cond_means.append(out["cond_means"].cpu().numpy())
            all_cond_logits.append(out["cond_mix_logits"].cpu().numpy())
            all_target_cond_z.append(y_cond_z.cpu().numpy())

    A = np.concatenate(all_alpha, axis=0)
    P_modal = A / A.sum(axis=-1, keepdims=True)
    T_modal = np.concatenate(all_target_modal, axis=0)
    ss_res = ((P_modal - T_modal) ** 2).sum(axis=0)
    ss_tot = ((T_modal - T_modal.mean(axis=0, keepdims=True)) ** 2).sum(axis=0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    presence = (T_modal > sparsity_threshold).mean(axis=0)

    cm = np.concatenate(all_cond_means, axis=0)
    cl = np.concatenate(all_cond_logits, axis=0)
    P_cond_z = _gmm_mean_1d(cm, cl)
    T_cond_z = np.concatenate(all_target_cond_z, axis=0)
    cond_mae_z = np.abs(P_cond_z - T_cond_z).mean(axis=0)
    cond_mae_raw = cond_mae_z * cond_std

    out: dict[str, float | list[float]] = {k: float(np.mean(v)) for k, v in accum.items()}
    out["r2_per_phase"] = r2.tolist()
    out["phase_presence"] = presence.tolist()
    out["cond_mae_z"] = cond_mae_z.tolist()
    out["cond_mae_raw"] = cond_mae_raw.tolist()
    return out
