"""Training loop primitives + stratified loader split.

Not a script — `scripts/03_train_base.py` imports these. Designed to be
small and reusable: when v1 adds more heads, only the loss function and
metric dict change; the loop body and EarlyStopping stay identical.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
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
