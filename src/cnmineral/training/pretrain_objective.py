"""
The SSL objective builder + the corpus statistics it needs.

`run_pretraining` calls `compute_loss_stats` once over its train split to
derive the per-element log-frac center/scale, then `make_loss_fn` to turn
those stats into a single `(predicted, target_frac, mask_positions) -> loss`
callable.  Keeping the builder here leaves the training loop and the held-out
test eval agnostic to the objective -- they consume an opaque `loss_fn` -- and
keeps the loss math in `cnmineral.losses.pretrain_loss` pure.  The objective is
unconditional (ADR 007): per-element standardized log-space masked MSE.  The
stats are plain CPU tensors so the same dict rebuilds the identical loss for
the test eval (`make_loss_fn` moves them to the model's device).
"""

from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from cnmineral.data.tokens import N_TOKENS
from cnmineral.losses.pretrain_loss import masked_element_log_mse

_LOG_FLOOR = 1e-12

LossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _materialize_frac(
    dataset: Dataset, indices: list[int], *, batch_size: int, num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack `frac` and a populated mask (`src != 0`) over `indices`."""
    loader = DataLoader(
        Subset(dataset, list(indices)), batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
    )
    fracs, srcs = [], []
    for batch in loader:
        fracs.append(batch["frac"])
        srcs.append(batch["src"])
    frac = torch.cat(fracs, dim=0)
    populated = torch.cat(srcs, dim=0) != 0
    return frac, populated


def log_frac_stats(
    dataset: Dataset, indices: list[int], *, batch_size: int, num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-element mean/std of `log10(frac)` over populated rows in `indices`.

    Populated is `src != 0` (tokens convention C).  Elements absent from every
    row keep the identity (mean 0, std 1).  This is the center/scale the
    standardized log-space objective (and the imputation head) consume.
    """
    frac, populated = _materialize_frac(
        dataset, indices, batch_size=batch_size, num_workers=num_workers,
    )
    log_frac = torch.log10(frac.clamp_min(_LOG_FLOOR))
    mean = torch.zeros(N_TOKENS)
    std = torch.ones(N_TOKENS)
    for j in range(N_TOKENS):
        col = log_frac[populated[:, j], j]
        if col.numel() > 0:
            mean[j] = col.mean()
            std[j] = col.std(unbiased=False) if col.numel() > 1 else 1.0
    return mean, std


def compute_loss_stats(
    dataset: Dataset,
    train_indices: list[int],
    *,
    batch_size: int,
    num_workers: int,
) -> dict:
    """Corpus stats the SSL objective needs: per-element log-frac center/scale.

    Computed once over the train split; returned as CPU tensors so the same
    dict can rebuild the loss for the test eval.
    """
    mean, std = log_frac_stats(
        dataset, train_indices,
        batch_size=batch_size, num_workers=num_workers,
    )
    return {"target_mean": mean, "target_std": std}


def make_loss_fn(
    stats: dict, *, device: torch.device | str | None = None,
) -> LossFn:
    """Build the `(predicted, target_frac, mask_positions) -> loss` callable.

    `stats` is the output of `compute_loss_stats`.  When `device` is set the
    stat tensors are moved once into the closure so the per-batch call needs no
    device juggling.
    """
    if stats is None:
        raise ValueError("make_loss_fn needs stats; got None")
    target_mean = stats["target_mean"]
    target_std = stats["target_std"]
    if device is not None:
        target_mean = target_mean.to(device)
        target_std = target_std.to(device)

    def loss_fn(predicted, target_frac, mask_positions):
        return masked_element_log_mse(
            predicted, target_frac, mask_positions,
            target_mean=target_mean, target_std=target_std,
        )
    return loss_fn
