"""
Pretrain loss selection + the corpus statistics each variant needs.

`run_pretraining` calls `compute_loss_stats` once over its train split to
derive whatever the configured objective needs (per-element log-frac
center/scale for `log_mse`, per-token inverse-variance weights for
`weighted_mse`; `mse` needs nothing), then `make_loss_fn` to turn
`config['loss']` + those stats into a single `(predicted, target_frac,
mask_positions) -> loss` callable.  Keeping selection here leaves the
training loop agnostic to which objective is running, and keeps the loss
math in `cnmineral.losses.pretrain_loss` pure.  The stats are plain CPU
tensors so the same dict rebuilds the identical loss for the held-out test
eval (`make_loss_fn` moves them to the model's device).
"""

from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from cnmineral.data.tokens import N_TOKENS
from cnmineral.losses.pretrain_loss import (
    masked_element_log_mse,
    masked_element_mse,
    masked_element_weighted_mse,
)

_LOG_FLOOR = 1e-12
# Guards 1/var against a literally-constant token (var == 0 -> inf).  It does
# NOT tame a merely near-constant token, whose weight would still dominate the
# mean-1 normalization: inverse-variance weighting assumes every token has
# genuine spread, which holds for the bulk-chemistry corpus (all 22 tokens are
# well-populated with orders-of-magnitude range).
_VAR_FLOOR = 1e-30

LossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _loss_config(config: dict) -> dict:
    """Normalize `config['loss']` to a dict with a `type` key.

    Accepts an absent key (-> `mse`), a bare string (`loss: log_mse`), or a
    dict (`loss: {type: log_mse, standardize: false}`).
    """
    spec = config.get("loss", "mse")
    if isinstance(spec, str):
        return {"type": spec}
    if isinstance(spec, dict):
        if "type" not in spec:
            raise ValueError(f"loss config dict missing 'type': {spec!r}")
        return spec
    raise TypeError(f"loss config must be str or dict, got {type(spec).__name__}")


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


def inverse_variance_weights(
    dataset: Dataset, indices: list[int], *, batch_size: int, num_workers: int,
) -> torch.Tensor:
    """Per-token `1/var(frac)` over populated rows, normalized to mean 1.

    Linear-space analog of per-element standardization: weighting each
    element's squared error by the inverse of its frac variance equalizes the
    per-element contribution to the loss, so the low-variance traces are
    amplified to parity with the ~1e9x-higher-variance majors.  Elements absent
    everywhere keep weight 1.
    """
    frac, populated = _materialize_frac(
        dataset, indices, batch_size=batch_size, num_workers=num_workers,
    )
    var = torch.ones(N_TOKENS)
    for j in range(N_TOKENS):
        col = frac[populated[:, j], j]
        if col.numel() > 1:
            var[j] = col.var(unbiased=False)
    weights = 1.0 / var.clamp_min(_VAR_FLOOR)
    return weights * (N_TOKENS / weights.sum())


def compute_loss_stats(
    config: dict,
    dataset: Dataset,
    train_indices: list[int],
    *,
    batch_size: int,
    num_workers: int,
) -> dict | None:
    """Corpus stats the configured objective needs, or None for plain `mse`.

    Computed once over the train split; returned as CPU tensors so the same
    dict can rebuild the loss for the test eval.
    """
    loss_cfg = _loss_config(config)
    ltype = loss_cfg["type"]
    if ltype == "mse":
        return None
    if ltype == "log_mse":
        if loss_cfg.get("standardize", True):
            mean, std = log_frac_stats(
                dataset, train_indices,
                batch_size=batch_size, num_workers=num_workers,
            )
        else:
            mean, std = torch.zeros(N_TOKENS), torch.ones(N_TOKENS)
        return {"target_mean": mean, "target_std": std}
    if ltype == "weighted_mse":
        weights = inverse_variance_weights(
            dataset, train_indices,
            batch_size=batch_size, num_workers=num_workers,
        )
        return {"token_weights": weights}
    raise ValueError(
        f"unknown loss type {ltype!r}; expected mse | log_mse | weighted_mse"
    )


def make_loss_fn(
    config: dict, stats: dict | None, *, device: torch.device | str | None = None,
) -> LossFn:
    """Build the `(predicted, target_frac, mask_positions) -> loss` callable.

    `stats` is the output of `compute_loss_stats` for the same `config`.  When
    `device` is set the stat tensors are moved once into the closure so the
    per-batch call needs no device juggling.
    """
    loss_cfg = _loss_config(config)
    ltype = loss_cfg["type"]

    if ltype == "mse":
        def loss_fn(predicted, target_frac, mask_positions):
            return masked_element_mse(predicted, target_frac, mask_positions)
        return loss_fn

    if ltype == "log_mse":
        if stats is None:
            raise ValueError("log_mse needs stats; got None")
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

    if ltype == "weighted_mse":
        if stats is None:
            raise ValueError("weighted_mse needs stats; got None")
        token_weights = stats["token_weights"]
        if device is not None:
            token_weights = token_weights.to(device)

        def loss_fn(predicted, target_frac, mask_positions):
            return masked_element_weighted_mse(
                predicted, target_frac, mask_positions,
                token_weights=token_weights,
            )
        return loss_fn

    raise ValueError(
        f"unknown loss type {ltype!r}; expected mse | log_mse | weighted_mse"
    )
