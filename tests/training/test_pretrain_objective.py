"""Tests for `cnmineral.training.pretrain_objective`: corpus stats + loss dispatch.

Contracts:
- `log_frac_stats` matches a hand-computed mean/std and keeps absent elements
  at the identity (mean 0, std 1).
- `inverse_variance_weights` is positive, mean-1 normalized, and up-weights a
  low-variance (trace-like) element over a high-variance (major-like) one.
- `compute_loss_stats` dispatches: None for mse, log center/scale for log_mse
  (identity when standardize off), token weights for weighted_mse.
- `make_loss_fn` returns a callable whose value matches the underlying loss for
  each objective, and raises on missing stats / unknown type.
"""

import pytest
import torch

from cnmineral.data.tokens import N_TOKENS
from cnmineral.losses.pretrain_loss import (
    masked_element_log_mse,
    masked_element_mse,
    masked_element_weighted_mse,
)
from cnmineral.training.pretrain_objective import (
    compute_loss_stats,
    inverse_variance_weights,
    log_frac_stats,
    make_loss_fn,
)


class _FracDataset(torch.utils.data.Dataset):
    """Minimal {src, frac} dataset for stats tests (no parquet load)."""

    def __init__(self, src: torch.Tensor, frac: torch.Tensor) -> None:
        self.src = src
        self.frac = frac

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, i: int) -> dict:
        return {"src": self.src[i], "frac": self.frac[i]}


def _single_element_dataset(values: list[float], token_pos: int = 0) -> _FracDataset:
    """One populated element at `token_pos`; every other token absent."""
    n = len(values)
    src = torch.zeros(n, N_TOKENS, dtype=torch.long)
    frac = torch.zeros(n, N_TOKENS, dtype=torch.float)
    src[:, token_pos] = 14  # any nonzero atomic number
    frac[:, token_pos] = torch.tensor(values, dtype=torch.float)
    return _FracDataset(src, frac)


def test_log_frac_stats_hand_computed() -> None:
    """log10([0.1, 0.001]) = [-1, -3] -> mean -2, std (population) 1.0 at pos 0;
    every absent element keeps the identity."""
    ds = _single_element_dataset([0.1, 0.001], token_pos=0)
    mean, std = log_frac_stats(ds, [0, 1], batch_size=8, num_workers=0)
    assert mean.shape == (N_TOKENS,) and std.shape == (N_TOKENS,)
    assert mean[0].item() == pytest.approx(-2.0)
    assert std[0].item() == pytest.approx(1.0)
    # absent elements untouched
    assert mean[1].item() == pytest.approx(0.0)
    assert std[1].item() == pytest.approx(1.0)


def test_inverse_variance_weights_upweights_low_variance() -> None:
    """A low-variance element gets a larger weight than a high-variance one;
    weights are positive and normalized to mean 1."""
    n = 2
    src = torch.zeros(n, N_TOKENS, dtype=torch.long)
    frac = torch.zeros(n, N_TOKENS, dtype=torch.float)
    src[:, 0] = 14  # high-variance "major-like"
    src[:, 1] = 38  # low-variance "trace-like"
    frac[:, 0] = torch.tensor([0.1, 0.3])      # var 0.01
    frac[:, 1] = torch.tensor([1.0e-6, 2.0e-6])  # var ~2.5e-13
    weights = inverse_variance_weights(
        _FracDataset(src, frac), [0, 1], batch_size=8, num_workers=0,
    )
    assert weights.shape == (N_TOKENS,)
    assert (weights > 0).all()
    assert weights.mean().item() == pytest.approx(1.0)
    assert weights[1].item() > weights[0].item()


def _batch(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    predicted = torch.rand(3, N_TOKENS, generator=g)
    target = torch.rand(3, N_TOKENS, generator=g)
    mask = torch.rand(3, N_TOKENS, generator=g) < 0.4
    return predicted, target, mask


def test_compute_loss_stats_dispatch() -> None:
    ds = _single_element_dataset([0.1, 0.001], token_pos=0)
    idx = [0, 1]
    assert compute_loss_stats({"loss": "mse"}, ds, idx, batch_size=8, num_workers=0) is None
    # absent loss key defaults to mse
    assert compute_loss_stats({}, ds, idx, batch_size=8, num_workers=0) is None

    log_stats = compute_loss_stats(
        {"loss": {"type": "log_mse"}}, ds, idx, batch_size=8, num_workers=0,
    )
    assert set(log_stats) == {"target_mean", "target_std"}
    assert log_stats["target_mean"][0].item() == pytest.approx(-2.0)

    plain_log = compute_loss_stats(
        {"loss": {"type": "log_mse", "standardize": False}}, ds, idx,
        batch_size=8, num_workers=0,
    )
    assert torch.equal(plain_log["target_mean"], torch.zeros(N_TOKENS))
    assert torch.equal(plain_log["target_std"], torch.ones(N_TOKENS))

    w_stats = compute_loss_stats(
        {"loss": {"type": "weighted_mse"}}, ds, idx, batch_size=8, num_workers=0,
    )
    assert set(w_stats) == {"token_weights"}


def test_compute_loss_stats_unknown_type_raises() -> None:
    ds = _single_element_dataset([0.1, 0.001])
    with pytest.raises(ValueError, match="unknown loss type"):
        compute_loss_stats({"loss": "bogus"}, ds, [0, 1], batch_size=8, num_workers=0)


def test_make_loss_fn_matches_underlying_losses() -> None:
    predicted, target, mask = _batch()
    ds = _single_element_dataset([0.1, 0.001], token_pos=0)
    idx = [0, 1]

    mse_fn = make_loss_fn({"loss": "mse"}, None)
    assert mse_fn(predicted, target, mask).item() == pytest.approx(
        masked_element_mse(predicted, target, mask).item()
    )

    cfg_log = {"loss": {"type": "log_mse"}}
    log_stats = compute_loss_stats(cfg_log, ds, idx, batch_size=8, num_workers=0)
    log_fn = make_loss_fn(cfg_log, log_stats)
    assert log_fn(predicted, target, mask).item() == pytest.approx(
        masked_element_log_mse(
            predicted, target, mask,
            target_mean=log_stats["target_mean"], target_std=log_stats["target_std"],
        ).item()
    )

    cfg_w = {"loss": {"type": "weighted_mse"}}
    w_stats = compute_loss_stats(cfg_w, ds, idx, batch_size=8, num_workers=0)
    w_fn = make_loss_fn(cfg_w, w_stats)
    assert w_fn(predicted, target, mask).item() == pytest.approx(
        masked_element_weighted_mse(
            predicted, target, mask, token_weights=w_stats["token_weights"],
        ).item()
    )


def test_make_loss_fn_requires_stats_for_rebalanced() -> None:
    with pytest.raises(ValueError, match="needs stats"):
        make_loss_fn({"loss": {"type": "log_mse"}}, None)
    with pytest.raises(ValueError, match="needs stats"):
        make_loss_fn({"loss": {"type": "weighted_mse"}}, None)
