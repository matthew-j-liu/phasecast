"""Tests for `cnmineral.training.pretrain_objective`: corpus stats + loss builder.

Contracts:
- `log_frac_stats` matches a hand-computed mean/std and keeps absent elements
  at the identity (mean 0, std 1).
- `compute_loss_stats` returns the log center/scale over the train split.
- `make_loss_fn` returns a callable whose value matches the underlying log loss,
  and raises on missing stats.
"""

import pytest
import torch

from cnmineral.data.tokens import N_TOKENS
from cnmineral.losses.pretrain_loss import masked_element_log_mse
from cnmineral.training.pretrain_objective import (
    compute_loss_stats,
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


def _batch(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    predicted = torch.rand(3, N_TOKENS, generator=g)
    target = torch.rand(3, N_TOKENS, generator=g)
    mask = torch.rand(3, N_TOKENS, generator=g) < 0.4
    return predicted, target, mask


def test_compute_loss_stats_returns_log_center_scale() -> None:
    ds = _single_element_dataset([0.1, 0.001], token_pos=0)
    log_stats = compute_loss_stats(ds, [0, 1], batch_size=8, num_workers=0)
    assert set(log_stats) == {"target_mean", "target_std"}
    assert log_stats["target_mean"][0].item() == pytest.approx(-2.0)


def test_make_loss_fn_matches_underlying_loss() -> None:
    predicted, target, mask = _batch()
    ds = _single_element_dataset([0.1, 0.001], token_pos=0)

    log_stats = compute_loss_stats(ds, [0, 1], batch_size=8, num_workers=0)
    log_fn = make_loss_fn(log_stats)
    assert log_fn(predicted, target, mask).item() == pytest.approx(
        masked_element_log_mse(
            predicted, target, mask,
            target_mean=log_stats["target_mean"], target_std=log_stats["target_std"],
        ).item()
    )


def test_make_loss_fn_requires_stats() -> None:
    with pytest.raises(ValueError, match="needs stats"):
        make_loss_fn(None)


def test_log_frac_stats_width_follows_frac() -> None:
    """Stats size from the batch width, not a global constant: a redox
    (23-wide) corpus gets 23-element stats."""
    n = 4
    src = torch.zeros(n, N_TOKENS + 1, dtype=torch.long)
    frac = torch.zeros(n, N_TOKENS + 1, dtype=torch.float)
    src[:, N_TOKENS] = 120
    frac[:, N_TOKENS] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    ds = _FracDataset(src, frac)
    mean, std = log_frac_stats(ds, list(range(n)), batch_size=8, num_workers=0)
    assert mean.shape == (N_TOKENS + 1,) and std.shape == (N_TOKENS + 1,)
    assert mean[N_TOKENS].item() != 0.0


def test_log_frac_stats_std_floor() -> None:
    """A near-constant populated column keeps a floored std so the
    standardized target cannot blow up."""
    ds = _single_element_dataset([0.25, 0.25, 0.25], token_pos=0)
    _mean, std = log_frac_stats(ds, [0, 1, 2], batch_size=8, num_workers=0)
    assert std[0].item() == pytest.approx(1e-6, rel=1e-3)
