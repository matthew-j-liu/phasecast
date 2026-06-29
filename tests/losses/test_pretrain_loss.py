"""Tests for the `cnmineral.losses.pretrain_loss` SSL objective.

Contracts for `masked_element_log_mse`:
- A batch with zero masked positions returns loss 0, not NaN.
- It is finite under -inf unmasked targets (frac=0 floored before log).
- It is zero on an exact standardized prediction.
"""

import pytest
import torch

from cnmineral.data.tokens import N_TOKENS
from cnmineral.losses.pretrain_loss import masked_element_log_mse


def _log_stats() -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform per-element center/scale for the log-space loss tests."""
    return torch.full((N_TOKENS,), -1.0), torch.full((N_TOKENS,), 0.5)


def test_log_mse_zero_when_no_mask() -> None:
    """An all-False mask yields loss 0 (not NaN)."""
    mean, std = _log_stats()
    predicted = torch.randn(4, N_TOKENS)
    target = torch.rand(4, N_TOKENS)
    mask = torch.zeros(4, N_TOKENS, dtype=torch.bool)
    loss = masked_element_log_mse(
        predicted, target, mask, target_mean=mean, target_std=std,
    )
    assert loss.item() == 0.0


def test_log_mse_finite_with_zero_unmasked_targets() -> None:
    """Unmasked positions hold frac=0 (log10 -> -inf); the loss stays finite
    because the target is floored before the log and those positions are
    masked out anyway."""
    mean, std = _log_stats()
    target = torch.zeros(2, N_TOKENS)
    target[:, 0] = 0.01  # only this position is populated + masked
    predicted = torch.randn(2, N_TOKENS)
    mask = torch.zeros(2, N_TOKENS, dtype=torch.bool)
    mask[:, 0] = True
    loss = masked_element_log_mse(
        predicted, target, mask, target_mean=mean, target_std=std,
    )
    assert torch.isfinite(loss).item()


def test_log_mse_zero_on_exact_standardized_prediction() -> None:
    """Predicting the standardized-log target exactly gives loss 0 -- confirms
    the per-element standardization the head emits is what the loss expects."""
    mean, std = _log_stats()
    target = torch.full((2, N_TOKENS), 0.01)  # log10 = -2
    mask = torch.ones(2, N_TOKENS, dtype=torch.bool)
    std_log_target = (torch.log10(target) - mean) / std  # = -2.0 everywhere
    loss = masked_element_log_mse(
        std_log_target, target, mask, target_mean=mean, target_std=std,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-10)
