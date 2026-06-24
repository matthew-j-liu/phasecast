"""Tests for the two `cnmineral.losses.pretrain_loss` objectives.

Shared contracts, checked per loss:
- Each reduction matches a hand-computed value on a small known case
  with uneven per-sample mask counts (where the reductions genuinely
  differ).
- A batch with zero masked positions returns loss 0, not NaN.
- Shape / dtype / reduction-name validation raises instead of
  silently broadcasting.
Plus per-loss specifics: the log loss is finite under -inf unmasked
targets and zero on an exact standardized prediction.
"""

import pytest
import torch

from cnmineral.data.tokens import N_TOKENS
from cnmineral.losses.pretrain_loss import (
    masked_element_log_mse,
    masked_element_mse,
)


def _known_case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hand-checkable case: squared errors at masked positions are
    0.16 (sample 0, one mask) and 0.0 + 0.09 (sample 1, two masks)."""
    predicted = torch.tensor([[0.5, 0.0, 0.0],
                              [0.2, 0.4, 0.0]])
    target = torch.tensor([[0.1, 0.0, 0.9],
                           [0.2, 0.1, 0.0]])
    mask = torch.tensor([[True, False, False],
                         [True, True, False]])
    return predicted, target, mask


def test_reductions_match_hand_computed_values() -> None:
    """sum = 0.25; mean_per_masked_position = 0.25/3; mean_per_sample
    = (0.16/1 + 0.09/2) / 2.  The latter two differ exactly because
    the mask counts are uneven across samples."""
    predicted, target, mask = _known_case()
    assert masked_element_mse(
        predicted, target, mask, reduction="sum",
    ).item() == pytest.approx(0.25)
    assert masked_element_mse(
        predicted, target, mask, reduction="mean_per_masked_position",
    ).item() == pytest.approx(0.25 / 3)
    assert masked_element_mse(
        predicted, target, mask, reduction="mean_per_sample",
    ).item() == pytest.approx((0.16 + 0.045) / 2)


def test_unmasked_positions_carry_no_gradient() -> None:
    """The position excluded from the mask contributes nothing: its
    prediction is wildly wrong (0.0 vs target 0.9) yet the loss only
    reflects the masked positions."""
    predicted, target, mask = _known_case()
    predicted = predicted.clone().requires_grad_(True)
    loss = masked_element_mse(predicted, target, mask)
    loss.backward()
    assert predicted.grad[0, 2] == 0.0
    assert predicted.grad[0, 0] != 0.0


def test_zero_masked_positions_returns_zero() -> None:
    """An all-False mask yields loss 0 under every reduction (the
    docstring promise; rests on clamp(min=1) in the denominators)."""
    predicted = torch.rand(4, 22)
    target = torch.rand(4, 22)
    mask = torch.zeros(4, 22, dtype=torch.bool)
    for reduction in ("sum", "mean_per_masked_position", "mean_per_sample"):
        loss = masked_element_mse(predicted, target, mask, reduction=reduction)
        assert loss.item() == 0.0, reduction


def test_validation_raises() -> None:
    predicted = torch.rand(2, 22)
    target = torch.rand(2, 22)
    mask = torch.zeros(2, 22, dtype=torch.bool)
    with pytest.raises(ValueError, match="shape mismatch"):
        masked_element_mse(predicted, torch.rand(2, 21), mask)
    with pytest.raises(ValueError, match="mask_positions shape"):
        masked_element_mse(predicted, target, torch.zeros(2, 21, dtype=torch.bool))
    with pytest.raises(TypeError, match="must be bool"):
        masked_element_mse(predicted, target, mask.float())
    with pytest.raises(ValueError, match="unknown reduction"):
        masked_element_mse(predicted, target, mask, reduction="mean")


def _log_stats() -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform per-element center/scale for the log-space loss tests."""
    return torch.full((N_TOKENS,), -1.0), torch.full((N_TOKENS,), 0.5)


def test_log_mse_zero_when_no_mask() -> None:
    """An all-False mask yields loss 0, mirroring the linear-space contract."""
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
