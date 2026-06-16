"""Tests for `cnmineral.models.imputation_head.LogImputationHead`.

Build-shape contracts for the linear and MLP read-outs, the
de-standardization round-trip the inference helpers rely on, the std floor,
and buffer serialization (the buffers must travel with the checkpoint).
"""

import pytest
import torch

from cnmineral.data.tokens import N_TOKENS
from cnmineral.models.imputation_head import LogImputationHead, _STD_FLOOR


def test_linear_build_forward_shape() -> None:
    """`hidden=None` builds a single linear layer emitting one value per token."""
    head = LogImputationHead(d_model=8, hidden=None)
    assert isinstance(head.net, torch.nn.Linear)
    out = head(torch.zeros(3, 8))
    assert out.shape == (3, N_TOKENS)


def test_mlp_build_forward_shape() -> None:
    """An int `hidden` builds a GELU MLP, still emitting one value per token."""
    head = LogImputationHead(d_model=8, hidden=4)
    assert isinstance(head.net, torch.nn.Sequential)
    out = head(torch.zeros(3, 8))
    assert out.shape == (3, N_TOKENS)


def test_destandardize_roundtrip() -> None:
    """`predict_log_frac` undoes the standardization; `predict_frac` exponentiates."""
    head = LogImputationHead(d_model=4, hidden=None)
    mean = torch.full((N_TOKENS,), -1.0)
    std = torch.full((N_TOKENS,), 0.5)
    head.fit_standardization(mean, std)
    h = torch.randn(2, 4)
    raw = head(h)
    log_frac = head.predict_log_frac(h)
    assert torch.allclose(log_frac, raw * std + mean)
    assert torch.allclose(head.predict_frac(h), torch.pow(10.0, log_frac))


def test_fit_standardization_floors_std() -> None:
    """A near-constant element's std is clamped so the target cannot blow up."""
    head = LogImputationHead(d_model=4)
    std = torch.ones(N_TOKENS)
    std[5] = 0.0
    head.fit_standardization(torch.zeros(N_TOKENS), std)
    assert head.target_std[5].item() == pytest.approx(_STD_FLOOR)


def test_fit_standardization_wrong_shape_raises() -> None:
    head = LogImputationHead(d_model=4)
    with pytest.raises(ValueError, match="standardization stats"):
        head.fit_standardization(torch.zeros(3), torch.ones(N_TOKENS))


def test_buffers_serialize_and_reload() -> None:
    """The standardization buffers are part of state_dict and reload exactly."""
    head = LogImputationHead(d_model=4, hidden=4)
    head.fit_standardization(torch.full((N_TOKENS,), 2.0), torch.full((N_TOKENS,), 3.0))
    state = head.state_dict()
    assert "target_mean" in state and "target_std" in state

    clone = LogImputationHead(d_model=4, hidden=4)
    clone.load_state_dict(state)
    assert torch.equal(clone.target_mean, head.target_mean)
    assert torch.equal(clone.target_std, head.target_std)
