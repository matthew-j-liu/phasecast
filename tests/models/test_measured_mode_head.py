"""Contract of the measured-label group head: shapes, positivity, and the
structural guarantee that its presence branch cannot see the support."""

import pytest
import torch

from cnmineral.data.phase_vocab import GROUPS
from cnmineral.models.measured_mode_head import MeasuredModeHead

D_MODEL = 32
N_GROUPS = len(GROUPS)


def _inputs(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    latent = torch.randn(n, D_MODEL, generator=g, dtype=torch.float64)
    support = (torch.rand(n, N_GROUPS, generator=g) > 0.4).double()
    return latent, support


@pytest.mark.parametrize("k", [1, 2, 4])
def test_output_is_always_mixture_shaped(k):
    """The scoring path takes (logits, alpha, log_w) from every arm; a head
    that dropped the component axis at K=1 would need a special case there."""
    head = MeasuredModeHead(D_MODEL, n_components=k).double()
    logits, alpha, log_w = head(*_inputs())
    assert logits.shape == (6, N_GROUPS)
    assert alpha.shape == (6, k, N_GROUPS)
    assert log_w.shape == (6, k)
    assert torch.allclose(log_w.exp().sum(-1), torch.ones(6, dtype=torch.float64))


def test_single_component_gate_is_exactly_zero():
    head = MeasuredModeHead(D_MODEL, n_components=1).double()
    _, _, log_w = head(*_inputs())
    assert (log_w == 0.0).all()


def test_concentrations_are_strictly_positive():
    """lgamma(0) is +inf and the masked Dirichlet forms mask * lgamma(alpha),
    so a zero here would be a NaN gradient rather than a no-op."""
    head = MeasuredModeHead(D_MODEL).double()
    _, alpha, _ = head(*_inputs())
    assert (alpha > 0).all()


def test_presence_branch_cannot_see_the_support():
    """It predicts the support, so reading it would be the leak that makes the
    hurdle's first factor free."""
    head = MeasuredModeHead(D_MODEL).double()
    latent, support = _inputs()
    support.requires_grad_(True)
    head(latent, support)[0].sum().backward()
    assert support.grad is None or torch.count_nonzero(support.grad) == 0


def test_standardizer_rides_on_the_module():
    """The fit standardizes on the drawn budget; a held-out slice must be
    scored through that same shift and scale, not its own."""
    head = MeasuredModeHead(D_MODEL).double()
    head.latent_mean = torch.full((D_MODEL,), 3.0, dtype=torch.float64)
    head.latent_std = torch.full((D_MODEL,), 2.0, dtype=torch.float64)
    latent = torch.full((2, D_MODEL), 7.0, dtype=torch.float64)
    assert torch.allclose(head.standardize(latent), torch.full_like(latent, 2.0))


def test_zero_components_is_rejected():
    with pytest.raises(ValueError, match="n_components"):
        MeasuredModeHead(D_MODEL, n_components=0)
