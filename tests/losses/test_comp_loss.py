import torch
from torch.distributions import Dirichlet

from cnmineral.losses.comp_loss import masked_comp_dirichlet_nll


def _random_case(batch=4, n_phases=5, n_ox=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    alpha = torch.rand(batch, n_phases, n_ox, generator=g) * 4 + 0.5
    target = torch.rand(batch, n_phases, n_ox, generator=g) + 0.05
    target = target / target.sum(-1, keepdim=True)
    return alpha, target


def test_matches_torch_dirichlet_on_present():
    alpha, target = _random_case()
    presence = torch.ones(4, 5)
    nll = masked_comp_dirichlet_nll(alpha, target, presence, reduction="none")
    ref = -Dirichlet(alpha).log_prob(target)
    assert torch.allclose(nll, ref, atol=1e-5)


def test_absent_phases_get_zero_loss_and_gradient():
    alpha, target = _random_case()
    alpha.requires_grad_(True)
    presence = torch.zeros(4, 5)
    presence[:, 0] = 1.0
    nll = masked_comp_dirichlet_nll(alpha, target, presence, reduction="none")
    assert (nll[:, 1:] == 0).all()
    nll.sum().backward()
    assert torch.all(alpha.grad[:, 1:] == 0)
    assert alpha.grad[:, 0].abs().sum() > 0


def test_mean_reduction_divides_by_present_pairs():
    alpha, target = _random_case()
    presence = torch.zeros(4, 5)
    presence[0, :2] = 1.0
    presence[1, 0] = 1.0
    per_pair = masked_comp_dirichlet_nll(alpha, target, presence, reduction="none")
    mean = masked_comp_dirichlet_nll(alpha, target, presence)
    assert torch.allclose(mean, per_pair.sum() / 3.0)
