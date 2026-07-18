import torch
from torch.distributions import Categorical, MixtureSameFamily, Normal

from cnmineral.losses.cond_loss import gmm_nll


def test_single_component_matches_normal():
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(6, 3, 1, generator=g)
    sigma = torch.rand(6, 3, 1, generator=g) + 0.5
    log_w = torch.zeros(6, 3, 1)
    target = torch.randn(6, 3, generator=g)
    nll = gmm_nll(log_w, mu, sigma, target, reduction="none")
    ref = -Normal(mu[..., 0], sigma[..., 0]).log_prob(target).sum(-1)
    assert torch.allclose(nll, ref, atol=1e-5)


def test_multi_component_matches_mixture_same_family():
    g = torch.Generator().manual_seed(1)
    k = 4
    logits = torch.randn(5, 3, k, generator=g)
    log_w = torch.log_softmax(logits, dim=-1)
    mu = torch.randn(5, 3, k, generator=g) * 3
    sigma = torch.rand(5, 3, k, generator=g) + 0.3
    target = torch.randn(5, 3, generator=g)
    nll = gmm_nll(log_w, mu, sigma, target, reduction="none")
    mix = MixtureSameFamily(Categorical(logits=logits), Normal(mu, sigma))
    ref = -mix.log_prob(target).sum(-1)
    assert torch.allclose(nll, ref, atol=1e-5)


def test_mean_reduction():
    log_w = torch.zeros(3, 3, 1)
    mu = torch.zeros(3, 3, 1)
    sigma = torch.ones(3, 3, 1)
    target = torch.zeros(3, 3)
    per_row = gmm_nll(log_w, mu, sigma, target, reduction="none")
    assert torch.allclose(gmm_nll(log_w, mu, sigma, target), per_row.mean())
