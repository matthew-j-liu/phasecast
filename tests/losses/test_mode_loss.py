import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet

from cnmineral.losses.mode_loss import (
    masked_dirichlet_nll,
    mixture_dirichlet_nll,
    mode_head_loss,
)


def _random_hurdle_batch(n=16, p=8, seed=0):
    """Alphas, present-renormalized targets, and masks with >= 2 present each."""
    g = torch.Generator().manual_seed(seed)
    alpha = torch.rand(n, p, generator=g) * 3 + 0.1
    presence = torch.zeros(n, p)
    for i in range(n):
        k = int(torch.randint(2, p + 1, (1,), generator=g))
        idx = torch.randperm(p, generator=g)[:k]
        presence[i, idx] = 1.0
    raw = torch.rand(n, p, generator=g) * presence + 1e-3 * presence
    magnitude = raw / raw.sum(-1, keepdim=True)
    return alpha, magnitude, presence


def test_full_support_matches_torch_dirichlet():
    torch.manual_seed(0)
    alpha = torch.rand(8, 5) * 3 + 0.1
    x = torch.distributions.Dirichlet(torch.ones(5)).sample((8,)).clamp_min(1e-4)
    x = x / x.sum(-1, keepdim=True)
    ours = masked_dirichlet_nll(alpha, x, torch.ones(8, 5), reduction="none")
    ref = -Dirichlet(alpha).log_prob(x)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_subcomposition_matches_sliced_dirichlet():
    alpha, magnitude, presence = _random_hurdle_batch()
    ours = masked_dirichlet_nll(alpha, magnitude, presence, reduction="none")
    for i in range(len(alpha)):
        s = presence[i] > 0
        ref = -Dirichlet(alpha[i, s]).log_prob(magnitude[i, s])
        assert torch.allclose(ours[i], ref, atol=1e-5)


def test_absent_phases_get_zero_gradient():
    alpha, magnitude, presence = _random_hurdle_batch()
    alpha = alpha.clone().requires_grad_(True)
    masked_dirichlet_nll(alpha, magnitude, presence).backward()
    assert torch.all(alpha.grad[presence == 0] == 0)
    assert alpha.grad[presence > 0].abs().sum() > 0


def test_single_phase_support_contributes_zero():
    alpha = torch.tensor([[2.0, 3.0]])
    magnitude = torch.tensor([[1.0, 0.0]])
    presence = torch.tensor([[1.0, 0.0]])
    nll = masked_dirichlet_nll(alpha, magnitude, presence)
    assert torch.isclose(nll, torch.tensor(0.0), atol=1e-6)


def test_mode_head_loss_composition():
    alpha, magnitude, presence = _random_hurdle_batch()
    logits = torch.randn_like(alpha)
    terms = mode_head_loss(logits, alpha, presence, magnitude, magnitude_weight=0.5)
    bce = F.binary_cross_entropy_with_logits(logits, presence)
    nll = masked_dirichlet_nll(alpha, magnitude, presence)
    assert torch.isclose(terms["presence_bce"], bce)
    assert torch.isclose(terms["dirichlet_nll"], nll)
    assert torch.isclose(terms["total"], bce + 0.5 * nll)


def test_mixture_k1_is_exactly_single_dirichlet():
    alpha, magnitude, presence = _random_hurdle_batch()
    log_w = torch.zeros(len(alpha), 1)
    mix = mixture_dirichlet_nll(log_w, alpha.unsqueeze(1), magnitude, presence,
                                reduction="none")
    single = masked_dirichlet_nll(alpha, magnitude, presence, reduction="none")
    assert torch.equal(mix, single)


def test_mixture_k2_matches_hand_logsumexp():
    alpha, magnitude, presence = _random_hurdle_batch()
    alpha2, _, _ = _random_hurdle_batch(seed=1)
    stacked = torch.stack([alpha, alpha2], dim=1)
    log_w = torch.log_softmax(torch.randn(len(alpha), 2,
                                          generator=torch.Generator().manual_seed(2)),
                              dim=-1)
    ours = mixture_dirichlet_nll(log_w, stacked, magnitude, presence,
                                 reduction="none")
    for i in range(len(alpha)):
        s = presence[i] > 0
        comp = torch.stack([
            Dirichlet(stacked[i, k, s]).log_prob(magnitude[i, s])
            for k in range(2)
        ])
        ref = -torch.logsumexp(log_w[i] + comp, dim=-1)
        assert torch.allclose(ours[i], ref, atol=1e-5)


def test_mixture_absent_phases_get_zero_gradient_per_component():
    alpha, magnitude, presence = _random_hurdle_batch()
    alpha2, _, _ = _random_hurdle_batch(seed=3)
    stacked = torch.stack([alpha, alpha2], dim=1).clone().requires_grad_(True)
    log_w = torch.log_softmax(torch.randn(len(alpha), 2), dim=-1)
    mixture_dirichlet_nll(log_w, stacked, magnitude, presence).backward()
    absent = (presence == 0).unsqueeze(1).expand_as(stacked)
    assert torch.all(stacked.grad[absent] == 0)
    assert stacked.grad[~absent].abs().sum() > 0


def test_gradient_reaches_unfrozen_encoder():
    from cnmineral.data.phase_vocab import N_PHASES
    from cnmineral.models.encoder import Encoder
    from cnmineral.models.mode_head import ModeHead

    torch.manual_seed(0)
    encoder = Encoder(d_model=16, n_layers=1, n_heads=2, frac_resolution=200)
    head = ModeHead(d_model=16)
    src = torch.tensor([[14, 13, 20, 0, 0], [14, 26, 0, 0, 0]], dtype=torch.long)
    frac = torch.tensor(
        [[0.5, 0.3, 0.2, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0, 0.0]], dtype=torch.float32
    )
    presence = torch.ones(2, N_PHASES)
    logits, alpha = head(encoder(src=src, frac=frac), presence)
    magnitude = torch.full_like(logits, 1.0 / logits.shape[-1])
    mode_head_loss(logits, alpha, presence, magnitude)["total"].backward()
    grads = [p.grad for p in encoder.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
