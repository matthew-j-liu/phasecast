import torch

from cnmineral.data.phase_vocab import N_PHASES
from cnmineral.models.comp_head import N_OXIDES, CompHead


def test_output_shape_and_positivity():
    head = CompHead(d_model=32)
    alpha = head(torch.randn(5, 32))
    assert alpha.shape == (5, N_PHASES, N_OXIDES)
    assert (alpha > 0).all()


def test_mean_is_simplex():
    head = CompHead(d_model=32)
    alpha = head(torch.randn(4, 32))
    mean = alpha / alpha.sum(-1, keepdim=True)
    assert torch.allclose(mean.sum(-1), torch.ones(4, N_PHASES))


def test_shared_concentration_scales_alpha_not_mean():
    torch.manual_seed(0)
    head = CompHead(d_model=32)
    h = torch.randn(3, 32)
    alpha = head(h)
    mean = alpha / alpha.sum(-1, keepdim=True)
    with torch.no_grad():
        head.conc_out.weight.mul_(0.0)
        head.conc_out.bias.fill_(5.0)
    alpha2 = head(h)
    mean2 = alpha2 / alpha2.sum(-1, keepdim=True)
    assert torch.allclose(mean, mean2, atol=1e-6)
    assert not torch.allclose(alpha, alpha2)


def test_default_hidden_is_half_d_model():
    head = CompHead(d_model=64)
    assert head.trunk[0].out_features == 32
    assert CompHead(d_model=64, hidden=7).trunk[0].out_features == 7


def test_layers_stack_the_trunk():
    deep = CompHead(d_model=32, layers=2)
    assert sum(isinstance(m, torch.nn.Linear) for m in deep.trunk) == 2
    alpha = deep(torch.randn(4, 32))
    assert alpha.shape == (4, N_PHASES, N_OXIDES) and (alpha > 0).all()
    shallow = sum(p.numel() for p in CompHead(d_model=32).parameters())
    assert sum(p.numel() for p in deep.parameters()) > shallow


def test_gradients_flow_to_all_parameters():
    head = CompHead(d_model=16)
    head(torch.randn(2, 16)).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
