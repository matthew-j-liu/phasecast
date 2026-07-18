import torch

from cnmineral.losses.closure_loss import closure_loss, closure_residual


def _consistent_case(batch=3, n_phases=4, n_ox=6, seed=0):
    """Fixings and bulk constructed so the residual is exactly zero."""
    g = torch.Generator().manual_seed(seed)
    modes = torch.rand(batch, n_phases, generator=g)
    modes = modes / modes.sum(-1, keepdim=True)
    comp = torch.rand(batch, n_phases, n_ox, generator=g)
    comp = comp / comp.sum(-1, keepdim=True)
    f_liq = torch.full((batch,), 0.4)
    f_fl = torch.full((batch,), 0.1)
    c_liq = torch.rand(batch, n_ox, generator=g)
    c_liq = c_liq / c_liq.sum(-1, keepdim=True)
    c_fl = torch.rand(batch, n_ox, generator=g)
    c_fl = c_fl / c_fl.sum(-1, keepdim=True)
    bulk = (
        (1 - f_liq - f_fl)[:, None] * torch.einsum("bp,bpo->bo", modes, comp)
        + f_liq[:, None] * c_liq
        + f_fl[:, None] * c_fl
    )
    return modes, comp, f_liq, f_fl, c_liq, c_fl, bulk


def test_consistent_inputs_give_zero_residual():
    args = _consistent_case()
    assert closure_residual(*args).abs().max() < 1e-6
    assert closure_loss(*args) < 1e-12


def test_gradient_reaches_modes_and_comp_not_fixings():
    modes, comp, f_liq, f_fl, c_liq, c_fl, bulk = _consistent_case()
    modes = (modes + 0.01).requires_grad_(True)
    comp.requires_grad_(True)
    loss = closure_loss(modes, comp, f_liq, f_fl, c_liq, c_fl, bulk)
    loss.backward()
    assert modes.grad is not None and modes.grad.abs().sum() > 0
    assert comp.grad is not None and comp.grad.abs().sum() > 0
    assert not f_liq.requires_grad and not c_liq.requires_grad


def test_loss_is_mean_squared_l2():
    args = _consistent_case()
    residual = closure_residual(*args) + 0.1
    bulk_shifted = args[-1] - 0.1
    loss = closure_loss(*args[:-1], bulk_shifted)
    assert torch.allclose(loss, residual.pow(2).sum(-1).mean(), atol=1e-6)
