"""RealRockAdapter: identity at init, exact tier nesting, and the amp pin."""

import pytest
import torch

from cnmineral.data.phase_vocab import GROUP_INDEX, GROUPS
from cnmineral.models.real_rock_adapter import TIERS, RealRockAdapter


def _inputs(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, len(GROUPS), generator=g)
    alpha = torch.rand(n, len(GROUPS), generator=g) * 5 + 0.1
    sio2 = 50 + torch.rand(n, generator=g) * 40
    return logits, alpha, sio2


def _randomize(adapter, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in adapter.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
        adapter.presence_scale.add_(1.0)
    return adapter


@pytest.mark.parametrize("tier", TIERS)
def test_adapter_is_the_identity_at_initialization(tier):
    """An unfit adapter must reproduce the zero-shot number exactly.

    That is what lets a fitted adapter's gain be read as nats recovered on the
    same scale the ledger reports.
    """
    logits, alpha, sio2 = _inputs()
    out_logits, out_alpha = RealRockAdapter(tier)(logits, alpha, sio2)
    assert torch.allclose(out_logits, logits)
    assert torch.allclose(out_alpha, alpha)


@pytest.mark.parametrize("tier", TIERS)
def test_structural_zeros_survive_the_magnitude_correction(tier):
    """Coarsening leaves alpha = 0 where a group has no member on the support.

    The correction is multiplicative for this reason: an additive one in log
    space would turn those into -inf and poison the Dirichlet term.
    """
    logits, alpha, sio2 = _inputs()
    alpha[:, GROUP_INDEX["ol"]] = 0.0
    _, out_alpha = _randomize(RealRockAdapter(tier))(logits, alpha, sio2)
    assert (out_alpha[:, GROUP_INDEX["ol"]] == 0).all()
    assert torch.isfinite(out_alpha).all()


@pytest.mark.parametrize("tier", ["linear", "mlp"])
def test_richer_tiers_contain_the_diagonal_exactly(tier):
    """diagonal in linear in mlp, so the 23-param affine is the degenerate case.

    Copying a fitted diagonal's parameters into a richer tier must reproduce it
    to floating point -- otherwise the tiers are rivals rather than a nesting
    and the curve compares three different things.
    """
    logits, alpha, sio2 = _inputs()
    diagonal = _randomize(RealRockAdapter("diagonal"))
    richer = RealRockAdapter(tier)
    with torch.no_grad():
        target = dict(richer.named_parameters())
        for name, p in diagonal.named_parameters():
            target[name].copy_(p)
    for got, want in zip(richer(logits, alpha, sio2),
                         diagonal(logits, alpha, sio2)):
        assert torch.allclose(got, want, atol=1e-6)


@pytest.mark.parametrize("tier", TIERS)
def test_amphibole_magnitude_is_pinned_at_every_tier(tier):
    """The pin holds through the mixing matrix and the MLP, not just the constant.

    Amphibole's magnitude offset is unidentifiable -- the mapping audit's Monte
    Carlo band straddles zero -- so a coupled correction reaching it would be
    the same unidentifiable quantity in another parameterization.
    """
    logits, alpha, sio2 = _inputs()
    amp = GROUP_INDEX["amp"]
    _, pinned = _randomize(RealRockAdapter(tier, pin_amp_magnitude=True))(
        logits, alpha, sio2)
    _, free = _randomize(RealRockAdapter(tier, pin_amp_magnitude=False))(
        logits, alpha, sio2)
    assert torch.allclose(pinned[:, amp], alpha[:, amp])
    assert not torch.allclose(free[:, amp], alpha[:, amp])


def test_off_support_groups_contribute_a_neutral_log_feature():
    """An off-support alpha must not reach the coupled tiers as log(tiny).

    Clamping to 1e-12 before the log hands the mixing matrix a -27 spike on
    groups carrying no information, and the linear / mlp fits diverge on it
    (val NLL ~1e20). The feature is 0 there instead, so a zero column cannot
    move the correction.
    """
    logits, alpha, sio2 = _inputs()
    alpha[:, GROUP_INDEX["ol"]] = 0.0
    adapter = _randomize(RealRockAdapter("linear"))
    _, baseline = adapter(logits, alpha, sio2)

    # scaling the mixing weights that read the off-support column must be a
    # no-op, which only holds if that column enters as a constant zero
    with torch.no_grad():
        adapter.magnitude_mix[GROUP_INDEX["ol"], :] *= 7.0
    _, moved = adapter(logits, alpha, sio2)
    assert torch.allclose(baseline, moved)


def test_unknown_tier_is_rejected():
    with pytest.raises(ValueError, match="unknown tier"):
        RealRockAdapter("quadratic")
