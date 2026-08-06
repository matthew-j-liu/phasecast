"""RealRockRecalibrator: identity at init, exact tier nesting, and the amp pin."""

import pytest
import torch

from cnmineral.data.phase_vocab import GROUP_INDEX, GROUPS
from cnmineral.models.real_rock_recalibrator import TIERS, RealRockRecalibrator


def _inputs(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, len(GROUPS), generator=g)
    alpha = torch.rand(n, len(GROUPS), generator=g) * 5 + 0.1
    sio2 = 50 + torch.rand(n, generator=g) * 40
    return logits, alpha, sio2


def _randomize(recalibrator, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in recalibrator.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
        recalibrator.presence_scale.add_(1.0)
    return recalibrator


@pytest.mark.parametrize("tier", TIERS)
def test_recalibrator_is_the_identity_at_initialization(tier):
    """An unfit recalibrator must reproduce the zero-shot number exactly.

    That is what lets a fitted recalibrator's gain be read as nats recovered on the
    same scale the ledger reports.  The tailed tier returns a two-component
    mixture, but its dilution factor is 1 at init, so both components equal the
    input and the mixture is the single Dirichlet distributionally.
    """
    logits, alpha, sio2 = _inputs()
    out_logits, out_alpha, log_w = RealRockRecalibrator(tier)(logits, alpha, sio2)
    assert torch.allclose(out_logits, logits)
    if tier == "tailed":
        assert out_alpha.shape == (len(alpha), 2, len(GROUPS))
        assert torch.allclose(out_alpha[:, 0], alpha)
        assert torch.allclose(out_alpha[:, 1], alpha)
        assert torch.allclose(log_w.exp().sum(-1), torch.ones(len(alpha)))
    else:
        assert torch.allclose(out_alpha, alpha)
        assert log_w is None


@pytest.mark.parametrize("tier", TIERS)
def test_structural_zeros_survive_the_magnitude_correction(tier):
    """Coarsening leaves alpha = 0 where a group has no member on the support.

    The correction is multiplicative for this reason: an additive one in log
    space would turn those into -inf and poison the Dirichlet term.
    """
    logits, alpha, sio2 = _inputs()
    alpha[:, GROUP_INDEX["ol"]] = 0.0
    _, out_alpha, _ = _randomize(RealRockRecalibrator(tier))(logits, alpha, sio2)
    assert (out_alpha[..., GROUP_INDEX["ol"]] == 0).all()
    assert torch.isfinite(out_alpha).all()


@pytest.mark.parametrize("tier", ["matrix", "mlp", "tailed"])
def test_richer_tiers_contain_the_vector_tier_exactly(tier):
    """vector in matrix in mlp / tailed, so the 23-param affine is the
    degenerate case of every richer tier.

    Copying a fitted vector tier's parameters into a richer tier must reproduce
    it to floating point -- otherwise the tiers are rivals rather than a nesting
    and the curve compares three different things.  For tailed the check is
    distributional: dilution 1 at init means both components must equal the
    vector tier's output.
    """
    logits, alpha, sio2 = _inputs()
    vector = _randomize(RealRockRecalibrator("vector"))
    richer = RealRockRecalibrator(tier)
    with torch.no_grad():
        target = dict(richer.named_parameters())
        for name, p in vector.named_parameters():
            target[name].copy_(p)
    got_logits, got_alpha, _ = richer(logits, alpha, sio2)
    want_logits, want_alpha, _ = vector(logits, alpha, sio2)
    assert torch.allclose(got_logits, want_logits, atol=1e-6)
    if tier == "tailed":
        assert torch.allclose(got_alpha[:, 0], want_alpha, atol=1e-6)
        assert torch.allclose(got_alpha[:, 1], want_alpha, atol=1e-6)
    else:
        assert torch.allclose(got_alpha, want_alpha, atol=1e-6)


@pytest.mark.parametrize("tier", TIERS)
def test_amphibole_magnitude_is_pinned_at_every_tier(tier):
    """The pin holds through the mixing matrix and the MLP, not just the constant.

    Amphibole's magnitude offset is unidentifiable -- the mapping audit's Monte
    Carlo band straddles zero -- so a coupled correction reaching it would be
    the same unidentifiable quantity in another parameterization.  For tailed,
    "pinned" means no amp-specific term: the sharp component's amp alpha is the
    input exactly, and the wide component applies the same global dilution to
    amp as to every other group.
    """
    logits, alpha, sio2 = _inputs()
    amp = GROUP_INDEX["amp"]
    _, pinned, _ = _randomize(RealRockRecalibrator(tier, pin_amp_magnitude=True))(
        logits, alpha, sio2)
    _, free, _ = _randomize(RealRockRecalibrator(tier, pin_amp_magnitude=False))(
        logits, alpha, sio2)
    if tier == "tailed":
        assert torch.allclose(pinned[:, 0, amp], alpha[:, amp])
        ratio = pinned[:, 1] / pinned[:, 0]
        assert torch.allclose(ratio, ratio[:, :1].expand_as(ratio))
        assert not torch.allclose(free[:, 0, amp], alpha[:, amp])
    else:
        assert torch.allclose(pinned[:, amp], alpha[:, amp])
        assert not torch.allclose(free[:, amp], alpha[:, amp])


def test_off_support_groups_contribute_a_neutral_log_feature():
    """An off-support alpha must not reach the coupled tiers as log(tiny).

    Clamping to 1e-12 before the log hands the mixing matrix a -27 spike on
    groups carrying no information, and the matrix / mlp fits diverge on it
    (val NLL ~1e20). The feature is 0 there instead, so a zero column cannot
    move the correction.
    """
    logits, alpha, sio2 = _inputs()
    alpha[:, GROUP_INDEX["ol"]] = 0.0
    recalibrator = _randomize(RealRockRecalibrator("matrix"))
    _, baseline, _ = recalibrator(logits, alpha, sio2)

    # scaling the mixing weights that read the off-support column must be a
    # no-op, which only holds if that column enters as a constant zero
    with torch.no_grad():
        recalibrator.magnitude_mix[GROUP_INDEX["ol"], :] *= 7.0
    _, moved, _ = recalibrator(logits, alpha, sio2)
    assert torch.allclose(baseline, moved)


def test_tailed_contamination_is_mean_preserving():
    """The wide component is the sharp one diluted: same mean, lower precision.

    That is the design's whole claim -- the tail fattens without the mean
    moving, so contamination cannot masquerade as a location correction.
    """
    logits, alpha, sio2 = _inputs()
    recalibrator = RealRockRecalibrator("tailed")
    with torch.no_grad():
        recalibrator.contam_logk.fill_(-1.5)
        recalibrator.contam_logit.fill_(1.0)
    _, out, log_w = recalibrator(logits, alpha, sio2)
    k = torch.exp(torch.tensor(-1.5))
    torch.testing.assert_close(out[:, 1], out[:, 0] * k)
    share = out / out.sum(-1, keepdim=True)
    torch.testing.assert_close(share[:, 1], share[:, 0])
    assert torch.allclose(log_w.exp().sum(-1), torch.ones(len(alpha)))


def test_tailed_composes_with_mixture_head_input():
    """K-component input comes back as a 2K mixture with normalized weights."""
    logits, alpha, sio2 = _inputs()
    alpha3 = torch.stack([alpha, alpha * 2.0], dim=1)
    log_w = torch.log(torch.full((len(alpha), 2), 0.5))
    recalibrator = _randomize(RealRockRecalibrator("tailed"))
    out_logits, out_alpha, out_log_w = recalibrator(
        logits, alpha3, sio2, log_w=log_w
    )
    assert out_alpha.shape == (len(alpha), 4, len(GROUPS))
    assert out_log_w.shape == (len(alpha), 4)
    assert torch.allclose(out_log_w.exp().sum(-1), torch.ones(len(alpha)))
    assert out_logits.shape == logits.shape


def test_unknown_tier_is_rejected():
    with pytest.raises(ValueError, match="unknown tier"):
        RealRockRecalibrator("quadratic")
