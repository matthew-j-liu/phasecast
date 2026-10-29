"""RealRockRecalibrator: identity at init, exact tier nesting, structural zeros."""

import pytest
import torch

from cnmineral.data.phase_vocab import GROUP_INDEX, GROUPS
from cnmineral.models.real_rock_recalibrator import TIERS, RealRockRecalibrator


def _inputs(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, len(GROUPS), generator=g)
    alpha = torch.rand(n, len(GROUPS), generator=g) * 5 + 0.1
    return logits, alpha


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
    same scale the ledger reports.
    """
    logits, alpha = _inputs()
    out_logits, out_alpha, log_w = RealRockRecalibrator(tier)(logits, alpha)
    assert torch.allclose(out_logits, logits)
    assert torch.allclose(out_alpha, alpha)
    assert log_w is None


@pytest.mark.parametrize("tier", TIERS)
def test_structural_zeros_survive_the_magnitude_correction(tier):
    """Coarsening leaves alpha = 0 where a group has no member on the support.

    The correction is multiplicative for this reason: an additive one in log
    space would turn those into -inf and poison the Dirichlet term.
    """
    logits, alpha = _inputs()
    alpha[:, GROUP_INDEX["ol"]] = 0.0
    _, out_alpha, _ = _randomize(RealRockRecalibrator(tier))(logits, alpha)
    assert (out_alpha[..., GROUP_INDEX["ol"]] == 0).all()
    assert torch.isfinite(out_alpha).all()


def test_matrix_tier_contains_the_vector_tier_exactly():
    """vector in matrix, so the 21-param affine is the degenerate case of the
    coupled tier.

    Copying a fitted vector tier's parameters into the matrix tier must
    reproduce it to floating point -- otherwise the tiers are rivals rather
    than a nesting and the curve compares two different things.
    """
    logits, alpha = _inputs()
    vector = _randomize(RealRockRecalibrator("vector"))
    richer = RealRockRecalibrator("matrix")
    with torch.no_grad():
        target = dict(richer.named_parameters())
        for name, p in vector.named_parameters():
            target[name].copy_(p)
    got_logits, got_alpha, _ = richer(logits, alpha)
    want_logits, want_alpha, _ = vector(logits, alpha)
    assert torch.allclose(got_logits, want_logits, atol=1e-6)
    assert torch.allclose(got_alpha, want_alpha, atol=1e-6)


def test_off_support_groups_contribute_a_neutral_log_feature():
    """An off-support alpha must not reach the coupled tiers as log(tiny).

    Clamping to 1e-12 before the log hands the mixing matrix a -27 spike on
    groups carrying no information, and the matrix fit diverges on it
    (val NLL ~1e20). The feature is 0 there instead, so a zero column cannot
    move the correction.
    """
    logits, alpha = _inputs()
    alpha[:, GROUP_INDEX["ol"]] = 0.0
    recalibrator = _randomize(RealRockRecalibrator("matrix"))
    _, baseline, _ = recalibrator(logits, alpha)

    # scaling the mixing weights that read the off-support column must be a
    # no-op, which only holds if that column enters as a constant zero
    with torch.no_grad():
        recalibrator.magnitude_mix[GROUP_INDEX["ol"], :] *= 7.0
    _, moved, _ = recalibrator(logits, alpha)
    assert torch.allclose(baseline, moved)


def test_unknown_tier_is_rejected():
    with pytest.raises(ValueError, match="unknown tier"):
        RealRockRecalibrator("quadratic")
