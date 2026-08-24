"""Adapting the frozen ModeHead in place: the identity it starts from, the
support it is forbidden to move, and the coarsening gradient the fit rides on.

The coarsening is the part worth distrusting.  Every other arm reads precomputed
group-space outputs, so this is the only place a noisy-OR has to carry a
gradient, and a wrong one would not fail loudly -- it would just fit badly and
read as "adapting the head does not help", which is the conclusion under test.
"""

import copy

import numpy as np
import pytest
import torch

from cnmineral.data.phase_vocab import (
    GROUP_INDEX,
    GROUPS,
    MODEL_TO_GROUP,
    N_PHASES,
    PHASE_INDEX,
)
from cnmineral.eval.real_rock import (
    coarsen_alpha,
    coarsen_presence,
    truth_forced_support,
)
from cnmineral.models.mode_head import ModeHead
from cnmineral.training.real_rock_recalibrator import (
    GroupPredictions,
    recalibrator_nll,
    score_predictions,
)

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "figures" / "src"))
from adapted_mode_head import (
    AdaptedModeHead,
    coarsen_alpha_on_support,
    coarsen_presence_logits,
    fit_adapted_head,
    group_map,
    score_adapted_head,
)

N_GROUPS = len(GROUPS)
D_MODEL = 16
N_RAW = 22

# float64 throughout so the finite-difference check has something to measure and
# the numpy-twin comparisons are not reading float32 rounding; dropout is live so
# a head accidentally left in train mode shows up as a nondeterministic forward.
def _head(dropout: float = 0.5, layers: int = 3) -> ModeHead:
    torch.manual_seed(0)
    return ModeHead(D_MODEL, hidden=8, layers=layers, dropout=dropout).double().eval()


def _predictions(n: int = 24, seed: int = 0, units_of: int = 3) -> GroupPredictions:
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(0.5, 8.0, (n, N_GROUPS))
    alpha[:, GROUP_INDEX["ol"]] = 0.0
    numeric = np.zeros((n, N_GROUPS), dtype=bool)
    numeric[:, [GROUP_INDEX["fsp"], GROUP_INDEX["px"], GROUP_INDEX["amp"]]] = True
    magnitude = np.zeros((n, N_GROUPS))
    magnitude[numeric] = rng.dirichlet(np.ones(3), n).reshape(-1)
    return GroupPredictions(
        logits7=torch.as_tensor(rng.normal(0, 1.5, (n, N_GROUPS))),
        alpha7=torch.as_tensor(alpha),
        sio2=torch.as_tensor(rng.uniform(50, 78, n)),
        presence=torch.as_tensor(numeric.astype(np.float64)),
        magnitude=torch.as_tensor(magnitude),
        numeric=torch.as_tensor(numeric.astype(np.float64)),
        unit=np.array([f"unit-{i // units_of}" for i in range(n)]),
        latent=torch.as_tensor(rng.normal(0, 1.0, (n, D_MODEL))),
        raw=torch.as_tensor(rng.normal(0, 1.0, (n, N_RAW))),
    )


def test_every_linear_layer_is_adapted():
    """The tree is walked rather than the eight layers named, so a head built at
    a different depth is adapted throughout instead of partially."""
    assert len(AdaptedModeHead(_head(layers=3)).adapters) == 8
    assert len(AdaptedModeHead(_head(layers=1)).adapters) == 4


def test_zero_init_reproduces_the_frozen_head_exactly():
    """B starts at zero, so an unfit adapter is the frozen head bit for bit --
    the contract that lets a fitted gain read as nats recovered against the
    zero-shot forecast rather than against a slightly different model."""
    head, data = _head(dropout=0.0), _predictions()
    adapted = AdaptedModeHead(head)
    support = adapted.pinned_support(data.latent, data.presence)
    with torch.no_grad():
        assert torch.equal(
            adapted.head.presence_logits(data.latent), head.presence_logits(data.latent)
        )
        assert torch.equal(
            adapted.head.magnitude_alpha(data.latent, support),
            head.magnitude_alpha(data.latent, support),
        )


def test_pinned_support_reproduces_the_frozen_heads_own_support():
    """The same `support21` `eval.real_rock.predict` builds, so this arm's
    Dirichlet sits on the support the correction arms' alpha7 was built on."""
    head, data = _head(dropout=0.0), _predictions()
    with torch.no_grad():
        p21 = torch.sigmoid(head.presence_logits(data.latent)).numpy()
    expected = truth_forced_support(data.presence.numpy(), p21)
    got = AdaptedModeHead(head).pinned_support(data.latent, data.presence)
    assert np.array_equal(got.numpy(), expected)


def test_the_support_does_not_move_when_the_adapters_do():
    """`truth_forced_support` is a threshold plus an argmax, so a support that
    tracked the adapters would carry no extra gradient -- only a discontinuity
    each time a member phase crossed 0.5 mid-fit."""
    head, data = _head(dropout=0.0), _predictions()
    adapted = AdaptedModeHead(head)
    before = adapted.pinned_support(data.latent, data.presence)
    with torch.no_grad():
        for lora in adapted.adapters:
            lora.b.normal_(0.0, 5.0)
    after = adapted.pinned_support(data.latent, data.presence)
    assert torch.equal(before, after)
    # and the adapters really are live otherwise, or the check above is vacuous
    with torch.no_grad():
        moved = adapted.head.presence_logits(data.latent)
        with adapted.frozen_path():
            base = adapted.head.presence_logits(data.latent)
    assert not torch.allclose(moved, base)


def test_coarsened_alpha_matches_the_numpy_twin():
    """Merging Dirichlet components sums concentrations, so the torch form is
    the numpy one exactly rather than approximately."""
    rng = np.random.default_rng(0)
    alpha = rng.uniform(0.1, 9.0, (12, N_PHASES))
    support = (rng.random((12, N_PHASES)) > 0.4).astype(np.float64)
    expected, _ = coarsen_alpha(alpha, support)
    got = coarsen_alpha_on_support(
        torch.as_tensor(alpha), torch.as_tensor(support), group_map()
    )
    assert np.array_equal(got.numpy(), expected)


def test_coarsened_presence_matches_the_numpy_twin_away_from_cancellation():
    """The log-space noisy-OR is the same function as `coarsen_presence`.

    Checked on moderate logits, where the numpy twin's `1 - p` is still
    accurate.  The two forms genuinely disagree once a group is nearly absent,
    by up to 0.01 in logit on the real benchmark -- that is float32 cancellation
    in the shipped path, and avoiding it in a gradient is why this form exists.
    """
    rng = np.random.default_rng(0)
    logits = rng.uniform(-6.0, 6.0, (12, N_PHASES))
    expected = coarsen_presence(1.0 / (1.0 + np.exp(-logits)))
    got = coarsen_presence_logits(torch.as_tensor(logits), group_map())
    assert np.allclose(torch.sigmoid(got).numpy(), expected, rtol=0, atol=1e-12)


def test_noisy_or_gradient_passes_gradcheck():
    """The analytic Jacobian against a central difference -- the actual proof
    for the coarsening, not an inference from the forward being right."""
    logits = torch.as_tensor(
        np.random.default_rng(0).uniform(-4.0, 4.0, (4, N_PHASES))
    ).requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda z: coarsen_presence_logits(z, group_map()), (logits,)
    )


def test_alpha_coarsening_gradient_passes_gradcheck():
    rng = np.random.default_rng(0)
    alpha = torch.as_tensor(rng.uniform(0.1, 9.0, (4, N_PHASES))).requires_grad_(True)
    support = torch.as_tensor((rng.random((4, N_PHASES)) > 0.4).astype(np.float64))
    assert torch.autograd.gradcheck(
        lambda a: coarsen_alpha_on_support(a, support, group_map()), (alpha,)
    )


def test_a_member_phase_moves_only_its_own_group():
    """A transposed or mis-keyed [21, 7] map passes gradcheck happily: the
    gradient of the wrong function is still its own gradient."""
    base = torch.full((1, N_PHASES), -1.0, dtype=torch.float64)
    before = coarsen_presence_logits(base, group_map())
    for phase, group in MODEL_TO_GROUP.items():
        bumped = base.clone()
        bumped[0, PHASE_INDEX[phase]] += 3.0
        after = coarsen_presence_logits(bumped, group_map())
        gi = GROUP_INDEX[group]
        assert after[0, gi] > before[0, gi]
        others = [i for i in range(N_GROUPS) if i != gi]
        assert torch.equal(after[0, others], before[0, others])


def test_end_to_end_gradient_matches_finite_difference():
    """Autograd against a central difference on the objective itself, so the
    pinned support and the coarsening's exact zeros are inside the check."""
    head, data = _head(dropout=0.0), _predictions()
    torch.manual_seed(0)
    adapted = AdaptedModeHead(head)
    support = adapted.pinned_support(data.latent, data.presence)
    with torch.no_grad():
        for lora in adapted.adapters:
            lora.b.normal_(0.0, 0.1)

    def loss() -> torch.Tensor:
        logits, alpha, log_w = adapted(data.latent, support)
        return recalibrator_nll(
            logits, alpha, data.presence, data.magnitude, data.numeric, log_w=log_w,
        )["joint"].mean()

    loss().backward()
    target = adapted.adapters[0].b
    analytic = target.grad[0, 0].item()

    h = 1e-6
    with torch.no_grad():
        target[0, 0] += h
    up = loss().item()
    with torch.no_grad():
        target[0, 0] -= 2 * h
    down = loss().item()
    assert (up - down) / (2 * h) == pytest.approx(analytic, rel=1e-5, abs=1e-9)


def test_only_the_adapters_receive_gradient():
    head, data = _head(dropout=0.0), _predictions()
    adapted = AdaptedModeHead(head)
    support = adapted.pinned_support(data.latent, data.presence)
    logits, alpha, log_w = adapted(data.latent, support)
    recalibrator_nll(
        logits, alpha, data.presence, data.magnitude, data.numeric, log_w=log_w,
    )["joint"].mean().backward()
    trainable = {id(p) for p in adapted.lora_parameters()}
    assert all(p.grad is not None for p in adapted.lora_parameters())
    assert all(
        p.grad is None for p in adapted.parameters() if id(p) not in trainable
    )


def test_shrinking_the_adapters_to_zero_returns_the_frozen_forecast():
    """No bias carve-out is needed here, unlike `fit_measured_head`: the fully
    shrunk model is the frozen head, which is already a forecast someone would
    make."""
    head, data = _head(dropout=0.0), _predictions()
    fitted = fit_adapted_head(data, head=head, weight_decay=1e6, epochs=300, lr=1e-2)
    assert max(p.abs().max().item() for p in fitted.lora_parameters()) < 1e-3
    at_init = AdaptedModeHead(head)
    support = at_init.pinned_support(data.latent, data.presence)
    frozen = score_predictions(*at_init(data.latent, support), data)
    assert score_adapted_head(fitted, data)["joint"].mean() == pytest.approx(
        frozen["joint"].mean(), abs=1e-6
    )


def test_the_head_stays_in_eval_mode():
    """Dropout is live on the production head; waking it would hand this arm a
    stochastic regularizer no other arm on the curve has."""
    head, data = _head(dropout=0.5), _predictions()
    adapted = AdaptedModeHead(head)
    assert not adapted.training
    support = adapted.pinned_support(data.latent, data.presence)
    first = adapted(data.latent, support)
    second = adapted(data.latent, support)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_fitting_reduces_the_loss_it_optimizes():
    """The same objective the recalibrator and the measured head minimize, so
    the three arms' levels are differenceable."""
    head, data = _head(dropout=0.0), _predictions()
    before = score_adapted_head(AdaptedModeHead(head), data)["joint"].mean()
    after = score_adapted_head(
        fit_adapted_head(data, head=head, weight_decay=1e-3, epochs=200), data
    )["joint"].mean()
    assert after < before


def test_the_injected_head_is_not_mutated():
    """One frozen head is shared across every draw and every arm of the curve,
    so adapting in place would leak one draw's fit into the next."""
    head, data = _head(dropout=0.0), _predictions()
    before = copy.deepcopy(head.state_dict())
    fit_adapted_head(data, head=head, weight_decay=1e-3, epochs=20)
    after = head.state_dict()
    assert set(before) == set(after)
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_a_mixture_head_is_rejected():
    """A K>1 head has no `magnitude_alpha`; adapting one would need a gate this
    arm deliberately does not add."""
    with pytest.raises(ValueError, match="n_components"):
        AdaptedModeHead(ModeHead(D_MODEL, hidden=8, n_components=2))


def test_a_latentless_slice_is_rejected_rather_than_silently_misfit():
    data = _predictions()
    stripped = GroupPredictions(
        logits7=data.logits7, alpha7=data.alpha7, sio2=data.sio2,
        presence=data.presence, magnitude=data.magnitude, numeric=data.numeric,
        unit=data.unit,
    )
    with pytest.raises(ValueError, match="latent"):
        fit_adapted_head(stripped, head=_head())
