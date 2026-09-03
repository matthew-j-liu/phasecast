"""Recalibrator fitting: the NaN guard, unit-clustered inference, honest sampling."""

import numpy as np
import pytest
import torch

from cnmineral.data.phase_vocab import (
    GROUP_INDEX, GROUPS, N_PHASES, PHASE_INDEX,
)
from cnmineral.eval.real_rock import coarsen_presence, real_rock_nll
from cnmineral.training.real_rock_recalibrator import (
    GroupPredictions,
    recalibrator_nll,
    draw_units,
    fit_recalibrator,
    lift_group_presence_to_phases,
    paired_bootstrap,
    score_recalibrator,
)

N_GROUPS = len(GROUPS)
D_MODEL = 16
N_RAW = 22


def _predictions(n=48, seed=0, units_of=3, zero_group="ol"):
    """A synthetic benchmark slice with a structural zero and repeated units."""
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(0.5, 8.0, (n, N_GROUPS))
    alpha[:, GROUP_INDEX[zero_group]] = 0.0
    numeric = np.zeros((n, N_GROUPS), dtype=bool)
    numeric[:, [GROUP_INDEX["fsp"], GROUP_INDEX["px"], GROUP_INDEX["amp"]]] = True
    magnitude = np.zeros((n, N_GROUPS))
    draws = rng.dirichlet(np.ones(3), n)
    magnitude[numeric] = draws.reshape(-1)
    presence = numeric.astype(np.float64)
    return GroupPredictions(
        logits7=torch.as_tensor(rng.normal(0, 1.5, (n, N_GROUPS))),
        alpha7=torch.as_tensor(alpha),
        sio2=torch.as_tensor(rng.uniform(50, 78, n)),
        presence=torch.as_tensor(presence),
        magnitude=torch.as_tensor(magnitude),
        numeric=torch.as_tensor(numeric.astype(np.float64)),
        unit=np.array([f"unit-{i // units_of}" for i in range(n)]),
        latent=torch.as_tensor(rng.normal(0, 1.0, (n, D_MODEL))),
        raw=torch.as_tensor(rng.normal(0, 1.0, (n, N_RAW))),
    )


def test_off_mask_zero_concentrations_do_not_make_the_loss_nan():
    """`mask * lgamma(0)` is `0 * inf` = NaN, which is a dead fit, not a bad one.

    Coarsening produces exact zeros wherever a group has no member on the
    support. The guard lives at this call site rather than in the shared
    `masked_dirichlet_nll`, which other heads use with strictly positive alpha.
    """
    data = _predictions()
    assert (data.alpha7[:, GROUP_INDEX["ol"]] == 0).all()
    out = recalibrator_nll(data.logits7, data.alpha7, data.presence, data.magnitude,
                      data.numeric)
    for term in ("bernoulli", "dirichlet", "joint"):
        assert torch.isfinite(out[term]).all()


def test_gradients_survive_the_nan_guard():
    """The guard must not detach the entries that are actually scored."""
    data = _predictions()
    alpha = data.alpha7.clone().requires_grad_(True)
    recalibrator_nll(data.logits7, alpha, data.presence, data.magnitude,
                data.numeric)["joint"].mean().backward()
    scored = data.numeric > 0
    assert torch.isfinite(alpha.grad).all()
    assert (alpha.grad[scored] != 0).any()
    # an off-mask group contributes nothing, so it must draw no gradient
    assert (alpha.grad[:, GROUP_INDEX["ol"]] == 0).all()


def test_torch_objective_matches_the_shipped_numpy_eval():
    """The fit's loss and the ledger's metric must be the same number.

    If they drift, "nats recovered" compares a recalibrator against a baseline
    measured on a different scale.
    """
    data = _predictions()
    p7 = torch.sigmoid(data.logits7).numpy()
    want = real_rock_nll(p7, data.alpha7.numpy(), data.presence.numpy(),
                         data.magnitude.numpy(), data.numeric.numpy().astype(bool))
    got = score_recalibrator(None, data)
    for term in ("bernoulli", "dirichlet", "joint"):
        assert np.allclose(want[term], got[term], atol=1e-9)


def test_fitting_reduces_the_loss_it_optimizes():
    data = _predictions()
    before = score_recalibrator(None, data)["joint"].mean()
    recalibrator = fit_recalibrator(data, tier="vector", weight_decay=1e-3, epochs=200)
    assert score_recalibrator(recalibrator, data)["joint"].mean() < before


def test_bootstrap_resamples_units_not_rocks():
    """Rocks inside a unit are near-identical chemistry.

    Resampling rocks would treat them as independent draws and understate the
    interval, so the interval must widen when the same rocks are declared to
    belong to fewer units.
    """
    rng = np.random.default_rng(0)
    n = 120
    baseline = rng.normal(10, 5, n)
    adapted = baseline - rng.normal(2, 5, n)
    per_rock = np.array([f"u{i}" for i in range(n)])
    clustered = np.array([f"u{i // 10}" for i in range(n)])

    wide = paired_bootstrap(baseline, adapted, clustered, n_boot=400)
    narrow = paired_bootstrap(baseline, adapted, per_rock, n_boot=400)
    assert wide["hi"] - wide["lo"] > narrow["hi"] - narrow["lo"]
    assert wide["n_units"] == 12 and narrow["n_units"] == n
    assert wide["mean"] == pytest.approx(narrow["mean"])


def test_lifting_the_group_correction_hits_the_target_noisy_or():
    """The 21-phase support must reproduce the group probability the recalibrator set.

    Shifting each member's logit by the group's logit shift -- the obvious
    thing -- is badly wrong, since the noisy-OR is not linear in logits: a
    group at 0.9999 corrected to 0.84 gives a -7.5 shift that drives both
    members to ~0, and the group vanishes from every sampled support. Caught by
    a scenario view whose top assemblages had no feldspar in a rock that is
    half feldspar.
    """
    data = _predictions(n=24)
    recalibrator = fit_recalibrator(data, tier="matrix", weight_decay=1e-2, epochs=100)
    rng = np.random.default_rng(0)
    p21 = rng.uniform(0.01, 0.99, (len(data), N_PHASES))
    # a group pinned near certainty is where the logit-shift version breaks
    p21[:, PHASE_INDEX["plag"]] = 0.995
    p21[:, PHASE_INDEX["afs"]] = 0.99

    lifted = lift_group_presence_to_phases(recalibrator, data.logits7, p21)
    with torch.no_grad():
        target = torch.sigmoid(
            recalibrator(data.logits7, data.alpha7)[0]
        ).numpy()
    assert np.allclose(coarsen_presence(lifted), target, atol=1e-6)
    assert ((lifted >= 0) & (lifted <= 1)).all()

    # phases no group claims are untouched
    for phase in ("g", "ru", "cor"):
        np.testing.assert_allclose(lifted[:, PHASE_INDEX[phase]],
                                   p21[:, PHASE_INDEX[phase]])


def test_draws_sample_whole_units_without_overshooting():
    """The curve's x-axis counts rocks, but rocks are harvested an eruption at a time."""
    data = _predictions(n=60, units_of=5)
    rng = np.random.default_rng(0)
    mask = draw_units(data, 20, rng)
    drawn = data.unit[mask]
    assert 0 < mask.sum() <= 20 * 1.2
    # every rock of a drawn unit comes along; none is split across the boundary
    for unit in np.unique(drawn):
        assert mask[data.unit == unit].all()


def test_a_unit_larger_than_the_budget_is_skipped_not_taken():
    """Otherwise one 66-rock unit decides the low end of the curve."""
    data = _predictions(n=60, units_of=30)
    mask = draw_units(data, 5, np.random.default_rng(0))
    assert mask.sum() == 0
