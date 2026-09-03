"""Tripwire for the few-shot adaptation curve: a src/ change that breaks it fails here."""

from functools import partial

import numpy as np
import pytest
import torch

from cnmineral.data.phase_vocab import GROUP_INDEX, GROUPS
from cnmineral.models.mode_head import ModeHead
from cnmineral.training.real_rock_recalibrator import GroupPredictions

from explorations.real_rock_fewshot import run_curve as run_curve_module
from explorations.real_rock_fewshot.adapted_mode_head import fit_adapted_head
from explorations.real_rock_fewshot.run_curve import (
    BUDGETS,
    CONTRAST_ARMS,
    CURVE_ARMS,
    WEIGHT_DECAY,
    crossing_budget,
    run_curve,
)
from explorations.real_rock_fewshot.run_residual_decomposition import (
    DELTA_GRID,
    interval_coverage,
    oracle_ceiling,
    shifted_beta_quantiles,
)

N_GROUPS = len(GROUPS)
D_MODEL = 16
N_RAW = 22


def _predictions(n, seed, units_of):
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


def test_weight_decay_is_recorded_for_every_arm():
    """The curve's only use of the val slice is one scalar per arm, fixed up
    front; an arm with no recorded value would silently take a default."""
    assert set(CURVE_ARMS) <= set(WEIGHT_DECAY)
    assert all(v > 0 for v in WEIGHT_DECAY.values())


SMALL_BUDGETS = (10, 30)
SMALL_DRAWS = 2


@pytest.fixture(scope="module")
def small_curve():
    """One curve shared by the assertions below -- seven arms x two budgets x
    two draws is a few hundred fits, and re-running it per test is the whole
    runtime of this file."""
    train = _predictions(90, seed=0, units_of=3)
    test = _predictions(40, seed=1, units_of=4)
    # only the adapted arm reads this; every other arm works from the
    # group-space outputs the fixture supplies directly
    head = ModeHead(D_MODEL, hidden=8, layers=3).double().eval()
    with pytest.MonkeyPatch.context() as patch:
        # the adapted arm is the only one that trains a head rather than a
        # handful of scalars, and at its shipped epoch budget it is the entire
        # runtime of this file.  What is under test here is the wiring -- that
        # the arm fits, scores, and pairs on the draw -- not its convergence.
        patch.setattr(run_curve_module, "fit_adapted_head",
                      partial(fit_adapted_head, epochs=25))
        return run_curve(
            train, test, budgets=SMALL_BUDGETS, n_draws=SMALL_DRAWS,
            weight_decay=WEIGHT_DECAY, head=head,
        )


def test_run_curve_produces_one_row_per_arm_budget_draw(small_curve):
    baseline, rows, _ = small_curve
    budgets = SMALL_BUDGETS
    assert len(rows) == len(CURVE_ARMS) * len(budgets) * 2
    assert {r["arm"] for r in rows} == set(CURVE_ARMS)
    assert np.isfinite(baseline["joint"]).all()

    for row in rows:
        assert row["n_rocks"] > 0
        assert row["n_units"] > 0
        assert np.isfinite(row["dirichlet"])
        # the paired interval must bracket its own point estimate
        assert row["recovered_lo"] <= row["recovered_dirichlet"] <= row["recovered_hi"]


def test_every_arm_at_one_draw_fits_the_identical_rocks(small_curve):
    """The whole basis for contrasting the arms is that they are paired on the
    draw; arms measured on different rocks are two overlapping bands, and at 25
    rocks which units you drew dominates the difference being reported."""
    _, rows, _ = small_curve
    for key in {(r["budget"], r["draw"]) for r in rows}:
        drawn = {(r["n_rocks"], r["n_units"]) for r in rows
                 if (r["budget"], r["draw"]) == key}
        assert len(drawn) == 1


def test_contrasts_are_reported_against_the_measured_arm(small_curve):
    _, _, contrasts = small_curve
    assert {c["arm"] for c in contrasts} == set(CURVE_ARMS) - {"measured"}
    assert len(contrasts) == len(CONTRAST_ARMS) * len(SMALL_BUDGETS) * SMALL_DRAWS
    for c in contrasts:
        assert c["lo"] <= c["minus_measured_joint"] <= c["hi"]


def test_crossing_reports_none_when_the_arms_never_cross():
    """"They never cross inside the swept range" is an answer, not a failure --
    it is the exchange rate reading "more than the budget we swept"."""
    rows = [{"arm": a, "budget": b, "n_rocks": b, "joint": j}
            for b in (10, 30)
            for a, j in (("matrix", 1.0), ("measured", 5.0))]
    assert crossing_budget(rows)["rocks"] is None


def test_budgets_are_ascending_so_the_curve_reads_left_to_right():
    assert list(BUDGETS) == sorted(BUDGETS)


def test_shifted_beta_quantiles_identity_matches_share_interval():
    """The oracle at (k=1, delta=0) reproduces the shipped interval, so a
    ceiling can never sit below the coverage it starts from."""
    from cnmineral.eval.real_rock import share_interval

    rng = np.random.default_rng(3)
    alpha = rng.uniform(0.5, 8.0, (50, N_GROUPS))
    mask = rng.uniform(size=(50, N_GROUPS)) < 0.6
    mask[:, :2] = True
    lo_si, hi_si = share_interval(alpha, mask.astype(float))
    on = alpha * mask
    for g in range(N_GROUPS):
        a = on[:, g]
        b = np.clip(on.sum(-1) - a, 1e-9, None)
        lo, hi = shifted_beta_quantiles(a, b)
        m = mask[:, g]
        np.testing.assert_allclose(lo[m], lo_si[m, g], rtol=1e-12)
        np.testing.assert_allclose(hi[m], hi_si[m, g], rtol=1e-12)


def test_width_scale_preserves_the_beta_mean_and_widens_coverage():
    """The precision knob is mean-preserving and monotone in interval width,
    which is what makes it a pure width knob."""
    a, b = np.array([2.0]), np.array([6.0])
    widths = []
    for k in (0.1, 1.0, 10.0):
        lo, hi = shifted_beta_quantiles(a, b, k=k)
        widths.append(float((hi - lo)[0]))
    assert widths[0] > widths[1] > widths[2]

    # at huge precision the interval pinches onto the preserved mean
    lo, hi = shifted_beta_quantiles(a, b, k=1e6)
    assert abs(lo[0] - 0.25) < 1e-3 and abs(hi[0] - 0.25) < 1e-3

    rng = np.random.default_rng(0)
    x = rng.beta(2.0, 6.0, 4000)
    an, bn = np.full(4000, 2.0), np.full(4000, 6.0)
    assert abs(interval_coverage(an, bn, x) - 0.8) < 0.03
    assert (interval_coverage(an, bn, x, k=0.2)
            > interval_coverage(an, bn, x)
            > interval_coverage(an, bn, x, k=5.0))


def test_oracle_separates_width_from_location():
    """A pure mean shift on a small-share group is invisible to the width-only
    oracle and recovered by the 2-d one -- the decomposition's whole claim."""
    rng = np.random.default_rng(7)
    n, nu, m0, shift = 1500, 6.0, 0.05, 1.5
    m1 = 1.0 / (1.0 + np.exp(-(np.log(m0 / (1 - m0)) + shift)))
    x = rng.beta(nu * m1, nu * (1 - m1), n)
    a = np.full(n, nu * m0)
    b = np.full(n, nu * (1 - m0))

    width_only = oracle_ceiling(a, b, x)
    both = oracle_ceiling(a, b, x, delta_grid=DELTA_GRID)
    assert width_only["cov"] < 0.6
    assert both["cov"] >= 0.78
    assert both["delta"] > 0.5
