"""Tripwire for the few-shot adaptation curve: a src/ change that breaks it fails here."""

import numpy as np
import torch

from cnmineral.data.phase_vocab import GROUP_INDEX, GROUPS
from cnmineral.training.real_rock_recalibrator import GroupPredictions

from explorations.real_rock_fewshot.run_curve import (
    BUDGETS,
    CURVE_TIERS,
    WEIGHT_DECAY,
    run_curve,
)
from explorations.real_rock_fewshot.run_residual_decomposition import (
    DELTA_GRID,
    interval_coverage,
    oracle_ceiling,
    shifted_beta_quantiles,
)

N_GROUPS = len(GROUPS)


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
    )


def test_weight_decay_is_recorded_for_every_tier():
    """The curve's only use of the val slice is one scalar per tier, fixed up
    front; a tier with no recorded value would silently take a default."""
    assert set(CURVE_TIERS) <= set(WEIGHT_DECAY)
    assert all(v > 0 for v in WEIGHT_DECAY.values())


def test_run_curve_produces_one_row_per_tier_budget_draw():
    train = _predictions(90, seed=0, units_of=3)
    test = _predictions(40, seed=1, units_of=4)
    budgets = (10, 30)
    baseline, rows = run_curve(
        train, test, budgets=budgets, n_draws=2, weight_decay=WEIGHT_DECAY,
    )
    assert len(rows) == len(CURVE_TIERS) * len(budgets) * 2
    assert {r["tier"] for r in rows} == set(CURVE_TIERS)
    assert np.isfinite(baseline["joint"]).all()

    for row in rows:
        assert row["n_rocks"] > 0
        assert row["n_units"] > 0
        assert np.isfinite(row["dirichlet"])
        # the paired interval must bracket its own point estimate
        assert row["recovered_lo"] <= row["recovered_dirichlet"] <= row["recovered_hi"]


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
