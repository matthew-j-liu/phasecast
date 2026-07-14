import numpy as np
import pytest

from cnmineral.data.condition_proposal import (
    P_HI,
    P_LO,
    W_CAP,
    draw_conditions,
    nl_rhyolite_solubility,
    q_density,
    w_max,
)


def test_standard_state_anchor():
    # N&L rhyolite standard state: 3.45 wt% dissolved at 799 bar / 850 C
    assert nl_rhyolite_solubility(0.799) == pytest.approx(3.45, abs=0.1)


def test_matches_reference_volatilecalc_across_live_range():
    # Values round-tripped through the DJRgeoscience VolatileCalc port (itself
    # checked against the USGS Excel original): sp('rhyolite', [S(P), 0, 850])
    # returned P within solver tolerance at 14 pressures, 2026-07-14.
    for p_kbar, wt in [(0.5, 2.604), (1.0, 3.880), (1.5, 4.940),
                       (2.0, 5.924), (2.5, 6.897), (3.0, 7.893)]:
        assert nl_rhyolite_solubility(p_kbar) == pytest.approx(wt, abs=0.005)


def test_solubility_monotonic_in_pressure():
    s = [nl_rhyolite_solubility(p) for p in np.linspace(P_LO, 3.0, 15)]
    assert np.all(np.diff(s) > 0)


def test_cap_binds_above_crossover():
    assert w_max(2.5) < W_CAP
    assert w_max(3.5) == W_CAP
    assert w_max(P_HI) == W_CAP


def test_draws_inside_support():
    p, w, q = draw_conditions(512, np.random.default_rng(0))
    assert np.all((p >= P_LO) & (p <= P_HI))
    assert np.all((w >= 0) & (w <= w_max(p) + 1e-12))
    assert np.all(q > 0)


def test_q_matches_closed_form():
    p, w, q = draw_conditions(512, np.random.default_rng(1))
    assert np.allclose(q, 1.0 / ((P_HI - P_LO) * w_max(p)))
    assert np.allclose(q, q_density(p, w))


def test_q_zero_outside_support():
    assert q_density(1.0, 7.9) == 0.0        # above S(1 kbar) ~ 3.9
    assert q_density(6.0, 8.5) == 0.0        # above the cap
    assert q_density(P_LO - 0.1, 1.0) == 0.0
    assert q_density(6.0, 7.9) > 0.0


def test_lhs_stratifies_p_marginal():
    n = 128
    p, _, _ = draw_conditions(n, np.random.default_rng(2))
    strata = np.floor((p - P_LO) / (P_HI - P_LO) * n).astype(int)
    assert len(np.unique(strata)) == n


def test_importance_weights_recover_support_area():
    # E_q[1/q] = area of the bounded region; checks sampler and density agree
    p, w, q = draw_conditions(4096, np.random.default_rng(3))
    pg = np.linspace(P_LO, P_HI, 2001)
    area = np.trapz(w_max(pg), pg)
    assert np.mean(1.0 / q) == pytest.approx(area, rel=0.03)


def test_draws_deterministic_under_seed():
    a = draw_conditions(64, np.random.default_rng(7))
    b = draw_conditions(64, np.random.default_rng(7))
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
