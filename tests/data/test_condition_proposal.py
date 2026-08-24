import numpy as np
import pytest

from cnmineral.data.condition_proposal import (
    P_HI,
    P_LO,
    W_CAP,
    ProposalKnobs,
    bulk_proposal,
    draw_conditions,
    draw_conditions_traced,
    nl_rhyolite_solubility,
    q_density,
    setting_posterior,
    w_max,
)

ARC_TRACES = {"th_ppm": 5.0, "nb_ppm": 8.0, "yb_ppm": 2.0}
DEPLETED_TRACES = {"th_ppm": 0.12, "nb_ppm": 2.3, "yb_ppm": 3.0}


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


def test_setting_posterior_sums_to_one_and_orders():
    w_arc, rung = setting_posterior(ARC_TRACES, 0.15)
    assert rung == "th_nb_yb"
    assert sum(w_arc) == pytest.approx(1.0)
    assert w_arc[0] > 0.9
    w_dep, _ = setting_posterior(DEPLETED_TRACES, 0.15)
    assert sum(w_dep) == pytest.approx(1.0)
    assert w_dep[1] == max(w_dep)


def test_cascade_rung_selection():
    _, rung = setting_posterior({"tio2": 1.5, "v_ppm": 300.0}, 0.15)
    assert rung == "ti_v"
    w, rung = setting_posterior({"nb_ppm": 5.0, "y_ppm": 30.0}, 0.15)
    assert rung == "nb_y"
    assert w[3] > 0.9          # subalkaline Nb/Y mass falls to the class tables
    w, rung = setting_posterior({"th_ppm": np.nan, "yb_ppm": 2.0}, 0.15)
    assert rung is None
    assert w == (0.0, 0.0, 0.0, 1.0)


def test_traced_draws_inside_support():
    bp = bulk_proposal("andesite", ARC_TRACES)
    p, w, q = bp.draw(512, np.random.default_rng(4))
    assert np.all((p >= P_LO) & (p <= P_HI))
    assert np.all((w >= 0) & (w <= w_max(p) + 1e-12))
    assert np.all(q > 0)


def test_traced_q_matches_closed_form():
    bp = bulk_proposal("basalt", DEPLETED_TRACES)
    p, w, q = bp.draw(512, np.random.default_rng(5))
    assert np.allclose(q, bp.q(p, w))


def test_floor_keeps_old_support():
    # anywhere the broad proposal has support, the traced q must be > 0
    bp = bulk_proposal("rhyolite", ARC_TRACES)
    p, w, _ = draw_conditions(512, np.random.default_rng(6))
    assert np.all(bp.q(p, w) >= bp.floor * q_density(p, w))
    assert np.all(bp.q(p, w) > 0)


def test_traced_importance_weights_recover_area():
    # support is the full broad region thanks to the floor, so E_q[1/q]
    # recovers the same area as the broad sampler's test above
    bp = bulk_proposal("andesite", ARC_TRACES)
    p, w, q = bp.draw(8192, np.random.default_rng(8))
    pg = np.linspace(P_LO, P_HI, 2001)
    area = np.trapz(w_max(pg), pg)
    assert np.mean(1.0 / q) == pytest.approx(area, rel=0.05)


def test_wet_box_p_floor():
    # arc-felsic box has w_lo = 3: no draws where solubility leaves no room
    bp = bulk_proposal("rhyolite", ARC_TRACES, ProposalKnobs(eps=0.0))
    p, w, q = bp.draw(512, np.random.default_rng(9))
    assert np.all(w <= w_max(p) + 1e-12)
    assert np.all(q > 0)


def test_fallback_covers_rosters():
    import pandas as pd
    from pathlib import Path

    corpus = Path(__file__).resolve().parents[2] / "data" / "corpus"
    if not (corpus / "train_bulks.csv").exists():
        pytest.skip("data/corpus roster tables absent")
    for roster in ("train_bulks.csv", "valtest_bulks.csv"):
        for qap in pd.read_csv(corpus / roster).qap_name.dropna().unique():
            bp = bulk_proposal(qap, {})
            assert bp.tier == "fallback"
            assert len(bp.boxes) == 1


def test_traced_deterministic_under_seed():
    a = draw_conditions_traced("andesite", ARC_TRACES, 64, np.random.default_rng(7))
    b = draw_conditions_traced("andesite", ARC_TRACES, 64, np.random.default_rng(7))
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
