"""Tripwire tests for the equilibrium-multimodality exploration.

Locks the two pieces of Python logic the verdict rides on: the near-solidus
endpoint pick (highest-T converged point below the liq threshold, else attrition)
and the phase-identity multimodality classifier (a silicate identity that flips
in/out across draws vs an assemblage where only proportions move). The MAGEMin
leg needs a live backend and is exercised by the CLI run, not here.
"""
import numpy as np
import pandas as pd

from explorations.equilibrium_multimodality.driver import P_HI, P_LO, W_HI, W_LO, draw_conditions
from explorations.equilibrium_multimodality.analyze import classify_bulk, pick_endpoints


def test_lhs_conditions_inside_box_and_stratified():
    P, W = draw_conditions(64, np.random.default_rng(0))
    assert (P >= P_LO).all() and (P <= P_HI).all()
    assert (W >= W_LO).all() and (W <= W_HI).all()
    for v, lo, hi in [(P, P_LO, P_HI), (W, W_LO, W_HI)]:
        bins = np.floor((v - lo) / (hi - lo) * 64).clip(0, 63)
        assert len(np.unique(bins)) == 64   # one draw per stratum on each axis


def _pts(rows):
    """rows: list of (bulk, draw, T, liq, ok, phases-dict)."""
    recs = [dict(bulk=b, draw=d, T_celsius=T, liq=liq, ok=ok, phases=ph,
                 qap_name="andesite", series="subalkaline", sio2_bulk=58.0,
                 P_kbar=5.0, H2O=3.0) for (b, d, T, liq, ok, ph) in rows]
    return pd.DataFrame(recs)


def test_endpoint_picks_highest_T_below_tau():
    ph = {"fsp": 0.5, "cpx": 0.3}
    pts = _pts([(1, 0, 1000, 0.50, True, {**ph, "liq": 0.5}),
                (1, 0, 900, 0.03, True, {**ph, "liq": 0.03}),
                (1, 0, 800, 0.005, True, {**ph, "liq": 0.005})])
    ep1 = pick_endpoints(pts, tau=0.01).iloc[0]
    ep5 = pick_endpoints(pts, tau=0.05).iloc[0]
    assert ep1.T_end == 800 and not ep1.attrition
    assert ep5.T_end == 900 and not ep5.attrition


def test_endpoint_attrition_when_never_below_tau():
    pts = _pts([(1, 0, 900, 0.30, True, {"fsp": 0.5, "liq": 0.30}),
                (1, 0, 800, 0.20, True, {"fsp": 0.6, "liq": 0.20})])
    ep = pick_endpoints(pts, tau=0.01).iloc[0]
    assert ep.attrition and ep.T_end == 800   # coldest, flagged


def test_endpoint_ignores_nonconverged_points():
    ph = {"fsp": 0.6}
    pts = _pts([(1, 0, 950, 0.001, False, {**ph, "liq": 0.001}),  # garbage, must be skipped
                (1, 0, 850, 0.004, True, {**ph, "liq": 0.004})])
    ep = pick_endpoints(pts, tau=0.01).iloc[0]
    assert ep.T_end == 850


def _ep_bulk(asms):
    return pd.DataFrame([dict(asm=a, attrition=False, liq_end=0.005, T_end=800.0) for a in asms])


def test_classifier_flags_amp_opx_identity_switch():
    # half the draws crystallize amphibole, half orthopyroxene (anti-correlated)
    asms = ([{"fsp": 0.5, "cpx": 0.3, "amp": 0.2}] * 6
            + [{"fsp": 0.5, "cpx": 0.3, "opx": 0.2}] * 6)
    v = classify_bulk(_ep_bulk(asms))
    assert v["phase_identity_multimodal"] and v["amp_opx_switch"]
    assert v["amp_only"] == 6 and v["opx_only"] == 6


def test_classifier_proportion_only_is_not_multimodal():
    # identical identity set on every draw; only the modal split moves
    rng = np.random.default_rng(1)
    asms = [{"fsp": 0.4 + 0.1 * rng.random(), "cpx": 0.3, "amp": 0.2} for _ in range(12)]
    v = classify_bulk(_ep_bulk(asms))
    assert not v["phase_identity_multimodal"]
    assert not v["amp_opx_switch"]
    assert v["amp_freq"] == 1.0   # amphibole present on every path -> not a flip
