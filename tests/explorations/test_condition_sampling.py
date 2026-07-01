"""Tripwire test for the `condition_sampling` exploration.

Locks the proposal-sampling contract: draws stay inside the declared `(P, H2O)`
envelope and the Latin-Hypercube marginals are stratified (unbiased coverage).
The MAGEMin/Julia legs need a live thermodynamic backend, so they are exercised
by the CLI run, not here.
"""

import numpy as np

from explorations.condition_sampling.gen_inputs import (
    P_HI,
    P_LO,
    W_HI,
    W_LO,
    grid,
    lhs,
    scale,
)


def test_lhs_draws_inside_envelope_and_stratified():
    rng = np.random.default_rng(0)
    n = 64
    P, W = scale(lhs(n, rng))
    assert P.shape == W.shape == (n,)
    assert (P >= P_LO).all() and (P <= P_HI).all()
    assert (W >= W_LO).all() and (W <= W_HI).all()
    # one draw per stratum on each axis -> exactly one point in each of n bins
    for v, lo, hi in [(P, P_LO, P_HI), (W, W_LO, W_HI)]:
        bins = np.floor((v - lo) / (hi - lo) * n).clip(0, n - 1)
        assert len(np.unique(bins)) == n


def test_grid_cell_centered_inside_envelope():
    P, W = grid(4, 8)
    assert P.shape == W.shape == (32,)
    assert (P > P_LO).all() and (P < P_HI).all()
    assert (W > W_LO).all() and (W < W_HI).all()
