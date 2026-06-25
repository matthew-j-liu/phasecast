"""Tripwire test for the `latent_anisotropy` exploration.

Locks the anisotropy metric contracts on synthetic matrices with known geometry
(isotropic -> effective rank ~ d and zero cone; rank-1 -> effective rank ~ 1 and
a cone) and doubles as a tripwire: a `src/`-side break in the exploration's reuse
fails at import.  The orchestrator `run_anisotropy` needs a real checkpoint, so
it is exercised by the CLI run, not here.
"""

import numpy as np

from explorations.latent_anisotropy.run_anisotropy import (
    first_order_anisotropy,
    second_order_anisotropy,
)


def test_isotropic_reference_is_nearly_isotropic() -> None:
    rng = np.random.default_rng(0)
    d = 64
    X = rng.standard_normal((8000, d)).astype(np.float32)

    so = second_order_anisotropy(X)
    assert so["dim"] == d
    # equal eigenvalues -> effective rank fills the space (finite-sample slack).
    assert so["effective_rank_frac"] > 0.9
    assert so["top1_var_frac"] < 0.05

    fo = first_order_anisotropy(X)
    # zero-mean, high-d -> random pairs are near-orthogonal, no shared direction.
    assert abs(fo["mean_pair_cosine"]) < 0.02
    assert fo["mean_vector_dominance"] < 0.05


def test_rank_one_matrix_collapses_to_effective_rank_one() -> None:
    rng = np.random.default_rng(1)
    direction = rng.standard_normal(32)
    scores = rng.standard_normal((4000, 1))
    X = (scores * direction).astype(np.float32)

    so = second_order_anisotropy(X)
    assert so["effective_rank"] < 1.05
    assert so["top1_var_frac"] > 0.99


def test_mean_offset_creates_first_order_cone_removed_by_centering() -> None:
    """A large shared offset is pure first-order anisotropy: cosine sees it,
    centering removes it, the covariance spectrum never does."""
    rng = np.random.default_rng(2)
    d = 32
    base = rng.standard_normal((6000, d)).astype(np.float32)
    offset = 20.0 * np.ones(d, dtype=np.float32)
    X = base + offset

    fo = first_order_anisotropy(X)
    fo_centered = first_order_anisotropy(X - X.mean(axis=0))
    assert fo["mean_pair_cosine"] > 0.8
    assert abs(fo_centered["mean_pair_cosine"]) < 0.05

    # the offset is a mean shift, not extra covariance variance, so the centered
    # spectrum stays full-rank.
    assert second_order_anisotropy(X)["effective_rank_frac"] > 0.9
