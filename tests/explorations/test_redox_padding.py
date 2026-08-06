"""Tripwire test for the `redox_padding` exploration.

Locks the pure pieces: the latent-geometry statistics against a synthetic
case whose answer is known by construction, and the missingness feature
build. Both run scripts need the real encoder and corpus, so the end-to-end
paths are exercised by their CLI runs, not here.
"""

import numpy as np

from explorations.redox_padding.run_geometry import RATIO_GRID, geometry
from explorations.redox_padding.run_missingness import input_features
from cnmineral.data.tokens import N_TOKENS, N_TOKENS_REDOX, REDOX_TOKEN_ID, VOCAB


def _straight_sweep(n: int = 40, d: int = 8, seed: int = 0):
    """Bulks whose latents move along one shared axis as the ratio rises.

    Each bulk sits at its own random origin and travels a fixed direction
    proportional to the ratio, so the nearest grid point to any query is
    exactly the ratio that generated it.
    """
    rng = np.random.default_rng(seed)
    origin = rng.normal(0, 10, size=(n, d)).astype(np.float32)
    direction = rng.normal(0, 1, size=d).astype(np.float32)
    swept = origin[:, None, :] + RATIO_GRID[None, :, None] * direction[None, None, :]
    return origin, direction, swept.astype(np.float32)


def test_padded_on_the_curve_recovers_its_ratio() -> None:
    """A padded latent placed exactly at grid ratio 0.30 must be reported
    as nearest to 0.30, at zero distance from the curve."""
    origin, direction, swept = _straight_sweep()
    padded = origin + 0.30 * direction
    r_true = np.full(len(padded), 0.30, dtype=np.float32)

    stats = geometry(padded, swept, r_true)

    assert stats["r_star"]["p50"] == 0.30
    assert stats["r_star_mode"] == 0.30
    assert stats["padded_to_curve_over_span"]["p90"] < 1e-5


def test_padded_tracking_the_rock_shows_up_as_correlation() -> None:
    """When each bulk's padded latent sits at that bulk's own ratio, the
    reported r_star-vs-r_true correlation must be ~1 -- the statistic that
    separates a chemistry-conditional guess from one global default."""
    origin, direction, swept = _straight_sweep()
    rng = np.random.default_rng(1)
    r_true = rng.choice(RATIO_GRID, size=len(origin)).astype(np.float32)
    padded = origin + r_true[:, None] * direction

    stats = geometry(padded, swept, r_true)

    assert stats["r_star_vs_r_true_pearson"] > 0.99
    assert stats["r_star_vs_r_true_spearman"] > 0.99


def test_constant_offset_reads_as_a_shared_direction() -> None:
    """A padded latent displaced off the curve by one common vector must
    report near-parallel displacements, and the offset correction must put
    it back on the curve."""
    origin, direction, swept = _straight_sweep()
    rng = np.random.default_rng(2)
    offset = rng.normal(0, 5, size=swept.shape[-1]).astype(np.float32)
    padded = origin + 0.30 * direction + offset
    r_true = np.full(len(padded), 0.30, dtype=np.float32)

    stats = geometry(padded, swept, r_true)

    assert stats["displacement_direction_cosine_mean"] > 0.99
    assert stats["offset_corrected"]["padded_to_curve_over_span"]["p90"] < 1e-5


def test_input_features_exclude_the_redox_slot() -> None:
    """The missingness arms must never see the target's own slot."""
    n = 5
    src = np.zeros((n, N_TOKENS_REDOX), dtype=np.int64)
    frac = np.zeros((n, N_TOKENS_REDOX), dtype=np.float32)
    src[:, :3] = 14
    frac[:, :3] = 0.5
    src[:, -1] = REDOX_TOKEN_ID
    frac[:, -1] = 0.42

    x, names = input_features(src, frac)

    assert x.shape == (n, 2 * N_TOKENS)
    assert len(names) == 2 * N_TOKENS
    assert not np.isclose(x, 0.42).any()
    assert names[0] == f"{VOCAB[0][0]}_value"
    assert names[N_TOKENS] == f"{VOCAB[0][0]}_present"
