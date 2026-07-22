import numpy as np

import pytest

from cnmineral.eval.summary_cards import (
    pick_ambiguous_bulk,
    pick_bulk_by_nll,
    sample_assemblages,
    top_support_patterns,
)


def test_sample_assemblages_rows_on_simplex():
    rng = np.random.default_rng(0)
    p = np.array([0.9, 0.5, 0.1, 0.0])
    alpha = np.array([2.0, 3.0, 1.0, 5.0])
    alpha_fn = lambda supports: np.tile(alpha, (len(supports), 1))
    modes = sample_assemblages(p, alpha_fn, 200, rng)
    sums = modes.sum(1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-9)
    assert (modes[:, 3] == 0).all()  # p=0 phase never enters a support
    modes2 = sample_assemblages(
        p, alpha_fn, 200, np.random.default_rng(0)
    )
    np.testing.assert_array_equal(modes, modes2)


def test_sample_assemblages_alpha_fn_sees_unique_supports():
    rng = np.random.default_rng(0)
    p = np.array([1.0, 0.5, 0.0])
    seen = []

    def alpha_fn(supports):
        seen.append(supports)
        return supports * 100.0 + 1.0

    modes = sample_assemblages(p, alpha_fn, 100, rng)
    assert len(seen) == 1
    uniq = seen[0]
    assert len(np.unique(uniq, axis=0)) == len(uniq)  # deduped patterns only
    assert set(map(tuple, uniq)) == {(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)}
    # alpha ~100 on the two-phase support makes its draws near-equal shares;
    # a shared bulk-level alpha could not produce that split
    two_phase = modes[modes[:, 1] > 0]
    assert len(two_phase)
    np.testing.assert_allclose(two_phase[:, 0], 0.5, atol=0.2)


def _dataset(pattern_counts_per_bulk):
    """presence/row_bulk/rung arrays from per-bulk lists of (pattern, count)."""
    presence, row_bulk, rung = [], [], []
    for b, patterns in enumerate(pattern_counts_per_bulk):
        for pattern, count in patterns:
            for _ in range(count):
                presence.append(pattern)
                row_bulk.append(b)
                rung.append(50)
    return (
        np.array(presence, dtype=np.float32),
        np.array(row_bulk),
        np.array(rung),
    )


def test_pick_ambiguous_bulk_prefers_even_split():
    a, b = [1, 1, 0, 0], [0, 0, 1, 1]
    presence, row_bulk, rung = _dataset([
        [(a, 10)],              # unimodal
        [(a, 9), (b, 1)],       # 90/10
        [(a, 5), (b, 5)],       # 50/50 -- winner
    ])
    bulk_ids = np.array([100, 200, 300])
    bulk, score = pick_ambiguous_bulk(presence, row_bulk, rung, bulk_ids, 50)
    assert bulk == 300
    assert score == 0.5


def test_pick_ambiguous_bulk_unimodal_fallback():
    a = [1, 0, 1, 0]
    presence, row_bulk, rung = _dataset([[(a, 8)], [(a, 4)]])
    bulk_ids = np.array([7, 9])
    bulk, score = pick_ambiguous_bulk(presence, row_bulk, rung, bulk_ids, 50)
    assert bulk == 7
    assert score == 0.0


def test_pick_bulk_by_nll_median_and_best():
    #                bulk 0 (mean 5)   bulk 1 (mean 1)   bulk 2 (mean 3)
    nll = np.array([4.0, 6.0,          1.0, 1.0,          3.0, 3.0,
                    99.0])  # bulk 0 row at the wrong rung, must be ignored
    row_bulk = np.array([0, 0, 1, 1, 2, 2, 0])
    rung = np.array([50, 50, 50, 50, 50, 50, 30])
    assert pick_bulk_by_nll(nll, row_bulk, rung, 50, "best") == 1
    assert pick_bulk_by_nll(nll, row_bulk, rung, 50, "median") == 2
    with pytest.raises(ValueError, match="pick must be"):
        pick_bulk_by_nll(nll, row_bulk, rung, 50, "worst")
    with pytest.raises(ValueError, match="no bulk has rows"):
        pick_bulk_by_nll(nll, row_bulk, rung, 99, "best")


def test_top_support_patterns_groups_and_shares():
    sampled = np.zeros((10, 3))
    sampled[:6, [0, 1]] = [0.7, 0.3]
    sampled[6:, [1, 2]] = [0.4, 0.6]
    out = top_support_patterns(sampled, k=2)
    assert out[0]["share"] == 0.6
    np.testing.assert_array_equal(out[0]["pattern"], [True, True, False])
    np.testing.assert_allclose(out[0]["mean_modes"], [0.7, 0.3, 0.0])
    assert out[1]["share"] == 0.4
    np.testing.assert_allclose(out[1]["mean_modes"], [0.0, 0.4, 0.6])
