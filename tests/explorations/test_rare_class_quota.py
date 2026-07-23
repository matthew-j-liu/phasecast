"""Tripwire test for the `rare_class_quota` exploration.

Locks the pure pieces: the take-all-tail hybrid allocation (the rejected
candidate whose seam justified alpha=0.5) and the rank-frequency figure.
`eligible_pool`/`main` need the gard2019 parquet and are exercised by the
CLI run, not here. `expansion_quotas` itself is production code, tested in
tests/data/test_volcanic_subset.py.
"""
import pandas as pd

from explorations.rare_class_quota.quota_analysis import (
    HYBRID_TAIL_CUTOFF,
    hybrid_quotas,
    powerlaw_figure,
)


def test_hybrid_takes_all_tail_then_smooths_the_rest():
    idx = ["big", "mid", "rare"]
    pool = pd.Series([10000, 2000, 100], index=idx)
    have = pd.Series([100, 30, 10], index=idx)
    avail = pd.Series([9900, 1970, 90], index=idx)
    assert avail["rare"] <= HYBRID_TAIL_CUTOFF < avail["mid"]
    q = hybrid_quotas(pool, have, avail, budget=1000)
    assert q["rare"] == avail["rare"]
    assert q.sum() == 1000
    assert (q <= avail).all()


def test_powerlaw_figure_fits_exact_law(tmp_path):
    rank = pd.Series([1000, 250, 1000 / 9, 62.5], index=list("abcd"))
    slope, r2 = powerlaw_figure(rank, tmp_path / "fit.png")
    assert abs(slope - (-2.0)) < 1e-6
    assert r2 > 0.999
    assert (tmp_path / "fit.png").stat().st_size > 0
