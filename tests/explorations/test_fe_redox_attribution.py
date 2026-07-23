"""Tripwire test for the `fe_redox_attribution` exploration.

Locks the pure pieces on synthetic data: the anhydrous per-bulk basis (H2O
dropped, renormalized, bulk-constancy enforced), the excess-O removal
renormalizing so the deficit can't leak the redox column back, the one
provenance-per-bulk join contract, group slicing, and the bootstrap CI. The
orchestrator `run_measurements` needs a real checkpoint, so it is exercised by
the CLI run, not here.
"""

import numpy as np
import pandas as pd
import pytest

from cnmineral.data.phase_vocab import N_PHASES
from explorations.fe_redox_attribution.run_measurements import (
    EXCESS_O_COL,
    anhydrous_bulk_basis,
    bootstrap_delta_ci,
    group_slices,
    provenance_by_bulk,
    without_excess_o,
)


def _rows_from_anhydrous(anhydrous: np.ndarray, h2o: np.ndarray) -> np.ndarray:
    """Build normalized bulk_11 rows from anhydrous [n, 10] + per-row H2O."""
    scaled = anhydrous * (1.0 - h2o[:, None])
    return np.column_stack([scaled, h2o])


def test_anhydrous_basis_drops_h2o_and_renormalizes() -> None:
    rng = np.random.default_rng(0)
    anhydrous = rng.dirichlet(np.ones(10), size=3)
    # each bulk appears twice with different draw H2O; anhydrous part fixed
    b11 = _rows_from_anhydrous(
        np.repeat(anhydrous, 2, axis=0), np.array([0.1, 0.3, 0.05, 0.2, 0.0, 0.15]),
    )
    out = anhydrous_bulk_basis(b11, np.array([0, 0, 1, 1, 2, 2]), 3)
    assert out.shape == (3, 10)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.allclose(out, anhydrous)


def test_anhydrous_basis_rejects_within_bulk_variation() -> None:
    rng = np.random.default_rng(1)
    anhydrous = rng.dirichlet(np.ones(10), size=2)  # two distinct chemistries
    b11 = _rows_from_anhydrous(anhydrous, np.array([0.1, 0.1]))
    with pytest.raises(ValueError, match="varies within a bulk"):
        anhydrous_bulk_basis(b11, np.array([0, 0]), 1)


def test_without_excess_o_renormalizes_away_the_deficit() -> None:
    rng = np.random.default_rng(2)
    bulk10 = rng.dirichlet(np.ones(10), size=4)
    out = without_excess_o(bulk10)
    assert out.shape == (4, 9)
    assert np.allclose(out.sum(axis=1), 1.0)
    expected = np.delete(bulk10, EXCESS_O_COL, axis=1)
    expected /= expected.sum(axis=1, keepdims=True)
    assert np.allclose(out, expected)


def test_provenance_by_bulk_join_contract(tmp_path) -> None:
    pd.DataFrame({
        "bulk": [7, 7, 9, 9],
        "draw": [0, 1, 0, 1],
        "fe_provenance": ["measured", "measured", "assumed_class", "assumed_class"],
    }).to_parquet(tmp_path / "chunk_0000_draws.parquet")
    prov = provenance_by_bulk(tmp_path)
    assert prov.to_dict() == {7: "measured", 9: "assumed_class"}

    pd.DataFrame({
        "bulk": [7, 7],
        "draw": [0, 1],
        "fe_provenance": ["measured", "assumed_class"],
    }).to_parquet(tmp_path / "chunk_0001_draws.parquet")
    with pytest.raises(ValueError, match="multiple fe_provenance"):
        provenance_by_bulk(tmp_path)


def test_group_slices_partitions_and_scores() -> None:
    rng = np.random.default_rng(3)
    truth = rng.uniform(size=(30, N_PHASES))
    pred = truth.copy()
    pred[:10] += 0.1  # first ten bulks get a uniform 0.1 error
    groups = np.array(["a"] * 10 + ["b"] * 20, dtype=object)
    out = group_slices(pred, truth, np.full(30, 128.0), groups)
    assert out["a"]["n"] == 10 and out["b"]["n"] == 20
    for phase_scores in out["a"]["scores"].values():
        assert abs(phase_scores["mae"] - 0.1) < 1e-12
    for phase_scores in out["b"]["scores"].values():
        assert phase_scores["mae"] == 0.0


def test_bootstrap_delta_ci_sign_and_null() -> None:
    zero = np.zeros(50)
    assert bootstrap_delta_ci(zero, zero, seed=0) == (0.0, 0.0)
    lo, hi = bootstrap_delta_ci(np.full(50, 0.3), np.full(50, 0.1), seed=0)
    assert abs(lo - 0.2) < 1e-12 and abs(hi - 0.2) < 1e-12
    rng = np.random.default_rng(4)
    lo, hi = bootstrap_delta_ci(
        rng.uniform(0.2, 0.4, size=200), rng.uniform(0.0, 0.2, size=200), seed=0,
    )
    assert lo > 0.0  # a real gap keeps the CI off zero
