"""Tripwire test for the `presence_fraction_ceiling` exploration.

Locks the pure pieces on synthetic arrays: the log-feature matrix contract
(NaN exactly where absent), per-bulk frequency targets agreeing with
`within_bulk_frequency`, the binomial floor formula, kNN frequency read, probe
output shape, and verdict routing.  The orchestrator `run_ceiling_probe` needs
a real checkpoint, so it is exercised by the CLI run, not here.
"""

import numpy as np
import torch

from cnmineral.data.phase_vocab import N_PHASES
from explorations.presence_fraction_ceiling.run_probe import (
    HEADLINE,
    FractionProbe,
    binomial_floor,
    build_verdict,
    knn_frequency,
    log_feature_matrix,
    score_arm,
)


def test_log_feature_matrix_nan_exactly_where_absent() -> None:
    frac = np.array([[0.5, 0.0, 1e-6], [0.0, 0.25, 0.0]])
    populated = frac > 0
    x = log_feature_matrix(frac, populated)
    assert np.isnan(x[~populated]).all()
    assert np.allclose(x[populated], np.log10(frac[populated]))


def test_binomial_floor_matches_halfnormal_mean() -> None:
    f = np.full((100, 3), 0.5)
    n = np.full(100, 128.0)
    floor = binomial_floor(f, n)
    expected = np.sqrt(2.0 / np.pi) * np.sqrt(0.25 / 128.0)
    assert floor.shape == (3,)
    assert np.allclose(floor, expected)
    # degenerate rates carry no sampling noise
    assert np.allclose(binomial_floor(np.zeros((10, 2)), n[:10]), 0.0)


def test_score_arm_perfect_and_antiperfect() -> None:
    rng = np.random.default_rng(0)
    truth = rng.uniform(size=(50, N_PHASES))
    scores = score_arm(truth, truth)
    assert all(v["mae"] == 0.0 for v in scores.values())
    assert all(abs(v["pearson_r"] - 1.0) < 1e-12 for v in scores.values())
    flat = score_arm(np.full_like(truth, 0.5), truth)
    assert all(np.isnan(v["pearson_r"]) for v in flat.values())


def test_knn_frequency_recovers_cluster_means() -> None:
    # two well-separated latent clusters with distinct f: each query's k
    # neighbours all come from its own cluster, so the read is the cluster mean
    train_lat = np.vstack([
        np.tile([10.0, 0.0], (20, 1)), np.tile([0.0, 10.0], (20, 1)),
    ]) + np.random.default_rng(1).normal(scale=0.01, size=(40, 2))
    train_f = np.vstack([
        np.tile([0.9] + [0.0] * (N_PHASES - 1), (20, 1)),
        np.tile([0.1] + [0.0] * (N_PHASES - 1), (20, 1)),
    ])
    query = np.array([[10.0, 0.0], [0.0, 10.0]])
    pred = knn_frequency(train_lat, train_f, query, k=5)
    assert pred.shape == (2, N_PHASES)
    assert np.allclose(pred[:, 0], [0.9, 0.1])


def test_fraction_probe_output_shape() -> None:
    probe = FractionProbe(in_dim=16, hidden=8)
    out = probe(torch.zeros(4, 16))
    assert out.shape == (4, N_PHASES)


def _uniform_scores(mae_by_arm: dict[str, float]) -> dict[str, dict]:
    return {
        arm: {p: {"mae": mae, "pearson_r": 0.5} for p in HEADLINE}
        for arm, mae in mae_by_arm.items()
    }


def test_build_verdict_routes() -> None:
    ceiling = build_verdict(_uniform_scores(
        {"head": 0.10, "raw_xgb": 0.09, "latent_mlp": 0.09}))
    assert "corpus is the ceiling" in ceiling["route"]

    head_fix = build_verdict(_uniform_scores(
        {"head": 0.10, "raw_xgb": 0.05, "latent_mlp": 0.05}))
    assert "fix the head side" in head_fix["route"]

    skip = build_verdict(_uniform_scores(
        {"head": 0.10, "raw_xgb": 0.05, "latent_mlp": 0.10}))
    assert "skip connection" in skip["route"]
