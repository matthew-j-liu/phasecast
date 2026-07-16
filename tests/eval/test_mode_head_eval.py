import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import beta as beta_dist
from torch.distributions import Dirichlet

from cnmineral.eval.mode_head import (
    beta_pit,
    conditional_mean,
    fp_fn_mass,
    hurdle_nll_rows,
    presence_scores,
    recall_by_band,
    reliability_table,
    stratified_mean,
    within_bulk_frequency,
)


def test_reliability_table_hand_case():
    p = np.array([0.05, 0.15, 0.95, 0.95])
    y = np.array([0.0, 1.0, 1.0, 1.0])
    t = reliability_table(p, y, n_bins=10)
    assert t["count"][0] == 1 and t["count"][1] == 1 and t["count"][9] == 2
    assert t["count"].sum() == 4
    assert np.isclose(t["mean_p"][9], 0.95)
    assert np.isclose(t["freq"][0], 0.0) and np.isclose(t["freq"][1], 1.0)
    assert np.isnan(t["freq"][5])
    # count-weighted |mean_p - freq|: (1*0.05 + 1*0.85 + 2*0.05) / 4
    assert np.isclose(t["ece"], 0.25)


def test_reliability_table_edge_probabilities_bin():
    t = reliability_table(np.array([0.0, 1.0]), np.array([0.0, 1.0]), n_bins=10)
    assert t["count"][0] == 1 and t["count"][9] == 1


def test_within_bulk_frequency():
    presence = np.array([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    row_bulk = np.array([0, 0, 2, 2])
    f, n_rows = within_bulk_frequency(presence, row_bulk, n_bulks=3)
    assert np.allclose(f[0], [1.0, 0.5])
    assert np.allclose(f[2], [0.0, 0.5])
    assert np.isnan(f[1]).all()
    assert n_rows.tolist() == [2, 0, 2]


def test_conditional_mean():
    alpha = np.array([[2.0, 3.0, 5.0]])
    mask = np.array([[1.0, 0.0, 1.0]])
    assert np.allclose(conditional_mean(alpha, mask), [[2 / 7, 0.0, 5 / 7]])
    empty = conditional_mean(alpha, np.zeros((1, 3)))
    assert np.allclose(empty, 0.0)


def test_beta_pit_matches_scipy_and_masks_degenerates():
    alpha = np.array([[2.0, 3.0, 4.0], [1.0, 5.0, 2.0]])
    presence = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    magnitude = np.array([[0.3, 0.7, 0.0], [0.0, 1.0, 0.0]])
    pit = beta_pit(alpha, presence, magnitude)
    assert np.isclose(pit[0, 0], beta_dist.cdf(0.3, 2.0, 3.0))
    assert np.isclose(pit[0, 1], beta_dist.cdf(0.7, 3.0, 2.0))
    assert np.isnan(pit[0, 2])
    assert np.isnan(pit[1]).all()  # single-phase support is a point mass


def test_fp_fn_mass_hand_case():
    p = np.array([[0.9, 0.9, 0.1]])
    alpha = np.array([[3.0, 1.0, 6.0]])
    presence = np.array([[1.0, 0.0, 1.0]])
    magnitude = np.array([[0.4, 0.0, 0.6]])
    out = fp_fn_mass(p, alpha, presence, magnitude, threshold=0.5)
    # predicted support {0, 1}: cond mean [0.75, 0.25]; phase 1 truly absent
    assert np.isclose(out["fp_mass"][0], 0.25)
    # phase 2 present with mass 0.6 but below threshold
    assert np.isclose(out["fn_mass"][0], 0.6)
    assert np.allclose(out["fp_mass_per_phase"], [[0.0, 0.25, 0.0]])


def test_presence_scores_nan_when_undefined():
    p = np.array([[0.9, 0.1, 0.2], [0.9, 0.9, 0.3]])
    presence = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    s = presence_scores(p, presence)
    assert np.isclose(s["recall"][0], 1.0)
    assert np.isclose(s["recall"][1], 1.0)
    assert np.isnan(s["recall"][2])  # never present
    assert np.isclose(s["precision"][0], 0.5)
    assert np.isnan(s["precision"][2])  # never predicted


def test_recall_by_band_separates_intermediate_frequencies():
    p = np.array([[0.9], [0.9], [0.2], [0.2]])
    presence = np.array([[1.0], [1.0], [1.0], [0.0]])
    f_rows = np.array([[0.95], [0.95], [0.5], [0.5]])
    bands = ((0.0, 0.9), (0.9, 1.01))
    out = recall_by_band(p, presence, f_rows, bands=bands)
    assert np.isclose(out["recall"][1, 0], 1.0)  # high-frequency rows detected
    assert np.isclose(out["recall"][0, 0], 0.0)  # intermediate row missed
    assert out["n_present"].tolist() == [[1], [2]]


def test_hurdle_nll_rows_matches_reference_assembly():
    torch.manual_seed(0)
    logits = torch.randn(6, 4)
    alpha = torch.rand(6, 4) * 3 + 0.2
    presence = torch.tensor([
        [1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1],
        [1, 0, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1],
    ], dtype=torch.float32)
    raw = torch.rand(6, 4) * presence + 1e-3 * presence
    magnitude = raw / raw.sum(-1, keepdim=True)

    out = hurdle_nll_rows(logits, alpha, presence, magnitude)
    for i in range(6):
        bern = F.binary_cross_entropy_with_logits(
            logits[i], presence[i], reduction="sum"
        )
        s = presence[i] > 0
        dirichlet = -Dirichlet(alpha[i, s]).log_prob(magnitude[i, s])
        assert np.isclose(out["bernoulli"][i], bern.item(), atol=1e-5)
        assert np.isclose(out["dirichlet"][i], dirichlet.item(), atol=1e-4)
        assert np.isclose(out["joint"][i], (bern + dirichlet).item(), atol=1e-4)
    assert np.allclose(out["bce_per_phase"].sum(-1), out["bernoulli"], atol=1e-5)


def test_stratified_mean():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    labels = np.array([50, 30, 50, 30])
    out = stratified_mean(values, labels)
    assert out["50"] == {"mean": 2.0, "n": 2}
    assert out["30"] == {"mean": 3.0, "n": 2}
