import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import beta as beta_dist
from torch.distributions import Dirichlet

import cnmineral.eval.joint_heads as joint_eval_mod
import cnmineral.eval.mode_head as eval_mod
import cnmineral.training.joint_heads as train_mod
from cnmineral.data.bulk_chemistry import BulkChemistryDataset
from cnmineral.eval.joint_heads import run_joint_eval
from cnmineral.eval.mode_head import (
    beta_pit,
    conditional_mean,
    fp_fn_mass,
    hurdle_nll_rows,
    magnitude_parity_stats,
    presence_scores,
    recall_by_band,
    reliability_table,
    sample_supports,
    select_overlay_exemplars,
    stratified_mean,
    support_group_pit_stats,
    within_bulk_conditional_sigma,
    within_bulk_frequency,
    within_bulk_sigma_profile,
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


def test_within_bulk_conditional_sigma():
    magnitude = np.array(
        [[0.2, 0.5, 0.0], [0.4, 0.5, 0.0], [0.6, 0.0, 0.0], [0.6, 0.0, 0.0]]
    )
    presence = np.array(
        [[1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=float
    )
    row_bulk = np.array([0, 0, 1, 1])
    s = within_bulk_conditional_sigma(magnitude, presence, row_bulk)
    assert np.isclose(s[0], np.sqrt(0.005))
    assert np.isclose(s[1], 0.0)
    assert np.isnan(s[2])


def test_within_bulk_sigma_profile():
    magnitude = np.array([[0.1], [0.1], [0.5], [0.9]])
    presence = np.ones((4, 1))
    row_bulk = np.array([0, 0, 1, 1])
    levels, sigmas = within_bulk_sigma_profile(
        magnitude, presence, row_bulk, n_bins=2
    )
    assert np.allclose(levels[0], [0.1, 0.7])
    assert np.allclose(sigmas[0], [0.0, 0.2])


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


def test_sample_supports_deterministic_extremes():
    rng = np.random.default_rng(0)
    p = np.array([[0.0, 1.0], [1.0, 0.0]])
    s = sample_supports(p, rng)
    assert np.array_equal(s, p)


def test_magnitude_parity_stats_hand_case():
    n_phases = len(eval_mod.SOLID_PHASES)
    cond_mean = np.zeros((3, n_phases))
    magnitude = np.zeros((3, n_phases))
    presence = np.zeros((3, n_phases))
    presence[:, 0] = 1.0
    cond_mean[:, 0] = [0.1, 0.2, 0.3]
    magnitude[:, 0] = [0.2, 0.4, 0.6]  # perfectly correlated, MAE 0.2
    stats = magnitude_parity_stats(cond_mean, magnitude, presence)
    assert np.isclose(stats["magnitude_mae"][0], 0.2)
    assert np.isclose(stats["magnitude_r"][0], 1.0)
    assert np.isnan(stats["magnitude_mae"][1])  # never-present phase


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


def test_support_group_pit_stats_classifies_hand_groups():
    # bulk 0 holds two supports (groups must split within a bulk):
    # group A piled near 0 -> shifted; group B at both extremes -> two_sided.
    # bulk 1 holds a uniform grid -> flat. bulk 2 has 5 rows -> filtered.
    pit_a = np.linspace(0.0, 0.09, 20)
    pit_b = np.array([0.02] * 8 + [0.98] * 8 + [0.5] * 4)
    pit_c = np.linspace(0.025, 0.975, 20)
    pit_d = np.full(5, 0.5)
    pit0 = np.concatenate([pit_a, pit_b, pit_c, pit_d])
    pit = np.column_stack([pit0, np.full(65, np.nan)])
    presence = np.zeros((65, 3))
    presence[:, 0] = 1.0
    presence[:20, 1] = 1.0
    presence[20:40, 2] = 1.0
    presence[40:, 1] = 1.0
    row_bulk = np.array([0] * 40 + [1] * 20 + [2] * 5)
    s = support_group_pit_stats(pit, presence, row_bulk, min_rows=20)

    assert s[0]["n_groups"] == 3
    assert np.isclose(s[0]["coverage"], 60 / 65)
    assert sorted(s[0]["group_class"]) == ["flat", "shifted", "two_sided"]
    assert np.isclose(s[0]["frac_shifted"], 1 / 3)
    assert np.isclose(s[0]["frac_two_sided"], 1 / 3)
    shifted_mean = s[0]["group_mean"][s[0]["group_class"] == "shifted"]
    assert np.isclose(shifted_mean, pit_a.mean())
    assert s[1]["n_groups"] == 0 and np.isnan(s[1]["frac_shifted"])


def test_support_group_pit_stats_uniform_false_positive_rate():
    rng = np.random.default_rng(0)
    n_groups, rows = 200, 20
    pit = rng.uniform(size=(n_groups * rows, 1))
    presence = np.ones((n_groups * rows, 2))
    row_bulk = np.repeat(np.arange(n_groups), rows)
    s = support_group_pit_stats(pit[:, :1], presence, row_bulk, min_rows=rows)
    assert s[0]["n_groups"] == n_groups
    # calibrated groups of the minimum size: misclassification stays bounded
    assert s[0]["frac_two_sided"] < 0.2
    assert s[0]["frac_shifted"] < 0.1


def test_beta_overlay_exemplar_selection():
    # bulk 0: two supports at means 0.2 / 0.7 -> support-separated winner
    # (constant within each support, so no within-support candidacy);
    # bulk 1: one support split 0.2 / 0.8 -> within-support winner.
    presence = np.zeros((80, 3))
    presence[:, 0] = 1.0
    presence[:25, 1] = 1.0
    presence[25:40, 2] = 1.0
    presence[40:, 1] = 1.0
    magnitude = np.zeros((80, 3))
    magnitude[:25, 0] = 0.2
    magnitude[25:40, 0] = 0.7
    magnitude[40:60, 0] = 0.2
    magnitude[60:, 0] = 0.8
    row_bulk = np.array([0] * 40 + [1] * 40)
    ex = select_overlay_exemplars(magnitude, presence, row_bulk, phase_idx=0)

    sep = ex["support_separated"]
    assert sep["bulk"] == 0
    assert len(sep["dom_rows"]) == 25 and len(sep["second_rows"]) == 15
    within = ex["within_support"]
    assert within["bulk"] == 1
    assert np.isclose(within["m_lo"], 0.2) and np.isclose(within["m_hi"], 0.8)


def test_run_mode_head_eval_end_to_end(
    tiny_pretrain_checkpoint, corpus_subset_path, synthetic_joint_corpus,
    tmp_path, monkeypatch,
):
    data_kwargs = {"csv_path": str(corpus_subset_path), "min_class_count": 5}
    chem = BulkChemistryDataset(
        csv_path=str(corpus_subset_path), require_all_majors=False, min_class_count=5
    )
    bulk_ids = [int(s) for s in chem.sample_ids[:12]]
    corpus_dir = synthetic_joint_corpus(tmp_path / "corpus", bulk_ids[:8])
    valtest_dir = synthetic_joint_corpus(tmp_path / "valtest", bulk_ids[8:])
    val_ids = np.array(bulk_ids[8:], dtype=np.int64)
    monkeypatch.setattr(train_mod, "load_valtest_split", lambda subset: val_ids)
    monkeypatch.setattr(eval_mod, "load_valtest_split", lambda subset: val_ids)
    monkeypatch.setattr(joint_eval_mod, "load_valtest_split", lambda subset: val_ids)

    config = {
        "encoder_checkpoint": str(tiny_pretrain_checkpoint),
        "corpus_dir": str(corpus_dir),
        "valtest_corpus_dir": str(valtest_dir),
        "epochs": 1,
        "batch_size": 16,
        "lr": 1e-3,
        "seed": 0,
        "data_kwargs": data_kwargs,
        "head_checkpoint_dir": str(tmp_path / "ckpt"),
        "head_checkpoint_filename": "heads_best.pt",
    }
    train_mod.train_joint_heads(config, torch.device("cpu"))
    ckpt_path = tmp_path / "ckpt" / "heads_best.pt"

    # the harness rebuilds the dataset against the default gard2019 csv;
    # inject the fixture subset without adding a production seam
    real_dataset = joint_eval_mod.JointHeadDataset
    monkeypatch.setattr(
        joint_eval_mod, "JointHeadDataset",
        lambda **kw: real_dataset(**kw, data_kwargs=data_kwargs),
    )
    monkeypatch.setattr(eval_mod, "RUNS_ROOT", tmp_path / "runs")
    metrics = run_joint_eval(
        ckpt_path, heads=("mode",), device=torch.device("cpu")
    )["mode"]

    assert metrics["split"] == {
        "csv": "src/cnmineral/data/valtest_split.csv",
        "subset": "val",
        "n_assigned_bulks": len(val_ids),
    }
    assert metrics["n_val_bulks"] == len(val_ids)
    assert metrics["n_val_rows"] == 3 * len(val_ids)
    out_dir = tmp_path / "runs" / "mode_head" / "ckpt"
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "reliability.png").exists()
    assert (out_dir / "magnitude_pit_by_support.png").exists()
