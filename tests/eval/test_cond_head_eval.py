import numpy as np
import torch
from scipy.stats import norm

import cnmineral.eval.cond_head as eval_mod
import cnmineral.eval.joint_heads as joint_eval_mod
import cnmineral.training.joint_heads as train_mod
from cnmineral.data.bulk_chemistry import BulkChemistryDataset
from cnmineral.eval.cond_head import (
    central_coverage,
    effective_components,
    gmm_nll_per_axis,
    gmm_pit,
    mass_below_zero,
    sample_supports,
)
from cnmineral.eval.joint_heads import run_joint_eval
from cnmineral.losses.cond_loss import gmm_nll


def _random_gmm(batch=6, k=3, seed=0):
    rng = np.random.default_rng(seed)
    log_w = np.log(rng.dirichlet(np.ones(k), size=(batch, 3)))
    mu = rng.normal(size=(batch, 3, k)) * 5
    sigma = rng.uniform(0.5, 2.0, size=(batch, 3, k))
    target = rng.normal(size=(batch, 3))
    return log_w, mu, sigma, target


def test_gmm_nll_per_axis_matches_loss_module():
    log_w, mu, sigma, target = _random_gmm()
    per_axis = gmm_nll_per_axis(log_w, mu, sigma, target)
    ref = gmm_nll(
        torch.as_tensor(log_w), torch.as_tensor(mu), torch.as_tensor(sigma),
        torch.as_tensor(target), reduction="none",
    ).numpy()
    assert np.allclose(per_axis.sum(-1), ref, atol=1e-6)


def test_gmm_pit_single_component_is_normal_cdf():
    rng = np.random.default_rng(1)
    mu = rng.normal(size=(4, 3, 1))
    sigma = rng.uniform(0.5, 2.0, size=(4, 3, 1))
    target = rng.normal(size=(4, 3))
    pit = gmm_pit(np.zeros((4, 3, 1)), mu, sigma, target)
    assert np.allclose(pit, norm.cdf((target - mu[..., 0]) / sigma[..., 0]))


def test_gmm_pit_calibrated_coverage():
    rng = np.random.default_rng(2)
    n = 5000
    mu = np.zeros((n, 3, 2))
    mu[..., 1] = 4.0
    sigma = np.ones((n, 3, 2))
    log_w = np.full((n, 3, 2), np.log(0.5))
    comp = rng.integers(0, 2, size=(n, 3))
    target = rng.normal(size=(n, 3)) + 4.0 * comp
    pit = gmm_pit(log_w, mu, sigma, target)
    cov = central_coverage(pit, (0.5, 0.9))
    assert np.allclose(cov["0.5"], 0.5, atol=0.03)
    assert np.allclose(cov["0.9"], 0.9, atol=0.03)


def test_central_coverage_hand_case():
    # only the 0.5 entry falls inside the central 50% interval
    pit = np.array([[0.5, 0.05, 0.95]]).T.reshape(3, 1)
    cov = central_coverage(pit, (0.5,))
    assert np.isclose(cov["0.5"][0], 1 / 3)


def test_effective_components_limits():
    one_hot = np.log(np.array([[[1.0 - 2e-12, 1e-12, 1e-12]]]))
    uniform = np.log(np.full((1, 1, 3), 1 / 3))
    assert np.isclose(effective_components(one_hot)[0, 0], 1.0, atol=1e-6)
    assert np.isclose(effective_components(uniform)[0, 0], 3.0, atol=1e-6)


def test_mass_below_zero():
    log_w = np.zeros((1, 1, 1))
    assert np.isclose(mass_below_zero(log_w, np.zeros((1, 1, 1)), np.ones((1, 1, 1)))[0, 0], 0.5)
    assert mass_below_zero(log_w, np.full((1, 1, 1), 100.0), np.ones((1, 1, 1)))[0, 0] < 1e-6


def test_sample_supports_deterministic_extremes():
    rng = np.random.default_rng(0)
    p = np.array([[0.0, 1.0], [1.0, 0.0]])
    s = sample_supports(p, rng)
    assert np.array_equal(s, p)


def test_run_cond_head_eval_end_to_end(
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

    real_dataset = joint_eval_mod.JointHeadDataset
    monkeypatch.setattr(
        joint_eval_mod, "JointHeadDataset",
        lambda **kw: real_dataset(**kw, data_kwargs=data_kwargs),
    )
    monkeypatch.setattr(eval_mod, "RESULTS_ROOT", tmp_path / "results")
    metrics = run_joint_eval(
        ckpt_path, heads=("cond",), device=torch.device("cpu")
    )["cond"]

    assert metrics["n_val_bulks"] == len(val_ids)
    assert metrics["n_val_rows"] == 3 * len(val_ids)
    assert np.isfinite(metrics["cond_nll"])
    axes = [r["axis"] for r in metrics["per_axis"]]
    assert axes == ["T", "P_kbar", "H2O"]
    for r in metrics["per_axis"]:
        assert np.isfinite(r["exposure_bias_nll_delta"])
    out_dir = tmp_path / "results" / "cond_head" / "ckpt"
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "cond_pit.png").exists()
    assert (out_dir / "cond_components.png").exists()
