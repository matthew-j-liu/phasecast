import dataclasses

import numpy as np
import pytest
import torch
from scipy.stats import beta as beta_dist

import cnmineral.eval.comp_head as eval_mod
import cnmineral.training.joint_heads as train_mod
from cnmineral.data.bulk_chemistry import BulkChemistryDataset
from cnmineral.data.magemin_corpus import COMP_OXIDES
from cnmineral.eval.comp_head import (
    an_number,
    comp_beta_pit,
    comp_nll_matrix,
    comp_parity_stats,
    run_comp_head_eval,
)


def test_comp_nll_matrix_masks_absent():
    alpha = np.full((2, 3, 4), 2.0)
    target = np.full((2, 3, 4), 0.25)
    presence = np.array([[1, 0, 1], [0, 0, 1]], dtype=float)
    nll = comp_nll_matrix(alpha, target, presence)
    assert np.isnan(nll[presence == 0]).all()
    assert np.isfinite(nll[presence == 1]).all()


def test_comp_beta_pit_uniform_alpha_analytic():
    # alpha = ones -> each marginal is Beta(1, O-1) with CDF 1-(1-x)^(O-1)
    n_ox = 5
    alpha = np.ones((1, 1, n_ox))
    x = 0.3
    target = np.full((1, 1, n_ox), x)
    presence = np.ones((1, 1))
    pit = comp_beta_pit(alpha, target, presence)
    assert np.allclose(pit, 1 - (1 - x) ** (n_ox - 1))


def test_comp_beta_pit_calibrated_samples_are_uniform():
    rng = np.random.default_rng(0)
    alpha_row = np.array([4.0, 2.0, 1.0, 3.0])
    samples = rng.dirichlet(alpha_row, size=4000)
    alpha = np.tile(alpha_row, (4000, 1, 1))
    pit = comp_beta_pit(alpha, samples[:, None, :], np.ones((4000, 1)))
    assert abs(np.nanmean(pit) - 0.5) < 0.02
    assert abs(np.nanmean(pit < 0.1) - 0.1) < 0.02


def test_comp_parity_stats_perfect_prediction():
    rng = np.random.default_rng(1)
    target = rng.dirichlet(np.ones(4), size=(10, 2)).reshape(10, 2, 4)
    presence = np.ones((10, 2))
    stats = comp_parity_stats(target.copy(), target, presence)
    assert np.allclose(stats["mae"], 0.0)
    assert np.allclose(stats["r"], 1.0)


def test_an_number_hand_case():
    comp = np.zeros(len(COMP_OXIDES))
    comp[COMP_OXIDES.index("CaO")] = 56.08
    comp[COMP_OXIDES.index("Na2O")] = 61.98
    # equal molar CaO and Na2O, no K2O -> An# = 0.5
    assert np.isclose(an_number(comp), 0.5)


def test_run_comp_head_eval_end_to_end(
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

    real_dataset = eval_mod.JointHeadDataset
    monkeypatch.setattr(
        eval_mod, "JointHeadDataset",
        lambda **kw: real_dataset(**kw, data_kwargs=data_kwargs),
    )
    monkeypatch.setattr(eval_mod, "RESULTS_ROOT", tmp_path / "results")
    metrics = run_comp_head_eval(ckpt_path, device=torch.device("cpu"))

    assert metrics["n_val_bulks"] == len(val_ids)
    assert metrics["n_val_rows"] == 3 * len(val_ids)
    assert np.isfinite(metrics["comp_nll"])
    assert metrics["closure_l2"]["mean"] >= 0
    out_dir = tmp_path / "results" / "comp_head" / "ckpt"
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "comp_parity_SiO2.png").exists()
    assert (out_dir / "closure_residual.png").exists()

    closure = eval_mod.run_predicted_closure_eval(
        ckpt_path, device=torch.device("cpu")
    )
    assert closure["n_val_rows"] == metrics["n_val_rows"]
    assert closure["closure_l2"]["mean"] >= 0
    assert set(closure["signed_mean_by_oxide"]) == set(
        metrics["closure_abs_by_oxide"]
    )

    stale = dataclasses.replace(
        train_mod.JointHeadCheckpoint.load(ckpt_path),
        label_config={k: v for k, v in metrics["label_config"].items()
                      if k != "comp_eps"},
    )
    stale_path = tmp_path / "ckpt_stale" / "heads_best.pt"
    stale.save(stale_path)
    with pytest.raises(ValueError, match="comp_eps"):
        run_comp_head_eval(stale_path, device=torch.device("cpu"))
