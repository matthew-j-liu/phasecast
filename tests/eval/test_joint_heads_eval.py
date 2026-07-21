import dataclasses

import numpy as np
import pytest
import torch

import cnmineral.eval.comp_head as comp_eval_mod
import cnmineral.eval.joint_heads as joint_eval_mod
import cnmineral.eval.mode_head as mode_eval_mod
import cnmineral.training.joint_heads as train_mod
from cnmineral.data.bulk_chemistry import BulkChemistryDataset
from cnmineral.eval.joint_heads import run_joint_eval, val_dataset_and_latents


def _train_tiny_joint(
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
    monkeypatch.setattr(mode_eval_mod, "load_valtest_split", lambda subset: val_ids)

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
    return tmp_path / "ckpt" / "heads_best.pt", data_kwargs, val_ids


def test_run_joint_eval_shares_one_dataset_build(
    tiny_pretrain_checkpoint, corpus_subset_path, synthetic_joint_corpus,
    tmp_path, monkeypatch,
):
    ckpt_path, data_kwargs, val_ids = _train_tiny_joint(
        tiny_pretrain_checkpoint, corpus_subset_path, synthetic_joint_corpus,
        tmp_path, monkeypatch,
    )
    real_dataset = joint_eval_mod.JointHeadDataset
    builds = []

    def counting_dataset(**kw):
        builds.append(1)
        return real_dataset(**kw, data_kwargs=data_kwargs)

    monkeypatch.setattr(joint_eval_mod, "JointHeadDataset", counting_dataset)
    monkeypatch.setattr(mode_eval_mod, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(comp_eval_mod, "RUNS_ROOT", tmp_path / "runs")

    metrics = run_joint_eval(
        ckpt_path, heads=("mode", "comp"), device=torch.device("cpu")
    )
    assert set(metrics) == {"mode", "comp"}
    assert len(builds) == 1
    assert metrics["mode"]["n_val_bulks"] == len(val_ids)
    assert metrics["comp"]["n_val_bulks"] == len(val_ids)
    assert (tmp_path / "runs" / "mode_head" / "ckpt" / "metrics.json").exists()
    assert (tmp_path / "runs" / "comp_head" / "ckpt" / "metrics.json").exists()


def test_run_joint_eval_rejects_unknown_head(tmp_path):
    with pytest.raises(ValueError, match="unknown heads"):
        run_joint_eval(tmp_path / "nope.pt", heads=("mode", "wumbo"))


def test_stale_label_config_raises(
    tiny_pretrain_checkpoint, corpus_subset_path, synthetic_joint_corpus,
    tmp_path, monkeypatch,
):
    ckpt_path, _, _ = _train_tiny_joint(
        tiny_pretrain_checkpoint, corpus_subset_path, synthetic_joint_corpus,
        tmp_path, monkeypatch,
    )
    ckpt = train_mod.JointHeadCheckpoint.load(ckpt_path)
    stale = dataclasses.replace(
        ckpt,
        label_config={k: v for k, v in ckpt.label_config.items()
                      if k != "comp_eps"},
    )
    with pytest.raises(ValueError, match="comp_eps"):
        val_dataset_and_latents(stale, torch.device("cpu"))
