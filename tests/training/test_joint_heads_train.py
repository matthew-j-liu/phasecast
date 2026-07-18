import numpy as np
import torch

import cnmineral.training.joint_heads as joint_heads_train
from cnmineral.data.bulk_chemistry import BulkChemistryDataset
from cnmineral.data.phase_vocab import N_PHASES
from cnmineral.models.comp_head import N_OXIDES
from cnmineral.training.joint_heads import JointHeadCheckpoint, train_joint_heads


def _config(tiny_pretrain_checkpoint, corpus_dir, valtest_dir, corpus_subset_path,
            tmp_path):
    return {
        "encoder_checkpoint": str(tiny_pretrain_checkpoint),
        "corpus_dir": str(corpus_dir),
        "valtest_corpus_dir": str(valtest_dir),
        "epochs": 2,
        "batch_size": 16,
        "lr": 1e-3,
        "lambda_cond": 0.01,
        "seed": 0,
        "data_kwargs": {"csv_path": str(corpus_subset_path), "min_class_count": 5},
        "head_checkpoint_dir": str(tmp_path / "ckpt"),
        "head_checkpoint_filename": "heads_best.pt",
    }


def test_train_smoke_and_checkpoint_roundtrip(
    tiny_pretrain_checkpoint, corpus_subset_path, synthetic_joint_corpus,
    tmp_path, monkeypatch,
):
    chem = BulkChemistryDataset(
        csv_path=str(corpus_subset_path), require_all_majors=False, min_class_count=5
    )
    bulk_ids = [int(s) for s in chem.sample_ids[:12]]
    corpus_dir = synthetic_joint_corpus(tmp_path / "corpus", bulk_ids[:8])
    valtest_dir = synthetic_joint_corpus(tmp_path / "valtest", bulk_ids[8:])
    monkeypatch.setattr(
        joint_heads_train, "load_valtest_split",
        lambda subset: np.array(bulk_ids[8:], dtype=np.int64),
    )
    config = _config(
        tiny_pretrain_checkpoint, corpus_dir, valtest_dir, corpus_subset_path,
        tmp_path,
    )

    ckpt = train_joint_heads(config, torch.device("cpu"))
    assert len(ckpt.monitor["train"]) == 2
    for term in ("presence_bce", "mode_nll", "comp_nll", "cond_nll", "closure"):
        assert len(ckpt.monitor[f"val_{term}"]) == 2
    assert ckpt.label_config["comp_eps"] == 1e-4
    assert ckpt.loss_config["lambda_cond"] == 0.01

    path = tmp_path / "ckpt" / "heads_best.pt"
    assert path.exists()
    loaded = JointHeadCheckpoint.load(path)
    assert loaded.head_config == ckpt.head_config
    assert loaded.encoder_checkpoint == str(tiny_pretrain_checkpoint)

    d_model = loaded.head_config["d_model"]
    logits, alpha = loaded.build_mode_head()(torch.randn(3, d_model))
    assert logits.shape == (3, N_PHASES) and torch.all(alpha > 0)
    comp_alpha = loaded.build_comp_head()(torch.randn(3, d_model))
    assert comp_alpha.shape == (3, N_PHASES, N_OXIDES) and torch.all(comp_alpha > 0)
    log_w, mu, sigma = loaded.build_cond_head()(
        torch.randn(3, d_model), torch.zeros(3, N_PHASES)
    )
    k = loaded.head_config["cond_n_components"]
    assert log_w.shape == mu.shape == sigma.shape == (3, 3, k)
