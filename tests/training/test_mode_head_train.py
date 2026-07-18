import numpy as np
import torch

import cnmineral.training.mode_head as mode_head_train
from cnmineral.data.bulk_chemistry import BulkChemistryDataset
from cnmineral.data.phase_vocab import N_PHASES
from cnmineral.training.mode_head import ModeHeadCheckpoint, train_mode_head


def _config(tiny_pretrain_checkpoint, corpus_dir, valtest_dir, corpus_subset_path,
            tmp_path):
    return {
        "encoder_checkpoint": str(tiny_pretrain_checkpoint),
        "corpus_dir": str(corpus_dir),
        "valtest_corpus_dir": str(valtest_dir),
        "epochs": 2,
        "batch_size": 16,
        "lr": 1e-3,
        "seed": 0,
        "data_kwargs": {"csv_path": str(corpus_subset_path), "min_class_count": 5},
        "head_checkpoint_dir": str(tmp_path / "ckpt"),
        "head_checkpoint_filename": "head_best.pt",
    }


def test_train_smoke_and_checkpoint_roundtrip(
    tiny_pretrain_checkpoint, corpus_subset_path, synthetic_mineralogy_corpus,
    tmp_path, monkeypatch,
):
    chem = BulkChemistryDataset(
        csv_path=str(corpus_subset_path), require_all_majors=False, min_class_count=5
    )
    bulk_ids = [int(s) for s in chem.sample_ids[:12]]
    corpus_dir = synthetic_mineralogy_corpus(tmp_path / "corpus", bulk_ids[:8])
    valtest_dir = synthetic_mineralogy_corpus(tmp_path / "valtest", bulk_ids[8:])
    monkeypatch.setattr(
        mode_head_train, "load_valtest_split",
        lambda subset: np.array(bulk_ids[8:], dtype=np.int64),
    )
    config = _config(
        tiny_pretrain_checkpoint, corpus_dir, valtest_dir, corpus_subset_path,
        tmp_path,
    )

    ckpt = train_mode_head(config, torch.device("cpu"))
    assert len(ckpt.monitor["train"]) == 2
    assert len(ckpt.monitor["val_nll"]) == 2
    assert ckpt.label_config["rungs"] == (50, 30)
    assert ckpt.label_config["presence_cutoff"] > 0
    assert ckpt.label_config["valtest_corpus_dir"] == str(valtest_dir)

    path = tmp_path / "ckpt" / "head_best.pt"
    assert path.exists()
    loaded = ModeHeadCheckpoint.load(path)
    assert loaded.head_config == ckpt.head_config
    assert loaded.encoder_checkpoint == str(tiny_pretrain_checkpoint)

    head = loaded.build_head()
    logits, alpha = head(torch.randn(3, loaded.head_config["d_model"]))
    assert logits.shape == (3, N_PHASES)
    assert torch.all(alpha > 0)
