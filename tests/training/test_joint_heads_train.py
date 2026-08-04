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


def _setup_config(tiny_pretrain_checkpoint, corpus_subset_path,
                  synthetic_joint_corpus, tmp_path, monkeypatch):
    """Synthetic train + valtest corpora, patched roster, ready-to-train config."""
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
    return _config(
        tiny_pretrain_checkpoint, corpus_dir, valtest_dir, corpus_subset_path,
        tmp_path,
    )


def test_train_smoke_and_checkpoint_roundtrip(
    tiny_pretrain_checkpoint, corpus_subset_path, synthetic_joint_corpus,
    tmp_path, monkeypatch,
):
    config = _setup_config(
        tiny_pretrain_checkpoint, corpus_subset_path, synthetic_joint_corpus,
        tmp_path, monkeypatch,
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
    for key in ("mode_hidden", "mode_layers", "comp_hidden", "comp_layers",
                "cond_hidden", "cond_layers"):
        assert key in loaded.head_config
    assert loaded.encoder_checkpoint == str(tiny_pretrain_checkpoint)

    d_model = loaded.head_config["d_model"]
    logits, alpha = loaded.build_mode_head()(
        torch.randn(3, d_model), torch.zeros(3, N_PHASES)
    )
    assert logits.shape == (3, N_PHASES) and torch.all(alpha > 0)
    comp_alpha = loaded.build_comp_head()(torch.randn(3, d_model))
    assert comp_alpha.shape == (3, N_PHASES, N_OXIDES) and torch.all(comp_alpha > 0)
    log_w, mu, sigma = loaded.build_cond_head()(
        torch.randn(3, d_model), torch.zeros(3, N_PHASES)
    )
    k = loaded.head_config["cond_n_components"]
    assert log_w.shape == mu.shape == sigma.shape == (3, 3, k)


def test_mixture_train_roundtrip_and_legacy_config_defaults_k1(
    tiny_pretrain_checkpoint, corpus_subset_path, synthetic_joint_corpus,
    tmp_path, monkeypatch,
):
    """K=2 trains, stamps its K, and rebuilds a gated head; a head_config
    without the key (every pre-mixture checkpoint) rebuilds the K=1 head."""
    config = _setup_config(
        tiny_pretrain_checkpoint, corpus_subset_path, synthetic_joint_corpus,
        tmp_path, monkeypatch,
    )
    ckpt = train_joint_heads(
        {**config, "mode_n_components": 2}, torch.device("cpu")
    )
    assert ckpt.head_config["mode_n_components"] == 2

    path = tmp_path / "ckpt" / "heads_best.pt"
    loaded = JointHeadCheckpoint.load(path)
    head = loaded.build_mode_head()
    assert head.n_components == 2
    d_model = loaded.head_config["d_model"]
    logits, alpha, log_w = head(torch.randn(3, d_model), torch.zeros(3, N_PHASES))
    assert alpha.shape == (3, 2, N_PHASES) and log_w.shape == (3, 2)

    # a pre-mixture checkpoint carries no mode_n_components key and a K=1
    # state dict; build_mode_head must reconstruct it strictly
    from cnmineral.models.mode_head import ModeHead
    k1 = ModeHead(d_model=d_model, hidden=loaded.head_config["mode_hidden"],
                  layers=loaded.head_config["mode_layers"])
    legacy = JointHeadCheckpoint(
        mode_state=k1.state_dict(), comp_state=loaded.comp_state,
        cond_state=loaded.cond_state,
        head_config={k: v for k, v in loaded.head_config.items()
                     if k != "mode_n_components"},
        loss_config=loaded.loss_config, encoder_checkpoint="",
        label_config={}, monitor={},
    )
    rebuilt = legacy.build_mode_head()
    assert rebuilt.n_components == 1 and not hasattr(rebuilt, "gate_out")


def test_per_head_selection_restores_each_heads_best(
    tiny_pretrain_checkpoint, corpus_subset_path, synthetic_joint_corpus,
    tmp_path, monkeypatch,
):
    """Each head's stamped state comes from its own best epoch, not a joint one.

    Run A stops after epoch 1; run B trains 3 epochs under a forced selection
    where mode's best stays at epoch 1 while comp/cond improve every epoch.
    Same seed means identical trajectories through epoch 1, so B's mode state
    must equal A's exactly while B's comp state must have moved on.
    """
    config = _setup_config(
        tiny_pretrain_checkpoint, corpus_subset_path, synthetic_joint_corpus,
        tmp_path, monkeypatch,
    )
    config.pop("head_checkpoint_dir")
    config.pop("head_checkpoint_filename")
    ckpt_a = train_joint_heads({**config, "epochs": 1}, torch.device("cpu"))

    def forced_selection(monitor, split="val"):
        n = len(monitor[f"{split}_mode_nll"])
        return {
            "mode": [0.0] + [1.0] * (n - 1),
            "comp": [-float(i) for i in range(n)],
            "cond": [-float(i) for i in range(n)],
        }

    monkeypatch.setattr(joint_heads_train, "selection_series", forced_selection)
    ckpt_b = train_joint_heads({**config, "epochs": 3}, torch.device("cpu"))

    def same(a, b):
        return all(torch.equal(a[k], b[k]) for k in a)

    assert same(ckpt_b.mode_state, ckpt_a.mode_state)
    assert not same(ckpt_b.comp_state, ckpt_a.comp_state)
    assert not same(ckpt_b.cond_state, ckpt_a.cond_state)


def test_early_stop_waits_for_slowest_head(
    tiny_pretrain_checkpoint, corpus_subset_path, synthetic_joint_corpus,
    tmp_path, monkeypatch,
):
    """Stopping requires every head to be `patience` epochs past its own best.

    Forced selection: mode/comp peak at epoch 1, cond keeps improving through
    epoch 4.  With patience 1 the run must survive to epoch 5 (cond's first
    non-improving epoch), not stop at epoch 2 when the fast heads plateau.
    """
    config = _setup_config(
        tiny_pretrain_checkpoint, corpus_subset_path, synthetic_joint_corpus,
        tmp_path, monkeypatch,
    )
    config.pop("head_checkpoint_dir")
    config.pop("head_checkpoint_filename")

    def forced_selection(monitor, split="val"):
        n = len(monitor[f"{split}_mode_nll"])
        flat = [0.0] + [1.0] * (n - 1)
        return {
            "mode": flat,
            "comp": list(flat),
            "cond": [-float(min(i, 3)) for i in range(n)],
        }

    monkeypatch.setattr(joint_heads_train, "selection_series", forced_selection)
    ckpt = train_joint_heads(
        {**config, "epochs": 8, "early_stop_patience": 1}, torch.device("cpu")
    )
    assert len(ckpt.monitor["val"]) == 5
