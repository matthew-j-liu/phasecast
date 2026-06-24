"""Tests for `cnmineral.training.imputation_head`.

The frozen-encoder contract (no gradient reaches the encoder) plus an
end-to-end head-training run against the tiny encoder checkpoint that
round-trips an `ImputationHeadCheckpoint`.
"""

from pathlib import Path

import torch

from cnmineral.data.tokens import N_TOKENS
from cnmineral.models.encoder import Encoder
from cnmineral.models.imputation_head import LogImputationHead
import pytest

from cnmineral.training.imputation_head import (
    ImputationHeadCheckpoint,
    _head_forward_loss,
    train_imputation_head,
)


def test_encoder_receives_no_gradient() -> None:
    """`_head_forward_loss` freezes the encoder: backward leaves every encoder
    parameter grad-free while the head's parameters do get gradients."""
    torch.manual_seed(0)
    encoder = Encoder(d_model=16, n_layers=1, n_heads=2, frac_resolution=200)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    head = LogImputationHead(d_model=16, hidden=8)

    src = torch.zeros(4, N_TOKENS, dtype=torch.long)
    frac = torch.zeros(4, N_TOKENS, dtype=torch.float32)
    src[:, :5] = torch.tensor([14, 13, 20, 12, 26])
    frac[:, :5] = 0.2
    # mask_prob=1.0 guarantees masked positions so the loss is nonzero.
    loss = _head_forward_loss(
        {"src": src, "frac": frac}, encoder, head, 1.0, torch.device("cpu"),
    )
    loss.backward()

    assert all(p.grad is None for p in encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())


def test_train_imputation_head_end_to_end(tiny_pretrain_checkpoint: Path) -> None:
    """Training on the tiny encoder checkpoint fills the monitor, fits the
    standardization buffers, and produces a reloadable head checkpoint that
    carries only the head (no encoder state)."""
    config = {
        "encoder_checkpoint": str(tiny_pretrain_checkpoint),
        "hidden": 8,
        "standardize": True,
        "epochs": 1,
        "batch_size": 32,
        "lr": 1e-3,
        "val_split": 0.25,
        "seed": 0,
    }
    ckpt = train_imputation_head(config, torch.device("cpu"))

    assert len(ckpt.monitor["train"]) == 1 and len(ckpt.monitor["val"]) == 1
    assert torch.isfinite(torch.tensor(ckpt.val_loss))
    assert ckpt.encoder_checkpoint == str(tiny_pretrain_checkpoint)
    assert "encoder" not in ckpt.head_state  # head-only artifact
    # standardization actually fit (buffers moved off the identity).
    assert not torch.equal(
        ckpt.head_state["target_mean"], torch.zeros(N_TOKENS),
    )

    head = ckpt.build_head()
    assert isinstance(head, LogImputationHead)
    assert head.predict_frac(torch.zeros(2, head.d_model)).shape == (2, N_TOKENS)


def test_early_stop_returns_best_val_weights(tiny_pretrain_checkpoint: Path) -> None:
    """With `early_stop_patience` set, the returned losses are the best-val
    epoch's (best-weight restore), and training records no more than `epochs`."""
    config = {
        "encoder_checkpoint": str(tiny_pretrain_checkpoint),
        "hidden": 8,
        "epochs": 4,
        "batch_size": 32,
        "lr": 1e-3,
        "val_split": 0.25,
        "early_stop_patience": 1,
        "seed": 0,
    }
    ckpt = train_imputation_head(config, torch.device("cpu"))
    assert len(ckpt.monitor["val"]) <= config["epochs"]
    assert ckpt.val_loss == pytest.approx(min(ckpt.monitor["val"]))


def test_dropout_roundtrips_through_checkpoint(tiny_pretrain_checkpoint: Path) -> None:
    """`dropout` is stamped into head_config and rebuilds via build_head; a
    pre-dropout checkpoint (key absent) still rebuilds at the 0.0 default."""
    head = LogImputationHead(d_model=16, hidden=8, dropout=0.25)
    ckpt = ImputationHeadCheckpoint.from_head(
        head, str(tiny_pretrain_checkpoint), 0.1, 0.2, {"train": [0.1], "val": [0.2]},
    )
    assert ckpt.head_config["dropout"] == 0.25
    assert ckpt.build_head().dropout == 0.25

    legacy = ImputationHeadCheckpoint(
        head_state=head.state_dict(),
        head_config={"d_model": 16, "hidden": 8, "standardize": True},
        encoder_checkpoint=str(tiny_pretrain_checkpoint),
        train_loss=0.1, val_loss=0.2, monitor={"train": [0.1], "val": [0.2]},
    )
    assert legacy.build_head().dropout == 0.0


def test_imputation_head_checkpoint_save_load(
    tiny_pretrain_checkpoint: Path, tmp_path: Path,
) -> None:
    """`ImputationHeadCheckpoint` survives a disk round-trip with its config
    and provenance intact."""
    head = LogImputationHead(d_model=16, hidden=8)
    head.fit_standardization(
        torch.full((N_TOKENS,), -1.0), torch.full((N_TOKENS,), 0.5),
    )
    ckpt = ImputationHeadCheckpoint.from_head(
        head, str(tiny_pretrain_checkpoint), 0.1, 0.2, {"train": [0.1], "val": [0.2]},
    )
    path = tmp_path / "head_best.pt"
    ckpt.save(path)

    loaded = ImputationHeadCheckpoint.load(path)
    assert loaded.head_config == {
        "d_model": 16, "hidden": 8, "standardize": True, "dropout": 0.0,
    }
    assert loaded.encoder_checkpoint == str(tiny_pretrain_checkpoint)
    assert torch.equal(
        loaded.build_head().target_std, head.target_std,
    )
