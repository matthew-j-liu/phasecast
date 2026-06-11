"""Tests for `cnmineral.training.{checkpoint, io}`.

Three contracts:
- `Checkpoint.from_models` + `.save` + `.load` + `.apply_to` round-trip
  the dict-shaped monitor, val_loss, optimizer state dict, etc.
- `checkpoint_tag` produces a stable, filesystem-safe namespacing tag.
- `evaluate_on_test` runs the in-memory model over the test split
  and returns a finite float.
"""

from pathlib import Path

import pytest
import torch

from conftest import _tiny_config

from cnmineral.models.encoder import Encoder
from cnmineral.models.pretrain_head import PretrainHead
from cnmineral.training import (
    Checkpoint,
    checkpoint_tag,
    create_dataset,
    evaluate_on_test,
    reserve_test,
    run_pretraining,
)


def test_checkpoint_class_roundtrip(tmp_path) -> None:
    """from_models + save + load + apply_to recovers the original weights
    bit-exact, and the carried metadata (epoch, train_loss, val_loss,
    monitor) survives the disk round-trip."""
    torch.manual_seed(0)
    encoder_a = Encoder(d_model=32, n_layers=2, n_heads=4,
                        frac_resolution=200)
    head_a = PretrainHead(d_model=32)
    optimizer = torch.optim.AdamW(
        list(encoder_a.parameters()) + list(head_a.parameters()), lr=1e-3,
    )
    config = _tiny_config()

    ckpt_path = tmp_path / "ckpt.pt"
    ckpt_a = Checkpoint.from_models(
        encoder_a, head_a, optimizer, epoch=1,
        config=config, train_loss=0.123, val_loss=0.31,
        monitor={"train": [0.5, 0.4, 0.3], "val": [0.55, 0.42, 0.31]},
    )
    ckpt_a.save(ckpt_path)

    torch.manual_seed(99)  # different init for the fresh instances
    encoder_b = Encoder(d_model=32, n_layers=2, n_heads=4,
                        frac_resolution=200)
    head_b = PretrainHead(d_model=32)
    ckpt_b = Checkpoint.load(ckpt_path)
    ckpt_b.apply_to(encoder_b, head_b)

    for (n_a, p_a), (n_b, p_b) in zip(
        encoder_a.named_parameters(), encoder_b.named_parameters(),
    ):
        assert n_a == n_b
        torch.testing.assert_close(p_a, p_b)
    for (n_a, p_a), (n_b, p_b) in zip(
        head_a.named_parameters(), head_b.named_parameters(),
    ):
        assert n_a == n_b
        torch.testing.assert_close(p_a, p_b)
    assert ckpt_b.epoch == 1
    assert ckpt_b.train_loss == pytest.approx(0.123)
    assert ckpt_b.val_loss == pytest.approx(0.31)
    assert ckpt_b.monitor == {
        "train": [0.5, 0.4, 0.3], "val": [0.55, 0.42, 0.31],
    }


def test_checkpoint_with_test_results_creates_new_instance() -> None:
    """with_test_results returns a fresh Checkpoint with updated test
    fields; the original instance is unchanged."""
    encoder = Encoder(d_model=32, n_layers=2, n_heads=4,
                      frac_resolution=200)
    head = PretrainHead(d_model=32)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()), lr=1e-3,
    )
    original = Checkpoint.from_models(
        encoder, head, optimizer, epoch=1, config=_tiny_config(),
        train_loss=0.1, val_loss=0.2, monitor={"train": [0.1], "val": [0.2]},
    )
    assert original.test_loss is None
    assert original.test_indices is None

    updated = original.with_test_results(0.05, [0, 1, 2])
    assert updated.test_loss == pytest.approx(0.05)
    assert updated.test_indices == [0, 1, 2]
    # original untouched
    assert original.test_loss is None
    assert original.test_indices is None
    # other fields unchanged
    assert updated.epoch == original.epoch
    assert updated.val_loss == original.val_loss


def test_checkpoint_tag_stable_and_filesystem_safe() -> None:
    """Tag derives from parent_dir + filename stem with non-alnum chars
    replaced by underscores; reruns on the same path produce the same
    string (study-storage namespacing depends on this)."""
    p = Path("checkpoints/pretrain_v3_smoke/epoch_001.pt")
    tag = checkpoint_tag(p)
    assert tag == "pretrain_v3_smoke_epoch_001"
    # idempotent
    assert checkpoint_tag(p) == tag
    # special chars are sanitized
    tricky = Path("checkpoints/run name (v3)/final ckpt.pt")
    sanitized = checkpoint_tag(tricky)
    assert all(c.isalnum() or c in "_-" for c in sanitized), sanitized


def test_evaluate_on_test_returns_finite() -> None:
    """Train a tiny model, evaluate the in-memory model on test, get finite float."""
    config = _tiny_config()
    dataset = create_dataset(config)
    splits = reserve_test(dataset, config)
    result = run_pretraining(
        dataset, splits["trainval"], config, device=torch.device("cpu"),
    )
    test_loss = evaluate_on_test(
        result["encoder"], result["head"],
        dataset, splits["test"], config,
        device=torch.device("cpu"),
    )
    assert isinstance(test_loss, float)
    assert torch.isfinite(torch.tensor(test_loss))
