"""Tests for `cnmineral.training.ssl_pretrain`: run_pretraining + run_single
+ the best/last checkpoint policy.

Eleven contracts:
- One forward/backward/step actually updates the encoder.
- The SSL log-space objective updates the encoder in one step.
- run_pretraining returns the expected result-dict keys.
- The objective runs end-to-end and returns matching-shaped loss_stats.
- The monitor dict is shaped right both in-memory and on disk.
- run_single returns finite test_loss + stamps the store's best.
- run_single writes a run_summary.json mirroring the stamped best.
- test_split=0 records None at the API surface.
- Only best + last land in the store directory (no per-epoch files).
- The store's best carries the minimum val loss seen across the run.
- The store's last stays without test_loss after run_single.
"""

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from conftest import _tiny_config, _write_yaml

from cnmineral.data.tokens import N_TOKENS, mask_for_ssl
from cnmineral.models.encoder import Encoder
from cnmineral.models.pretrain_head import PretrainHead
from cnmineral.training import (
    CheckpointStore,
    create_dataset,
    reserve_test,
    run_pretraining,
    run_single,
)
from cnmineral.training.pretrain_objective import compute_loss_stats, make_loss_fn


def test_one_step_runs() -> None:
    """End-to-end one training step; encoder weights must change."""
    torch.manual_seed(0)
    encoder = Encoder(d_model=32, n_layers=2, n_heads=4, dropout=0.0,
                      frac_resolution=200)
    head = PretrainHead(d_model=32)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=3e-4,
    )

    ds = create_dataset(_tiny_config())
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    snapshot = {n: p.detach().clone() for n, p in encoder.named_parameters()}

    masked = mask_for_ssl(batch["src"], batch["frac"], mask_prob=0.5)
    h = encoder(masked.src_corrupted, masked.frac_corrupted)
    predicted = head(h)
    # trivial differentiable masked MSE -- this test pins the train step
    # mechanics, not the production objective (covered separately).
    m = masked.mask_positions
    loss = ((predicted - masked.target_frac) ** 2 * m).sum() / m.sum().clamp(min=1)
    assert torch.isfinite(loss), f"loss not finite: {loss.item()}"

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    changed = [
        name for name, param in encoder.named_parameters()
        if not torch.equal(param.detach(), snapshot[name])
    ]
    assert changed, (
        "no encoder parameter changed after one optimizer step; "
        "training is silently a no-op"
    )


def test_val_split_zero_raises() -> None:
    """run_pretraining requires a val curve (HPO prunes on it and the
    best-checkpoint policy monitors it); val_split <= 0 must fail
    loudly at entry instead of producing an empty val loader."""
    config = _tiny_config()
    config["val_split"] = 0
    dataset = create_dataset(config)
    with pytest.raises(ValueError, match="val_split"):
        run_pretraining(
            dataset, list(range(len(dataset))), config,
            device=torch.device("cpu"),
        )


def test_run_pretraining_returns_result_dict() -> None:
    config = _tiny_config()
    dataset = create_dataset(config)
    splits = reserve_test(dataset, config)
    result = run_pretraining(
        dataset, splits["trainval"], config, device=torch.device("cpu"),
    )
    assert set(result.keys()) >= {
        "val_loss", "train_loss", "encoder", "head", "monitor", "final_epoch",
    }
    assert isinstance(result["val_loss"], float)
    assert torch.isfinite(torch.tensor(result["val_loss"]))
    assert result["final_epoch"] == 1
    assert isinstance(result["encoder"], Encoder)
    assert isinstance(result["head"], PretrainHead)


def test_run_pretraining_monitor_shape(tmp_path: Path) -> None:
    """monitor in result dict + on disk has dict shape with train+val keys."""
    config = _tiny_config()
    config["epochs"] = 2
    dataset = create_dataset(config)
    splits = reserve_test(dataset, config)
    store = CheckpointStore(
        tmp_path,
        best_filename="epoch_best.pt",
        last_filename="epoch_last.pt",
    )
    result = run_pretraining(
        dataset, splits["trainval"], config, device=torch.device("cpu"),
        store=store,
    )
    # in-memory monitor
    assert set(result["monitor"].keys()) == {"train", "val"}
    assert len(result["monitor"]["train"]) == 2
    assert len(result["monitor"]["val"]) == 2

    # on-disk monitor round-trips the dict shape (the last Checkpoint
    # lands once after the loop and reflects the final-epoch state)
    assert store.last_exists()
    last = store.load_last()
    assert set(last.monitor.keys()) == {"train", "val"}
    assert len(last.monitor["train"]) == 2


def test_run_single_returns_test_loss(tmp_path: Path) -> None:
    """run_single returns a dict carrying finite val_loss, train_loss, test_loss."""
    config = _tiny_config(checkpoint_dir=str(tmp_path / "ckpts"))
    config_path = _write_yaml(tmp_path / "cfg.yaml", config)

    result = run_single(config_path, device=torch.device("cpu"))

    assert set(result.keys()) >= {"val_loss", "train_loss", "test_loss"}
    assert isinstance(result["test_loss"], float)
    assert torch.isfinite(torch.tensor(result["test_loss"]))

    # store's best now carries the test_loss field populated
    store = CheckpointStore(
        tmp_path / "ckpts",
        best_filename="epoch_best.pt",
        last_filename="epoch_last.pt",
    )
    assert store.best_exists()
    best = store.load_best()
    assert best.test_loss == pytest.approx(result["test_loss"])
    assert best.test_indices is not None


def test_run_single_writes_results_artifacts(
    tmp_path: Path, pretrain_runs_root: Path,
) -> None:
    """run_single writes run_summary.json + loss_curve.png under the
    per-run results dir.  Headline fields match the stamped best
    checkpoint / returned result dict; the monitor is the full-run
    curve, not the best checkpoint's truncated copy."""
    config = _tiny_config(checkpoint_dir=str(tmp_path / "ckpts"))
    config_path = _write_yaml(tmp_path / "cfg.yaml", config)

    result = run_single(config_path, device=torch.device("cpu"))

    results_dir = pretrain_runs_root / "ckpts_epoch_best"
    summary_path = results_dir / "run_summary.json"
    assert summary_path.exists()
    with open(summary_path) as f:
        summary = json.load(f)
    assert summary["test_loss"] == pytest.approx(result["test_loss"])
    assert summary["config"]["checkpoint_dir"] == str(tmp_path / "ckpts")
    assert set(summary["monitor"]) == {"train", "val"}
    assert len(summary["monitor"]["val"]) == config["epochs"]
    assert set(summary) == {
        "epoch", "train_loss", "val_loss", "test_loss",
        "monitor", "git_sha", "saved_at", "config",
    }

    curve_path = results_dir / "loss_curve.png"
    assert curve_path.exists()
    assert curve_path.stat().st_size > 0


def test_test_split_zero_records_none(tmp_path: Path) -> None:
    """test_split=0 should produce test_loss=None (no nan in user-facing API)."""
    config = _tiny_config(checkpoint_dir=str(tmp_path / "ckpts"))
    config["test_split"] = 0.0
    config_path = _write_yaml(tmp_path / "cfg.yaml", config)

    result = run_single(config_path, device=torch.device("cpu"))

    assert result["test_loss"] is None
    # Store's best is stamped with test_loss=None + test_indices=[]
    # under test_split=0: the stamp-step still fires (no guard against
    # empty splits), but evaluate_on_test produced NaN -> None and the
    # reserved-test list is empty.
    store = CheckpointStore(
        tmp_path / "ckpts",
        best_filename="epoch_best.pt",
        last_filename="epoch_last.pt",
    )
    best = store.load_best()
    assert best.test_loss is None
    assert best.test_indices == []


def test_best_and_last_are_the_only_files_after_run(tmp_path: Path) -> None:
    """After a multi-epoch run, the store holds both a best and a last
    snapshot, and no other .pt files appear in the directory."""
    config = _tiny_config()
    config["epochs"] = 3
    dataset = create_dataset(config)
    splits = reserve_test(dataset, config)
    store = CheckpointStore(
        tmp_path,
        best_filename="epoch_best.pt",
        last_filename="epoch_last.pt",
    )
    run_pretraining(
        dataset, splits["trainval"], config, device=torch.device("cpu"),
        store=store,
    )
    assert store.best_exists()
    assert store.last_exists()
    # And nothing else lands in the directory.
    other_pt_files = sorted(
        p.name for p in tmp_path.glob("*.pt")
        if p != store.best_path and p != store.last_path
    )
    assert other_pt_files == [], other_pt_files


def test_epoch_best_holds_minimum_val_loss(tmp_path: Path) -> None:
    """The store's best snapshot carries the minimum val_loss seen
    across the run, and its `epoch` field matches the epoch number at
    which the minimum was reached."""
    config = _tiny_config()
    config["epochs"] = 3
    dataset = create_dataset(config)
    splits = reserve_test(dataset, config)
    store = CheckpointStore(
        tmp_path,
        best_filename="epoch_best.pt",
        last_filename="epoch_last.pt",
    )
    result = run_pretraining(
        dataset, splits["trainval"], config, device=torch.device("cpu"),
        store=store,
    )
    val_curve = result["monitor"]["val"]
    expected_best_epoch = val_curve.index(min(val_curve)) + 1
    assert result["best_epoch"] == expected_best_epoch
    assert result["best_val_loss"] == pytest.approx(min(val_curve))

    best = store.load_best()
    assert best.epoch == expected_best_epoch
    assert best.val_loss == pytest.approx(min(val_curve))


def test_best_and_last_have_distinct_model_states(
    tmp_path: Path,
) -> None:
    """When val isn't monotonic, the store's best and last snapshots
    must hold genuinely different model weights.  This pins that the
    best/last policy actually captures both checkpoints (not
    accidentally writing the same state to both), which is a
    precondition for the test_loss-on-best fix to matter.
    """
    config = _tiny_config()
    config["epochs"] = 3
    # lr high enough that val overshoots after its minimum, so the
    # best epoch lands before the final one and the two snapshots
    # genuinely differ (at the default lr the corpus-fixture val
    # curve improves monotonically and this test would self-skip).
    config["lr"] = 0.1
    dataset = create_dataset(config)
    splits = reserve_test(dataset, config)
    store = CheckpointStore(
        tmp_path,
        best_filename="epoch_best.pt",
        last_filename="epoch_last.pt",
    )
    result = run_pretraining(
        dataset, splits["trainval"], config, device=torch.device("cpu"),
        store=store,
    )
    if result["best_epoch"] == result["final_epoch"]:
        pytest.skip("best == last by epoch; can't pin distinct states")

    best = store.load_best()
    last = store.load_last()
    differ_somewhere = any(
        not torch.equal(best.encoder_state[k], last.encoder_state[k])
        for k in best.encoder_state
    )
    assert differ_somewhere, (
        "store best + last have identical encoder weights despite "
        "best_epoch != final_epoch"
    )


def test_run_single_stamps_test_results_on_store_best(
    tmp_path: Path,
) -> None:
    """End-to-end: drive run_single against a tmp store, then verify
    `store.load_best().test_loss` is populated AND matches the value
    returned in the run_single result dict.  Pins that the stamp
    travels through CheckpointStore (not back-door file writes)."""
    config = _tiny_config(checkpoint_dir=str(tmp_path / "ckpts"))
    config["epochs"] = 3
    config_path = _write_yaml(tmp_path / "cfg.yaml", config)
    result = run_single(config_path, device=torch.device("cpu"))

    store = CheckpointStore(
        tmp_path / "ckpts",
        best_filename="epoch_best.pt",
        last_filename="epoch_last.pt",
    )
    best = store.load_best()
    assert best.test_loss is not None
    assert best.test_loss == pytest.approx(result["test_loss"])
    assert best.test_indices is not None
    assert len(best.test_indices) > 0


def test_last_carries_test_loss_none_after_run_single(
    tmp_path: Path,
) -> None:
    """The store's best gets the test_loss stamp after run_single;
    the last snapshot stays without it (it's the resume snapshot,
    not the headline)."""
    config = _tiny_config(checkpoint_dir=str(tmp_path / "ckpts"))
    config_path = _write_yaml(tmp_path / "cfg.yaml", config)
    run_single(config_path, device=torch.device("cpu"))

    store = CheckpointStore(
        tmp_path / "ckpts",
        best_filename="epoch_best.pt",
        last_filename="epoch_last.pt",
    )
    last = store.load_last()
    assert last.test_loss is None
    assert last.test_indices is None


def test_run_single_writes_to_configured_filenames(tmp_path: Path) -> None:
    """Custom filenames in the YAML land on disk; the default names
    do NOT.  Pins the contract that lets two runs share a
    `checkpoint_dir` without overwriting each other."""
    config = _tiny_config(checkpoint_dir=str(tmp_path / "ckpts"))
    config["checkpoint_best_filename"] = "attempt2_best.pt"
    config["checkpoint_last_filename"] = "attempt2_last.pt"
    config_path = _write_yaml(tmp_path / "cfg.yaml", config)

    run_single(config_path, device=torch.device("cpu"))

    landed = sorted(p.name for p in (tmp_path / "ckpts").glob("*.pt"))
    assert landed == ["attempt2_best.pt", "attempt2_last.pt"], landed


def test_log_loss_one_step_updates_encoder() -> None:
    """A standardized log-space objective produces nonzero encoder gradients --
    the rebalanced loss trains the encoder (the trace-supervision fix), not just
    the head."""
    torch.manual_seed(0)
    ds = create_dataset(_tiny_config())
    encoder = Encoder(d_model=32, n_layers=2, n_heads=4, dropout=0.0,
                      frac_resolution=200)
    head = PretrainHead(d_model=32)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()), lr=3e-4,
    )

    stats = compute_loss_stats(
        ds, list(range(len(ds))), batch_size=32, num_workers=0,
    )
    loss_fn = make_loss_fn(stats)

    batch = next(iter(DataLoader(ds, batch_size=8, shuffle=False)))
    snapshot = {n: p.detach().clone() for n, p in encoder.named_parameters()}

    masked = mask_for_ssl(batch["src"], batch["frac"], mask_prob=0.5)
    predicted = head(encoder(masked.src_corrupted, masked.frac_corrupted))
    loss = loss_fn(predicted, masked.target_frac, masked.mask_positions)
    assert torch.isfinite(loss), f"loss not finite: {loss.item()}"

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    changed = [
        name for name, param in encoder.named_parameters()
        if not torch.equal(param.detach(), snapshot[name])
    ]
    assert changed, "log-space loss left the encoder unchanged after one step"


def test_run_pretraining_returns_loss_stats() -> None:
    """The objective runs end-to-end and returns matching-shaped loss_stats."""
    config = _tiny_config()
    dataset = create_dataset(config)
    splits = reserve_test(dataset, config)
    result = run_pretraining(
        dataset, splits["trainval"], config, device=torch.device("cpu"),
    )
    assert torch.isfinite(torch.tensor(result["val_loss"]))
    assert torch.isfinite(torch.tensor(result["train_loss"]))

    stats = result["loss_stats"]
    assert set(stats) == {"target_mean", "target_std"}
    assert stats["target_mean"].shape == (N_TOKENS,)
    assert stats["target_std"].shape == (N_TOKENS,)
