"""Unit tests for the SSL pretrain driver.

These pin the contracts the script relies on:

- The forward/backward/step recipe actually trains (gradient reaches
  the encoder; loss is finite).
- The synthetic dataset satisfies the v3 padding convention.
- reserve_test splits a dataset into trainval + test, deterministic on seed.
- Checkpoint save/load round-trips the dict-shaped monitor + val_loss.
- run_pretraining returns a finite val loss and tracks both train + val curves.
- The Optuna objective integration end-to-end.
- evaluate_on_test loads a checkpoint and returns a finite test loss.
- An Optuna study at a JournalStorage path survives recreation
  (load_if_exists=True).

Run with:
    pytest tests/test_pretrain_script.py -v
"""

import json
import sys
from pathlib import Path

import optuna
import pytest
import torch
import yaml
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from torch.utils.data import DataLoader

from cnmineral.data.synthetic import SyntheticBulkChemistryDataset
from cnmineral.data.tokens import (
    N_TOKENS,
    assert_padding_convention,
    mask_for_ssl,
)
from cnmineral.losses.pretrain_loss import masked_element_mse
from cnmineral.models.encoder import Encoder
from cnmineral.models.pretrain_head import PretrainHead

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pretrain  # noqa: E402  -- needs the sys.path insert above


def _tiny_config(checkpoint_dir: Path | None = None) -> dict:
    """A config small enough that run_pretraining finishes in milliseconds.

    synthetic_n=16 with test_frac=0.25 -> 12 trainval / 4 test;
    val_frac=0.25 inside run_pretraining -> 9 train / 3 val.  All
    non-empty so the split + train + val passes all exercise.
    """
    return {
        "epochs": 1,
        "batch_size": 4,
        "lr": 3e-4,
        "mask_prob": 0.25,
        "weight_decay": 0.01,
        "dropout": 0.0,
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 4,
        "frac_resolution": 200,
        "data_source": "synthetic",
        "synthetic_n": 16,
        "test_frac": 0.25,
        "val_frac": 0.25,
        "num_workers": 0,
        "seed": 0,
        "checkpoint_dir": checkpoint_dir,
    }


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

    ds = SyntheticBulkChemistryDataset(n=8, seed=0)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    snapshot = {n: p.detach().clone() for n, p in encoder.named_parameters()}

    masked = mask_for_ssl(batch["src"], batch["frac"], mask_prob=0.5)
    h = encoder(masked.src_corrupted, masked.frac_corrupted)
    predicted = head(h)
    loss = masked_element_mse(
        predicted, masked.target_frac, masked.mask_positions,
    )
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


class TestSyntheticDataset:
    def test_shape_and_dtype(self) -> None:
        ds = SyntheticBulkChemistryDataset(n=4, seed=0)
        assert len(ds) == 4
        item = ds[0]
        assert set(item.keys()) == {"src", "frac", "sample_id"}
        assert item["src"].shape == (N_TOKENS,)
        assert item["frac"].shape == (N_TOKENS,)
        assert item["src"].dtype == torch.long
        assert item["frac"].dtype == torch.float32
        assert isinstance(item["sample_id"], int)

    def test_obeys_padding_convention(self) -> None:
        ds = SyntheticBulkChemistryDataset(n=8, seed=1)
        loader = DataLoader(ds, batch_size=4)
        for batch in loader:
            assert_padding_convention(batch["src"], batch["frac"])

    def test_reproducible(self) -> None:
        a = SyntheticBulkChemistryDataset(n=4, seed=7)
        b = SyntheticBulkChemistryDataset(n=4, seed=7)
        torch.testing.assert_close(a[0]["frac"], b[0]["frac"])


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    encoder_a = Encoder(d_model=32, n_layers=2, n_heads=4,
                        frac_resolution=200)
    head_a = PretrainHead(d_model=32)
    optimizer = torch.optim.AdamW(
        list(encoder_a.parameters()) + list(head_a.parameters()), lr=1e-3,
    )
    config = _tiny_config()

    ckpt_path = tmp_path / "ckpt.pt"
    pretrain.save_checkpoint(
        ckpt_path, encoder_a, head_a, optimizer, epoch=1,
        config=config, train_loss=0.123, val_loss=0.31,
        monitor={"train": [0.5, 0.4, 0.3], "val": [0.55, 0.42, 0.31]},
    )

    torch.manual_seed(99)  # different init for the fresh instances
    encoder_b = Encoder(d_model=32, n_layers=2, n_heads=4,
                        frac_resolution=200)
    head_b = PretrainHead(d_model=32)
    payload = pretrain.load_checkpoint(ckpt_path, encoder_b, head_b)

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
    assert payload["epoch"] == 1
    assert payload["train_loss"] == pytest.approx(0.123)
    assert payload["val_loss"] == pytest.approx(0.31)
    assert payload["monitor"] == {
        "train": [0.5, 0.4, 0.3], "val": [0.55, 0.42, 0.31],
    }


def test_reserve_test_split_sizes() -> None:
    """reserve_test splits to expected sizes, deterministic on seed."""
    config = _tiny_config()
    dataset = pretrain.create_dataset(config)
    splits_a = pretrain.reserve_test(dataset, config)
    splits_b = pretrain.reserve_test(dataset, config)
    # n=16, test_frac=0.25 -> 4 test, 12 trainval
    assert len(splits_a["test"]) == 4
    assert len(splits_a["trainval"]) == 12
    # disjoint + cover everything
    assert set(splits_a["test"]).isdisjoint(set(splits_a["trainval"]))
    assert sorted(splits_a["test"] + splits_a["trainval"]) == list(range(16))
    # same seed -> same indices
    assert splits_a["test"] == splits_b["test"]
    assert splits_a["trainval"] == splits_b["trainval"]


def test_reserve_test_synthetic_falls_back_to_unstratified() -> None:
    """Synthetic datasets lack qap_names; _qap_labels_for_indices
    returns None so train_test_split runs unstratified.  This test
    pins that the synthetic code path still works end-to-end after
    the stratify wiring landed."""
    from cnmineral.data.synthetic import SyntheticBulkChemistryDataset
    dataset = SyntheticBulkChemistryDataset(n=16, seed=0)
    assert not hasattr(dataset, "qap_names")
    labels = pretrain._qap_labels_for_indices(dataset, range(len(dataset)))
    assert labels is None


def test_reserve_test_stratifies_when_qap_names_present(
    corpus_subset_path: Path,
) -> None:
    """On a dataset that carries qap_names, reserve_test preserves
    per-class proportions in train/test within tolerance.

    Uses min_class_count=5 so every surviving class is large enough
    that stratification can actually divide it; uses test_frac=0.25
    so smallest classes (>=5 samples) land >=1 row in test."""
    from cnmineral.data.bulk_chemistry import BulkChemistryDataset
    dataset = BulkChemistryDataset(
        csv_path=corpus_subset_path, min_class_count=5,
    )
    config = {"seed": 0, "test_frac": 0.25}
    splits = pretrain.reserve_test(dataset, config)
    test_set = set(splits["test"])
    trainval_set = set(splits["trainval"])

    from collections import Counter
    total = Counter(dataset.qap_names)
    in_test = Counter(dataset.qap_names[i] for i in test_set)

    # every class with >= 5 samples must appear in BOTH splits
    for cls, n in total.items():
        assert in_test[cls] >= 1, f"class {cls!r} (n={n}) missing from test"
        assert in_test[cls] < n, f"class {cls!r} (n={n}) entirely in test"
        # proportional preservation: in_test/total approximately test_frac
        # tolerance is wide for small classes (5-sample class -> 1 in test
        # is 20%, ratio swings if class has 6 samples -> 1 in test = 16.7%)
        ratio = in_test[cls] / n
        assert 0.10 <= ratio <= 0.40, (
            f"class {cls!r} (n={n}) test ratio {ratio:.2f} outside "
            f"[0.10, 0.40]; expected near 0.25"
        )

    # train + test = full corpus, no overlap
    assert test_set.isdisjoint(trainval_set)
    assert len(test_set) + len(trainval_set) == len(dataset)


def test_run_pretraining_returns_result_dict() -> None:
    config = _tiny_config()
    dataset = pretrain.create_dataset(config)
    splits = pretrain.reserve_test(dataset, config)
    result = pretrain.run_pretraining(
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
    config = _tiny_config(checkpoint_dir=tmp_path)
    config["epochs"] = 2
    dataset = pretrain.create_dataset(config)
    splits = pretrain.reserve_test(dataset, config)
    result = pretrain.run_pretraining(
        dataset, splits["trainval"], config, device=torch.device("cpu"),
    )
    # in-memory monitor
    assert set(result["monitor"].keys()) == {"train", "val"}
    assert len(result["monitor"]["train"]) == 2
    assert len(result["monitor"]["val"]) == 2

    # on-disk monitor round-trips the dict shape
    ckpt_path = tmp_path / "epoch_002.pt"
    assert ckpt_path.exists()
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert set(payload["monitor"].keys()) == {"train", "val"}
    assert len(payload["monitor"]["train"]) == 2


def test_sample_hyperparameter_dispatches_each_type() -> None:
    """_sample_hyperparameter routes float, int, categorical to the right trial method."""
    trial = optuna.trial.FixedTrial({"lr": 1e-4, "n_layers": 3, "batch_size": 64})
    assert pretrain._sample_hyperparameter(
        trial, "lr", {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
    ) == 1e-4
    assert pretrain._sample_hyperparameter(
        trial, "n_layers", {"type": "int", "low": 2, "high": 4},
    ) == 3
    assert pretrain._sample_hyperparameter(
        trial, "batch_size", {"type": "categorical", "choices": [32, 64, 128]},
    ) == 64

    with pytest.raises(ValueError, match="unknown search_space type"):
        pretrain._sample_hyperparameter(trial, "x", {"type": "bogus"})


def test_objective_end_to_end() -> None:
    """Tiny search_space-driven study: optimize 1 trial on synthetic, get finite val."""
    base = _tiny_config()
    dataset = pretrain.create_dataset(base)
    splits = pretrain.reserve_test(dataset, base)
    search_space = {
        "lr": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
        "mask_prob": {"type": "float", "low": 0.2, "high": 0.4},
    }

    def objective(trial: optuna.trial.Trial) -> float:
        trial_hyperparameters = {
            name: pretrain._sample_hyperparameter(trial, name, spec)
            for name, spec in search_space.items()
        }
        merged = {**base, **trial_hyperparameters}
        result = pretrain.run_pretraining(
            dataset, splits["trainval"], merged,
            device=torch.device("cpu"), trial=trial,
        )
        return result["val_loss"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1)
    assert len(study.trials) == 1
    assert torch.isfinite(torch.tensor(study.best_value))


def test_evaluate_on_test_returns_finite() -> None:
    """Train a tiny model, evaluate the in-memory model on test, get finite float."""
    config = _tiny_config()
    dataset = pretrain.create_dataset(config)
    splits = pretrain.reserve_test(dataset, config)
    result = pretrain.run_pretraining(
        dataset, splits["trainval"], config, device=torch.device("cpu"),
    )
    test_loss = pretrain.evaluate_on_test(
        result["encoder"], result["head"],
        dataset, splits["test"], config,
        device=torch.device("cpu"),
    )
    assert isinstance(test_loss, float)
    assert torch.isfinite(torch.tensor(test_loss))


def _write_yaml(path: Path, payload: dict) -> Path:
    with open(path, "w") as f:
        yaml.safe_dump(payload, f)
    return path


def test_run_single_returns_test_loss(tmp_path: Path) -> None:
    """run_single returns a dict carrying finite val_loss, train_loss, test_loss."""
    config = _tiny_config(checkpoint_dir=str(tmp_path / "ckpts"))
    config_path = _write_yaml(tmp_path / "cfg.yaml", config)

    result = pretrain.run_single(config_path, device=torch.device("cpu"))

    assert set(result.keys()) >= {"val_loss", "train_loss", "test_loss"}
    assert isinstance(result["test_loss"], float)
    assert torch.isfinite(torch.tensor(result["test_loss"]))

    # final checkpoint should now carry the test_loss field populated
    final_ckpt = tmp_path / "ckpts" / "epoch_001.pt"
    assert final_ckpt.exists()
    payload = torch.load(final_ckpt, map_location="cpu", weights_only=False)
    assert payload["test_loss"] == pytest.approx(result["test_loss"])
    assert payload["test_indices"] is not None


def test_run_hpo_summary_carries_test_loss(tmp_path: Path) -> None:
    """run_hpo's summary JSON includes the final production run + test loss."""
    base = _tiny_config(checkpoint_dir=str(tmp_path / "ckpts"))
    hpo_config = {
        "study_name": "smoke_test",
        "storage_path": str(tmp_path / "smoke.journal"),
        "sampler_seed": 42,
        "n_trials": 1,
        "n_startup_trials": 1,
        "pruner_warmup_steps": 1,
        "hpo_run_epochs": 1,
        "base": base,
        "search_space": {
            "lr": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
        },
    }
    config_path = _write_yaml(tmp_path / "hpo.yaml", hpo_config)

    pretrain.run_hpo(config_path, device=torch.device("cpu"))

    summary_path = tmp_path / "smoke_test_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    for key in (
        "best_trial_val_loss", "final_train_loss",
        "final_val_loss", "final_test_loss",
    ):
        assert key in summary, f"summary missing {key}"
        assert summary[key] is not None
        assert torch.isfinite(torch.tensor(float(summary[key])))


def test_test_frac_zero_records_none(tmp_path: Path) -> None:
    """test_frac=0 should produce test_loss=None (no nan in user-facing API)."""
    config = _tiny_config(checkpoint_dir=str(tmp_path / "ckpts"))
    config["test_frac"] = 0.0
    config_path = _write_yaml(tmp_path / "cfg.yaml", config)

    result = pretrain.run_single(config_path, device=torch.device("cpu"))

    assert result["test_loss"] is None
    final_ckpt = tmp_path / "ckpts" / "epoch_001.pt"
    payload = torch.load(final_ckpt, map_location="cpu", weights_only=False)
    assert payload["test_loss"] is None
    assert payload["test_indices"] == []


def test_study_resumes(tmp_path: Path) -> None:
    """A JournalStorage study must survive recreation with load_if_exists=True."""
    storage_path = tmp_path / "study.journal"

    def tiny_objective(trial: optuna.trial.Trial) -> float:
        x = trial.suggest_float("x", 0.0, 1.0)
        return (x - 0.5) ** 2

    storage_a = JournalStorage(JournalFileBackend(str(storage_path)))
    study_a = optuna.create_study(
        study_name="resume_test",
        storage=storage_a,
        direction="minimize",
        load_if_exists=True,
    )
    study_a.optimize(tiny_objective, n_trials=2)
    assert len(study_a.trials) == 2

    storage_b = JournalStorage(JournalFileBackend(str(storage_path)))
    study_b = optuna.create_study(
        study_name="resume_test",
        storage=storage_b,
        direction="minimize",
        load_if_exists=True,
    )
    assert len(study_b.trials) == 2
