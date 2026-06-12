"""Tests for `cnmineral.training.hpo`: Optuna sweep + study resume.

Six contracts:
- `_sample_hyperparameter` routes each spec type to the right
  `trial.suggest_*`.
- The Optuna objective integrates end-to-end with `run_pretraining`.
- `run_hpo` writes a summary JSON populated with final losses.
- HPO trials don't write to base.checkpoint_dir (only the post-sweep
  production run does).
- The production run restores `base.epochs` over `hpo_run_epochs`.
- A `JournalStorage` study survives recreation (load_if_exists=True).
"""

import json
from pathlib import Path

import optuna
import pytest
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from conftest import _tiny_config, _write_yaml

from cnmineral.training import (
    CheckpointStore,
    create_dataset,
    reserve_test,
    run_hpo,
    run_pretraining,
)
from cnmineral.training.hpo import _sample_hyperparameter


def test_sample_hyperparameter_dispatches_each_type() -> None:
    """_sample_hyperparameter routes float, int, categorical to the right trial method."""
    trial = optuna.trial.FixedTrial({"lr": 1e-4, "n_layers": 3, "batch_size": 64})
    assert _sample_hyperparameter(
        trial, "lr", {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
    ) == 1e-4
    assert _sample_hyperparameter(
        trial, "n_layers", {"type": "int", "low": 2, "high": 4},
    ) == 3
    assert _sample_hyperparameter(
        trial, "batch_size", {"type": "categorical", "choices": [32, 64, 128]},
    ) == 64

    with pytest.raises(ValueError, match="unknown search_space type"):
        _sample_hyperparameter(trial, "x", {"type": "bogus"})


def test_objective_end_to_end() -> None:
    """Tiny search_space-driven study: optimize 1 trial on the corpus fixture, get finite val."""
    base = _tiny_config()
    dataset = create_dataset(base)
    splits = reserve_test(dataset, base)
    search_space = {
        "lr": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
        "mask_prob": {"type": "float", "low": 0.2, "high": 0.4},
    }

    def objective(trial: optuna.trial.Trial) -> float:
        trial_hyperparameters = {
            name: _sample_hyperparameter(trial, name, spec)
            for name, spec in search_space.items()
        }
        merged = {**base, **trial_hyperparameters}
        result = run_pretraining(
            dataset, splits["trainval"], merged,
            device=torch.device("cpu"), trial=trial,
        )
        return result["val_loss"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1)
    assert len(study.trials) == 1
    assert torch.isfinite(torch.tensor(study.best_value))


def test_run_hpo_summary_carries_test_loss(
    tmp_path: Path, pretrain_results_root: Path,
) -> None:
    """run_hpo's summary JSON includes the final production run + test
    loss, the study figures land next to the summary, and the
    production run writes the same per-run artifacts as run_single."""
    base = _tiny_config(checkpoint_dir=str(tmp_path / "ckpts"))
    hpo_config = {
        "study_name": "smoke_test",
        "storage_path": str(tmp_path / "smoke.journal"),
        "output_dir": str(tmp_path / "out"),
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

    run_hpo(config_path, device=torch.device("cpu"))

    summary_path = tmp_path / "out" / "smoke_test" / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["best_trial_value"] is not None
    assert torch.isfinite(torch.tensor(float(summary["best_trial_value"])))
    for key in ("train_loss", "val_loss", "test_loss"):
        assert key in summary["final"], f"final missing {key}"
        assert summary["final"][key] is not None
        assert torch.isfinite(torch.tensor(float(summary["final"][key])))

    for figure in ("hpo_history.png", "hpo_trial_curves.png"):
        assert (summary_path.parent / figure).stat().st_size > 0

    production_results = pretrain_results_root / "ckpts_epoch_best"
    assert (production_results / "run_summary.json").exists()
    assert (production_results / "loss_curve.png").stat().st_size > 0


def test_hpo_trial_does_not_write_checkpoints(tmp_path: Path) -> None:
    """run_hpo's objective drops `checkpoint_dir` for trial runs, so an
    HPO trial against `base.checkpoint_dir` writes nothing during trial
    execution.  The post-sweep production run (if it fires) is what
    populates the directory afterwards."""
    base = _tiny_config(checkpoint_dir=str(tmp_path / "ckpts"))
    hpo_config = {
        "study_name": "hpo_no_trial_writes",
        "storage_path": str(tmp_path / "hpo.journal"),
        "output_dir": str(tmp_path / "out"),
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
    run_hpo(config_path, device=torch.device("cpu"))

    # The only on-disk files in ckpts/ are the post-sweep production
    # run's best + last snapshots -- never per-trial writes.
    store = CheckpointStore(
        tmp_path / "ckpts",
        best_filename="epoch_best.pt",
        last_filename="epoch_last.pt",
    )
    assert store.best_exists()
    assert store.last_exists()
    other = sorted(
        p.name for p in (tmp_path / "ckpts").glob("*.pt")
        if p != store.best_path and p != store.last_path
    )
    assert other == [], other


def test_production_run_restores_base_epochs(tmp_path: Path) -> None:
    """Trials run at `hpo_run_epochs`; the post-sweep production run
    must be restored to `base.epochs`.  Pinned with the two values
    deliberately different -- if the restore breaks, a real sweep
    silently ships a production checkpoint trained at the trial
    budget."""
    base = _tiny_config(checkpoint_dir=str(tmp_path / "ckpts"))
    base["epochs"] = 2
    hpo_config = {
        "study_name": "epochs_restore",
        "storage_path": str(tmp_path / "epochs.journal"),
        "output_dir": str(tmp_path / "out"),
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
    run_hpo(config_path, device=torch.device("cpu"))

    # the trial reported exactly hpo_run_epochs intermediate values
    storage = JournalStorage(
        JournalFileBackend(str(tmp_path / "epochs.journal")),
    )
    study = optuna.load_study(study_name="epochs_restore", storage=storage)
    assert len(study.trials[0].intermediate_values) == 1

    # the production run trained for base.epochs
    store = CheckpointStore(
        tmp_path / "ckpts",
        best_filename="epoch_best.pt",
        last_filename="epoch_last.pt",
    )
    last = store.load_last()
    assert last.epoch == 2
    assert len(last.monitor["val"]) == 2
    assert last.config["epochs"] == 2


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
