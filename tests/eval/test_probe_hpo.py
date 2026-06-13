"""Tests for `cnmineral.eval.probe.{hpo, calibration}`.

End-to-end: a 2-trial sweep against a small random-init encoder on
the corpus fixture writes a usable summary JSON and respects
resumability (`load_if_exists=True`).
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

from cnmineral.eval.probe import build_report, run_calibration, run_hpo


def _build_smoke_pretrain_config(
    fixture_path: Path, ckpt_dir: Path,
) -> dict:
    """The smallest valid pretrain config that produces a checkpoint
    + dataset the probe HPO can drive against.  Mirrors the shape of
    `configs/pretrain/pretrain_production.yaml` but with a tiny
    encoder + the corpus fixture instead of the real CSV."""
    return {
        "seed": 0,
        "epochs": 1,
        "batch_size": 16,
        "lr": 3.0e-4,
        "mask_prob": 0.25,
        "weight_decay": 0.01,
        "dropout": 0.0,
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 2,
        "frac_resolution": 200,
        "num_workers": 0,
        "test_split": 0.20,
        "val_split": 0.20,
        "data_kwargs": {
            "csv_path": str(fixture_path),
            "rock_group": "igneous",
            "require_all_majors": True,
            "min_class_count": 2,
        },
        "checkpoint_dir": str(ckpt_dir),
        "checkpoint_best_filename": "epoch_best.pt",
        "checkpoint_last_filename": "epoch_last.pt",
    }


def _train_fixture_checkpoint(
    corpus_subset_path: Path, ckpt_dir: Path,
) -> Path:
    """Drive `cnmineral.training.run_single` for 1 epoch on the
    corpus fixture so we have a real checkpoint to probe-HPO against."""
    import yaml

    from cnmineral.training import run_single

    cfg_path = ckpt_dir / "pretrain_test.yaml"
    cfg = _build_smoke_pretrain_config(corpus_subset_path, ckpt_dir)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    run_single(cfg_path, device="cpu")
    return ckpt_dir / "epoch_best.pt"


def _write_probe_hpo_yaml(
    path: Path, study_name: str, storage_dir: Path, output_dir: Path,
    *, n_trials: int = 2,
) -> Path:
    """Hand-write a minimal probe_hpo.yaml that doesn't depend on
    `configs/probe/probe_hpo.yaml` parsing -- this way the test breaks
    if `run_hpo` mishandles the YAML, not if I edit the prod config.

    `output_dir` must be absolute because `probe_hpo.run_hpo` resolves it
    relative to `REPO_ROOT`; test fixtures stamp the absolute tmp path.
    """
    import yaml

    spec = {
        "study_name": study_name,
        "storage_path": str(storage_dir / "probe_hpo_{encoder_tag}.journal"),
        "output_dir": str(output_dir),
        "sampler_seed": 7,
        "n_trials": n_trials,
        "n_startup_trials": 2,
        "pruner_warmup_steps": 0,
        "probe_target": "qap_name",
        "base": {
            "probe_epochs": 3,
            "batch_size": 64,
            "val_split": 0.1,
            "early_stop_patience": 2,
            "seed": 0,
        },
        "search_space": {
            "hidden_dim": {"type": "categorical", "choices": [16, 32]},
            "dropout": {"type": "float", "low": 0.0, "high": 0.3},
            "lr": {
                "type": "float", "low": 1e-4, "high": 5e-3, "log": True,
            },
            "weight_decay": {
                "type": "float", "low": 1e-5, "high": 1e-3, "log": True,
            },
            "n_hidden_layers": {"type": "categorical", "choices": [1, 2]},
            "label_smoothing": {
                "type": "float", "low": 0.0, "high": 0.1,
            },
            "l2_normalize_inputs": {
                "type": "categorical", "choices": [False, True],
            },
            "use_cosine_schedule": {
                "type": "categorical", "choices": [False, True],
            },
        },
    }
    with open(path, "w") as f:
        yaml.safe_dump(spec, f)
    return path


def test_run_probe_hpo_end_to_end_on_fixture(
    corpus_subset_path: Path, tmp_path: Path,
) -> None:
    """2-trial sweep against a 1-epoch-fixture checkpoint produces a
    well-formed summary JSON carrying best_params, final_mlp, and
    baseline_lr.  Drives the whole pipeline -- not just the objective.
    """
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    ckpt_path = _train_fixture_checkpoint(corpus_subset_path, ckpt_dir)
    assert ckpt_path.exists()

    storage_dir = tmp_path / "optuna"
    storage_dir.mkdir()
    output_dir = tmp_path / "out"
    hpo_cfg_path = _write_probe_hpo_yaml(
        tmp_path / "probe_hpo.yaml",
        study_name="probe_hpo_test",
        storage_dir=storage_dir,
        output_dir=output_dir,
    )

    summary = run_hpo(
        checkpoint_path=ckpt_path,
        hpo_config_path=hpo_cfg_path,
        device="cpu",
    )

    assert summary["n_trials_complete"] == 2
    assert summary["probe_target"] == "qap_name"
    assert summary["best_params"], "expected non-empty best_params"
    assert summary["final_mlp"] is not None
    assert 0.0 <= summary["final_mlp"]["test_accuracy"] <= 1.0
    assert 0.0 <= summary["baseline_lr"]["test_accuracy"] <= 1.0

    summary_files = list(output_dir.glob("*/summary.json"))
    assert len(summary_files) == 1
    with open(summary_files[0]) as f:
        on_disk = json.load(f)
    assert on_disk["n_trials_complete"] == 2

    # study figure + production-MLP loss curve land next to the summary
    encoder_dir = summary_files[0].parent
    assert (encoder_dir / "hpo_history.png").stat().st_size > 0
    assert (encoder_dir / "loss_curve.png").stat().st_size > 0

    # storage file should exist + be re-openable (resumability prereq).
    journal_files = list(storage_dir.glob("*.journal"))
    assert len(journal_files) == 1


def _fake_summary(tag: str, best_params: dict) -> dict:
    """Hand-build a probe-HPO summary fixture that `build_report` can
    consume without driving the full sweep.  Only the keys the
    aggregator reads are present."""
    return {
        "encoder_tag": tag,
        "best_params": best_params,
        "best_trial_value": 0.5,
        "final_mlp": {"test_accuracy": 0.5},
        "baseline_lr": {"test_accuracy": 0.4},
    }


def test_compare_probe_hpos_flags_converged_vs_scattered() -> None:
    """build_report's convergence flags fire correctly on a hand-crafted
    set of three encoder summaries: hidden_dim agrees 3/3, n_hidden_layers
    agrees 2/3 (N-1-of-N rule still passes), lr scatters across two
    decades."""
    summaries = [
        _fake_summary("enc_a", {
            "hidden_dim": 512, "n_hidden_layers": 2,
            "dropout": 0.20, "lr": 3.0e-4, "weight_decay": 1.0e-4,
            "label_smoothing": 0.05,
            "l2_normalize_inputs": True, "use_cosine_schedule": True,
        }),
        _fake_summary("enc_b", {
            "hidden_dim": 512, "n_hidden_layers": 2,
            "dropout": 0.22, "lr": 4.0e-4, "weight_decay": 1.5e-4,
            "label_smoothing": 0.04,
            "l2_normalize_inputs": True, "use_cosine_schedule": True,
        }),
        _fake_summary("enc_c", {
            "hidden_dim": 512, "n_hidden_layers": 1,
            "dropout": 0.18, "lr": 3.0e-2, "weight_decay": 1.2e-4,
            "label_smoothing": 0.06,
            "l2_normalize_inputs": True, "use_cosine_schedule": True,
        }),
    ]
    report = build_report(summaries)
    assert report["n_encoders"] == 3
    hp = report["hyperparameters"]

    # hidden_dim agrees unanimously
    assert hp["hidden_dim"]["converged"] is True
    assert hp["hidden_dim"]["mode"] == 512

    # n_hidden_layers: 2/3 agree, passes N-1 rule (2 >= 3-1)
    assert hp["n_hidden_layers"]["converged"] is True

    # dropout cluster tight -> CV small -> converged
    assert hp["dropout"]["converged"] is True

    # lr spans 3e-4 to 3e-2 (two decades) -> log-CV high -> scattered
    assert hp["lr"]["converged"] is False

    # weight_decay clusters tight in log space -> converged
    assert hp["weight_decay"]["converged"] is True

    # consensus dict contains every HP
    assert set(report["consensus"]) == set(hp)


def test_compare_probe_hpos_writes_report_and_yaml(tmp_path: Path) -> None:
    """run_calibration writes both the markdown report and calibrated YAML."""
    summaries = [
        _fake_summary(f"enc_{i}", {
            "hidden_dim": 512,
            "n_hidden_layers": 2,
            "dropout": 0.2 + 0.01 * i,
            "lr": 3.0e-4,
            "weight_decay": 1.0e-4,
            "label_smoothing": 0.05,
            "l2_normalize_inputs": True,
            "use_cosine_schedule": True,
        })
        for i in range(3)
    ]
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    summary_paths = []
    for s in summaries:
        p = in_dir / f"{s['encoder_tag']}_summary.json"
        with open(p, "w") as f:
            json.dump(s, f)
        summary_paths.append(p)

    out_dir = tmp_path / "out"
    report = run_calibration(summary_paths, out_dir)
    assert (out_dir / "calibration_report.md").exists()
    assert (out_dir / "calibrated_probe.yaml").exists()
    assert report["consensus"]["hidden_dim"] == 512


def test_run_probe_hpo_zero_trials_returns_baseline_only(
    corpus_subset_path: Path, tmp_path: Path,
) -> None:
    """n_trials=0 short-circuits the sweep; the LR baseline is still
    written so the operator gets at least one number per encoder."""
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    ckpt_path = _train_fixture_checkpoint(corpus_subset_path, ckpt_dir)

    storage_dir = tmp_path / "optuna"
    storage_dir.mkdir()
    hpo_cfg_path = _write_probe_hpo_yaml(
        tmp_path / "probe_hpo.yaml",
        study_name="probe_hpo_test_zero",
        storage_dir=storage_dir,
        output_dir=tmp_path / "out",
        n_trials=0,
    )

    summary = run_hpo(
        checkpoint_path=ckpt_path,
        hpo_config_path=hpo_cfg_path,
        device="cpu",
    )
    assert summary["n_trials_complete"] == 0
    assert summary["best_params"] == {}
    assert summary["final_mlp"] is None
    assert 0.0 <= summary["baseline_lr"]["test_accuracy"] <= 1.0
