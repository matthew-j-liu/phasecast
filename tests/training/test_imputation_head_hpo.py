"""Tests for `cnmineral.training.imputation_head_hpo`.

End-to-end: a 2-trial sweep against the tiny encoder checkpoint writes a
well-formed summary JSON (best_params, best_value, baseline) and a resumable
journal.  Exercises the whole pipeline -- split, per-trial head train, val LOO
scoring -- not just the objective.
"""

import json
from pathlib import Path

import torch
import yaml

from cnmineral.training.imputation_head_hpo import run_hpo


def _write_hpo_yaml(
    path: Path, encoder_ckpt: Path, storage_dir: Path, output_dir: Path,
    *, n_trials: int = 2, with_baseline: bool = True,
) -> Path:
    """Minimal hand-written HPO config so the test breaks on `run_hpo`
    mishandling the YAML, not on edits to the production config.  Absolute
    storage / output paths because `run_hpo` resolves them against REPO_ROOT
    (an absolute path passes straight through)."""
    spec = {
        "encoder_checkpoint": str(encoder_ckpt),
        "study_name": "imputation_head_hpo_test",
        "storage_path": str(storage_dir / "imp_hpo_{encoder_tag}.journal"),
        "output_dir": str(output_dir),
        "sampler_seed": 7,
        "n_trials": n_trials,
        "n_startup_trials": 2,
        "score_batch_size": 256,
        "base": {
            "standardize": True,
            "batch_size": 64,
            "val_split": 0.25,
            "seed": 0,
            "epochs": 2,
            "early_stop_patience": 1,
        },
        "search_space": {
            "hidden": {"type": "categorical", "choices": [8, 16]},
            "lr": {"type": "float", "low": 1e-4, "high": 5e-3, "log": True},
            "dropout": {"type": "float", "low": 0.0, "high": 0.3},
        },
    }
    if with_baseline:
        spec["baseline"] = {"hidden": 16, "lr": 1e-3, "epochs": 2}
    with open(path, "w") as f:
        yaml.safe_dump(spec, f)
    return path


def test_run_imputation_hpo_end_to_end(
    tiny_pretrain_checkpoint: Path, tmp_path: Path,
) -> None:
    """2-trial sweep on the tiny encoder produces a well-formed summary with
    best_params, a finite best_value, the baseline reference, and a journal."""
    storage_dir = tmp_path / "optuna"
    storage_dir.mkdir()
    output_dir = tmp_path / "out"
    cfg_path = _write_hpo_yaml(
        tmp_path / "imp_hpo.yaml", tiny_pretrain_checkpoint, storage_dir, output_dir,
    )

    summary = run_hpo(cfg_path, device=torch.device("cpu"))

    assert summary["n_trials_complete"] == 2
    assert summary["objective"] == "median_trace_log_r2_val"
    assert summary["best_params"], "expected non-empty best_params"
    assert "hidden" in summary["best_params"]
    assert torch.isfinite(torch.tensor(summary["best_value"]))
    assert summary["baseline"] is not None
    assert "median_trace_log_r2" in summary["baseline"]

    summary_files = list(output_dir.glob("*/summary.json"))
    assert len(summary_files) == 1
    with open(summary_files[0]) as f:
        on_disk = json.load(f)
    assert on_disk["n_trials_complete"] == 2
    assert (summary_files[0].parent / "hpo_history.png").stat().st_size > 0

    journal_files = list(storage_dir.glob("*.journal"))
    assert len(journal_files) == 1
