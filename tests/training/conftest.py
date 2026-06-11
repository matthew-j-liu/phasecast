"""Shared fixtures + helper builders for training-subpackage tests.

Both `_tiny_config` and `_write_yaml` are pytest-internal helpers
shared by multiple test files in `tests/training/`.  Per phase-edge
convention (`from conftest import ...`), tests import them by name.
"""

from pathlib import Path

import yaml


def _tiny_config(checkpoint_dir: Path | str | None = None) -> dict:
    """A config small enough that run_pretraining finishes in milliseconds.

    synthetic_n=16 with test_frac=0.25 -> 12 trainval / 4 test;
    val_frac=0.25 inside run_pretraining -> 9 train / 3 val.  All
    non-empty so the split + train + val passes all exercise.

    When `checkpoint_dir` is set, the two filename keys must be
    set too -- production callers will `KeyError` otherwise, so
    fixtures must mirror that contract.
    """
    config = {
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
    if checkpoint_dir is not None:
        config["checkpoint_best_filename"] = "epoch_best.pt"
        config["checkpoint_last_filename"] = "epoch_last.pt"
    return config


def _write_yaml(path: Path, payload: dict) -> Path:
    with open(path, "w") as f:
        yaml.safe_dump(payload, f)
    return path
