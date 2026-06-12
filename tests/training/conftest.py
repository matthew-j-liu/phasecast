"""Shared fixtures + helper builders for training-subpackage tests.

Both `_tiny_config` and `_write_yaml` are pytest-internal helpers
shared by multiple test files in `tests/training/`.  Per phase-edge
convention (`from conftest import ...`), tests import them by name.
"""

from pathlib import Path

import yaml

CORPUS_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "complete_subset.parquet"
)


def _tiny_config(checkpoint_dir: Path | str | None = None) -> dict:
    """A config small enough that run_pretraining finishes in ~a second.

    The corpus fixture yields ~413 rows under these data_kwargs;
    test_frac=0.25 -> ~310 trainval / ~103 test, then val_frac=0.25
    inside run_pretraining -> ~232 train / ~78 val.  All non-empty so
    the split + train + val passes all exercise.

    `min_class_count=5` (not the fixture-wide minimum of 2) because
    both splits stratify by qap_name: a 2-member class leaves a
    singleton in trainval after the outer split, and sklearn rejects
    stratifying a singleton in the inner train/val split.

    When `checkpoint_dir` is set, the two filename keys must be
    set too -- production callers will `KeyError` otherwise, so
    fixtures must mirror that contract.
    """
    config = {
        "epochs": 1,
        "batch_size": 32,
        "lr": 3e-4,
        "mask_prob": 0.25,
        "weight_decay": 0.01,
        "dropout": 0.0,
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 4,
        "frac_resolution": 200,
        "data_kwargs": {
            "csv_path": str(CORPUS_FIXTURE),
            "rock_group": "igneous",
            "require_all_majors": True,
            "min_class_count": 5,
        },
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
