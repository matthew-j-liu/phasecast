"""pytest auto-loaded configuration.

pytest auto-loads this file before any test module imports run, which
makes it the right place to set the ``KMP_DUPLICATE_LIB_OK=TRUE`` env
var.  pytest's own collection machinery imports numpy, which in turn
loads libomp; setting the env var inside ``cnmineral/__init__.py`` is
too late because the test collector pulls numpy in before any
``cnmineral`` import.

Background: macOS's PyTorch and PetThermoTools (via MAGEMin) both link
against libomp.  Concurrent loading aborts with "OpenMP already
initialized" unless this env var is set.  See ``CLAUDE.md`` macOS
gotcha section.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# torch and xgboost each link their own libomp; with both runtimes live in one
# process, xgboost's first parallel region segfaults once torch has spun up its
# thread pool (e.g. an encoder test running before an xgboost test).  Pinning to
# one OpenMP thread sidesteps the dual-runtime crash; test tensors are tiny so
# there is no speed cost.  Production scripts keep full threading.
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Load xgboost's libomp first, before torch/sklearn spin up their own OpenMP
# runtimes.  On macOS, an xgboost parallel region entered after sklearn (or
# torch) has initialized its pool segfaults; importing xgboost up front makes
# its runtime the one already live when the others load.
import xgboost  # noqa: F401

from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from cnmineral.models.encoder import Encoder
from cnmineral.models.pretrain_head import PretrainHead
from cnmineral.training.checkpoint import Checkpoint

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def pretrain_results_root(monkeypatch, tmp_path: Path) -> Path:
    """Redirect `run_single`'s artifact writes into the test's tmp dir.

    Autouse so no test (including fixtures that drive `run_single` as
    setup) can litter the real `results/pretrain/` tree.  Tests
    asserting on the summary artifact request this fixture by name to
    learn the patched root.
    """
    from cnmineral.training import ssl_pretrain

    root = tmp_path / "results"
    monkeypatch.setattr(ssl_pretrain, "RESULTS_ROOT", root)
    return root


@pytest.fixture(scope="session")
def corpus_subset_path() -> Path:
    """Path to the 2000-row deterministic parquet subset of the bulk corpus.

    Built by ``scripts/build/build_test_fixture.py`` (seed=42).  Tests
    that want a built ``BulkChemistryDataset`` instantiate it themselves
    with the filter combo they need.
    """
    path = REPO_ROOT / "tests" / "fixtures" / "complete_subset.parquet"
    if not path.exists():
        pytest.fail(
            f"test fixture missing: {path}\n"
            f"rebuild with: python scripts/build/build_test_fixture.py"
        )
    return path


class SyntheticTokenDataset(Dataset):
    """Yields `{src, frac, sample_id}` from prebuilt `[N, 22]` tensors."""

    def __init__(self, src: torch.Tensor, frac: torch.Tensor) -> None:
        self._src = src
        self._frac = frac

    def __len__(self) -> int:
        return int(self._src.shape[0])

    def __getitem__(self, idx: int) -> dict:
        return {
            "src": self._src[idx].clone(),
            "frac": self._frac[idx].clone(),
            "sample_id": int(idx),
        }


@pytest.fixture
def synthetic_token_dataset():
    """The `SyntheticTokenDataset` class, for building tiny token datasets."""
    return SyntheticTokenDataset


def _tiny_config(corpus_subset_path: Path) -> dict:
    return {
        "epochs": 1, "batch_size": 32, "lr": 3e-4, "mask_prob": 0.25,
        "weight_decay": 0.01, "dropout": 0.0, "d_model": 16, "n_layers": 1,
        "n_heads": 2, "frac_resolution": 200,
        "data_kwargs": {
            "csv_path": str(corpus_subset_path), "rock_group": "igneous",
            "require_all_majors": True, "min_class_count": 5,
        },
        "test_split": 0.25, "val_split": 0.25, "num_workers": 0, "seed": 0,
    }


@pytest.fixture
def tiny_pretrain_checkpoint(tmp_path: Path, corpus_subset_path: Path) -> Path:
    """Mint a tiny but real encoder+head checkpoint on disk, return its path."""
    torch.manual_seed(0)
    config = _tiny_config(corpus_subset_path)
    encoder = Encoder(
        d_model=config["d_model"], n_layers=config["n_layers"],
        n_heads=config["n_heads"], frac_resolution=config["frac_resolution"],
    )
    head = PretrainHead(d_model=config["d_model"])
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()), lr=1e-3,
    )
    ckpt = Checkpoint.from_models(
        encoder, head, optimizer, epoch=1, config=config,
        train_loss=0.2, val_loss=0.3, monitor={"train": [0.2], "val": [0.3]},
    )
    path = tmp_path / "epoch_best.pt"
    ckpt.save(path)
    return path
