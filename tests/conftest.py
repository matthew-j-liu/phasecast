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

import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from cnmineral.models.encoder import Encoder
from cnmineral.models.pretrain_head import PretrainHead
from cnmineral.training.checkpoint import Checkpoint

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def pretrain_runs_root(monkeypatch, tmp_path: Path) -> Path:
    """Redirect `run_single`'s artifact writes into the test's tmp dir.

    Autouse so no test (including fixtures that drive `run_single` as
    setup) can litter the real `runs/pretrain/` tree.  Tests
    asserting on the summary artifact request this fixture by name to
    learn the patched root.
    """
    from cnmineral.training import ssl_pretrain

    root = tmp_path / "runs"
    monkeypatch.setattr(ssl_pretrain, "RUNS_ROOT", root)
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


# per-phase 11-oxide mass fractions (magemin_corpus.COMP_OXIDES order), each
# summing to 1; plag/afs embed solvus-side CaO/Na2O/K2O ratios so feldspar
# instances route deterministically (see feldspar_rule)
SYNTH_COMPS = {
    "liq": [0.62, 0.16, 0.04, 0.02, 0.04, 0.03, 0.04, 0.01, 0.0, 0.0, 0.04],
    "fl": [0.10, 0.02, 0.0, 0.0, 0.0, 0.01, 0.02, 0.0, 0.0, 0.0, 0.85],
    "plag": [0.55, 0.28, 0.12, 0.0, 0.009, 0.001, 0.04, 0.0, 0.0, 0.0, 0.0],
    "afs": [0.66, 0.19, 0.005, 0.0, 0.005, 0.10, 0.03, 0.01, 0.0, 0.0, 0.0],
    "cpx": [0.52, 0.03, 0.20, 0.14, 0.10, 0.0, 0.005, 0.005, 0.0, 0.0, 0.0],
    "amp": [0.45, 0.10, 0.10, 0.15, 0.15, 0.005, 0.025, 0.01, 0.0, 0.0, 0.01],
    "q": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

# (phase-as-written, instance, mode_wt) per (draw, rung, T_end); fsp instances
# carry plag/afs comps so the vocabulary split has one of each at rung 50
_SYNTH_GROUPS = [
    (0, 50, 900.0, [("liq", "liq", 0.45), ("fl", "fl", 0.05),
                    ("fsp", "plag", 0.25), ("fsp", "afs", 0.05),
                    ("cpx", "cpx", 0.15), ("amp", "amp", 0.05)]),
    (0, 30, 800.0, [("liq", "liq", 0.30), ("cpx", "cpx", 0.40),
                    ("fsp", "plag", 0.20), ("q", "q", 0.10)]),
    (1, 50, 950.0, [("liq", "liq", 0.50), ("cpx", "cpx", 0.50)]),
]

_SYNTH_DRAW_CONDITIONS = {0: (6.0, None), 1: (2.0, None)}


def _wt_columns(comp_key: str) -> dict:
    from cnmineral.data.magemin_corpus import COMP_OXIDES

    return {f"wt_{o}": v for o, v in zip(COMP_OXIDES, SYNTH_COMPS[comp_key])}


def write_synthetic_rungs(corpus_dir: Path, bulk_ids: list[int]) -> Path:
    """Write one chunk_0000_rungs.parquet with two draws per bulk.

    Draw 0 has rows at rungs 50 and 30 (mixed assemblage, one plag + one afs
    feldspar instance at rung 50, a fluid row); draw 1 reaches only rung 50
    (attrition). Every row carries the full 11-oxide composition and T_end.
    Returns `corpus_dir`.
    """
    rows = []
    for bulk in bulk_ids:
        for draw, rung, t_end, phases in _SYNTH_GROUPS:
            rows += [
                dict(bulk=bulk, draw=draw, rung=rung, T_end=t_end, phase=phase,
                     instance=i, mode_wt=mode, **_wt_columns(comp_key))
                for i, (phase, comp_key, mode) in enumerate(phases)
            ]
    corpus_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(corpus_dir / "chunk_0000_rungs.parquet")
    return corpus_dir


def write_synthetic_draws(corpus_dir: Path, bulk_ids: list[int]) -> Path:
    """Write the chunk_0000_draws.parquet matching `write_synthetic_rungs`.

    Each draw's input oxides are the rung-50 reconstruction of its rows
    (mode-weighted comps, scaled to wt%), so rung-50 groups close exactly
    under the mass-balance identity; rung-30 groups close only approximately.
    All iron is ferrous (`in_fe2o3` = 0). Returns `corpus_dir`.
    """
    import numpy as np

    from cnmineral.data.magemin_corpus import COMP_OXIDES

    in_cols = ["in_sio2", "in_al2o3", "in_cao", "in_mgo", "in_feo",
               "in_k2o", "in_na2o", "in_tio2", "in_cr2o3"]
    slot = {c: COMP_OXIDES.index(o) for c, o in zip(
        in_cols, ["SiO2", "Al2O3", "CaO", "MgO", "FeO",
                  "K2O", "Na2O", "TiO2", "Cr2O3"])}
    rows = []
    for bulk in bulk_ids:
        for draw, rung, _, phases in _SYNTH_GROUPS:
            if rung != 50:
                continue
            b = np.zeros(len(COMP_OXIDES))
            for _, comp_key, mode in phases:
                b += mode * np.array(SYNTH_COMPS[comp_key])
            b *= 100.0
            p_kbar, _ = _SYNTH_DRAW_CONDITIONS[draw]
            rows.append(dict(
                bulk=bulk, draw=draw, P_kbar=p_kbar,
                H2O=float(b[COMP_OXIDES.index("H2O")]), in_fe2o3=0.0,
                **{c: float(b[i]) for c, i in slot.items()},
            ))
    corpus_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(corpus_dir / "chunk_0000_draws.parquet")
    return corpus_dir


@pytest.fixture
def synthetic_joint_corpus():
    """Builder writing both rungs and draws tables (for `load_joint_labels`)."""

    def _build(corpus_dir: Path, bulk_ids: list[int]) -> Path:
        write_synthetic_rungs(corpus_dir, bulk_ids)
        write_synthetic_draws(corpus_dir, bulk_ids)
        return corpus_dir

    return _build


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
