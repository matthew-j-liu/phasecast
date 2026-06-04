"""Unit tests for the SSL pretrain driver.

These pin the contracts the user is about to lean on while filling in
the per-step training loop and the HPO suggest_hyperparams body:

- The 6-step forward/backward/step recipe actually trains (gradient
  reaches the encoder; loss is finite).
- The synthetic dataset satisfies the v3 padding convention.
- Checkpoint save/load is a clean roundtrip.
- pretrain_run and the Optuna objective both return a finite float on
  synthetic data, even with the stubbed-out inner loop.
- An Optuna study at a JournalStorage path survives recreation
  (load_if_exists=True).

Run with:
    pytest tests/test_pretrain_script.py -v
"""

import copy
import importlib.util
import sys
from pathlib import Path

import optuna
import pytest
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from torch.utils.data import DataLoader

from cnmineral.data.synthetic import SyntheticUnlabeledDataset
from cnmineral.data.tokens import (
    N_TOKENS,
    assert_padding_convention,
    mask_for_ssl,
)
from cnmineral.losses.pretrain import masked_element_mse
from cnmineral.models.encoder import Encoder
from cnmineral.models.pretrain_head import PretrainHead


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "01_pretrain.py"


def _load_pretrain_module():
    """Import scripts/01_pretrain.py by path (numeric prefix blocks normal import)."""
    spec = importlib.util.spec_from_file_location("pretrain_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pretrain_script"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pretrain_module():
    return _load_pretrain_module()


def _tiny_config(checkpoint_dir: Path | None = None) -> dict:
    """A config small enough that pretrain_run finishes in milliseconds."""
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
        "num_workers": 0,
        "seed": 0,
        "checkpoint_dir": checkpoint_dir,
    }


def test_one_step_runs() -> None:
    """End-to-end one training step; encoder weights must change."""
    torch.manual_seed(0)
    encoder = Encoder(d_model=32, n_layers=2, n_heads=4, dropout=0.0,
                      frac_resolution=200)
    head = PretrainHead(d_model=32, hidden=16)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=3e-4,
    )

    ds = SyntheticUnlabeledDataset(n=8, seed=0)
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
        ds = SyntheticUnlabeledDataset(n=4, seed=0)
        assert len(ds) == 4
        item = ds[0]
        assert set(item.keys()) == {"src", "frac", "sample_id"}
        assert item["src"].shape == (N_TOKENS,)
        assert item["frac"].shape == (N_TOKENS,)
        assert item["src"].dtype == torch.long
        assert item["frac"].dtype == torch.float32
        assert isinstance(item["sample_id"], int)

    def test_obeys_padding_convention(self) -> None:
        ds = SyntheticUnlabeledDataset(n=8, seed=1)
        loader = DataLoader(ds, batch_size=4)
        for batch in loader:
            assert_padding_convention(batch["src"], batch["frac"])

    def test_reproducible(self) -> None:
        a = SyntheticUnlabeledDataset(n=4, seed=7)
        b = SyntheticUnlabeledDataset(n=4, seed=7)
        torch.testing.assert_close(a[0]["frac"], b[0]["frac"])


def test_checkpoint_roundtrip(tmp_path: Path, pretrain_module) -> None:
    torch.manual_seed(0)
    encoder_a = Encoder(d_model=32, n_layers=2, n_heads=4,
                        frac_resolution=200)
    head_a = PretrainHead(d_model=32, hidden=16)
    optimizer = torch.optim.AdamW(
        list(encoder_a.parameters()) + list(head_a.parameters()), lr=1e-3,
    )
    config = _tiny_config()

    ckpt_path = tmp_path / "ckpt.pt"
    pretrain_module.save_checkpoint(
        ckpt_path, encoder_a, head_a, optimizer, epoch=1,
        config=config, train_loss=0.123,
    )

    torch.manual_seed(99)  # different init for the fresh instances
    encoder_b = Encoder(d_model=32, n_layers=2, n_heads=4,
                        frac_resolution=200)
    head_b = PretrainHead(d_model=32, hidden=16)
    payload = pretrain_module.load_checkpoint(ckpt_path, encoder_b, head_b)

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


def test_pretrain_run_returns_float(pretrain_module) -> None:
    config = _tiny_config()
    loss = pretrain_module.pretrain_run(config, device=torch.device("cpu"))
    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))


def test_objective_returns_float(pretrain_module) -> None:
    """suggest_hyperparams + pretrain_run smoke (HPO objective body)."""
    base = _tiny_config()
    trial = optuna.trial.FixedTrial({
        "lr": 1e-4,
        "mask_prob": 0.25,
        "weight_decay": 1e-2,
        "dropout": 0.0,
        "batch_size": 4,
    })
    overrides = pretrain_module.suggest_hyperparams(trial)
    assert isinstance(overrides, dict)
    merged = {**base, **overrides}
    loss = pretrain_module.pretrain_run(
        merged, device=torch.device("cpu"), trial=trial,
    )
    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))


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
