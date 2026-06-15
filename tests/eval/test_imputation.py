"""Tests for `cnmineral.eval.imputation`.

Structural tests on the leave-one-out core + back-transform, plus an
end-to-end `run_imputation` against the 2000-row corpus fixture.  The
tiny-encoder predictions are not meaningful; the end-to-end test guards
wiring, the test-split reconstruction, and artifact emission.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from cnmineral.data.bulk_chemistry import OXIDE_TO_CATION_FACTOR
from cnmineral.data.tokens import MASK_TOKEN_ID, N_TOKENS
from cnmineral.eval import imputation
from cnmineral.eval.imputation import (
    _back_transform,
    _impute_token,
    run_imputation,
)
from cnmineral.eval.latent import (
    load_encoder_and_head_from_checkpoint,
    load_encoder_from_checkpoint,
)
from cnmineral.models.encoder import Encoder
from cnmineral.models.pretrain_head import PretrainHead


class _SpyEncoder(nn.Module):
    """Records the (src, frac) it is called with; returns zero latents."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, *, src: torch.Tensor, frac: torch.Tensor) -> torch.Tensor:
        self.seen.append((src.clone(), frac.clone()))
        return torch.zeros(src.shape[0], self.d_model)


class _ConstHead(nn.Module):
    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return torch.zeros(pooled.shape[0], N_TOKENS)


def test_loader_returns_head_with_22_outputs(tiny_pretrain_checkpoint: Path) -> None:
    """The head-returning loader yields a PretrainHead emitting one frac per token."""
    encoder, head, ckpt = load_encoder_and_head_from_checkpoint(tiny_pretrain_checkpoint)
    assert isinstance(head, PretrainHead)
    out = head(torch.zeros(3, ckpt.config["d_model"]))
    assert out.shape == (3, N_TOKENS)


def test_load_encoder_from_checkpoint_unchanged_pair(
    tiny_pretrain_checkpoint: Path,
) -> None:
    """The head-less loader still returns (encoder, ckpt) after the refactor,
    and its encoder weights match the head-returning loader's bit-exact."""
    encoder_pair, ckpt = load_encoder_from_checkpoint(tiny_pretrain_checkpoint)
    assert isinstance(encoder_pair, Encoder)
    encoder_full, _head, _ckpt = load_encoder_and_head_from_checkpoint(tiny_pretrain_checkpoint)
    for a, b in zip(encoder_pair.state_dict().values(),
                    encoder_full.state_dict().values()):
        assert torch.equal(a, b)


def test_impute_token_masks_target_and_keeps_populated_rows(
    synthetic_token_dataset,
) -> None:
    """`_impute_token` masks exactly the target token and returns only the rows
    where that element was originally populated (mapped to dataset indices)."""
    src = torch.zeros(4, N_TOKENS, dtype=torch.long)
    frac = torch.zeros(4, N_TOKENS, dtype=torch.float32)
    # token 0 populated in rows 0 and 2 only
    src[[0, 2], 0] = 14
    frac[[0, 2], 0] = 0.5
    dataset = synthetic_token_dataset(src, frac)
    indices = np.array([0, 1, 2, 3], dtype=np.int64)

    encoder = _SpyEncoder(d_model=8)
    head = _ConstHead()
    kept, true, pred = _impute_token(
        encoder, head, dataset, indices, 0, device="cpu", batch_size=2,
    )
    assert kept.tolist() == [0, 2]
    assert true.tolist() == [0.5, 0.5]
    assert pred.shape == (2,)
    for seen_src, seen_frac in encoder.seen:
        assert torch.all(seen_src[:, 0] == MASK_TOKEN_ID)
        assert torch.all(seen_frac[:, 0] == 0.0)


def test_impute_token_empty_when_token_unpopulated(synthetic_token_dataset) -> None:
    """A token padded in every row yields no comparisons."""
    src = torch.zeros(3, N_TOKENS, dtype=torch.long)
    frac = torch.zeros(3, N_TOKENS, dtype=torch.float32)
    dataset = synthetic_token_dataset(src, frac)
    kept, true, pred = _impute_token(
        _SpyEncoder(4), _ConstHead(), dataset,
        np.array([0, 1, 2]), 5, device="cpu", batch_size=2,
    )
    assert kept.size == 0 and true.size == 0 and pred.size == 0


def test_back_transform_inverts_normalization() -> None:
    """Majors invert to oxide wt%, Fe to cation wt%, traces to ppm."""
    si, _ = _back_transform(0, np.array([0.45], dtype=np.float32))
    assert si[0] == pytest.approx(0.45 * 100.0 / OXIDE_TO_CATION_FACTOR["sio2"], rel=1e-5)

    fe_val, fe_label = _back_transform(4, np.array([0.05], dtype=np.float32))
    assert fe_val[0] == pytest.approx(5.0, rel=1e-5)
    assert fe_label == "Fe (cation wt%)"

    sr_val, sr_label = _back_transform(9, np.array([1e-4], dtype=np.float32))
    assert sr_val[0] == pytest.approx(100.0, rel=1e-5)
    assert sr_label == "Sr (ppm)"


def test_run_imputation_end_to_end_on_fixture(
    tiny_pretrain_checkpoint: Path, tmp_path: Path, monkeypatch,
) -> None:
    """`run_imputation` reconstructs the test split, imputes every populated
    element, and emits the metrics JSON + a major scatter grid."""
    monkeypatch.setattr(imputation, "RESULTS_ROOT", tmp_path / "results")

    metrics = run_imputation(tiny_pretrain_checkpoint, device=torch.device("cpu"))

    per_element = metrics["per_element"]
    assert per_element, "expected at least the majors imputed"
    assert "Si" in per_element
    si = per_element["Si"]
    assert si["n"] > 0
    assert np.isfinite(si["backtransformed"]["mae"])
    assert np.isfinite(si["frac"]["rmse"])

    out_dir = tmp_path / "results" / "imputation"
    tag_dir = next(out_dir.iterdir())
    assert (tag_dir / "imputation_metrics.json").exists()
    assert (tag_dir / "scatter_majors.png").exists()
