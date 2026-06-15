"""Tests for the trace-imputation loss-gate exploration (`run_gate`).

Unit tests on the feature-construction and the masked-encode leakage guard
(the invariant that makes 1b an honest baseline), plus an end-to-end smoke run
against the corpus fixture that guards wiring, the three runs, and artifact
emission.  The tiny-encoder numbers are not meaningful.  The on-disk checkpoint
and synthetic dataset come from the root `tests/conftest.py`.
"""

from pathlib import Path

import numpy as np
import torch

from cnmineral.data.tokens import N_TOKENS
from cnmineral.models.encoder import Encoder
from explorations.trace_imputation_loss_gate import run_gate
from explorations.trace_imputation_loss_gate.run_gate import (
    _encode_with_masked_token,
    _log_feature_matrix,
    run_imputation_diagnostic,
)


def test_log_feature_matrix_absent_is_nan() -> None:
    """Populated entries map to log10(frac); absent entries become NaN."""
    frac = np.array([[0.1, 0.0, 0.001], [0.0, 0.5, 0.0]], dtype=np.float64)
    populated = frac > 0
    xlog = _log_feature_matrix(frac, populated)
    assert xlog[0, 0] == np.log10(0.1)
    assert xlog[0, 2] == np.log10(0.001)
    assert np.isnan(xlog[0, 1])
    assert np.isnan(xlog[1, 0]) and np.isnan(xlog[1, 2])


def test_masked_encode_cannot_leak_target(synthetic_token_dataset) -> None:
    """The 1b invariant: masking token e makes the latent independent of e's
    input value, so 1b cannot read the held-out element straight off `h`."""
    torch.manual_seed(0)
    encoder = Encoder(d_model=16, n_layers=1, n_heads=2, frac_resolution=200)
    encoder.eval()

    src = torch.zeros(5, N_TOKENS, dtype=torch.long)
    frac = torch.zeros(5, N_TOKENS, dtype=torch.float32)
    src[:, [0, 1, 2]] = torch.tensor([14, 13, 20])
    frac[:, [0, 1, 2]] = 0.3
    idx = np.arange(5)

    # perturb only token 2 (both src and frac); masked-on-2 latent must not move
    src2 = src.clone()
    frac2 = frac.clone()
    src2[:, 2] = 26
    frac2[:, 2] = 0.9

    h_a = _encode_with_masked_token(
        encoder, synthetic_token_dataset(src, frac), idx, 2, device="cpu", batch_size=4)
    h_b = _encode_with_masked_token(
        encoder, synthetic_token_dataset(src2, frac2), idx, 2, device="cpu", batch_size=4)
    assert np.allclose(h_a, h_b, atol=1e-6)


def test_run_diagnostic_end_to_end_on_fixture(
    tiny_pretrain_checkpoint: Path, tmp_path: Path, monkeypatch,
) -> None:
    """`run_imputation_diagnostic` produces all three runs per element, a
    populated gate block, and the metrics JSON + R2 chart."""
    monkeypatch.setattr(run_gate, "RESULTS_ROOT", tmp_path / "results")

    metrics = run_imputation_diagnostic(
        tiny_pretrain_checkpoint, device=torch.device("cpu"),
        n_trials=2, study_subsample=500,
    )

    per_element = metrics["per_element"]
    assert per_element and "Si" in per_element
    si = per_element["Si"]
    for run in ("run_1a", "run_1b", "run_1c"):
        assert run in si
        assert np.isfinite(si[run]["r2"])
    assert si["run_1c"]["n_clipped"] >= 0
    assert metrics["space"] == "log10"
    assert "verdict" in metrics["gate"]

    out_dir = tmp_path / "results"
    tag_dir = next(out_dir.iterdir())
    assert (tag_dir / "diagnostic_metrics.json").exists()
    assert (tag_dir / "r2_by_element.png").exists()
