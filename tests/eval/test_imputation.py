"""Tests for `cnmineral.eval.imputation`.

An end-to-end log-space scoring run of a trained imputation head against the
tiny encoder checkpoint (guards wiring, the test-split reconstruction, and
artifact emission; the tiny head's numbers are not meaningful), the
back-transform helper, plus the gate-reference loader's present / absent paths.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cnmineral.data.bulk_chemistry import OXIDE_TO_CATION_FACTOR
from cnmineral.data.tokens import N_TOKENS
from cnmineral.eval import imputation
from cnmineral.eval.imputation import (
    _back_transform,
    _load_gate_references,
    run_imputation,
)
from cnmineral.models.imputation_head import LogImputationHead
from cnmineral.training.imputation_head import ImputationHeadCheckpoint


def _make_head_checkpoint(encoder_ckpt: Path, dest: Path) -> Path:
    head = LogImputationHead(d_model=16, hidden=8)
    head.fit_standardization(
        torch.full((N_TOKENS,), -2.0), torch.full((N_TOKENS,), 1.0),
    )
    ImputationHeadCheckpoint.from_head(
        head, str(encoder_ckpt), 0.1, 0.2, {"train": [0.1], "val": [0.2]},
    ).save(dest)
    return dest


def test_run_logspace_end_to_end_on_fixture(
    tiny_pretrain_checkpoint: Path, tmp_path: Path, monkeypatch,
) -> None:
    """`run_imputation` reconstructs the encoder's test split, scores every
    populated element in log space, and writes the metrics JSON + plot under a
    `readout/` subdir."""
    monkeypatch.setattr(imputation, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(imputation, "GATE_RESULTS_ROOT", tmp_path / "gate")
    head_ckpt = _make_head_checkpoint(tiny_pretrain_checkpoint, tmp_path / "head_best.pt")

    metrics = run_imputation(head_ckpt, device=torch.device("cpu"))

    per_element = metrics["per_element"]
    assert per_element and "Si" in per_element
    assert np.isfinite(per_element["Si"]["log_r2"])
    assert metrics["space"] == "log10"
    assert metrics["trace_summary"]["n_traces"] >= 0

    out_dir = tmp_path / "results" / "imputation"
    readout_dir = next(out_dir.iterdir()) / "readout"
    assert (readout_dir / "readout_logspace_metrics.json").exists()
    assert (readout_dir / "r2_by_element.png").exists()


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


def test_load_gate_references_reads_run_1a_1b(tmp_path: Path, monkeypatch) -> None:
    """The loader maps the gate's run_1a / run_1b R2 to the reference keys."""
    monkeypatch.setattr(imputation, "GATE_RESULTS_ROOT", tmp_path)
    tag = "enc_epoch_best"
    gate_dir = tmp_path / tag
    gate_dir.mkdir()
    payload = {"per_element": {"Sr": {"run_1a": {"r2": 0.9}, "run_1b": {"r2": 0.84}}}}
    (gate_dir / "diagnostic_metrics.json").write_text(json.dumps(payload))

    refs = _load_gate_references(tag)
    assert refs["Sr"]["input_xgb_ceiling_r2"] == 0.9
    assert refs["Sr"]["frozen_latent_xgb_r2"] == 0.84


def test_load_gate_references_missing_returns_empty(tmp_path: Path, monkeypatch) -> None:
    """No gate file -> empty refs, so the eval still runs without reference lines."""
    monkeypatch.setattr(imputation, "GATE_RESULTS_ROOT", tmp_path)
    assert _load_gate_references("nonexistent_tag") == {}
