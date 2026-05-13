"""Shape-check smoke test for BaseModel + TrainMageminDataset.

Runs in <1s. Catches dtype/shape errors before launching a full training run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cnmineral.models.dataset import CATIONS_ORDER, oxides_to_cations
from cnmineral.models.losses import mode_loss
from cnmineral.models.model import BaseModel


def test_oxides_to_cations_shape() -> None:
    """11 oxide cols → 10 cation cols; FeO+Fe2O3 should sum into Fe."""
    cols = [
        "sio2", "al2o3", "cao", "mgo", "feo", "fe2o3",
        "k2o", "na2o", "tio2", "cr2o3", "h2o_tot",
        "T_celsius", "P_kbar",     # should be ignored
    ]
    N = 4
    # Set FeO and Fe2O3 both to non-zero so we can check they sum
    X = np.zeros((N, len(cols)), dtype=np.float32)
    X[:, cols.index("feo")] = 10.0      # 10 wt% FeO  → 7.773 wt% Fe
    X[:, cols.index("fe2o3")] = 1.0     #  1 wt% Fe2O3 → 0.6994 wt% Fe
    X[:, cols.index("sio2")] = 50.0     # 50 wt% SiO2 → 23.375 wt% Si

    cat = oxides_to_cations(X, cols)
    assert cat.shape == (N, 10), f"expected (4, 10), got {cat.shape}"

    fe_idx = [i for i, (sym, _z) in enumerate(CATIONS_ORDER) if sym == "Fe"][0]
    si_idx = [i for i, (sym, _z) in enumerate(CATIONS_ORDER) if sym == "Si"][0]

    expected_fe = 10.0 * 0.7773 + 1.0 * 0.6994
    expected_si = 50.0 * 0.4675
    np.testing.assert_allclose(cat[0, fe_idx], expected_fe, rtol=1e-4)
    np.testing.assert_allclose(cat[0, si_idx], expected_si, rtol=1e-4)


def test_base_model_forward_shape() -> None:
    """BaseModel(src, frac) should produce [B, n_minerals]."""
    torch.manual_seed(0)
    n_minerals = 22
    batch = 4
    n_cations = len(CATIONS_ORDER)

    model = BaseModel(n_minerals=n_minerals, d_model=64, N=1, heads=2, dropout=0.0)
    src = torch.tensor(
        [[z for _sym, z in CATIONS_ORDER]] * batch,
        dtype=torch.long,
    )
    # Dataset feeds cation mass fractions in [0, 1] summing to ~1 (not wt%
    # in [0, 100] — CrabNet's FractionalEncoder expects normalized fractions).
    frac = torch.rand(batch, n_cations, dtype=torch.float32)
    frac = frac / frac.sum(dim=-1, keepdim=True)

    model.eval()
    with torch.no_grad():
        out = model(src, frac)

    assert out.shape == (batch, n_minerals), f"expected ({batch}, {n_minerals}), got {out.shape}"
    # Note: untrained outputs can be negative. The sigmoid gate suppresses
    # magnitude but doesn't constrain sign — the value path is unconstrained.
    # The training loss is what pushes outputs into the [0, 1] regime.
    assert torch.all(torch.isfinite(out)), "outputs must be finite"


def test_mode_loss() -> None:
    """mode_loss returns (scalar, dict[str, float]) with named components."""
    torch.manual_seed(0)
    B, P = 8, 22
    pred = torch.softmax(torch.randn(B, P), dim=-1)   # sums to 1
    target = torch.softmax(torch.randn(B, P), dim=-1)

    total, components = mode_loss(pred, target, lambda_sum=0.01)

    assert total.ndim == 0, "mode_loss total must be scalar"
    assert "mse" in components and "sum_violation" in components
    # Since both pred and target sum to 1, sum_violation should be ~0
    np.testing.assert_allclose(components["sum_violation"], 0.0, atol=1e-5)


if __name__ == "__main__":
    test_oxides_to_cations_shape()
    test_base_model_forward_shape()
    test_mode_loss()
    print("all shape tests passed")
