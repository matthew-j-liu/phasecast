"""Tests for `cnmineral.models.fractional_encoder.FractionalEncoder`.

Three contracts:
- Output shape is `[B, T, d_model // 2]` (CrabNet's half-width
  convention; two encoders concatenate to d_model).
- Both the linear and log10 branches stay finite at the extreme real
  inputs: 0 (padding), 1e-6 (ppm-scale trace), and 1.0.  The log10
  branch in particular survives `log2(0) = -inf` only through its
  clamp chain, which nothing else pins.
- The encoding is deterministic and resolves distinct abundances to
  distinct vectors at production resolution.
"""

import pytest
import torch

from cnmineral.models.fractional_encoder import FractionalEncoder


def test_output_shape_is_half_d_model() -> None:
    fe = FractionalEncoder(d_model=64, resolution=200)
    out = fe(torch.rand(2, 22))
    assert out.shape == (2, 22, 32)


@pytest.mark.parametrize("log10", [False, True])
def test_extreme_fracs_stay_finite(log10: bool) -> None:
    fe = FractionalEncoder(d_model=64, resolution=5000, log10=log10)
    x = torch.tensor([[0.0, 1e-6, 1 / 5000, 0.5, 1.0]])
    out = fe(x)
    assert torch.isfinite(out).all(), f"non-finite encoding (log10={log10})"


def test_deterministic_and_distinct_at_production_resolution() -> None:
    fe = FractionalEncoder(d_model=64, resolution=5000)
    x = torch.tensor([[0.1, 0.2]])
    a, b = fe(x), fe(x)
    torch.testing.assert_close(a, b)
    assert not torch.equal(a[0, 0], a[0, 1]), (
        "0.1 and 0.2 mapped to the same encoding; resolution binning "
        "is collapsing distinct abundances"
    )
