"""BaseModel for v0: wraps kingcrab.SubCrab to predict mineral wt% from
cation wt% input.

This is essentially upstream CrabNet-Mineral's BaseCrabNet
(alignn/src/base_model.py, lines 16-52), adapted to our parameter names. The
sigmoid-gate sparsity behavior (value × sigmoid(logits)) comes for free from
SubCrab's existing output head — see kingcrab.py:542-544.

The model expects pre-converted cation wt%, not oxide wt%. See
TrainMageminDataset for the conversion.

When v1 adds CompHead / GsysHead / ClassHead, this file is replaced with a
new model.py that separates the Encoder from per-head outputs and shares the
encoder across heads. Not in v0 scope.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

# Vendored kingcrab.py lives at src/cnmineral/kingcrab.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kingcrab import SubCrab


class BaseModel(nn.Module):
    """Wraps SubCrab; predicts mineral wt% from atomic numbers + cation fractions.

    Parameters
    ----------
    n_minerals : int
        Number of mineral phases to predict (= H5's /Y_cols length, e.g. 22).
        Internally `out_dims` is set to `2 * n_minerals` so SubCrab's
        `value × sigmoid(gate)` splits cleanly: first half is value, second is
        gate logits.
    d_model, N, heads, dropout, elem_prop : standard CrabNet params.

    Notes
    -----
    The compute_device argument forwarded to SubCrab matters when SubCrab loads
    its element-property embedding (mat2vec) — let SubCrab decide based on
    the parent module's device after `.to(device)`.
    """

    def __init__(
        self,
        n_minerals: int,
        d_model: int = 256,
        N: int = 2,
        heads: int = 8,
        dropout: float = 0.1,
        elem_prop: str = "mat2vec",
    ) -> None:
        super().__init__()
        self.n_minerals = n_minerals
        self.subcrab = SubCrab(
            d_model=d_model,
            N=N,
            heads=heads,
            out_dims=2 * n_minerals,        # value + gate, sigmoid-gated inside SubCrab
            fractional=True,
            attention=True,
            dropout=dropout,
            elem_prop=elem_prop,
        )

    def forward(self, src: torch.Tensor, frac: torch.Tensor) -> torch.Tensor:
        """Predict mineral wt% vector.

        Parameters
        ----------
        src : LongTensor [batch, n_cations]
            Atomic numbers; same vocabulary per sample. Zeros are treated as
            padding (SubCrab masks them in pooling).
        frac : FloatTensor [batch, n_cations]
            Cation wt% in the same order as `src`.

        Returns
        -------
        FloatTensor [batch, n_minerals]
            Per-mineral predicted wt fractions. The sigmoid gate inside SubCrab
            zeros out absent phases; values are in roughly [0, 1] but not
            constrained to sum exactly to 1 — the soft sum-to-1 regularizer in
            the loss handles that.
        """
        return self.subcrab(src, frac)
