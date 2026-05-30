"""Shared bulk-only encoder for v1.

Wraps kingcrab's Encoder transformer + a masked mean over real cation tokens,
returning a single pooled [B, d_model] latent. Heads (ModeHead, CondHead)
consume this latent.

Position note: T, P, H2O do NOT enter here. The model's forward graph only
ever sees 11 oxide wt% (10 cation tokens after Fe collapse). T/P/H2O are
per-row supervision targets for CondHead, not inputs.
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kingcrab import Encoder as KingcrabEncoder


class Encoder(nn.Module):
    """SubCrab transformer + masked-mean pooling.

    Output: [B, d_model] pooled latent. No output head, no sigmoid gating;
    those belong to ModeHead and CondHead.

    Parameters mirror the relevant SubCrab knobs. `extra_features` and the
    `out_dims`-related machinery are absent because pooled latent is the only
    output.
    """

    def __init__(
        self,
        d_model: int = 256,
        N: int = 2,
        heads: int = 8,
        dropout: float = 0.1,
        elem_prop: str = "mat2vec",
        dim_feedforward: int = 2048,
        pe_resolution: int = 5000,
        ple_resolution: int = 5000,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.encoder = KingcrabEncoder(
            d_model=d_model,
            N=N,
            heads=heads,
            attention=True,
            compute_device=None,
            pe_resolution=pe_resolution,
            ple_resolution=ple_resolution,
            elem_prop=elem_prop,
            emb_scaler=1.0,
            pos_scaler=1.0,
            pos_scaler_log=1.0,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(
        self, src: torch.Tensor, frac: torch.Tensor, return_tokens: bool = False,
    ) -> torch.Tensor:
        """Encode + mask-mean pool.

        Parameters
        ----------
        src : LongTensor [B, T]
            Atomic numbers; zeros treated as padding.
        frac : FloatTensor [B, T]
            Cation mass fractions in same order as `src` (already / 100 by the
            dataset to match kingcrab's [0, 1] expectation).
        return_tokens : bool
            If False (default, backward-compatible), return the pooled latent
            [B, d_model]. If True, return the pre-pool per-token output
            [B, T, d_model] with padding positions zeroed — used by the SSL
            pretraining head for per-token reconstruction.

        Returns
        -------
        FloatTensor [B, d_model] if return_tokens=False, else [B, T, d_model].
        """
        h = self.encoder(src, frac)
        pad_mask = (src == 0).unsqueeze(-1)
        h = h.masked_fill(pad_mask, 0.0)
        if return_tokens:
            return h
        n_real = (~pad_mask).sum(dim=1).clamp(min=1).to(h.dtype)
        return h.sum(dim=1) / n_real
