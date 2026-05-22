"""BaseModel (v0) + BaseModelV1 (v1, inverse-inference architecture).

v0 — `BaseModel` wraps kingcrab.SubCrab end-to-end (encoder + sigmoid-gated
ResidualNetwork head); predicts only modal wt%. Kept here so the v0
checkpoint at `checkpoints/base_v0/best.pt` can be re-loaded for comparison
without code juggling.

v1 — `BaseModelV1` is the inverse-inference architecture defined in ADR 006:
shared SubCrab Encoder -> {Dirichlet ModeHead, GMM CondHead}. Forward returns
a dict so heads can be added later (CompHead v1.1, GsysHead v2) without
breaking the call signature.
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kingcrab import SubCrab

from cnmineral.models.encoder import Encoder
from cnmineral.models.heads import ModeHead, CondHead


class BaseModel(nn.Module):
    """v0 ModeHead-only model. Preserved for the v0 baseline checkpoint."""

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
            out_dims=2 * n_minerals,
            fractional=True,
            attention=True,
            dropout=dropout,
            elem_prop=elem_prop,
        )

    def forward(self, src: torch.Tensor, frac: torch.Tensor) -> torch.Tensor:
        return self.subcrab(src, frac)


class BaseModelV1(nn.Module):
    """Encoder + Dirichlet ModeHead + 3-axis 1D-GMM CondHead.

    Forward graph takes 11 oxide wt% (as 10-cation `src`/`frac` after Fe
    collapse) only. T, P, H2O are NOT inputs — they are per-row training
    supervision for the CondHead.

    Parameters
    ----------
    n_minerals : int
        Modal vocabulary size from the corpus (18 in v1).
    K_gmm : int
        Mixture components per condition axis. ADR 006 expects HPO over {2,3,4}.
    d_model, N, heads, dropout, elem_prop, dim_feedforward
        Encoder knobs.
    mode_hidden, cond_hidden : list[int]
        ResidualNetwork hidden sizes for each head.

    Returns
    -------
    dict
        {"alpha": [B, n_minerals],
         "cond_means": [B, 3, K_gmm],
         "cond_log_sigmas": [B, 3, K_gmm],
         "cond_mix_logits": [B, 3, K_gmm]}
    """

    def __init__(
        self,
        n_minerals: int,
        K_gmm: int = 3,
        d_model: int = 256,
        N: int = 2,
        heads: int = 8,
        dropout: float = 0.1,
        elem_prop: str = "mat2vec",
        dim_feedforward: int = 2048,
        mode_hidden: list[int] | None = None,
        cond_hidden: list[int] | None = None,
        alpha_floor: float = 1e-3,
    ) -> None:
        super().__init__()
        self.n_minerals = n_minerals
        self.K_gmm = K_gmm
        self.encoder = Encoder(
            d_model=d_model,
            N=N,
            heads=heads,
            dropout=dropout,
            elem_prop=elem_prop,
            dim_feedforward=dim_feedforward,
        )
        self.mode_head = ModeHead(
            d_model=d_model,
            n_minerals=n_minerals,
            hidden=mode_hidden,
            alpha_floor=alpha_floor,
        )
        self.cond_head = CondHead(
            d_model=d_model,
            K_gmm=K_gmm,
            n_axes=3,
            hidden=cond_hidden,
        )

    def forward(self, src: torch.Tensor, frac: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(src, frac)
        alpha = self.mode_head(h)
        cond = self.cond_head(h)
        return {
            "alpha": alpha,
            "cond_means": cond["means"],
            "cond_log_sigmas": cond["log_sigmas"],
            "cond_mix_logits": cond["mix_logits"],
        }
