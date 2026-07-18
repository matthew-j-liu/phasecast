"""CompHead: per-phase Dirichlet over the 11 MAGEMin output oxides.

Reads the encoder's pooled latent and emits one Dirichlet per solid phase
over `magemin_corpus.COMP_OXIDES`. Composition is bulk-controlled, so the
head reads `h` only -- no conditioning on support or conditions (heads.md).
Parameterized as mean x shared concentration: a softmax over oxides gives
the Dirichlet mean, one softplus scalar per phase scales certainty. Training
lives in `cnmineral.training.joint_heads`; the loss in
`cnmineral.losses.comp_loss`.
"""

import torch
import torch.nn.functional as F
from torch import nn

from ..data.magemin_corpus import COMP_OXIDES
from ..data.phase_vocab import N_PHASES

N_OXIDES = len(COMP_OXIDES)


class CompHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        hidden: int | None = None,
        dropout: float = 0.0,
        alpha_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        hidden = d_model // 2 if hidden is None else hidden
        self.trunk = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.mean_out = nn.Linear(hidden, N_PHASES * N_OXIDES)
        self.conc_out = nn.Linear(hidden, N_PHASES)
        self.alpha_eps = alpha_eps

    def forward(self, pooled_latent: torch.Tensor) -> torch.Tensor:
        """[B, d_model] -> alpha [B, N_PHASES, N_OXIDES] > 0.

        `alpha / alpha.sum(-1)` is the softmax mean, invariant to the
        concentration scalar; `alpha.sum(-1)` is the per-phase concentration.
        """
        z = self.trunk(pooled_latent)
        mean = F.softmax(
            self.mean_out(z).view(-1, N_PHASES, N_OXIDES), dim=-1
        )
        conc = F.softplus(self.conc_out(z)) + self.alpha_eps
        return conc.unsqueeze(-1) * mean
