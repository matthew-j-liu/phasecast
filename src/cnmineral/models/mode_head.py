"""Hurdle ModeHead: presence Bernoullis x support-conditioned Dirichlet.

Reads the encoder's pooled latent and emits the two factors of the hurdle
likelihood P(modal) = P(present) * P(modal | present). The presence branch
sees the latent only (it predicts the support, so it must not see it); the
magnitude branch is conditioned on the assemblage support alongside the
latent (chain rule, heads.md) -- teacher-forced true mask at train,
ModeHead-sampled at inference -- so its Dirichlet describes "how much,
given this rock and this mineral set" instead of one bulk-level compromise.
The conditional Dirichlet is evaluated on the present support only
(Aitchison sub-composition). Training lives in
`cnmineral.training.joint_heads`; the loss in `cnmineral.losses.mode_loss`.
"""

import torch
import torch.nn.functional as F
from torch import nn

from ..data.phase_vocab import N_PHASES


def _trunk(in_dim: int, hidden: int, layers: int, dropout: float) -> nn.Sequential:
    blocks: list[nn.Module] = []
    for _ in range(layers):
        blocks += [nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        in_dim = hidden
    return nn.Sequential(*blocks)


class ModeHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        hidden: int | None = None,
        layers: int = 1,
        dropout: float = 0.0,
        alpha_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}")
        hidden = d_model // 2 if hidden is None else hidden
        self.presence_trunk = _trunk(d_model, hidden, layers, dropout)
        self.presence_out = nn.Linear(hidden, N_PHASES)
        self.magnitude_trunk = _trunk(d_model + N_PHASES, hidden, layers, dropout)
        self.alpha_out = nn.Linear(hidden, N_PHASES)
        self.alpha_eps = alpha_eps

    def presence_logits(self, pooled_latent: torch.Tensor) -> torch.Tensor:
        """[B, d_model] -> presence logits [B, N_PHASES]."""
        return self.presence_out(self.presence_trunk(pooled_latent))

    def magnitude_alpha(
        self, pooled_latent: torch.Tensor, presence: torch.Tensor
    ) -> torch.Tensor:
        """([B, d_model], [B, N_PHASES]) -> alpha [B, N_PHASES] > 0."""
        z = self.magnitude_trunk(torch.cat([pooled_latent, presence], dim=-1))
        return F.softplus(self.alpha_out(z)) + self.alpha_eps

    def forward(self, pooled_latent: torch.Tensor, presence: torch.Tensor):
        """(presence logits, alpha), each [B, N_PHASES]."""
        logits = self.presence_logits(pooled_latent)
        return logits, self.magnitude_alpha(pooled_latent, presence)
