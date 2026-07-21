"""Hurdle ModeHead: presence Bernoullis x conditional Dirichlet over the solids.

Reads the encoder's pooled latent and emits the two factors of the hurdle
likelihood P(modal) = P(present) * P(modal | present): per-phase presence
logits and per-phase Dirichlet concentrations. The conditional Dirichlet is
evaluated on the present support only (Aitchison sub-composition), so the
alphas are meaningful relative to whichever support they are sliced to.
Training lives in `cnmineral.training.joint_heads`; the loss in
`cnmineral.losses.mode_loss`.
"""

import torch
import torch.nn.functional as F
from torch import nn

from ..data.phase_vocab import N_PHASES


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
        blocks: list[nn.Module] = []
        in_dim = d_model
        for _ in range(layers):
            blocks += [nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
            in_dim = hidden
        self.trunk = nn.Sequential(*blocks)
        self.presence_out = nn.Linear(hidden, N_PHASES)
        self.alpha_out = nn.Linear(hidden, N_PHASES)
        self.alpha_eps = alpha_eps

    def forward(self, pooled_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, d_model] -> presence logits [B, N_PHASES], alpha [B, N_PHASES] > 0."""
        z = self.trunk(pooled_latent)
        logits = self.presence_out(z)
        alpha = F.softplus(self.alpha_out(z)) + self.alpha_eps
        return logits, alpha
