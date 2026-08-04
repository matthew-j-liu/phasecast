"""Hurdle ModeHead: presence Bernoullis x support-conditioned Dirichlet.

Reads the encoder's pooled latent and emits the two factors of the hurdle
likelihood P(modal) = P(present) * P(modal | present). The presence branch
sees the latent only (it predicts the support, so it must not see it); the
magnitude branch is conditioned on the assemblage support alongside the
latent (chain rule, heads.md) -- teacher-forced true mask at train,
ModeHead-sampled at inference -- so its Dirichlet describes "how much,
given this rock and this mineral set" instead of one bulk-level compromise.
The conditional Dirichlet is evaluated on the present support only
(Aitchison sub-composition). At `n_components` > 1 the magnitude branch
emits K alpha vectors plus gate log-weights from the same trunk (a mixture
of Dirichlets over the support); K=1 is the historical single-Dirichlet
head, bit-identical in parameters and outputs. Training lives in
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
        n_components: int = 1,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}")
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        hidden = d_model // 2 if hidden is None else hidden
        self.presence_trunk = _trunk(d_model, hidden, layers, dropout)
        self.presence_out = nn.Linear(hidden, N_PHASES)
        self.magnitude_trunk = _trunk(d_model + N_PHASES, hidden, layers, dropout)
        # K=1 keeps exactly this module tree (names and shapes) so every
        # pre-mixture checkpoint loads unchanged; the gate exists only at K>1
        self.alpha_out = nn.Linear(hidden, n_components * N_PHASES)
        if n_components > 1:
            self.gate_out = nn.Linear(hidden, n_components)
        self.alpha_eps = alpha_eps
        self.n_components = n_components

    def presence_logits(self, pooled_latent: torch.Tensor) -> torch.Tensor:
        """[B, d_model] -> presence logits [B, N_PHASES]."""
        return self.presence_out(self.presence_trunk(pooled_latent))

    def magnitude_alpha(
        self, pooled_latent: torch.Tensor, presence: torch.Tensor
    ) -> torch.Tensor:
        """([B, d_model], [B, N_PHASES]) -> alpha [B, N_PHASES] > 0.

        Single-Dirichlet head only; the mixture head's alphas are ambiguous
        without their gate, so K>1 callers must use `magnitude_mixture`.
        """
        if self.n_components > 1:
            raise ValueError("K>1 head: use magnitude_mixture, not magnitude_alpha")
        z = self.magnitude_trunk(torch.cat([pooled_latent, presence], dim=-1))
        return F.softplus(self.alpha_out(z)) + self.alpha_eps

    def magnitude_mixture(
        self, pooled_latent: torch.Tensor, presence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """([B, d_model], [B, N_PHASES]) -> (alpha [B, K, N_PHASES], log_w [B, K]).

        Valid at any K; at K=1 the single log-weight is exactly 0.
        """
        z = self.magnitude_trunk(torch.cat([pooled_latent, presence], dim=-1))
        alpha = F.softplus(self.alpha_out(z)) + self.alpha_eps
        alpha = alpha.view(-1, self.n_components, N_PHASES)
        if self.n_components == 1:
            log_w = alpha.new_zeros(alpha.shape[0], 1)
        else:
            log_w = F.log_softmax(self.gate_out(z), dim=-1)
        return alpha, log_w

    def forward(self, pooled_latent: torch.Tensor, presence: torch.Tensor):
        """K=1: (presence logits, alpha), each [B, N_PHASES] -- the historical
        contract. K>1: (presence logits, alpha [B, K, N_PHASES], log_w [B, K]).
        """
        logits = self.presence_logits(pooled_latent)
        if self.n_components == 1:
            return logits, self.magnitude_alpha(pooled_latent, presence)
        alpha, log_w = self.magnitude_mixture(pooled_latent, presence)
        return logits, alpha, log_w
