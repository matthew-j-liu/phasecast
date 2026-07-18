"""CondHead: support-conditioned per-axis GMM over (T, P, H2O).

The inverse-inference posterior. Input is the pooled latent concatenated
with the assemblage presence vector (chain rule, heads.md): teacher-forced
true mask at train, ModeHead-sampled at inference; the support enters this
head only, never the encoder. Each axis gets an independent K-component
Gaussian mixture in linear units (no log transforms -- negative samples are
diagnostics). Training lives in `cnmineral.training.joint_heads`; the loss
in `cnmineral.losses.cond_loss`.
"""

import torch
import torch.nn.functional as F
from torch import nn

from ..data.phase_vocab import N_PHASES

COND_AXES: tuple[str, ...] = ("T", "P_kbar", "H2O")
N_AXES = len(COND_AXES)


class CondHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        n_components: int = 3,
        hidden: int | None = None,
        dropout: float = 0.0,
        sigma_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        hidden = d_model // 2 if hidden is None else hidden
        self.trunk = nn.Sequential(
            nn.Linear(d_model + N_PHASES, hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.weight_out = nn.Linear(hidden, N_AXES * n_components)
        self.mu_out = nn.Linear(hidden, N_AXES * n_components)
        self.sigma_out = nn.Linear(hidden, N_AXES * n_components)
        self.n_components = n_components
        self.sigma_eps = sigma_eps

    @torch.no_grad()
    def init_output_bias(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Seed mu/sigma biases from per-axis target stats ([A] each).

        Targets are raw-unit (T ~ 1e3), so a zero-initialized head starts
        thousands of sigmas off and Adam needs ~mu/lr steps to close the gap;
        seeding makes the NLL sane from step one. Component means spread
        across +/-1 std so the mixture starts separated.
        """
        k = self.n_components
        offsets = torch.linspace(-1.0, 1.0, k, device=mean.device)
        mu = mean.unsqueeze(-1) + std.unsqueeze(-1) * offsets
        self.mu_out.bias.copy_(mu.reshape(-1))
        s = std.clamp_min(10 * self.sigma_eps).unsqueeze(-1).expand(-1, k).reshape(-1)
        inv_softplus = torch.where(s > 20.0, s, torch.log(torch.expm1(s)))
        self.sigma_out.bias.copy_(inv_softplus)

    def forward(
        self, pooled_latent: torch.Tensor, presence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """([B, d_model], [B, N_PHASES]) -> (log_w, mu, sigma), each [B, 3, K].

        Axes follow `COND_AXES` order; `log_w` normalizes over K.
        """
        k = self.n_components
        z = self.trunk(torch.cat([pooled_latent, presence], dim=-1))
        log_w = F.log_softmax(self.weight_out(z).view(-1, N_AXES, k), dim=-1)
        mu = self.mu_out(z).view(-1, N_AXES, k)
        sigma = F.softplus(self.sigma_out(z).view(-1, N_AXES, k)) + self.sigma_eps
        return log_w, mu, sigma
