"""Hurdle mode head in group space, supervised directly on measured modes.

The alternative to correcting the simulator-supervised `ModeHead`: same frozen
encoder, same rocks, same hurdle likelihood, but the parameters are fit from
measured modal labels instead of MAGEMin ones.  Where `RealRockRecalibrator`
starts from the frozen head's forecast and learns a correction, this starts
from nothing and reads the pooled latent directly, so the contrast between them
isolates what the simulator supervision was worth (#53).

It speaks only the seven groups measured modal data resolves, against the
simulator-supervised head's 21 phases plus per-phase compositions plus
formation conditions -- a real asymmetry, and one to state rather than engineer
away.  Fitting lives in `cnmineral.training.measured_mode_head`.
"""

import torch
import torch.nn.functional as F
from torch import nn

from ..data.phase_vocab import GROUPS


class MeasuredModeHead(nn.Module):
    """Pooled latent -> (7 presence logits, Dirichlet mixture over the support).

    Mirrors `ModeHead`'s split: the presence branch sees the latent only, since
    it predicts the support and must not see it, while the magnitude branch is
    conditioned on the support alongside the latent.  Output is always
    mixture-shaped so `training.real_rock_recalibrator.score_predictions` takes
    it unchanged; at `n_components` 1 the single log-weight is exactly 0.

    `n_components` defaults to 2 to match the canonical `tailed` recalibration
    tier, whose entire margin over `matrix` came from a two-parameter precision
    contamination.  A single Dirichlet cannot express that tail, and scoring the
    two arms against each other with only one of them able to would measure tail
    flexibility rather than label source.

    No dropout: the fit is full-batch with no per-draw early stopping, so weight
    decay is deliberately the one selected regularizer, exactly as it is for the
    recalibrator.
    """

    def __init__(
        self,
        d_model: int,
        *,
        hidden: int = 64,
        n_components: int = 2,
        alpha_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        n = len(GROUPS)
        self.presence_trunk = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU())
        self.presence_out = nn.Linear(hidden, n)
        self.magnitude_trunk = nn.Sequential(nn.Linear(d_model + n, hidden), nn.GELU())
        self.alpha_out = nn.Linear(hidden, n_components * n)
        if n_components > 1:
            self.gate_out = nn.Linear(hidden, n_components)
        self.alpha_eps = alpha_eps
        self.n_components = n_components
        self.n_groups = n

        # Filled by `fit_measured_head` from the drawn budget's latents and
        # carried on the module, so scoring a held-out slice applies the same
        # shift and scale the fit was written against.
        self.register_buffer("latent_mean", torch.zeros(d_model))
        self.register_buffer("latent_std", torch.ones(d_model))

    def standardize(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.latent_mean) / self.latent_std

    def forward(
        self, latent: torch.Tensor, support: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """([B, d_model], [B, 7]) -> (logits [B, 7], alpha [B, K, 7], log_w [B, K])."""
        z = self.standardize(latent)
        logits = self.presence_out(self.presence_trunk(z))
        h = self.magnitude_trunk(torch.cat([z, support], dim=-1))
        alpha = F.softplus(self.alpha_out(h)) + self.alpha_eps
        alpha = alpha.view(-1, self.n_components, self.n_groups)
        if self.n_components == 1:
            log_w = alpha.new_zeros(alpha.shape[0], 1)
        else:
            log_w = F.log_softmax(self.gate_out(h), dim=-1)
        return logits, alpha, log_w

    def extra_repr(self) -> str:
        return f"n_groups={self.n_groups}, n_components={self.n_components}"
