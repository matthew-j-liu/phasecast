"""Probabilistic heads for v1: Dirichlet ModeHead + 1D-GMM CondHead.

Both heads consume the shared [B, d_model] latent from `encoder.Encoder`.

ModeHead: per-sample Dirichlet concentration `alpha` of shape [B, n_minerals].
The Dirichlet is the natural distribution on the simplex; mineral wt% labels
sum to 1, so the head's output induces a valid distribution by construction.
Aleatoric uncertainty per phase = Dirichlet variance. Multimodality is added
across the K=5 deep ensemble.

CondHead: per-axis (T, P, H2O) Gaussian mixture with `K_gmm` components.
Independent per axis (Alt 10 in ADR 006: joint MVN-GMM deferred). Linear-
space (no log transform) per ADR. Heteroscedastic — log_sigma is per-sample,
per-component, not a global parameter.
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kingcrab import ResidualNetwork


COND_AXES = ("T", "P", "H2O")


def _residual_mlp(in_dim: int, out_dim: int, hidden: list[int]) -> nn.Module:
    """Thin wrapper picking ResidualNetwork (matches v0 head topology).

    Falls back to a plain Linear if hidden is empty, since ResidualNetwork
    requires at least one hidden layer.
    """
    if not hidden:
        return nn.Linear(in_dim, out_dim, bias=False)
    return ResidualNetwork(in_dim, out_dim, hidden, bias=False)


class ModeHead(nn.Module):
    """Dirichlet over modal wt%: outputs `alpha` of shape [B, n_minerals].

    Uses softplus + small floor to keep alpha strictly positive. The floor
    also caps how concentrated the predicted distribution can be at the
    simplex corners (alpha -> 0+ implies near-degenerate masses on those
    vertices, which is numerically nasty under NLL).
    """

    def __init__(
        self,
        d_model: int,
        n_minerals: int,
        hidden: list[int] | None = None,
        alpha_floor: float = 1e-3,
    ) -> None:
        super().__init__()
        hidden = hidden if hidden is not None else [256, 128]
        self.net = _residual_mlp(d_model, n_minerals, hidden)
        self.alpha_floor = alpha_floor

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.net(h)
        return F.softplus(logits) + self.alpha_floor


class CondHead(nn.Module):
    """3-axis (T, P, H2O) independent 1D-GMM head with K_gmm components per axis.

    Returns a dict with three [B, 3, K_gmm] tensors:
      - `means`       : per-component means in z-scored target space
      - `log_sigmas`  : per-component log std deviations (clamped at forward)
      - `mix_logits`  : per-component mixture weights as raw logits

    Axes are independent (per-axis Categorical × Normal mixture); the dataset
    z-scores T, P, H2O independently so the model doesn't see drastically
    different scales across axes. Inference de-normalizes via the dataset's
    saved (mean, std) per axis.
    """

    def __init__(
        self,
        d_model: int,
        K_gmm: int = 3,
        n_axes: int = 3,
        hidden: list[int] | None = None,
        log_sigma_min: float = -5.0,
        log_sigma_max: float = 3.0,
    ) -> None:
        super().__init__()
        hidden = hidden if hidden is not None else [256, 128]
        self.K_gmm = K_gmm
        self.n_axes = n_axes
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        out_dim = n_axes * 3 * K_gmm
        self.net = _residual_mlp(d_model, out_dim, hidden)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        B = h.shape[0]
        raw = self.net(h).view(B, self.n_axes, 3, self.K_gmm)
        means = raw[:, :, 0, :]
        log_sigmas = raw[:, :, 1, :].clamp(self.log_sigma_min, self.log_sigma_max)
        mix_logits = raw[:, :, 2, :]
        return {"means": means, "log_sigmas": log_sigmas, "mix_logits": mix_logits}
