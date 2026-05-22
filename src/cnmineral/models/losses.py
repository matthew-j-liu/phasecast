"""Loss functions: v0 mode_loss preserved + v1 joint NLL (Dirichlet + GMM).

v0: MSE + soft sum-to-1 regularizer on modal wt%. Used by the legacy
BaseModel checkpoint at checkpoints/base_v0.

v1: joint NLL of the inverse-inference architecture in ADR 006.
- ModeHead -> Dirichlet NLL with label smoothing eps to avoid log(0) on
  exact-zero phase labels.
- CondHead -> per-axis 1D-GMM NLL, summed across (T, P, H2O). Targets are
  expected in z-scored space (dataset z-scores conditions on load).
- Joint: 0.5 * NLL_dir + 0.5 * lambda_cond * NLL_gmm. lambda_cond is an HPO
  knob (ADR 006 search dim).

Each loss returns (scalar, dict) where the dict carries named components for
logging and HPO objective decomposition.
"""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Dirichlet, MixtureSameFamily, Normal


def mode_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_sum: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    """v0 MSE + soft sum-to-1 regularizer. Keep verbatim for v0 reproducibility."""
    mse = F.mse_loss(pred, target)
    sum_violation = (pred.sum(dim=-1) - 1.0).pow(2).mean()
    total = mse + lambda_sum * sum_violation
    return total, {"mse": float(mse.item()), "sum_violation": float(sum_violation.item())}


def smooth_simplex(y: torch.Tensor, eps: float) -> torch.Tensor:
    """Label-smooth a sum-to-1 vector so every component is strictly positive.

    y' = (1 - eps) * y + eps / K. Preserves sum-to-1 exactly.
    """
    K = y.shape[-1]
    return (1.0 - eps) * y + eps / K


def dirichlet_nll(
    alpha: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Mean Dirichlet NLL across the batch.

    `target` must sum to 1 along the last dim (mineral wt% from MAGEMin do, up
    to FP error). We label-smooth into the open simplex by `eps` to avoid
    log(0) on zero phases.
    """
    y = smooth_simplex(target, eps)
    y = y / y.sum(dim=-1, keepdim=True)
    dist = Dirichlet(alpha)
    return -dist.log_prob(y).mean()


def gmm_nll_1d(
    means: torch.Tensor,
    log_sigmas: torch.Tensor,
    mix_logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean 1D-GMM NLL across batch x axes.

    means, log_sigmas, mix_logits : [B, A, K]    (A = n_axes, K = n_components)
    target                        : [B, A]      (z-scored)

    Independent mixture per axis. Returns scalar mean over (B * A).
    """
    sigmas = log_sigmas.exp()
    component = Normal(means, sigmas)
    mixture = Categorical(logits=mix_logits)
    gmm = MixtureSameFamily(mixture, component)
    log_p = gmm.log_prob(target)
    return -log_p.mean()


def joint_loss(
    out: dict[str, torch.Tensor],
    target_modal: torch.Tensor,
    target_cond: torch.Tensor,
    lambda_cond: float = 1.0,
    eps_smooth: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """v1 joint NLL.

    Parameters
    ----------
    out : dict
        BaseModelV1.forward output. Requires keys: alpha, cond_means,
        cond_log_sigmas, cond_mix_logits.
    target_modal : [B, n_minerals]
        Mineral wt% (sums to ~1; smoothed inside dirichlet_nll).
    target_cond : [B, 3]
        Z-scored (T, P, H2O).
    lambda_cond : float
        Weight on the GMM term. 1.0 means equal weighting after the
        intrinsic 0.5/0.5 split below.
    eps_smooth : float
        Dirichlet label smoothing.

    Returns
    -------
    total : scalar tensor
    components : dict[str, float]
        Loggable: nll_dir, nll_gmm, lambda_cond_used. Useful for the HPO
        objective which needs the two losses separately to z-score them.
    """
    nll_dir = dirichlet_nll(out["alpha"], target_modal, eps=eps_smooth)
    nll_gmm = gmm_nll_1d(
        out["cond_means"],
        out["cond_log_sigmas"],
        out["cond_mix_logits"],
        target_cond,
    )
    total = 0.5 * nll_dir + 0.5 * lambda_cond * nll_gmm
    return total, {
        "nll_dir": float(nll_dir.item()),
        "nll_gmm": float(nll_gmm.item()),
        "lambda_cond": float(lambda_cond),
    }
