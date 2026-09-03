"""Hurdle ModeHead loss: presence BCE + present-masked Dirichlet NLL.

The two surrogate terms of the hurdle likelihood, trained on separate clean
targets and never multiplied: BCE on the per-phase presence logits, and the
NLL of the Dirichlet sub-composition restricted to the label's present
support (teacher-forced mask; absent phases contribute zero magnitude
gradient). `mode_head_loss` combines them and is shaped to compose into the
future joint multi-head loss. The joint hurdle NLL for evaluation is a
different assembly and lives with the eval code, not here.
"""

import torch
import torch.nn.functional as F


def masked_dirichlet_nll(
    alpha: torch.Tensor,
    magnitude: torch.Tensor,
    presence: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """NLL of Dir(alpha_S) against the present-only renormalized target.

    `alpha` [B, P] positive; `magnitude` [B, P] the present-renormalized
    modes (zeros at absent phases); `presence` [B, P] the 0/1 support mask.
    A Dirichlet sub-composition is itself Dirichlet on the retained
    parameters, so slicing by the mask is a proper density. Single-phase
    supports contribute exactly 0 (degenerate point mass).
    """
    m = presence.to(alpha.dtype)
    log_beta = (m * torch.lgamma(alpha)).sum(-1) - torch.lgamma((m * alpha).sum(-1))
    log_kernel = torch.xlogy(m * (alpha - 1.0), magnitude.clamp_min(1e-12)).sum(-1)
    nll = log_beta - log_kernel
    if reduction == "mean":
        return nll.mean()
    if reduction == "none":
        return nll
    raise ValueError(f"unknown reduction: {reduction}")


def mixture_dirichlet_nll(
    log_w: torch.Tensor,
    alpha: torch.Tensor,
    magnitude: torch.Tensor,
    presence: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """NLL of a K-component Dirichlet mixture on the present support.

    `log_w` [B, K] normalized log gate weights; `alpha` [B, K, P] positive;
    `magnitude` / `presence` [B, P] as in `masked_dirichlet_nll`. Each
    component is the same masked sub-composition density; the mixture
    combines them in log space (`cond_loss.gmm_nll` pattern). At K=1 with
    `log_w` zeros this reproduces `masked_dirichlet_nll` exactly.
    """
    comp_nll = masked_dirichlet_nll(
        alpha, magnitude.unsqueeze(1), presence.unsqueeze(1), reduction="none"
    )
    nll = -torch.logsumexp(log_w - comp_nll, dim=-1)
    if reduction == "mean":
        return nll.mean()
    if reduction == "none":
        return nll
    raise ValueError(f"unknown reduction: {reduction}")


def mode_head_loss(
    logits: torch.Tensor,
    alpha: torch.Tensor,
    presence: torch.Tensor,
    magnitude: torch.Tensor,
    *,
    magnitude_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Combined ModeHead loss; returns each term for logging.

    `total` is what to backpropagate; `presence_bce` and `dirichlet_nll` are
    the unweighted terms.
    """
    bce = F.binary_cross_entropy_with_logits(logits, presence)
    nll = masked_dirichlet_nll(alpha, magnitude, presence)
    return {
        "total": bce + magnitude_weight * nll,
        "presence_bce": bce,
        "dirichlet_nll": nll,
    }
