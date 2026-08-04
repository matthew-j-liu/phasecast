"""Joint multi-head loss composing mode, comp, cond, and closure terms.

    L = L_mode + lambda_comp * L_comp + lambda_cond * L_cond
        + lambda_balance * L_closure

where L_mode = BCE + magnitude_weight * masked Dirichlet NLL (mode_loss).
Weights come from the training config; `joint_heads_hpo` sweeps them
(ADR 006).
The cond NLL is in raw (T, P, H2O) units, so `lambda_cond` also absorbs the
scale gap to the simplex NLLs.
"""

import torch

from .closure_loss import closure_loss
from .comp_loss import masked_comp_dirichlet_nll
from .cond_loss import gmm_nll
from .mode_loss import mode_head_loss


def joint_heads_loss(
    mode_logits: torch.Tensor,
    mode_alpha: torch.Tensor,
    comp_alpha: torch.Tensor,
    cond_params: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    presence: torch.Tensor,
    magnitude: torch.Tensor,
    comp_target: torch.Tensor,
    cond_target: torch.Tensor,
    f_liq: torch.Tensor,
    f_fl: torch.Tensor,
    c_liq: torch.Tensor,
    c_fl: torch.Tensor,
    bulk_11: torch.Tensor,
    *,
    magnitude_weight: float = 1.0,
    lambda_comp: float = 1.0,
    lambda_cond: float = 1.0,
    lambda_balance: float = 1.0,
    mode_log_w: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """All-head loss; returns each unweighted term for logging.

    Presence is the teacher-forced true mask everywhere it appears: the mode
    magnitude slice, the comp mask, the CondHead input (upstream), and the
    closure modes. `mode_log_w` is None for the single-Dirichlet ModeHead
    (mode_alpha [B, P]); the mixture head passes gate log-weights with
    mode_alpha [B, K, P], and closure uses the gate-weighted mixture mean.
    """
    mode = mode_head_loss(
        mode_logits, mode_alpha, presence, magnitude,
        magnitude_weight=magnitude_weight, log_w=mode_log_w,
    )
    comp_nll = masked_comp_dirichlet_nll(comp_alpha, comp_target, presence)
    cond_nll = gmm_nll(*cond_params, cond_target)

    if mode_log_w is None:
        masked_alpha = presence * mode_alpha
        cond_modes = masked_alpha / masked_alpha.sum(-1, keepdim=True).clamp_min(1e-12)
    else:
        masked_alpha = presence.unsqueeze(1) * mode_alpha
        component_modes = masked_alpha / masked_alpha.sum(-1, keepdim=True).clamp_min(1e-12)
        cond_modes = (mode_log_w.exp().unsqueeze(-1) * component_modes).sum(1)
    comp_mean = comp_alpha / comp_alpha.sum(-1, keepdim=True)
    closure = closure_loss(
        cond_modes, comp_mean, f_liq, f_fl, c_liq, c_fl, bulk_11
    )

    total = (
        mode["total"]
        + lambda_comp * comp_nll
        + lambda_cond * cond_nll
        + lambda_balance * closure
    )
    return {
        "total": total,
        "presence_bce": mode["presence_bce"],
        "mode_nll": mode["dirichlet_nll"],
        "comp_nll": comp_nll,
        "cond_nll": cond_nll,
        "closure": closure,
    }
