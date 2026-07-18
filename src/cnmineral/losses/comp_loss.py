"""CompHead loss: present-masked Dirichlet NLL over the oxide simplex.

Full 11-simplex Dirichlet NLL per (row, phase), counted only where the label
marks the phase present (teacher-forced mask; absent phases contribute zero
gradient). Unlike the ModeHead magnitude loss there is no sub-composition
slicing on the oxide axis: targets are eps-clipped to full support by the
label builder, so every oxide participates.
"""

import torch


def masked_comp_dirichlet_nll(
    alpha: torch.Tensor,
    comp_target: torch.Tensor,
    presence: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """NLL of Dir(alpha) against eps-clipped oxide fractions, present phases only.

    `alpha` [B, P, O] positive; `comp_target` [B, P, O] rows on the simplex
    (arbitrary at absent phases); `presence` [B, P] the 0/1 support mask.
    Mean reduction divides by the number of present (row, phase) pairs.
    """
    log_beta = torch.lgamma(alpha).sum(-1) - torch.lgamma(alpha.sum(-1))
    log_kernel = torch.xlogy(alpha - 1.0, comp_target.clamp_min(1e-12)).sum(-1)
    nll = presence.to(alpha.dtype) * (log_beta - log_kernel)
    if reduction == "mean":
        return nll.sum() / presence.sum().clamp_min(1.0)
    if reduction == "none":
        return nll
    raise ValueError(f"unknown reduction: {reduction}")
