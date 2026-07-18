"""CondHead loss: per-axis Gaussian-mixture NLL over (T, P, H2O).

Each axis is an independent K-component mixture in linear units; the row NLL
sums the three axis NLLs. Targets are raw corpus values (T_end in C, P in
kbar, H2O in wt%) -- no standardization here, so the scale of this loss is
set by the T axis; the joint loss weights it via `lambda_cond`.
"""

import math

import torch


def gmm_nll(
    log_w: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """NLL of the per-axis mixture; parameters [B, A, K], `target` [B, A]."""
    t = target.unsqueeze(-1)
    log_norm = -0.5 * ((t - mu) / sigma) ** 2 - sigma.log() - 0.5 * math.log(2 * math.pi)
    nll = -torch.logsumexp(log_w + log_norm, dim=-1).sum(-1)
    if reduction == "mean":
        return nll.mean()
    if reduction == "none":
        return nll
    raise ValueError(f"unknown reduction: {reduction}")
