"""Mass-balance closure: predicted solids + teacher-forced melt vs input bulk.

    B = (1 - F_liq - F_fl) * sum_solid(m_i * C_i) + F_liq*C_liq + F_fl*C_fl

`m` is the ModeHead conditional Dirichlet mean over the teacher-forced true
support (crystals-renormalized), `C` the CompHead Dirichlet means; the melt
and fluid fractions and compositions come from the corpus row, so the term
constrains the predicted solids instead of acting as a free error sink
(heads.md). The penalty is the squared L2 norm of the per-oxide residual --
a constraint penalty, not a head likelihood, so the no-MSE rule for head
losses does not apply. Training-time only.
"""

import torch


def closure_residual(
    cond_modes: torch.Tensor,
    comp_mean: torch.Tensor,
    f_liq: torch.Tensor,
    f_fl: torch.Tensor,
    c_liq: torch.Tensor,
    c_fl: torch.Tensor,
    bulk_11: torch.Tensor,
) -> torch.Tensor:
    """Per-oxide residual [B, O] of the reconstructed bulk against the input.

    `cond_modes` [B, P] sums to 1 over the true support; `comp_mean` [B, P, O]
    rows on the oxide simplex; fixings `f_liq`/`f_fl` [B] and `c_liq`/`c_fl`/
    `bulk_11` [B, O] in the same 11-oxide mass-fraction basis.
    """
    solids = torch.einsum("bp,bpo->bo", cond_modes, comp_mean)
    f_solid = (1.0 - f_liq - f_fl).unsqueeze(-1)
    recon = f_solid * solids + f_liq.unsqueeze(-1) * c_liq + f_fl.unsqueeze(-1) * c_fl
    return recon - bulk_11


def closure_loss(
    cond_modes: torch.Tensor,
    comp_mean: torch.Tensor,
    f_liq: torch.Tensor,
    f_fl: torch.Tensor,
    c_liq: torch.Tensor,
    c_fl: torch.Tensor,
    bulk_11: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Squared L2 norm of the closure residual per row."""
    sq = closure_residual(
        cond_modes, comp_mean, f_liq, f_fl, c_liq, c_fl, bulk_11
    ).pow(2).sum(-1)
    if reduction == "mean":
        return sq.mean()
    if reduction == "none":
        return sq
    raise ValueError(f"unknown reduction: {reduction}")
