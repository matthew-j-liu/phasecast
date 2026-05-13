"""Loss functions for v0 ModeHead training.

Primary task is mineral wt% regression: MSE between predicted and true vectors,
plus a soft sum-to-1 regularizer that pushes the per-sample sum toward 1
without hard-constraining it. The hard alternative — softmax — would force
mass redistribution across all phases and defeat the sigmoid-gate sparsity we
want.

Each loss function returns (scalar Tensor, dict[str, float]). The dict
contains named loss components useful for logging — they don't affect
gradients.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mode_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_sum: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    """MSE on mineral wt% + soft sum-to-1 regularizer.

    Parameters
    ----------
    pred : FloatTensor [batch, n_minerals]
        Model output. Values are in roughly [0, 1] but not constrained to sum
        to exactly 1.
    target : FloatTensor [batch, n_minerals]
        Ground-truth wt fractions, summing to ~1 (MAGEMin returns 0.997-1.003
        per row; our pipeline LOI-normalized inputs sum to exactly 100 wt%).
    lambda_sum : float
        Weight on the sum-to-1 regularizer. 0.01 default keeps the regularizer
        secondary to the per-mineral MSE; raise it if predictions drift
        too far from sum=1.

    Returns
    -------
    total_loss : scalar Tensor
        For backprop.
    components : dict[str, float]
        For logging: `mse` and `sum_violation` (sum-to-1 MSE term).
    """
    mse = F.mse_loss(pred, target)
    sum_violation = (pred.sum(dim=-1) - 1.0).pow(2).mean()
    total = mse + lambda_sum * sum_violation
    return total, {"mse": float(mse.item()), "sum_violation": float(sum_violation.item())}
