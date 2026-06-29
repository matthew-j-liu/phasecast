"""
Masked-element loss on the positions that were masked during the SSL
corruption step.  The predicted value for non-masked positions is ignored
(no gradient through the loss).

`masked_element_log_mse` is the SSL pretrain objective: a per-element
standardized log-space MSE.  The log space lifts the traces off the floor
that a linear-frac MSE pins them to (major fracs are ~1e7-1e8x larger, so a
linear loss leaves traces effectively unsupervised); per-element
standardization is a smaller refinement that keeps the high-variance majors
from consuming the global gradient-clip budget.  `target_mean` / `target_std`
are the corpus log-frac center/scale built by
`cnmineral.training.pretrain_objective`.
"""

import torch

_LOG_FLOOR = 1e-12


def _reduce_masked(
    squared_error: torch.Tensor,
    mask_positions: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """Apply `mask_positions` and the chosen reduction to a `[B, N]` SE grid."""
    if reduction == "sum":
        return (squared_error * mask_positions.float()).sum()
    if reduction == "mean_per_masked_position":
        total_masked = mask_positions.sum().clamp(min=1)
        return (squared_error * mask_positions.float()).sum() / total_masked
    if reduction == "mean_per_sample":
        per_sample_sum = (squared_error * mask_positions.float()).sum(dim=1)
        per_sample_count = mask_positions.sum(dim=1).clamp(min=1).float()
        return (per_sample_sum / per_sample_count).mean()
    raise ValueError(f"unknown reduction: {reduction!r}")


def masked_element_log_mse(
    predicted_std_log: torch.Tensor,
    target_frac: torch.Tensor,
    mask_positions: torch.Tensor,
    *,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    reduction: str = "mean_per_masked_position",
) -> torch.Tensor:
    """Per-element standardized log-space MSE at masked positions.

    The head emits standardized `log10(frac)`; this builds the matching
    standardized-log target from `target_frac` and compares them at the
    masked positions.  The log space is what lifts the traces off the floor
    that linear-frac MSE pins them to; per-element standardization (via the
    head's `target_mean` / `target_std`) is a smaller refinement.  For a
    linear head each output is an independent affine map, so standardization
    barely changes R2 on its own -- it earns its place through the global
    gradient clip, where it stops the high-variance majors from consuming the
    clip budget and starving the low-variance heavy REE.

    Parameters
    ----------
    predicted_std_log
        `[B, N_TOKENS]` head output, `LogImputationHead.forward`.
    target_frac
        `[B, N_TOKENS]` original (pre-mask) frac, from
        `cnmineral.data.tokens.MaskedBatch.target_frac`.  Masked positions
        carry the value to predict; unmasked/padding positions are `0`.
    mask_positions
        `[B, N_TOKENS]` bool.  Only `True` positions contribute.
    target_mean, target_std
        `[N_TOKENS]` per-element `log10(frac)` center/scale (the head's
        buffers).  `target_std` is assumed already floored.

    Returns
    -------
    loss
        Scalar tensor.  `0` (zero gradient) when nothing is masked.

    Notes
    -----
    Unmasked/padding positions hold `frac=0`, whose `log10` is `-inf`;
    `-inf * 0` is `nan`, so the target is clamped to a tiny floor before the
    log.  Those positions are masked out of the reduction regardless -- the
    floor only keeps the masked-out arithmetic finite.
    """
    if predicted_std_log.shape != target_frac.shape:
        raise ValueError(
            f"predicted/target shape mismatch: "
            f"{predicted_std_log.shape} vs {target_frac.shape}"
        )
    if mask_positions.shape != target_frac.shape:
        raise ValueError(
            f"mask_positions shape {mask_positions.shape} does not match "
            f"target_frac shape {target_frac.shape}"
        )
    if mask_positions.dtype != torch.bool:
        raise TypeError(f"mask_positions must be bool, got {mask_positions.dtype}")

    log_target = torch.log10(target_frac.clamp_min(_LOG_FLOOR))
    std_log_target = (log_target - target_mean) / target_std
    squared_error = (predicted_std_log - std_log_target) ** 2
    return _reduce_masked(squared_error, mask_positions, reduction)
