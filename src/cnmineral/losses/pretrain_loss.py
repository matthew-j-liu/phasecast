"""
Masked-element MSE on the positions that were masked during the SSL
corruption step.  The predicted `frac` for non-masked positions is
ignored (no gradient through the loss)

`masked_element_mse` is linear-frac space (the SSL pretrain objective).
`masked_element_log_mse` is the per-element standardized log-space variant
used to train the frozen-encoder imputation head -- linear-frac MSE lets the
majors dominate and pins traces to a floor, which the log-space objective
fixes.
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


def masked_element_mse(
    predicted_frac: torch.Tensor,
    target_frac: torch.Tensor,
    mask_positions: torch.Tensor,
    reduction: str = "mean_per_masked_position",
) -> torch.Tensor:
    """
    Mean-squared error between predicted and target frac at masked positions.

    Parameters
    ----------
    predicted_frac
        `[B, N_TOKENS]` predicted frac, output of
        `cnmineral.models.pretrain_head.PretrainHead`.
    target_frac
        `[B, N_TOKENS]` original (pre-mask) frac, from
        `cnmineral.data.tokens.MaskedBatch.target_frac`.
    mask_positions
        `[B, N_TOKENS]` boolean.  `True` where the position was
        populated AND selected for masking.  Only these positions
        contribute to the loss.
    reduction
        - `"mean_per_masked_position"` (default): sum of squared errors
          divided by the total count of `True` in `mask_positions`.
          Standard MEM convention; per-batch loss scale is invariant to
          how many positions happened to be masked.
        - `"sum"`: raw sum of squared errors over masked positions.
        - `"mean_per_sample"`: per-sample mean over its masked positions,
          then mean across the batch.  Penalises samples with few masks
          equally to samples with many; rarely useful for MEM.

    Returns
    -------
    loss
        Scalar tensor.  If no positions are masked anywhere in the batch
        the loss is `0` (with gradient `0`); the SSL training loop
        should skip the optimizer step in that case rather than fail.
    """
    if predicted_frac.shape != target_frac.shape:
        raise ValueError(
            f"predicted/target shape mismatch: "
            f"{predicted_frac.shape} vs {target_frac.shape}"
        )
    if mask_positions.shape != target_frac.shape:
        raise ValueError(
            f"mask_positions shape {mask_positions.shape} does not match "
            f"target_frac shape {target_frac.shape}"
        )
    if mask_positions.dtype != torch.bool:
        raise TypeError(
            f"mask_positions must be bool, got {mask_positions.dtype}"
        )

    squared_error = (predicted_frac - target_frac) ** 2  # [B, N_TOKENS]
    return _reduce_masked(squared_error, mask_positions, reduction)


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
    masked positions.  Standardizing per element (via the head's
    `target_mean` / `target_std`) equalizes the majors' and traces'
    contributions to the shared head's gradient.

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
