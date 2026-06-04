"""V3 SSL pretrain loss.

Masked-element MSE on the positions that were masked during the SSL
corruption step.  The predicted `frac` for non-masked positions is
ignored (no gradient through the loss).

See `../crabnet-mineral-v2-legacy/src/cnmineral/models/pretrain_losses.py`
for the v2 reference implementation.  The v2 version had a per-token
gradient path; v3 uses the pooled-latent decoder so the loss shape is
unchanged but the reconstruction surface is different.
"""

import torch


def masked_element_mse(
    predicted_frac: torch.Tensor,
    target_frac: torch.Tensor,
    mask_positions: torch.Tensor,
    reduction: str = "mean_per_masked_position",
) -> torch.Tensor:
    """Mean-squared error between predicted and target frac at masked positions.

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

    if reduction == "sum":
        return (squared_error * mask_positions.float()).sum()

    if reduction == "mean_per_masked_position":
        total_masked = mask_positions.sum().clamp(min=1)
        return (squared_error * mask_positions.float()).sum() / total_masked

    if reduction == "mean_per_sample":
        per_sample_sum = (squared_error * mask_positions.float()).sum(dim=1)
        per_sample_count = mask_positions.sum(dim=1).clamp(min=1).float()
        per_sample_mean = per_sample_sum / per_sample_count
        return per_sample_mean.mean()

    raise ValueError(f"unknown reduction: {reduction!r}")
